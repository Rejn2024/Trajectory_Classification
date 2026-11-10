"""Frozen intent labels, joint gradients, single-fit notebook execution and reload."""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from bvr_behavior_prediction import nn_duel_skill_forecasting as duel
from bvr_behavior_prediction import paired_skill_forecasting as paired
from bvr_behavior_prediction import skill_intent_forecasting as intent
from bvr_behavior_prediction.enemy_missile_models import HGTIntentTransformer
from tests.test_nn_duel_skill_forecasting import write_source
from tests.test_hgt_missile_models import target_example

ROOT = Path(__file__).parents[1]
PREP = "26_prepare_skill_intent_target_relative_missile_nn_duel_skill_forecasting.ipynb"
TRAIN = "27_skill_intent_target_relative_missile_hgt_nn_duel_skill_forecasting.ipynb"
CLASSES = ["launch", "maintain_heading", "notch_left", "notch_right"]


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def prepare_base(tmp_path, monkeypatch):
    source = write_source(tmp_path, frames=181)
    # Preserve the collector's episode row groups while assigning real taxonomy skills.
    for side in duel.SIDES:
        path = source / side / "target_decisions/part-00000.parquet"
        original = pq.ParquetFile(path)
        groups = [original.read_row_group(i).to_pandas() for i in range(original.num_row_groups)]
        del original
        for group in groups:
            group["skill"] = [CLASSES[(int(step) // 10) % len(CLASSES)] for step in group.step]
        with pq.ParquetWriter(path, pa.Table.from_pandas(groups[0], preserve_index=False).schema) as writer:
            for group in groups:
                writer.write_table(pa.Table.from_pandas(group, preserve_index=False))
    manifest = duel.read_json(source / "manifest.json")
    for name in manifest["files"]:
        manifest["files"][name]["sha256"] = duel.sha256(source / name)
    duel.write_json(source / "manifest.json", manifest)
    monkeypatch.setattr(duel, "catalogue", lambda root: CLASSES)
    base = tmp_path / "base"
    duel.prepare_dataset(tmp_path, base, source, nonoverlapping_train=True, enemy_missiles=True,
                         observer_acceleration=True, target_relative_missiles=True)
    return base


def test_taxonomy_covers_the_real_catalogue_and_masks_missing_labels():
    classes = paired.catalogue(ROOT)
    policy = intent.intent_policy(classes)
    assert set(classes) == set(intent.SKILL_TO_INTENT) and len(classes) == 33
    assert [len(intent.INTENT_GROUPS[c]) for c in intent.INTENT_CLASSES] == [9, 11, 13]
    assert policy["mapping"]["preemptive_crank"] == "defensive"
    assert policy["mapping"]["crank_target_left"] == "context_dependent"
    assert policy["mapping"]["lock_target"] == "offensive"
    labels = np.array([[classes.index("launch"), classes.index("notch_left"), -1],
                        [-1, classes.index("maintain_heading"), classes.index("turn_cold")]])
    np.testing.assert_array_equal(intent.map_intent_targets(labels, policy["skill_to_intent"]),
                                  [[0, 1, -1], [-1, 2, 1]])
    with pytest.raises(ValueError, match="Unknown skills"):
        intent.intent_policy(["unknown_skill"])


def test_overlay_reuses_base_arrays_and_checks_taxonomy_labels_and_parent(tmp_path, monkeypatch):
    base = prepare_base(tmp_path, monkeypatch)
    before = paired.sha256(base / "manifest.json")
    destination = tmp_path / "intent"
    intent.prepare_intent_dataset(base, destination)
    data = intent.IntentNNDuelData(destination)
    original = duel.TargetRelativeMissileNNDuelData(base)
    assert data.frame_directory == base
    assert not (destination / "features.npy").exists()
    assert Path(data.features.filename) == base / "features.npy"
    np.testing.assert_array_equal(data.targets, original.targets)
    np.testing.assert_array_equal(data.anchors, original.anchors)
    np.testing.assert_array_equal(data.mean, original.mean)
    np.testing.assert_array_equal(data.std, original.std)
    pd.testing.assert_frame_equal(data.samples, original.samples)
    np.testing.assert_array_equal(data.targets < 0, data.intent_targets < 0)
    np.testing.assert_array_equal(data.targets < 0, data.family_targets < 0)
    assert (data.direction_targets[data.targets < 0] == -1).all()
    assert paired.sha256(base / "manifest.json") == before
    # Spawned prefetch workers must open the base feature files, not the small overlay directory.
    config = {**paired.DEFAULT_TRAINING, "data_backend": "prefetch", "num_workers": 2}
    ids = data.indices["train"][:3]
    with paired.BatchPipeline(data, "cpu", config) as pipeline:
        x, y = next(iter(pipeline.batches(ids, 3)))
    expected_x, expected_y = data.batch(ids, "cpu")
    torch.testing.assert_close(x, expected_x, atol=0, rtol=0)
    torch.testing.assert_close(y, expected_y)
    manifest = intent.read_json(destination / "manifest.json")
    labels = np.load(destination / "intent_targets.npy")
    labels[0, 0] = (labels[0, 0] + 1) % 3
    np.save(destination / "intent_targets.npy", labels)
    manifest["files"]["intent_targets.npy"] = paired.sha256(destination / "intent_targets.npy")
    intent.write_json(destination / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Intent labels"):
        intent.IntentNNDuelData(destination)
    manifest["intent_supervision"]["skill_to_intent"][0] = 2
    intent.write_json(destination / "manifest.json", manifest)
    with pytest.raises(ValueError, match="taxonomy"):
        intent.IntentNNDuelData(destination)
    parent_manifest = paired.read_json(base / "manifest.json")
    parent_manifest["changed"] = True
    paired.write_json(base / "manifest.json", parent_manifest)
    with pytest.raises(ValueError, match="Parent prepared dataset changed"):
        intent.IntentNNDuelData(destination)


@pytest.mark.parametrize("all_missing", [False, True])
def test_joint_loss_is_cross_entropy_with_exact_masks_and_both_head_gradients(all_missing):
    torch.manual_seed(4)
    skill = torch.randn(2, 3, 4, requires_grad=True)
    coarse = torch.randn(2, 3, 3, requires_grad=True)
    targets = torch.tensor([[0, -1, 2], [3, 1, -1]])
    if all_missing:
        targets[:] = -1
    mapping = torch.tensor([0, 1, 2, 0])
    weights = torch.tensor([[1., 2., 3., 4.]] * 3)
    expected_intent = torch.where(targets >= 0, mapping[targets.clamp_min(0)], -1)
    skill_reference = paired.masked_loss(skill, targets, weights)
    intent_reference = paired.masked_loss(coarse, expected_intent)
    actual = paired.masked_intent_loss(skill, coarse, targets, mapping, .25, weights)
    torch.testing.assert_close(actual, skill_reference + .25 * intent_reference)
    actual.backward()
    assert torch.isfinite(skill.grad).all() and torch.isfinite(coarse.grad).all()
    assert not skill.grad[targets < 0].any() and not coarse.grad[targets < 0].any()
    if not all_missing:
        assert skill.grad.abs().sum() > 0 and coarse.grad.abs().sum() > 0


@pytest.mark.parametrize("autocast", [False, True])
def test_shared_hgt_encoder_receives_auxiliary_gradient_and_default_forward_is_skills(autocast):
    raw, mean, std = target_example()
    model = HGTIntentTransformer(width=12, heads=6, layers=1, graph_width=8, graph_heads=2,
                                 graph_layers=1, dropout=0., mean=mean, std=std,
                                 observer_acceleration=True, target_relative_missiles=True,
                                 target_relative_missile_velocities=True).eval()
    x = (raw - mean) / std
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        skill, coarse = model(x, return_intent=True)
    assert skill.shape == (2, 3, 33) and coarse.shape == (2, 3, 3)
    if not autocast:
        torch.testing.assert_close(model(x), skill, atol=0, rtol=0)
    paired.masked_loss(coarse, torch.zeros(2, 3, dtype=torch.long)).backward()
    assert model.intent_head[-1].weight.grad.abs().sum() > 0
    assert model.encoder.layers[0].linear1.weight.grad.abs().sum() > 0
    assert model.messages[0].edge_attention.weight.grad.abs().sum() > 0
    assert model.head[-1].weight.grad is None  # Intent supervision reaches the shared trunk directly.
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_both_notebooks_prepare_one_fit_evaluate_reload_and_report(tmp_path, monkeypatch):
    import IPython.display
    import nbformat

    base = prepare_base(tmp_path, monkeypatch)
    package = tmp_path / "bvr_behavior_prediction"
    package.mkdir()
    for name in ("nn_duel_skill_forecasting.py", "paired_skill_forecasting.py"):
        shutil.copy2(ROOT / "bvr_behavior_prediction" / name, package / name)
    monkeypatch.setenv("BVR_NN_ROOT", str(tmp_path))
    monkeypatch.setenv("BVR_NN_TARGET_RELATIVE_MISSILE_DATA", str(base))
    monkeypatch.setenv("BVR_NN_INTENT_MLFLOW", "0")
    monkeypatch.delenv("BVR_NN_INTENT_DATA", raising=False)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(IPython.display, "display", lambda *args: None)
    original_train = paired.train_model
    fit_count = []

    def count_fit(*args, **kwargs):
        fit_count.append(kwargs.get("resume", False))
        return original_train(*args, **kwargs)

    monkeypatch.setattr(paired, "train_model", count_fit)
    namespace = {}
    for name in (PREP, TRAIN):
        nb = nbformat.read(ROOT / "notebooks" / name, as_version=4)
        nbformat.validate(nb)
        assert len({c.id for c in nb.cells}) == len(nb.cells)
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            assert cell.execution_count is None and not cell.outputs and cell.source.startswith("%%time\n")
            exec(compile(cell.source.removeprefix("%%time\n"), name, "exec"), namespace)
            if name == TRAIN and i == 1:
                assert namespace["SEED"] == 42
                assert "TRAINING_SEEDS" not in namespace and "RUNS" not in namespace
                assert namespace["CONFIG"]["intent_loss_weight"] == .25
                assert namespace["CONFIG"]["architecture"] == "hgt_hierarchical_intent"
                assert namespace["CONFIG"]["hierarchical_loss"] == "partial_credit"
                assert namespace["CONFIG"]["family_reward"] == namespace["CONFIG"]["handedness_reward"] == 1.
                assert namespace["CONFIG"]["class_weighting"] == "family_sqrt_inverse"
                assert namespace["CONFIG"]["handedness_weighting"] == "sqrt_inverse"
                assert namespace["CONFIG"]["epoch_streaming_metrics"] is True
                assert namespace["CONFIG"]["evaluation_frame_cache"] is True
                namespace["CONFIG"]["epochs"] = 2
    assert fit_count == [False]
    data, run = namespace["data"], namespace["RUN"]
    assert len(list((namespace["EXPERIMENT_ROOT"] / "runs").iterdir())) == 1
    assert (run / "intent_diagnostics/manifest.json").is_file()
    assert (run / "training_frequency_intent_baseline.csv").is_file()
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    assert checkpoint["intent_supervision"] == data.manifest["intent_supervision"]
    assert checkpoint["skill_hierarchy"] == data.manifest["skill_hierarchy"]
    assert checkpoint["hierarchical_objective"] == {"hierarchical_loss": "partial_credit",
                                                     "family_reward": 1., "handedness_reward": 1.}
    assert checkpoint["scheduler_state"] is not None
    assert (run / "evaluation_source.py").is_file()
    assert checkpoint["evaluation_source_sha256"] == paired.sha256(run / "evaluation_source.py")
    assert checkpoint["handedness_weighting"] == namespace["HANDEDNESS_WEIGHTING"]
    assert checkpoint["class_weighting"]["weighting_level"] == "skill_family"
    assert (run / "handedness_weights.json").is_file() and (run / "handedness_weights.csv").is_file()
    restored = paired.make_model(len(data.classes), checkpoint["model_config"], checkpoint["mean"], checkpoint["std"])
    restored.load_state_dict(checkpoint["model_state"])
    restored.eval()
    x, _ = data.batch(data.indices["validation"], "cpu")
    with torch.inference_mode(), paired.evaluation_context("cpu", "math"):
        skills, groups = restored(x, return_intent=True)
        hierarchy = restored(x, return_hierarchy=True)
    for filename, logits in (("final_validation.npz", skills), ("final_intent_validation.npz", groups)):
        with np.load(run / filename) as stored:
            np.testing.assert_array_equal(stored["probabilities"], logits.softmax(-1).numpy())
    with np.load(run / "final_family_validation.npz") as stored:
        np.testing.assert_array_equal(stored["probabilities"], hierarchy["family_logits"].softmax(-1).numpy())
        np.testing.assert_array_equal(stored["targets"], data.family_targets[data.indices["validation"]])
    with np.load(run / "final_direction_validation.npz") as stored:
        np.testing.assert_array_equal(stored["targets"], data.direction_targets[data.indices["validation"]])
    receipt = intent.read_json(namespace["EXPERIMENT_ROOT"] / "comparisons/latest_hgt_hierarchical_intent_run.json")
    assert receipt["seed"] == 42 and receipt["run"] == str(run)
    assert (run / "final_hierarchical_credit_metrics.csv").is_file()
    history = pd.read_csv(run / "history.csv")
    np.testing.assert_allclose(history.train_data_loss,
                               history.train_skill_data_loss + .25 * history.train_intent_nll)
    np.testing.assert_allclose(history.train_skill_data_loss,
                               history.train_family_data_loss + history.train_handedness_data_loss)
    assert checkpoint["test_nll"] == pytest.approx(history.test_nll.min())
    assert ((history.test_hierarchical_credit >= 0) & (history.test_hierarchical_credit <= 1)).all()
    evaluated = paired.evaluate(restored, data, "validation", "cpu", intent_loss_weight=.25,
                                 **checkpoint["hierarchical_objective"])
    assert evaluated["skill_data_loss"] < evaluated["joint_skill_data_loss"]
    balanced = paired.evaluate(restored, data, "validation", "cpu", intent_loss_weight=.25,
        handedness_weights=paired.handedness_weight_tensor(checkpoint["handedness_weighting"], "cpu"),
        **checkpoint["hierarchical_objective"])
    pd.testing.assert_frame_equal(evaluated["summary"], balanced["summary"])
    pd.testing.assert_frame_equal(evaluated["hierarchical_reward"], balanced["hierarchical_reward"])
    # Reuse and report-only cells must not fit another seed or rewrite the prepared data.
    before = paired.sha256(data.directory / "manifest.json")
    prep = nbformat.read(ROOT / "notebooks" / PREP, as_version=4)
    for i in (1, 3, 5):
        exec(compile(prep.cells[i].source.removeprefix("%%time\n"), "reuse-intent-data", "exec"), namespace)
    assert paired.sha256(data.directory / "manifest.json") == before
    namespace.update(TRAIN_MODEL=False, EXISTING_RUN=run, RUN=run)
    train = nbformat.read(ROOT / "notebooks" / TRAIN, as_version=4)
    for i in (3, 5):
        exec(compile(train.cells[i].source.removeprefix("%%time\n"), "report-only-intent", "exec"), namespace)
    assert fit_count == [False]
    resumed = original_train(data, run, namespace["CONFIG"], device="cpu", resume=True)
    assert len(resumed) == len(history)  # Completed resume performs no extra optimizer updates.
    last = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    last["handedness_weighting"]["weights"][0][0][0] += .1
    torch.save(last, run / "last.pt")
    with pytest.raises(ValueError, match="original training-fitted handedness weights"):
        original_train(data, run, namespace["CONFIG"], device="cpu", resume=True)
