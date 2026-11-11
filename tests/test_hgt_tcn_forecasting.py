"""Causal temporal replacement, hierarchy gradients, checkpoint reuse and notebook 28."""

import copy
import json
import shutil
import sys

import nbformat
import numpy as np
import pandas as pd
import pytest
import torch

from bvr_behavior_prediction import paired_skill_forecasting as paired
from bvr_behavior_prediction import skill_intent_forecasting as intent
from bvr_behavior_prediction.enemy_missile_models import CausalTemporalConvEncoder
from tests.test_forecasting_evaluation import assert_equivalent, example
from tests.test_hgt_missile_models import target_example
from tests.test_skill_intent_forecasting import ROOT, prepare_base


NOTEBOOK = "28_skill_intent_target_relative_missile_hgt_tcn_nn_duel_skill_forecasting.ipynb"


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def configuration(classes):
    return paired.model_configuration(dict(
        architecture="hgt_hierarchical_intent_tcn", width=12, dropout=0.,
        tcn_kernel_size=3, tcn_dilations=[1, 2, 4, 8], graph_width=8,
        graph_heads=2, graph_layers=1, observer_acceleration=True,
        target_relative_missiles=True, target_relative_missile_velocities=True),
        82 + len(classes), intent.hierarchy_policy(classes))


def test_temporal_convolutions_are_causal_and_last_output_reaches_first_frame():
    torch.manual_seed(4)
    encoder = CausalTemporalConvEncoder(12, dropout=0).eval()
    frames = torch.randn(2, 51, 12, requires_grad=True)
    changed = frames.detach().clone()
    changed[:, 21:] = torch.randn_like(changed[:, 21:]) * 100
    output = encoder(frames)
    assert output.shape == frames.shape and encoder.receptive_field == 61
    torch.testing.assert_close(output[:, :21], encoder(changed)[:, :21], atol=0, rtol=0)
    output[:, -1].square().sum().backward()
    assert frames.grad[:, 0].abs().sum() > 0
    assert torch.isfinite(frames.grad).all()


@pytest.mark.parametrize("settings", [
    {"tcn_kernel_size": True}, {"tcn_kernel_size": 1}, {"tcn_kernel_size": 2.5},
    {"tcn_dilations": []}, {"tcn_dilations": [1, 0]},
    {"tcn_dilations": [True]}, {"tcn_dilations": [1, 2.5]},
    {"tcn_dilations": [1, 2]},  # Valid dilation values but insufficient history coverage.
])
def test_invalid_tcn_settings_fail_before_training(settings):
    config = configuration(["launch", "notch_left", "notch_right"])
    with pytest.raises(ValueError, match="tcn_|TCN receptive field"):
        paired.make_model(3, {**config, **settings})


@pytest.mark.parametrize("autocast", [False, True])
def test_hgt_tcn_heads_gradients_and_factory_checkpoint_reload(autocast):
    classes = paired.catalogue(ROOT)
    settings = configuration(classes)
    assert not {"heads", "layers"} & settings.keys()
    raw, mean, std = target_example()
    model = paired.make_model(33, settings, mean, std).eval()
    assert not hasattr(model, "position")
    assert not any(isinstance(m, (torch.nn.TransformerEncoder, torch.nn.MultiheadAttention))
                   for m in model.modules())
    x = (raw - mean) / std
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        outputs = model(x, return_hierarchy=True)
    assert outputs["family_logits"].shape == (2, 3, 29)
    assert outputs["direction_logits"].shape == (2, 3, 4, 2)
    assert outputs["intent_logits"].shape == (2, 3, 3)
    torch.testing.assert_close(outputs["skill_log_probs"].exp().sum(-1), torch.ones(2, 3))
    labels = torch.full((2, 3), classes.index("notch_right"), dtype=torch.long)
    loss = paired.masked_hierarchical_skill_loss(
        outputs["family_logits"], outputs["direction_logits"], labels,
        paired.hierarchy_mapping_tensors(model.skill_hierarchy, "cpu"), hierarchical_loss="partial_credit")
    loss = loss + .25 * paired.masked_loss(outputs["intent_logits"], torch.ones_like(labels))
    loss.backward()
    for parameter in (model.family_head[-1].weight, model.direction_head[-1].weight,
                      model.intent_head[-1].weight, model.encoder.blocks[0].convolutions[0].weight,
                      model.messages[0].edge_attention.weight):
        assert parameter.grad.abs().sum() > 0 and torch.isfinite(parameter.grad).all()
    restored = paired.make_model(33, json.loads(json.dumps(settings)), mean, std).eval()
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    with torch.inference_mode():
        torch.testing.assert_close(restored(x), model(x), atol=0, rtol=0)
        skill, coarse = restored(x, return_intent=True)
    assert skill.shape == (2, 3, 33) and coarse.shape == (2, 3, 3)


def test_tcn_cached_streaming_and_full_metrics_agree():
    data, _ = example()
    model = paired.make_model(len(data.classes), configuration(data.classes), data.mean, data.std)
    options = dict(intent_loss_weight=.25, hierarchical_loss="partial_credit", evaluation_attention="math")
    baseline = paired.evaluate(model, data, "test", "cpu", batch_size=3, **options)
    cached = paired.evaluate(model, data, "test", "cpu", batch_size=2, frame_cache=True,
                             cache_max_frames=153, frame_batch_size=17, **options)
    streamed = paired.evaluate(model, data, "test", "cpu", batch_size=4, frame_cache=True,
                               streaming=True, cache_max_frames=153, frame_batch_size=17, **options)
    assert_equivalent(baseline, cached, full=True)
    assert_equivalent(baseline, streamed)


@pytest.mark.skipif(not hasattr(torch.export, "Dim"), reason="Dynamic export requires newer PyTorch")
def test_tcn_dynamic_export_retains_batch_dimension():
    model = paired.make_model(3, configuration(["launch", "notch_left", "notch_right"])).eval()
    windows = torch.zeros(2, 51, 85)
    program = torch.export.export(model, (windows,), dynamic_shapes=(
        {0: torch.export.Dim("batch", min=1, max=4)},))
    with torch.inference_mode():
        torch.testing.assert_close(program.module()(windows[:1]), model(windows[:1]))


def test_notebook28_trains_reports_reloads_and_reuses_one_tcn_run(tmp_path, monkeypatch):
    import IPython.display

    base = prepare_base(tmp_path, monkeypatch)
    overlay = tmp_path / "intent"
    intent.prepare_intent_dataset(base, overlay)
    package = tmp_path / "bvr_behavior_prediction"
    package.mkdir()
    shutil.copy2(ROOT / "bvr_behavior_prediction/nn_duel_skill_forecasting.py", package)
    monkeypatch.setenv("BVR_NN_ROOT", str(tmp_path))
    monkeypatch.setenv("BVR_NN_INTENT_DATA", str(overlay))
    monkeypatch.setenv("BVR_NN_INTENT_TCN_MLFLOW", "0")
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(IPython.display, "display", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    notebook = nbformat.read(ROOT / "notebooks" / NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    namespace, fit_count = {}, []
    original_train = paired.train_model

    def train(*args, **kwargs):
        fit_count.append(True)
        return original_train(*args, **kwargs)

    monkeypatch.setattr(paired, "train_model", train)
    before = paired.sha256(overlay / "manifest.json")
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        assert cell.execution_count is None and not cell.outputs
        assert cell.source.startswith("%%time\n")
        exec(compile(cell.source.removeprefix("%%time\n"), NOTEBOOK, "exec"), namespace)
        if i == 1:
            config = namespace["CONFIG"]
            assert config["architecture"] == "hgt_hierarchical_intent_tcn"
            assert config["tcn_dilations"] == [1, 2, 4, 8]
            assert config["width"] == 96 and namespace["TCN_RECEPTIVE_FIELD"] == 61
            assert config["max_class_weight_ratio"] == 100 and config["max_handedness_weight_ratio"] == 20
            assert namespace["EXPERIMENT_ROOT"].name == "nn_duel_skill_intent_tcn_forecasting"
            config.update(epochs=2, width=12, graph_width=8, graph_heads=2, graph_layers=1,
                          batch_size=8, evaluation_batch_size=32, cpu_threads=1,
                          data_backend="direct", num_workers=0)
    assert fit_count == [True] and paired.sha256(overlay / "manifest.json") == before
    data, run, config = namespace["data"], namespace["RUN"], namespace["CONFIG"]
    saved = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    assert saved["model_config"]["architecture"] == "hgt_hierarchical_intent_tcn"
    assert saved["model_config"]["tcn_dilations"] == [1, 2, 4, 8]
    assert not any(k.startswith("position") for k in saved["model_state"])
    assert saved["skill_hierarchy"] == data.manifest["skill_hierarchy"]
    assert saved["handedness_weighting"] == namespace["HANDEDNESS_WEIGHTING"]
    for task in ("", "intent_", "family_", "direction_", "hierarchical_credit_"):
        table = pd.read_csv(run / f"final_{task}metrics.csv")
        assert len(table) == 9 and set(table.split) == set(paired.SPLITS)
    for relative in ("training_frequency_baseline.csv", "training_frequency_intent_baseline.csv",
                     "intent_diagnostics/summary.csv", "diagnostics/learning_curves.png"):
        assert (run / relative).is_file()
    receipt = namespace["EXPERIMENT_ROOT"] / "comparisons/latest_hgt_hierarchical_intent_tcn_run.json"
    assert paired.read_json(receipt)["architecture"] == "hgt_hierarchical_intent_tcn"
    restored = paired.make_model(len(data.classes), saved["model_config"], saved["mean"], saved["std"]).eval()
    restored.load_state_dict(saved["model_state"])
    x, _ = data.batch(data.indices["validation"], "cpu")
    with torch.inference_mode(), np.load(run / "final_validation.npz") as predictions:
        np.testing.assert_allclose(restored(x).softmax(-1).numpy(), predictions["probabilities"], atol=2e-6)
    history = pd.read_csv(run / "history.csv")
    assert saved["test_nll"] == pytest.approx(history.test_nll.min())
    np.testing.assert_allclose(history.train_data_loss,
                               history.train_skill_data_loss + .25 * history.train_intent_nll)
    namespace["TRAIN_MODEL"] = False
    for i in (3, 5):
        exec(compile(notebook.cells[i].source.removeprefix("%%time\n"), NOTEBOOK, "exec"), namespace)
    assert fit_count == [True]
    resumed = original_train(data, run, config, device="cpu", resume=True)
    assert len(resumed) == 2
    with pytest.raises(ValueError, match="[Rr]esume"):
        original_train(data, run, {**config, "tcn_dilations": [1, 2, 4, 16]}, device="cpu", resume=True)
