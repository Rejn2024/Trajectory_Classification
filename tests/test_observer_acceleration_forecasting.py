"""Causal observer acceleration, the replacement schema and notebook 22/23 execution."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from bvr_behavior_prediction import nn_duel_skill_forecasting as duel
from bvr_behavior_prediction import paired_skill_forecasting as paired
from bvr_behavior_prediction.enemy_missile_models import TypedGraphTransformer
from tests.test_enemy_missile_models import example
from tests.test_nn_duel_skill_forecasting import CLASSES, write_source


ROOT = Path(__file__).parents[1]
PREP = "22_prepare_observer_acceleration_enemy_missile_nn_duel_skill_forecasting.ipynb"
TRAIN = "23_observer_acceleration_enemy_missile_graph_nn_duel_skill_forecasting.ipynb"


def test_observer_acceleration_matches_opponent_difference_and_is_past_only():
    raw = np.zeros((21, 51), np.float32)
    time = np.arange(len(raw)) * .1
    raw[:, 3:6] = (time[:, None] ** 2 * [2., -3., 4.]) / 600
    acceleration = duel.observer_acceleration_features(raw)
    # With zero relative velocity the opponent has the same world acceleration.
    np.testing.assert_array_equal(acceleration, paired.observation_features(raw)[:, 51:54])
    previous = np.maximum(0, np.arange(len(raw)) - 5)
    elapsed = np.maximum(1, np.arange(len(raw)) - previous) * .1
    expected = (time ** 2 - time[previous] ** 2) / elapsed
    np.testing.assert_allclose(acceleration, expected[:, None] * [2., -3., 4.] / 100,
                               rtol=1e-6, atol=1e-8)
    assert not acceleration[0].any()
    changed = raw.copy()
    changed[11:, 3:6] += 100
    changed[:, :3] = 999
    np.testing.assert_array_equal(duel.observer_acceleration_features(changed)[:11], acceleration[:11])


def test_preparation_removes_only_positions_adds_acceleration_and_preserves_protocol(tmp_path, monkeypatch):
    from tests import test_nn_duel_skill_forecasting as fixture

    original_frames = fixture.frames_and_decisions

    def accelerating_frames(*args, **kwargs):
        observations, tracks, decisions = original_frames(*args, **kwargs)
        time = observations.time_s.to_numpy()[:, None]
        observations[["obs_03", "obs_04", "obs_05"]] = (time**2 * [2, -3, 4]) / 600
        return observations, tracks, decisions

    monkeypatch.setattr(fixture, "frames_and_decisions", accelerating_frames)
    monkeypatch.setattr(duel, "catalogue", lambda root: CLASSES)
    source = write_source(tmp_path, frames=181)
    analysis = duel.analyze_corpus(tmp_path, source)
    old_path, new_path = tmp_path / "old", tmp_path / "new"
    for path, use_acceleration in ((old_path, False), (new_path, True)):
        duel.prepare_dataset(tmp_path, path, source, analysis=analysis,
                             nonoverlapping_train=True, enemy_missiles=True,
                             observer_acceleration=use_acceleration)
    old = duel.EnemyMissileNNDuelData(old_path)
    new = duel.ObserverAccelerationEnemyMissileNNDuelData(new_path)
    assert new.features.shape == old.features.shape
    assert new.manifest["features"][:48] == old.manifest["features"][3:51]
    assert new.manifest["features"][48:51] == old.manifest["features"][51:54]
    assert new.manifest["features"][51:54] == duel.OBSERVER_ACCELERATION_FEATURES
    assert not set(duel.OBSERVER_ACCELERATION_POLICY["replaces"]) & set(new.manifest["features"])
    np.testing.assert_array_equal(new.features[:, :51], old.features[:, 3:54])
    np.testing.assert_array_equal(new.features[:, 54:], old.features[:, 54:])
    expected_acceleration = np.concatenate([
        duel.observer_acceleration_features(old.features[start:start + 181, :51])
        for start in range(0, len(old.features), 181)])
    np.testing.assert_array_equal(new.features[:, 51:54], expected_acceleration)
    assert np.any(new.features[:, 51:54] != 0)
    # A derivative must restart separately at each physical-flight/view boundary.
    assert not new.features[::181, 51:54].any()
    np.testing.assert_array_equal(new.targets, old.targets)
    np.testing.assert_array_equal(new.anchors, old.anchors)
    pd.testing.assert_frame_equal(new.samples, old.samples)
    for name in ("scenario_splits.json", "nonoverlap_audit.json"):
        assert duel.read_json(old_path / name) == duel.read_json(new_path / name)
    for name in ("enemy_missile_audit.csv", "skill_support_by_split.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(old_path / name), pd.read_csv(new_path / name))
    np.testing.assert_array_equal(new.mean[:51], old.mean[3:54])
    np.testing.assert_array_equal(new.std[:51], old.std[3:54])
    np.testing.assert_array_equal(new.mean[54:], old.mean[54:])
    np.testing.assert_array_equal(new.std[54:], old.std[54:])
    selected_frames = np.concatenate([
        np.arange(end - 50, end + 1) for end in new.anchors[new.indices["train"]]])
    values = new.features[selected_frames, 51:54].astype(float)
    np.testing.assert_allclose(new.mean[51:54], values.mean(0), rtol=1e-6)
    np.testing.assert_allclose(new.std[51:54], np.maximum(values.std(0), .01), rtol=1e-6)
    for loader, path in ((duel.EnemyMissileNNDuelData, new_path),
                         (duel.ObserverAccelerationEnemyMissileNNDuelData, old_path)):
        with pytest.raises(ValueError, match="Expected own-skill-conditioned"):
            loader(path)
    with pytest.raises(ValueError, match="observer_acceleration"):
        duel.prepare_dataset(tmp_path, tmp_path / "invalid", observer_acceleration=True)
    new.manifest["observer_acceleration"]["difference_steps"] = 6
    duel.write_json(new_path / "manifest.json", new.manifest)
    with pytest.raises(ValueError, match="causal observer-acceleration policy"):
        duel.ObserverAccelerationEnemyMissileNNDuelData(new_path)


def test_graph_remapping_preserves_geometry_masks_and_consumes_observer_acceleration():
    torch.set_num_threads(1)
    raw, mean, std = example()
    settings = dict(width=12, heads=6, layers=1, dropout=0, graph_width=8, graph_layers=2)
    old = TypedGraphTransformer(**settings, mean=mean, std=std).eval()
    # Give the new acceleration channels the old position values solely to check
    # exact decoder equivalence after permuting observer-projection columns.
    order = [*range(3, 54), 0, 1, 2, *range(54, 115)]
    new = TypedGraphTransformer(**settings, mean=mean[order], std=std[order],
                                observer_acceleration=True).eval()
    state = old.state_dict()
    state["feature_mean"], state["feature_std"] = mean[order], std[order]
    observer_order = [*range(3, 12), 0, 1, 2, *range(12, 45)]
    state["observer.0.weight"] = state["observer.0.weight"][:, observer_order]
    new.load_state_dict(state)
    x = ((raw[:, :, order] - mean[order]) / std[order]).requires_grad_()
    before = old((raw - mean) / std)
    after = new(x)
    torch.testing.assert_close(before, after, atol=2e-6, rtol=2e-6)
    after.square().mean().backward()
    assert x.grad[:, :, 51:54].abs().sum() > 0
    assert torch.isfinite(x.grad).all()
    _, _, _, present = new._decode(x)
    assert present[:, :2].all()
    assert not present[:, 3].any()  # Original absent friendly slot stays absent.


@pytest.mark.parametrize("target_relative", [False, True])
def test_new_notebooks_prepare_train_reload_and_report(tmp_path, monkeypatch, target_relative):
    """Execute the real notebook cells, shortening only the epoch budget."""
    import shutil
    import IPython.display
    import nbformat

    prep_name = ("24_prepare_target_relative_missile_nn_duel_skill_forecasting.ipynb"
                 if target_relative else PREP)
    train_name = ("25_target_relative_missile_graph_nn_duel_skill_forecasting.ipynb"
                  if target_relative else TRAIN)
    prefix = "BVR_NN_TARGET_RELATIVE_MISSILE" if target_relative else "BVR_NN_OBSERVER_ACCELERATION"
    contract = (duel.TARGET_RELATIVE_MISSILE_DATA_CONTRACT if target_relative
                else duel.OBSERVER_ACCELERATION_DATA_CONTRACT)
    monkeypatch.setattr(duel, "catalogue", lambda root: CLASSES)
    source = write_source(tmp_path, frames=181)
    package = tmp_path / "bvr_behavior_prediction"
    package.mkdir()
    for name in ("nn_duel_skill_forecasting.py", "paired_skill_forecasting.py"):
        shutil.copy2(ROOT / "bvr_behavior_prediction" / name, package / name)
    monkeypatch.setenv("BVR_NN_ROOT", str(tmp_path))
    monkeypatch.setenv("BVR_NN_FORECAST_SOURCE", str(source))
    monkeypatch.setenv("BVR_NN_FORECAST_DOWNLOAD", "0")
    monkeypatch.setenv(f"{prefix}_MLFLOW", "0")
    monkeypatch.delenv(f"{prefix}_DATA", raising=False)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(IPython.display, "display", lambda *args: None)
    namespace = {}
    for name in (prep_name, train_name):
        notebook = nbformat.read(ROOT / "notebooks" / name, as_version=4)
        nbformat.validate(notebook)
        ids = [cell.id for cell in notebook.cells]
        assert len(ids) == len(set(ids))
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            assert cell.source.startswith("%%time\n")
            assert cell.execution_count is None and cell.outputs == []
            exec(compile(cell.source.removeprefix("%%time\n"), name, "exec"), namespace)
            if name == train_name and index == 1:
                config = namespace["CONFIG"]
                assert config["observer_acceleration"] is True
                assert config["architecture"] == ("hgt" if target_relative else "typed_graph")
                if target_relative:
                    assert config["graph_heads"] == 4
                    assert config["missile_distance_floor_m"] == 1000.0
                    assert config["missile_distance_power"] == 1.0
                    assert config["hgt_version"] == 2
                    assert config["lr_schedule"] == "warmup_cosine"
                    assert config["warmup_fraction"] == .05
                    assert config["early_stopping_min_epochs"] == config["overtraining_min_epochs"] == 20
                assert config["width"] == 96 and config["graph_width"] == 64
                assert config.get("target_relative_missiles", False) is target_relative
                assert config.get("target_relative_missile_velocities", False) is target_relative
                config["epochs"] = 1
    data = namespace["data"]
    assert isinstance(data, duel.ObserverAccelerationEnemyMissileNNDuelData)
    assert len(namespace["RUNS"]) == 3
    experiment = namespace["EXPERIMENT_ROOT"]
    assert experiment.name == ("nn_duel_target_relative_missile_forecasting" if target_relative
                               else "nn_duel_observer_acceleration_enemy_missile_forecasting")
    for run in namespace["RUNS"].values():
        saved = duel.read_json(run / "run.json")
        checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
        assert checkpoint["data_contract"] == contract
        assert checkpoint["model_config"]["observer_acceleration"] is True
        assert checkpoint["model_config"].get("target_relative_missiles", False) is target_relative
        assert checkpoint["model_config"].get("target_relative_missile_velocities", False) is target_relative
        if target_relative:
            assert checkpoint["model_config"]["architecture"] == "hgt"
            assert checkpoint["model_config"]["graph_heads"] == 4
            assert checkpoint["model_config"]["missile_distance_floor_m"] == 1000.0
            assert checkpoint["model_config"]["missile_distance_power"] == 1.0
            assert checkpoint["model_config"]["hgt_version"] == 2
            assert checkpoint["model_config"]["threat_cpa_horizon_s"] == 30.
            assert checkpoint["scheduler_state"]["last_epoch"] == checkpoint["optimizer_steps"]
            assert checkpoint["learning_rate_schedule"]["name"] == "warmup_cosine"
        restored = paired.make_model(len(data.classes), checkpoint["model_config"],
                                     checkpoint["mean"], checkpoint["std"])
        restored.load_state_dict(checkpoint["model_state"])
        restored.eval()
        x, _ = data.batch(data.indices["validation"], "cpu")
        with torch.inference_mode(), paired.evaluation_context("cpu", "math"):
            expected = restored(x).softmax(-1).numpy()
        with np.load(run / "final_validation.npz") as result:
            np.testing.assert_array_equal(result["probabilities"], expected)
        samples = pd.read_csv(run / "diagnostics/samples.csv")
        origins = data.features[data.anchors[samples.sample_index]]
        np.testing.assert_array_equal(samples.friendly_missiles_present,
                                      np.any(origins[:, 16:44] != 0, axis=1))
        np.testing.assert_array_equal(samples.enemy_missiles_present,
                                      np.any(origins[:, 54:82] != 0, axis=1))
        assert saved["feature_names"] == data.manifest["features"]
        assert (run / "diagnostics/manifest.json").is_file()
        assert (run / "training_frequency_baseline.csv").is_file()
    # Schema identity, not input width, prevents old-layout training on new data.
    with pytest.raises(ValueError, match="matching notebook 17 or 22"):
        paired.train_model(data, tmp_path / "wrong-layout", {
            **namespace["CONFIG"], "observer_acceleration": False,
        }, device="cpu")
    architecture = "hgt" if target_relative else "typed_graph"
    receipt = duel.read_json(experiment / f"comparisons/latest_{architecture}_runs.json")
    assert receipt["data_manifest_sha256"] == duel.sha256(data.directory / "manifest.json")
    assert not (tmp_path / "artifacts/nn_duel_enemy_missile_forecasting").exists()
    # Completed preparation can be verified and reused without rewriting files.
    prep = nbformat.read(ROOT / "notebooks" / prep_name, as_version=4)
    before = duel.sha256(data.directory / "manifest.json")
    for index in (1, 4, 6):
        exec(compile(prep.cells[index].source.removeprefix("%%time\n"), "reuse", "exec"), namespace)
    assert duel.sha256(data.directory / "manifest.json") == before
