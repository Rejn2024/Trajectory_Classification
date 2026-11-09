"""The 09b training path also accepts observer-skill feature channels."""

import numpy as np
import pandas as pd
import pytest
import torch

from bvr_behavior_prediction import paired_skill_forecasting as paired


def prepared_data(tmp_path):
    directory = tmp_path / "data"
    directory.mkdir()
    names = [*paired.FEATURES, "observer_skill_alpha", "observer_skill_beta"]
    rng = np.random.default_rng(17)
    features = rng.normal(size=(183, len(names))).astype(np.float32)
    features[:, -2:] = np.eye(2, dtype=np.float32)[np.arange(len(features)) % 2]
    np.save(directory / "features.npy", features)
    np.save(directory / "anchors.npy", np.array([50, 60, 111, 121, 172, 182]))
    np.save(directory / "targets.npy", np.array([[0, 1, -1], [1, 0, 1]] * 3))
    pd.DataFrame({
        "episode_id": np.repeat(["e_train", "e_test", "e_validation"], 2),
        "scenario_group": np.repeat(["g_train", "g_test", "g_validation"], 2),
        "split": np.repeat(paired.SPLITS, 2),
    }).to_parquet(directory / "samples.parquet", index=False)
    paired.write_json(directory / "scaling.json", {
        "mean": [0.0] * len(names), "std": [1.0] * len(names),
    })
    manifest = {
        "classes": ["alpha", "beta"], "features": names,
        "data_contract": "observer-skills-test-v1",
        "input_contract": "Observations plus the observer's current skill",
        "target_contract": "Opposing aircraft future selections",
        "files": {p.name: paired.sha256(p) for p in directory.iterdir()},
    }
    paired.write_json(directory / "manifest.json", manifest)
    return paired.PairedData(directory)


@pytest.mark.parametrize("l1_lambda", [0.0, 1e-6])
def test_expanded_features_train_resume_reload_and_benchmark(tmp_path, l1_lambda):
    data = prepared_data(tmp_path)
    output = tmp_path / "run"
    config = {
        "epochs": 2, "width": 12, "heads": 6, "layers": 1, "dropout": 0.0,
        "batch_size": 2, "evaluation_batch_size": 2, "cpu_threads": 1,
        "precision": "fp32", "data_backend": "direct",
        "l1_lambda": l1_lambda, "overtraining_shutdown": True,
        "early_stopping": True, "early_stopping_min_epochs": 0,
    }
    history = paired.train_model(data, output, config, device="cpu")
    if l1_lambda:
        assert (history.train_l1_penalty > 0).all()
    else:
        assert (history.train_l1_penalty == 0).all()
    np.testing.assert_allclose(history.train_objective, history.train_nll + history.train_l1_penalty)
    saved = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    assert saved["model_config"]["input_features"] == len(data.manifest["features"])
    assert saved["feature_names"] == data.manifest["features"]
    assert saved["data_contract"] == data.manifest["data_contract"]
    run = paired.read_json(output / "run.json")
    for field in ("data_contract", "input_contract", "target_contract"):
        assert run[field] == data.manifest[field]
    model = paired.PairedTransformer(num_classes=2, **saved["model_config"])
    model.load_state_dict(saved["model_state"])
    model.eval()
    windows, _ = data.batch(data.indices["validation"], "cpu")
    with torch.inference_mode():
        expected = model(windows).softmax(-1).numpy()
    _, _, _, results = paired.final_evaluation(data, output, device="cpu")
    np.testing.assert_allclose(results["validation"]["probabilities"], expected)
    resumed = paired.train_model(data, output, config, device="cpu", resume=True)
    pd.testing.assert_frame_equal(history, resumed)
    benchmark = paired.benchmark_batches(
        data, config, device="cpu", batch_sizes=(2,), warmup=1, repeats=1,
    )
    assert benchmark.status.tolist() == ["ok"]


def test_model_uses_extra_channels_and_checks_input_width():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = paired.PairedTransformer(
        num_classes=2, width=12, heads=6, layers=1, dropout=0.0,
        input_features=len(paired.FEATURES) + 2,
    )
    windows = torch.randn(2, 51, len(paired.FEATURES) + 2)
    paired.masked_loss(model(windows), torch.tensor([[0, 1, 0], [1, 0, 1]])).backward()
    assert model.projection.weight.grad[:, -2:].abs().sum() > 0
    with pytest.raises(ValueError, match="56"):
        model(windows[:, :, :-1])


def test_tracking_records_overtraining_metrics_and_stop_reason(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from bvr_behavior_prediction.paired_skill_tracking import RunTracker

    tracker = RunTracker(tmp_path, {"enabled": True, "log_per_class": False})
    tracker.state = {"run_id": "fixture", "last_logged_epoch": 0}
    recorded, tags = {}, {}
    monkeypatch.setattr(tracker, "metrics", lambda values, step: recorded.update(values))
    tracker.client = SimpleNamespace(set_tag=lambda run, key, value: tags.update({key: value}))
    row = {"epoch": 4, "train_nll": .6, "test_nll": 1.3,
           "train_macro_f1": .8, "test_macro_f1": .5, "seconds": 1,
           "epochs_without_improvement": 3, "improved": False,
           "train_l1_penalty": .01, "train_objective": .61, "generalization_gap": .7,
           "overtraining_epochs": 3, "overtraining_detected": True}
    tracker.epoch(row, pd.DataFrame(), pd.DataFrame())
    for key in ("train_l1_penalty", "train_objective", "generalization_gap",
                "overtraining_epochs", "overtraining_detected"):
        assert recorded[f"epoch/{key}"] == row[key]
    paired.write_json(tmp_path / "training_status.json", {
        "completed_epochs": 4, "status": "early_stopped", "best_epoch": 1,
        "best_test_nll": 1.0, "stop_reason": "overtraining"})
    monkeypatch.setattr(tracker, "sync_artifacts", lambda: None)
    monkeypatch.setattr(tracker, "save_model", lambda: None)
    tracker.finish_training()
    assert tags["stop_reason"] == "overtraining"


@pytest.mark.parametrize("width", [0, -1, True, 1.5])
def test_invalid_input_feature_count(width):
    with pytest.raises(ValueError, match="input_features"):
        paired.PairedTransformer(input_features=width)


@pytest.mark.parametrize("contract,pipeline", [
    (paired.DATA_CONTRACT, "trained_pilot_skill_forecasting"),
    ("observer-skills-test-v1", "observer-skills-test-v1"),
])
def test_tracking_metadata_uses_actual_dataset_contract(tmp_path, monkeypatch, contract, pipeline):
    import sys
    from types import ModuleType, SimpleNamespace

    from bvr_behavior_prediction.paired_skill_tracking import RunTracker

    # Exercise metadata recording without requiring the optional MLflow package.
    entities = ModuleType("mlflow.entities")
    entities.Param = lambda key, value: SimpleNamespace(key=key, value=value)
    monkeypatch.setitem(sys.modules, "mlflow.entities", entities)
    data = prepared_data(tmp_path)
    data.manifest["data_contract"] = contract
    output = tmp_path / "tracking"
    output.mkdir()
    tracker = RunTracker(output, {"enabled": True})
    tags, params = {}, {}
    tracker.state = {"run_id": "fixture"}
    tracker.client = SimpleNamespace(
        set_tag=lambda run, key, value: tags.__setitem__(key, value),
        log_batch=lambda run, params: recorded_params.update({p.key: p.value for p in params}),
    )
    recorded_params = params
    monkeypatch.setattr(tracker, "artifact", lambda *args: None)
    tracker.begin({"classes": data.classes, "training": {},
                   "data_manifest_sha256": "dataset", "source_sha256": "training"}, data)
    assert tags["data_contract"] == contract and tags["pipeline"] == pipeline
    assert params["data_contract"] == contract
    assert params["input_features"] == str(len(data.manifest["features"]))
