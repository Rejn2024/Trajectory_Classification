"""Train/reload the actual model families through the shared, masked trainer."""

import numpy as np
import pandas as pd
import pytest
import torch

from bvr_behavior_prediction import paired_skill_forecasting as paired
from bvr_behavior_prediction.nn_duel_skill_forecasting import (
    ENEMY_MISSILE_DATA_CONTRACT, feature_names,
)


def prepared_enemy_data(tmp_path):
    directory = tmp_path / "data"
    directory.mkdir()
    classes = ["defend", "turn"]
    names = feature_names(classes, enemy_missiles=True)
    rng = np.random.default_rng(31)
    raw = np.zeros((183, len(names)), np.float32)
    raw[:, :19] = rng.normal(0, .1, (183, 19))
    raw[:, 47:54] = rng.normal(0, .1, (183, 7))
    # A friendly and an enemy missile with compact slots absent elsewhere.
    raw[:, 19:26] = [.1, .02, .03, .3, -.2, .01, .11]
    raw[:, 54:61] = [-.2, .01, .1, -.4, .1, -.01, .21]
    raw[:, 82:] = np.eye(2)[np.arange(183) % 2]
    np.save(directory / "features.npy", raw)
    np.save(directory / "anchors.npy", np.array([50, 60, 111, 121, 172, 182]))
    np.save(directory / "targets.npy", np.array([[0, 1, -1], [1, 0, 1]] * 3))
    pd.DataFrame({
        "episode_id": np.repeat(["train_flight", "test_flight", "validation_flight"], 2),
        "scenario_group": np.repeat(["train_group", "test_group", "validation_group"], 2),
        "observer_id": ["A01", "B01"] * 3,
        "split": np.repeat(paired.SPLITS, 2), "step": [60, 120] * 3,
    }).to_parquet(directory / "samples.parquet", index=False)
    mean, std = raw[:61].mean(0), np.maximum(raw[:61].std(0), .01)
    mean[82:], std[82:] = 0, 1
    paired.write_json(directory / "scaling.json", {"mean": mean.tolist(), "std": std.tolist()})
    manifest = {
        "classes": classes, "features": names, "data_contract": ENEMY_MISSILE_DATA_CONTRACT,
        "horizons_s": [1, 2, 5], "history_steps": 51,
        "input_contract": "Observer history, current own skill and same-time missile tracks",
        "target_contract": "Opponent skill selections at exact future times",
        "files": {p.name: paired.sha256(p) for p in directory.iterdir()},
    }
    paired.write_json(directory / "manifest.json", manifest)
    return paired.PairedData(directory)


@pytest.mark.parametrize("architecture", ["missile_set", "typed_graph"])
def test_training_resume_reload_and_benchmark(tmp_path, architecture, monkeypatch):
    data = prepared_enemy_data(tmp_path)
    config = {
        "architecture": architecture, "graph_width": 8, "graph_layers": 2,
        "epochs": 2, "width": 12, "heads": 6, "layers": 1, "dropout": 0.0,
        "batch_size": 2, "evaluation_batch_size": 2, "cpu_threads": 1,
        "precision": "fp32", "data_backend": "direct", "l1_lambda": 1e-6,
        "class_weighting": "sqrt_inverse", "early_stopping": True,
        "early_stopping_min_epochs": 0, "overtraining_shutdown": True,
    }
    output = tmp_path / architecture
    history = paired.train_model(data, output, config, device="cpu")
    assert (history.train_l1_penalty > 0).all()
    run = paired.read_json(output / "run.json")
    assert run["architecture"] == architecture and run["parameter_count"] > 0
    assert (output / "architecture_source.py").is_file()
    checkpoint = torch.load(output / "best.pt", weights_only=False, map_location="cpu")
    model = paired.make_model(len(data.classes), checkpoint["model_config"],
                              checkpoint["mean"], checkpoint["std"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    x, _ = data.batch(data.indices["validation"], "cpu")
    with torch.inference_mode():
        expected = model(x).softmax(-1).numpy()
    _, _, _, result = paired.final_evaluation(data, output, device="cpu")
    np.testing.assert_array_equal(result["validation"]["probabilities"], expected)
    pd.testing.assert_frame_equal(history, paired.train_model(
        data, output, config, device="cpu", resume=True))
    benchmark = paired.benchmark_batches(data, config, device="cpu", batch_sizes=(2,),
                                          warmup=1, repeats=1)
    assert benchmark.status.tolist() == ["ok"]
    original = paired.sha256
    monkeypatch.setattr(paired, "sha256", lambda p:
                        "changed-model-source" if str(p).endswith("enemy_missile_models.py")
                        else original(p))
    with pytest.raises(ValueError, match="same configuration, data and source"):
        paired.train_model(data, output, config, device="cpu", resume=True)


def test_graph_rejects_other_input_contract(tmp_path):
    data = prepared_enemy_data(tmp_path)
    data.manifest["data_contract"] = "unrelated"
    with pytest.raises(ValueError, match="notebook 17"):
        paired.train_model(data, tmp_path / "run", {"architecture": "typed_graph"}, device="cpu")


def test_unknown_model_and_historical_configuration():
    config = {**paired.DEFAULT_TRAINING, "width": 12, "heads": 6, "layers": 1}
    old = paired.model_configuration(config, 54)
    assert "architecture" not in old
    assert isinstance(paired.make_model(2, old), paired.PairedTransformer)
    with pytest.raises(ValueError, match="Unknown"):
        paired.model_configuration({**config, "architecture": "unknown"}, 115)


def test_three_family_saved_runs_execute_comparison_notebook(tmp_path, monkeypatch):
    """Exercise notebook21 against genuine trained/reloaded/persisted model runs."""
    import json
    import shutil
    import sys
    from pathlib import Path
    import IPython.display
    from bvr_behavior_prediction.enemy_missile_diagnostics import save_diagnostics

    data = prepared_enemy_data(tmp_path)
    repository = Path(__file__).parents[1]
    package = tmp_path / "bvr_behavior_prediction"
    package.mkdir()
    shutil.copy2(repository / "bvr_behavior_prediction/enemy_missile_diagnostics.py", package)
    root = tmp_path / "artifacts/nn_duel_enemy_missile_forecasting"
    for family in ("transformer", "missile_set", "typed_graph"):
        config = {
            "architecture": family, "graph_width": 8, "graph_layers": 2,
            "epochs": 1, "width": 12, "heads": 6, "layers": 1, "dropout": 0.0,
            "batch_size": 2, "evaluation_batch_size": 2, "cpu_threads": 1,
            "precision": "fp32", "data_backend": "direct", "l1_lambda": 1e-6,
            "class_weighting": "sqrt_inverse", "early_stopping_min_epochs": 0,
        }
        run = root / "runs" / family
        paired.train_model(data, run, config, device="cpu")
        paired.final_evaluation(data, run, device="cpu")
        save_diagnostics(run, data)
        paired.write_json(root / "comparisons" / f"latest_{family}_runs.json", {
            "architecture": family, "runs": [str(run)],
            "data_manifest_sha256": paired.sha256(data.directory / "manifest.json"),
        })
    monkeypatch.setenv("BVR_NN_ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(IPython.display, "display", lambda *args: None)
    namespace = {}
    notebook = json.loads((repository / "notebooks/21_compare_enemy_missile_nn_duel_skill_forecasting.ipynb").read_text())
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]).removeprefix("%%time\n"),
                         "comparison-notebook", "exec"), namespace)
            if index == 1:
                namespace["BOOTSTRAP_RESAMPLES"] = 10
    output = namespace["OUTPUT"]
    manifest = paired.read_json(output / "manifest.json")
    assert manifest["seeds"] == [42]
    assert set(namespace["TABLES"]["metrics"].architecture) == {
        "transformer", "missile_set", "typed_graph"}
    for name in ("metrics.csv", "aggregate.csv", "per_class_deltas.csv",
                 "paired_bootstrap.csv", "architecture_comparison.png", "input_run_lists.json"):
        assert (output / name).is_file()
