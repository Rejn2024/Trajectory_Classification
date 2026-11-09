"""Portable, aligned prediction reports must not train or choose validation winners."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bvr_behavior_prediction import enemy_missile_diagnostics as diagnostics
from bvr_behavior_prediction.paired_skill_forecasting import metric_tables


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def saved_run(tmp_path, architecture="transformer", seed=42):
    directory = tmp_path / f"{architecture}_{seed}"
    directory.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    write_json(data_dir / "manifest.json", {"fixture": "same prepared data"})
    classes = ["alpha", "beta", "gamma"]
    training = {"architecture": architecture, "seed": seed, "width": 96, "heads": 6,
                "layers": 2, "dropout": .25, "class_weighting": "sqrt_inverse",
                "l1_lambda": 1e-6, "epochs": 100, "graph_width": 64, "graph_layers": 2}
    run = {"architecture": architecture, "training": training, "classes": classes,
           "parameter_count": 100, "data_contract": "fixture-enemy-missile",
           "feature_names": [f"f{i}" for i in range(115)], "horizons_s": [1, 2, 5],
           "data_manifest_sha256": digest(data_dir / "manifest.json")}
    write_json(directory / "run.json", run)
    write_json(directory / "selection.json", {"epoch": 1, "validation_used_for_selection": False})
    write_json(directory / "training_status.json", {"status": "completed", "completed_epochs": 2})
    pd.DataFrame({"epoch": [1, 2], "train_nll": [.6, .5], "test_nll": [.7, .8],
                  "train_data_loss": [.65, .55], "train_l1_penalty": [.1, .09],
                  "train_objective": [.75, .64], "generalization_gap": [.1, .3]}).to_csv(
        directory / "history.csv", index=False)
    summary_tables, class_tables, metadata, indices = [], [], [], {}
    for s, split in enumerate(diagnostics.SPLITS):
        ids = np.arange(s * 4, (s + 1) * 4)
        targets = np.tile(np.array([0, 1, 2, 0])[:, None], (1, 3))
        targets[0, 2] = -1
        probabilities = np.full((4, 3, 3), .1, np.float32)
        for i, label in enumerate([0, 1, 2, 0]):
            probabilities[i, :, label] = .8
        if architecture != "transformer":
            probabilities[1, 0] = [.2, .7, .1]
        summary, per_class, confusion = metric_tables(targets, probabilities, classes, split)
        np.savez_compressed(directory / f"final_{split}.npz", targets=targets,
                            probabilities=probabilities, sample_indices=ids, confusion=confusion)
        summary_tables.append(summary)
        class_tables.append(per_class)
        indices[split] = ids
        metadata.extend({"split": split, "episode_id": f"{split}_flight_{i//2}",
                         "scenario_group": f"{split}_scenario_{i//2}", "step": 60 + i * 10}
                        for i in range(4))
    pd.concat(summary_tables).to_csv(directory / "final_metrics.csv", index=False)
    pd.concat(class_tables).to_csv(directory / "final_per_class.csv", index=False)
    features = np.zeros((12, 115), np.float32)
    features[::2, 19] = .1
    features[::3, 54] = .1
    data = SimpleNamespace(directory=data_dir, samples=pd.DataFrame(metadata), indices=indices,
                           anchors=np.arange(12), features=features)
    return directory, data


def test_persists_prediction_diagnostics_calibration_and_scenario_metadata(tmp_path):
    run, data = saved_run(tmp_path)
    before = digest(run / "final_validation.npz")
    result = diagnostics.save_diagnostics(run, data)
    assert digest(run / "final_validation.npz") == before
    destination = run / "diagnostics"
    assert len(list(destination.glob("*.png"))) == 7
    assert (destination / "learning_curves.png").stat().st_size > 100
    assert len(result["summary"]) == 9
    row = result["summary"].query("split == 'validation' and horizon_s == 1").iloc[0]
    assert row.accuracy == 1
    assert row.ece == pytest.approx(.2)
    assert row.brier == pytest.approx(.06)
    assert row.top3_accuracy == 1
    assert result["reliability"].groupby(["split", "horizon_s"])["count"].sum().tolist() == [4, 4, 3] * 3
    samples = pd.read_csv(destination / "samples.csv")
    assert samples.scenario_group.nunique() == 6
    assert samples.friendly_missiles_present.sum() == 6
    assert samples.enemy_missiles_present.sum() == 4
    cohorts = pd.read_csv(destination / "missile_cohorts.csv")
    assert len(cohorts) == 36
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["sources"]["final_validation.npz"] == before
    assert manifest["artifacts"]["samples.csv"] == digest(destination / "samples.csv")
    assert "never" in manifest["partition_roles"]["validation"]
    # Report-only regeneration preserves previously exported sample/cohort metadata.
    diagnostics.save_diagnostics(run)
    assert pd.read_csv(destination / "samples.csv").equals(samples)


def make_three(tmp_path, monkeypatch):
    # Unit comparisons exercise numerical/validation logic without rendering all
    # diagnostic figures for each otherwise identical synthetic architecture.
    monkeypatch.setattr(diagnostics, "_plot_diagnostics", lambda *args: None)
    runs = {}
    for architecture in diagnostics.ARCHITECTURES:
        run, data = saved_run(tmp_path, architecture)
        diagnostics.save_diagnostics(run, data)
        runs[architecture] = [run]
    return runs


def test_comparison_matches_examples_and_bootstraps_whole_scenarios(tmp_path, monkeypatch):
    runs = make_three(tmp_path, monkeypatch)
    originals = {path: digest(path) for paths in runs.values() for run in paths for path in run.rglob("*") if path.is_file()}
    result = diagnostics.compare_architectures(runs, tmp_path / "comparison", bootstrap_resamples=20)
    assert len(result["metrics"]) == 27
    assert len(result["aggregate"]) == 27
    assert len(result["per_class_deltas"]) == 81
    bootstrap = result["paired_bootstrap"]
    assert len(bootstrap) == 72  # 3 paired contrasts x 3 splits x 4 horizons x 2 metrics.
    assert bootstrap.scenario_groups.eq(2).all()
    assert bootstrap.resamples.eq(20).all()
    direct = bootstrap.query("architecture == 'typed_graph' and reference_architecture == 'missile_set'")
    assert direct[["estimate", "lower", "upper"]].eq(0).all().all()
    assert len(result["horizon_means"]) == 9
    assert len(result["horizon_mean_aggregate"]) == 9
    assert "logged_fit_evaluation_seconds" in result["metrics"]
    assert not any("winner" in c or "rank" in c for c in result["metrics"].columns)
    assert originals == {path: digest(path) for path in originals}
    manifest = json.loads((tmp_path / "comparison" / "manifest.json").read_text())
    assert manifest["seeds"] == [42]
    assert "no best" in manifest["validation_policy"]


@pytest.mark.parametrize("field,value", [
    ("data_manifest_sha256", "other-dataset"),
    ("feature_names", ["different_order"]),
])
def test_comparison_rejects_mismatched_data_contract(tmp_path, monkeypatch, field, value):
    runs = make_three(tmp_path, monkeypatch)
    path = runs["typed_graph"][0] / "run.json"
    run = json.loads(path.read_text())
    run[field] = value
    write_json(path, run)
    with pytest.raises(ValueError, match="protocol mismatch"):
        diagnostics.compare_architectures(runs, tmp_path / "out", 0)


def test_comparison_rejects_protocol_differences_and_duplicate_seeds(tmp_path, monkeypatch):
    runs = make_three(tmp_path, monkeypatch)
    path = runs["typed_graph"][0] / "run.json"
    original = json.loads(path.read_text())
    altered = json.loads(path.read_text())
    altered["training"]["l1_lambda"] = 0
    write_json(path, altered)
    with pytest.raises(ValueError, match="protocol mismatch"):
        diagnostics.compare_architectures(runs, tmp_path / "out", 0)
    write_json(path, original)
    extra = tmp_path / "duplicate"
    import shutil
    shutil.copytree(runs["typed_graph"][0], extra)
    runs["typed_graph"].append(extra)
    with pytest.raises(ValueError, match="Duplicate seed"):
        diagnostics.compare_architectures(runs, tmp_path / "out", 0)


def test_comparison_rejects_reordered_prediction_examples(tmp_path, monkeypatch):
    runs = make_three(tmp_path, monkeypatch)
    run = runs["typed_graph"][0]
    path = run / "final_test.npz"
    with np.load(path) as f:
        arrays = {k: f[k] for k in f.files}
    arrays["sample_indices"] = arrays["sample_indices"][::-1]
    np.savez_compressed(path, **arrays)
    diagnostics.save_diagnostics(run)
    with pytest.raises(ValueError, match="examples/targets differ"):
        diagnostics.compare_architectures(runs, tmp_path / "out", 0)


def test_comparison_rejects_changed_predictions_and_missing_exports(tmp_path, monkeypatch):
    runs = make_three(tmp_path, monkeypatch)
    path = runs["typed_graph"][0] / "final_test.npz"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed since diagnostic export"):
        diagnostics.compare_architectures(runs, tmp_path / "out", 0)
    (runs["transformer"][0] / "diagnostics" / "manifest.json").unlink()
    with pytest.raises(FileNotFoundError, match="save_diagnostics"):
        diagnostics.compare_architectures(runs, tmp_path / "out", 0)


def test_missing_run_artifacts_produce_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="finish this run"):
        diagnostics.save_diagnostics(tmp_path)
    with pytest.raises(ValueError, match="exactly these architectures"):
        diagnostics.compare_architectures({}, tmp_path / "comparison")


@pytest.mark.parametrize("alteration", ["scenario", "duplicate", "missing_column", "split"])
def test_comparison_validates_sample_scenario_metadata(tmp_path, monkeypatch, alteration):
    runs = make_three(tmp_path, monkeypatch)
    run = runs["typed_graph"][0]
    path = run / "diagnostics" / "samples.csv"
    samples = pd.read_csv(path)
    if alteration == "scenario":
        samples.loc[0, "scenario_group"] = "different_scenario"
    elif alteration == "duplicate":
        samples.loc[1, "sample_index"] = samples.loc[0, "sample_index"]
    elif alteration == "missing_column":
        samples = samples.drop(columns="scenario_group")
    else:
        samples.loc[0, "split"] = "validation"
    samples.to_csv(path, index=False)
    manifest_path = run / "diagnostics" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["samples.csv"] = digest(path)
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="[Mm]alformed sample|metadata differs"):
        diagnostics.compare_architectures(runs, tmp_path / "out", 0)


def test_paired_scenario_bootstrap_has_correct_known_delta():
    # Every scenario has the same +1 NLL difference, so its paired interval is
    # exactly log(2), regardless of scenario size and the bootstrap draw.
    y = np.zeros((5, 3), dtype=int)
    reference = {"targets": y, "probabilities": np.tile([.8, .2], (5, 3, 1))}
    candidate = {"targets": y, "probabilities": np.tile([.4, .6], (5, 3, 1))}
    result = pd.DataFrame(diagnostics._bootstrap(reference, candidate,
        ["large", "large", "large", "small", "small"], 50, np.random.default_rng(1)))
    nll = result.query("metric == 'probability_nll'")
    np.testing.assert_allclose(nll[["estimate", "lower", "upper"]], np.log(2))
    np.testing.assert_allclose(result.query("metric == 'accuracy'")[["estimate", "lower", "upper"]], -1)
    assert result.scenario_groups.eq(2).all()


def test_comparison_rejects_unmatched_seed_sets(tmp_path, monkeypatch):
    runs = make_three(tmp_path, monkeypatch)
    run, data = saved_run(tmp_path, "typed_graph", seed=43)
    diagnostics.save_diagnostics(run, data)
    runs["typed_graph"].append(run)
    with pytest.raises(ValueError, match="same training seeds"):
        diagnostics.compare_architectures(runs, tmp_path / "out", 0)


def test_logit_nll_is_preserved_when_saved_probability_underflows(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "_plot_diagnostics", lambda *args: None)
    run, _ = saved_run(tmp_path)
    path = run / "final_validation.npz"
    with np.load(path) as stored:
        arrays = {k: stored[k] for k in stored.files}
    arrays["probabilities"][0, 0] = [0., .6, .4]
    _, _, arrays["confusion"] = metric_tables(arrays["targets"], arrays["probabilities"],
                                              ["alpha", "beta", "gamma"], "validation")
    np.savez_compressed(path, **arrays)
    summary = pd.read_csv(run / "final_metrics.csv")
    summary.loc[(summary.split == "validation") & (summary.horizon_s == 1), "nll"] = 125.
    summary.to_csv(run / "final_metrics.csv", index=False)
    per_class = pd.read_csv(run / "final_per_class.csv")
    per_class.loc[(per_class.split == "validation") & (per_class.horizon_s == 1)
                  & (per_class.skill == "alpha"), "class_nll"] = 250.
    per_class.to_csv(run / "final_per_class.csv", index=False)
    result = diagnostics.save_diagnostics(run)
    row = result["summary"].query("split == 'validation' and horizon_s == 1").iloc[0]
    assert row.nll == 125.
    assert row.probability_nll < 20.
    row = result["per_class"].query("split == 'validation' and horizon_s == 1 and skill == 'alpha'").iloc[0]
    assert row.class_nll == 250.
    assert row.probability_class_nll < 40.


def test_authoritative_nll_requires_matching_counts_and_identities(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "_plot_diagnostics", lambda *args: None)
    run, _ = saved_run(tmp_path)
    path = run / "final_metrics.csv"
    summary = pd.read_csv(path)
    summary.loc[0, "count"] += 1
    summary.to_csv(path, index=False)
    with pytest.raises(ValueError, match="counts/support differ"):
        diagnostics.save_diagnostics(run)
