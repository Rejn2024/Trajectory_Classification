"""Causal aircraft-role and end-to-end data contract tests for notebooks 12/13."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bvr_behavior_prediction import nn_duel_skill_forecasting as duel


CLASSES = ["defend", "turn"]


def test_training_notebook_schema_code_and_stopping_defaults():
    import ast
    import nbformat
    from bvr_behavior_prediction.paired_skill_forecasting import DEFAULT_TRAINING

    path = Path(__file__).parents[1] / "notebooks/13_nn_duel_skill_forecasting.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    namespace = {"DEFAULT_TRAINING": DEFAULT_TRAINING, "DEVICE": "cpu"}
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source.removeprefix("%%time\n"), str(path))
        compile(tree, str(path), "exec")
        for statement in tree.body:
            # Execute the actual notebook's CONFIG assignment/update in isolation.
            if (isinstance(statement, ast.Assign) and
                    any(isinstance(t, ast.Name) and t.id == "CONFIG" for t in statement.targets)
                    or isinstance(statement, ast.Expr) and
                    isinstance(statement.value, ast.Call) and
                    isinstance(statement.value.func, ast.Attribute) and
                    isinstance(statement.value.func.value, ast.Name) and
                    statement.value.func.value.id == "CONFIG"):
                exec(compile(ast.Module(body=[statement], type_ignores=[]), str(path), "exec"), namespace)
    config = namespace["CONFIG"]
    assert config["l1_lambda"] == 1e-6
    assert config["early_stopping"] and config["early_stopping_min_epochs"] == 0
    assert config["overtraining_shutdown"] and config["overtraining_patience"] == 3


def frames_and_decisions(observer="A01", frames=121):
    key = {"episode_id": ["flight"] * frames, "step": np.arange(frames),
           "time_s": np.arange(frames) * .1}
    observation = pd.DataFrame({**key, **{name: np.full(frames, .2)
                                        for name in duel.OBS_COLUMNS}})
    track = pd.DataFrame({**key, "observer_alive": True, "target_alive": True})
    steps = np.arange(0, frames - 1, 10)
    decisions = pd.DataFrame({"episode_id": "flight", "step": steps,
                              "time_s": steps * .1, "aircraft_id": observer,
                              "skill": [CLASSES[(i // 10) % 2] for i in steps]})
    return observation, track, decisions


def test_own_skill_changes_at_decision_not_before_and_never_uses_future():
    obs, tracks, own = frames_and_decisions("A01")
    _, _, target = frames_and_decisions("B01")
    target["skill"] = "defend"
    features, ends, labels = duel.episode_examples(obs, tracks, target, own, CLASSES,
                                                  "A01", "B01")
    np.testing.assert_array_equal(features[9, -2:], [1, 0])
    np.testing.assert_array_equal(features[10, -2:], [0, 1])
    np.testing.assert_array_equal(features[120, -2:], [0, 0])
    assert ends[0] == 60
    np.testing.assert_array_equal(labels[0], [0, 0, 0])
    # Alter an opponent selection at the present origin: only future target
    # timestamps can affect labels, and opponent labels never change inputs.
    target.loc[target.step.eq(60), "skill"] = "turn"
    changed, _, changed_labels = duel.episode_examples(obs, tracks, target, own, CLASSES,
                                                       "A01", "B01")
    np.testing.assert_array_equal(changed, features)
    np.testing.assert_array_equal(changed_labels, labels)


def test_own_skill_gaps_unknown_labels_and_aircraft_swaps_fail():
    _, _, own = frames_and_decisions()
    for damaged in (own[own.step.ne(30)], own.assign(aircraft_id="B01"),
                    own.assign(skill="unknown")):
        with pytest.raises(ValueError):
            duel.current_own_skills(damaged, "flight", "A01", 121, CLASSES)


def test_missing_exact_target_masked_without_forward_fill():
    obs, tracks, own = frames_and_decisions("A01")
    _, _, target = frames_and_decisions("B01")
    target = target[target.step.ne(80)]
    _, ends, labels = duel.episode_examples(obs, tracks, target, own, CLASSES, "A01", "B01")
    assert labels[ends.tolist().index(60), 1] == -1
    assert labels[ends.tolist().index(60), 0] >= 0
    # Dead-history exclusion applies to all raw 56 frames.
    tracks.loc[tracks.step.eq(10), "target_alive"] = False
    _, dead_ends, _ = duel.episode_examples(obs, tracks, target, own, CLASSES, "A01", "B01")
    assert 60 not in dead_ends
    assert 70 in dead_ends


def write_source(root, flights=10, frames=121):
    source = root / "shared/corpora/nn-duels-fixture/series"
    source.mkdir(parents=True)
    cases, episodes = [], []
    tables = {(side, part): [] for side in duel.SIDES
              for part in ("observations", "tracks", "target_decisions")}
    for i in range(flights):
        episode_id = f"flight_{i:03d}"
        cases.append({"episode_id": episode_id, "scenario_group": episode_id,
                      "initial": {"A01": {"position": [0, 0, 0]},
                                  "B01": {"position": [1000 + i, 0, 0]}}})
        for side, target_id in (("A01", "B01"), ("B01", "A01")):
            obs, track, decisions = frames_and_decisions(target_id, frames=frames)
            for table in (obs, track, decisions):
                table["episode_id"] = episode_id
            # Two viewpoints have deliberately different own-state features.
            obs["obs_00"] = i + (100 if side == "A01" else 200)
            if target_id == "B01":
                decisions["skill"] = "defend"
            for part, table in zip(("observations", "tracks", "target_decisions"),
                                   (obs, track, decisions)):
                tables[side, part].append(pa.Table.from_pandas(table, preserve_index=False))
            episodes.append({"episode_id": episode_id, "scenario_group": episode_id,
                             "observer_id": side, "target_id": target_id,
                             "pilot_id": f"pilot_{target_id}", "frames": frames,
                             "family": "fixture", "duration_s": (frames - 1) * .1,
                             "shard": "part-00000"})
    for (side, part), groups in tables.items():
        path = source / side / part / "part-00000.parquet"
        path.parent.mkdir(parents=True)
        with pq.ParquetWriter(path, groups[0].schema) as writer:
            for group in groups:
                writer.write_table(group)
    pd.DataFrame(episodes).to_parquet(source / "episodes.parquet", index=False)
    duel.write_json(source / "config.json", {"sample_interval_s": .1,
                                             "decision_interval_s": 1, "cases": cases})
    duel.write_json(source / "schema.json", {"schema_version": "nn-duels-v1"})
    manifest = {"name": "nn-duels-v1", "complete": True, "physical_flights": flights,
                "requested_flights": flights, "perspective_episodes": 2 * flights,
                "config_sha256": duel.sha256(source / "config.json")}
    manifest["files"] = {p.relative_to(source).as_posix(): {"sha256": duel.sha256(p)}
                         for p in source.rglob("*") if p.is_file()}
    duel.write_json(source / "manifest.json", manifest)
    catalog = root / "bvr_sim_source/bvr_sim/agents/skill_manager.py"
    catalog.parent.mkdir(parents=True)
    catalog.write_text("# fixture catalogue\n")
    return source


def test_end_to_end_both_views_grouping_scaling_and_integrity(tmp_path, monkeypatch):
    monkeypatch.setattr(duel, "catalogue", lambda root: CLASSES)
    source = write_source(tmp_path)
    assert duel.discover_corpus(tmp_path) == source
    analysis = duel.analyze_corpus(tmp_path, source)
    groups = duel.split_scenarios(analysis)
    assert groups == duel.split_scenarios(analysis)
    assert [sum(g["split"] == split for g in groups.values()) for split in duel.SPLITS] == [4, 3, 3]
    output = tmp_path / "prepared"
    manifest = duel.prepare_dataset(tmp_path, output, source, analysis=analysis)
    data = duel.NNDuelData(output)
    assert data.features.shape == (20 * 121, 56)
    assert manifest["source_physical_flights"] == 10
    assert data.samples.groupby("episode_id").split.nunique().eq(1).all()
    np.testing.assert_array_equal(data.mean[-2:], [0, 0])
    np.testing.assert_array_equal(data.std[-2:], [1, 1])
    ids = data.indices["train"][:2]
    x, y = data.batch(ids, "cpu")
    assert tuple(x.shape) == (2, 51, 56)
    assert tuple(y.shape) == (2, 3)
    # Scaling includes unique frames in training histories only, both views.
    used = set()
    for anchor in data.anchors[data.indices["train"]]:
        used.update(range(int(anchor) - 50, int(anchor) + 1))
    np.testing.assert_allclose(data.mean[:54], data.features[sorted(used), :54].mean(axis=0),
                               rtol=1e-5, atol=1e-5)
    with (output / "targets.npy").open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="Prepared data changed"):
        duel.NNDuelData(output)


def test_discovery_excludes_partial_and_explicit_path_overrides(tmp_path):
    source = write_source(tmp_path)
    partial = tmp_path / "shared/corpora/nn-duels-other/newer"
    partial.mkdir(parents=True)
    duel.write_json(partial / "manifest.json", {"name": "nn-duels-v1", "complete": False})
    assert duel.discover_corpus(tmp_path) == source
    assert duel.discover_corpus(tmp_path, source) == source
    with pytest.raises(ValueError, match="No completed"):
        duel.discover_corpus(tmp_path, partial)


def test_repeated_ranges_cannot_leave_an_unsplittable_stratum():
    count = 50
    names = [f"group_{i:03d}" for i in range(count)]
    distance = [1000] * 40 + [2000] * 9 + [3000]
    analysis = {"episodes": pd.DataFrame({"scenario_group": names}),
                "counts": np.ones((count, 3, 2), dtype=np.int64),
                "cases": [{"scenario_group": name,
                           "initial": {"A01": {"position": [0, 0, 0]},
                                       "B01": {"position": [r, 0, 0]}}}
                          for name, r in zip(names, distance)]}
    groups = duel.split_scenarios(analysis)
    assert len(groups) == count
    assert {value["split"] for value in groups.values()} == set(duel.SPLITS)


def test_nonoverlap_schedule_covers_raw_histories_across_views_without_label_selection():
    ends = np.arange(60, 400, 10)
    for episode_id in ("flight_a", "flight_b"):
        chosen = []
        for side in duel.SIDES:
            mask = duel.nonoverlapping_train_mask(ends, episode_id, side)
            chosen.extend((int(end), side) for end in ends[mask])
            # Eligibility gaps must never shift later scheduled windows.
            np.testing.assert_array_equal(
                duel.nonoverlapping_train_mask(ends[2:], episode_id, side), mask[2:])
        chosen.sort()
        assert [end for end, _ in chosen] == list(range(60, 400, 60))
        assert all(a[1] != b[1] for a, b in zip(chosen, chosen[1:]))
        used = set()
        for end, _ in chosen:
            raw = set(range(end - 55, end + 1))
            assert not raw & used
            used.update(raw)


@pytest.mark.parametrize("steps", [[60, 60], [60, 110], [60, 115]])
def test_nonoverlap_audit_rejects_shared_acceleration_frames_and_paired_views(steps):
    # Even disjoint 51-frame model inputs can share raw acceleration frames.
    samples = pd.DataFrame({"episode_id": ["flight"] * 2,
                            "observer_id": ["A01", "B01"],
                            "split": ["train"] * 2, "step": steps})
    with pytest.raises(ValueError, match="Overlapping training raw histories"):
        duel.nonoverlap_audit(samples)


def test_nonoverlap_preparation_matches_dense_holdouts_scaling_and_retained_support(tmp_path, monkeypatch):
    monkeypatch.setattr(duel, "catalogue", lambda root: CLASSES)
    source = write_source(tmp_path, frames=361)
    analysis = duel.analyze_corpus(tmp_path, source)
    dense_path, sparse_path = tmp_path / "dense", tmp_path / "sparse"
    duel.prepare_dataset(tmp_path, dense_path, source, analysis=analysis)
    manifest = duel.prepare_dataset(tmp_path, sparse_path, source, analysis=analysis,
                                    nonoverlapping_train=True)
    dense, sparse = duel.NNDuelData(dense_path), duel.NonoverlappingNNDuelData(sparse_path)
    assert manifest["data_contract"] == duel.NONOVERLAP_DATA_CONTRACT
    assert len(sparse.indices["train"]) < len(dense.indices["train"]) / 6
    assert sparse.samples.groupby("episode_id").split.nunique().eq(1).all()
    assert set(sparse.samples.loc[sparse.samples.split.eq("train"), "observer_id"]) == set(duel.SIDES)
    assert duel.read_json(dense_path / "scenario_splits.json") == duel.read_json(sparse_path / "scenario_splits.json")
    for split in ("test", "validation"):
        a, b = dense.indices[split], sparse.indices[split]
        pd.testing.assert_frame_equal(dense.samples.iloc[a].reset_index(drop=True),
                                      sparse.samples.iloc[b].reset_index(drop=True))
        np.testing.assert_array_equal(dense.anchors[a], sparse.anchors[b])
        np.testing.assert_array_equal(dense.targets[a], sparse.targets[b])
    train = sparse.samples.loc[sparse.samples.split.eq("train")]
    for _, rows in train.groupby("episode_id"):
        used = set()
        for end in rows.step:
            raw = set(range(end - 55, end + 1))
            assert not raw & used
            used.update(raw)
    selected_frames = np.concatenate([
        np.arange(end - 50, end + 1) for end in sparse.anchors[sparse.indices["train"]]])
    expected = sparse.features[selected_frames, :54].astype(float)
    np.testing.assert_allclose(sparse.mean[:54], expected.mean(0), atol=1e-6)
    np.testing.assert_allclose(sparse.std[:54], np.maximum(expected.std(0), .01), atol=1e-6)
    np.testing.assert_array_equal(sparse.mean[54:], [0, 0])
    np.testing.assert_array_equal(sparse.std[54:], [1, 1])
    support = pd.read_csv(sparse_path / "skill_support_by_split.csv")
    for split in duel.SPLITS:
        labels = sparse.targets[sparse.indices[split]]
        for horizon, seconds in enumerate(duel.HORIZONS):
            for label, skill in enumerate(CLASSES):
                value = support.query("split == @split and horizon_s == @seconds and skill == @skill")
                assert value.windows.item() == int((labels[:, horizon] == label).sum())
    audit = pd.read_csv(sparse_path / "window_sampling_audit.csv").groupby("split").sum(numeric_only=True)
    assert audit.loc["train", "retained_windows"] == len(sparse.indices["train"])
    assert audit.loc["test", "dropped_windows"] == audit.loc["validation", "dropped_windows"] == 0
    assert duel.nonoverlap_audit(sparse.samples)["minimum_origin_spacing_steps"] == 60
    with pytest.raises(ValueError, match="Expected own-skill-conditioned"):
        duel.NonoverlappingNNDuelData(dense_path)
    with pytest.raises(ValueError, match="Expected own-skill-conditioned"):
        duel.NNDuelData(sparse_path)
    # The loader rechecks disjointness even if file hashes are updated.
    damaged = sparse.samples.copy()
    damaged.loc[sparse.indices["train"][1], "step"] = damaged.loc[sparse.indices["train"][0], "step"]
    damaged.to_parquet(sparse_path / "samples.parquet", index=False)
    manifest["files"]["samples.parquet"] = duel.sha256(sparse_path / "samples.parquet")
    duel.write_json(sparse_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Overlapping training"):
        duel.NonoverlappingNNDuelData(sparse_path)


@pytest.mark.parametrize("training_number,training_notebook", [
    (15, "15_nonoverlap_nn_duel_skill_forecasting.ipynb"),
    (16, "16_balanced_nonoverlap_nn_duel_skill_forecasting.ipynb"),
])
def test_nonoverlap_notebooks_run_prepare_train_and_report(tmp_path, monkeypatch,
                                                          training_number, training_notebook):
    """Run their real Python cells on small data with the requested architecture."""
    import shutil
    import sys
    import nbformat
    import torch
    import IPython.display

    repository = Path(__file__).parents[1]
    monkeypatch.setattr(duel, "catalogue", lambda root: CLASSES)
    source = write_source(tmp_path, frames=361)
    package = tmp_path / "bvr_behavior_prediction"
    package.mkdir()
    for name in ("paired_skill_forecasting.py", "nn_duel_skill_forecasting.py"):
        shutil.copy2(repository / "bvr_behavior_prediction" / name, package / name)
    monkeypatch.setenv("BVR_NN_ROOT", str(tmp_path))
    monkeypatch.setenv("BVR_NN_FORECAST_SOURCE", str(source))
    monkeypatch.setenv("BVR_NN_FORECAST_DOWNLOAD", "0")
    monkeypatch.setenv("BVR_NN_NONOVERLAP_MLFLOW", "0")
    monkeypatch.setenv("BVR_NN_BALANCED_MLFLOW", "0")
    monkeypatch.delenv("BVR_NN_NONOVERLAP_DATA", raising=False)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    displayed = []
    monkeypatch.setattr(IPython.display, "display", lambda *args: displayed.extend(args))
    namespace = {}
    for number, name in ((14, "14_prepare_nonoverlap_nn_duel_skill_forecasting.ipynb"),
                         (training_number, training_notebook)):
        notebook = nbformat.read(repository / "notebooks" / name, as_version=4)
        nbformat.validate(notebook)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            assert cell.source.startswith("%%time\n")
            code = compile(cell.source.removeprefix("%%time\n"), name, "exec")
            exec(code, namespace)
            if number == training_number and index == 1:
                config = namespace["CONFIG"]
                assert config["width"] == 96 and config["dropout"] == .25
                assert config["heads"] == 6 and config["layers"] == 2
                assert config["l1_lambda"] == 1e-6
                assert config["early_stopping_min_epochs"] == 0 and config["overtraining_shutdown"]
                assert config["class_weighting"] == ("sqrt_inverse" if training_number == 16 else "none")
                # Only reduce run duration; retain the actual model and all three seeds.
                config["epochs"] = 2
    assert isinstance(namespace["data"], duel.NonoverlappingNNDuelData)
    assert len(namespace["RUNS"]) == 3
    for run in namespace["RUNS"].values():
        checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
        assert checkpoint["model_config"]["width"] == 96
        assert checkpoint["model_config"]["dropout"] == .25
        assert checkpoint["data_contract"] == duel.NONOVERLAP_DATA_CONTRACT
        if training_number == 16:
            state = duel.read_json(run / "class_weights.json")
            assert checkpoint["class_weighting"] == state
            assert (run / "class_weights.csv").is_file()
            labels = namespace["data"].targets[namespace["data"].indices["train"]]
            for h in range(3):
                expected = np.bincount(labels[labels[:, h] >= 0, h], minlength=len(CLASSES))
                np.testing.assert_array_equal(state["counts"][h], expected)
        assert (run / "final_validation.npz").is_file()
        assert (run / "training_frequency_baseline.csv").is_file()
    assert len(list((namespace["RUN"] / "diagnostics").glob("*.png"))) == 5
    assert not (tmp_path / "artifacts/nn_duel_skill_forecasting").exists()
    if training_number == 16:
        assert namespace["EXPERIMENT_ROOT"].name == "nn_duel_balanced_forecasting"
        assert namespace["DATA_ROOT"].name == "nn_duel_nonoverlap_forecasting"
    # Preparation can be rerun without rewriting the existing completed dataset.
    prep = nbformat.read(repository / "notebooks/14_prepare_nonoverlap_nn_duel_skill_forecasting.ipynb", as_version=4)
    for index in (1, 4, 6):
        exec(compile(prep.cells[index].source.removeprefix("%%time\n"), "reuse", "exec"), namespace)
