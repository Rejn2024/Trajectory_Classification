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


def write_source(root, flights=10):
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
            obs, track, decisions = frames_and_decisions(target_id)
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
                             "pilot_id": f"pilot_{target_id}", "frames": 121,
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
