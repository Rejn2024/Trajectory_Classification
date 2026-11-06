"""Protect physical-flight counts, two-sided labels, restart and atomic publication."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import generate_nn_duel_corpus as duel


def test_scenarios_cover_pairs_families_and_exclude_old_geometry():
    cases = duel.scenarios()
    assert cases == duel.scenarios()
    assert cases != duel.scenarios(seed=42)
    assert len({c["scenario_group"] for c in cases}) == 10_000
    assert len({json.dumps(c["initial"]) for c in cases}) == 10_000
    from collections import Counter

    assert set(Counter(tuple(c["pilots"].values()) for c in cases).values()) == {100}
    assert set(Counter(c["family"] for c in cases).values()) == {1000}
    for case in cases:
        a, b = (case["initial"][s] for s in duel.SIDES)
        horizontal = np.linalg.norm(np.subtract(a["position"][:2], b["position"][:2]))
        assert 5999 < horizontal < 65001
        assert abs(((b["heading_nwu_deg"] - a["heading_nwu_deg"]) % 360) - 180) >= 5
        for state in (a, b):
            assert 1500 <= state["position"][2] <= 11500
            assert 200 <= state["speed_mps"] <= 380


def test_exact_horizons_never_fill_missing_decisions():
    track = pd.DataFrame({"observer_alive": [True] * 121, "target_alive": [True] * 121})
    decisions = pd.DataFrame(
        {"step": np.arange(0, 120, 10), "time_s": np.arange(0, 12), "skill": ["a", "b"] * 6}
    )
    count = duel.forecast_counts(track, decisions, ["a", "b", "missing"])
    assert count.sum(axis=1).tolist() == [5, 4, 1]
    assert count[:, 2].tolist() == [0, 0, 0]
    reduced = decisions.loc[decisions.step.ne(80)]
    assert duel.forecast_counts(track, reduced, ["a", "b", "missing"]).sum(axis=1).tolist() == [
        4,
        3,
        1,
    ]
    track.loc[30, "target_alive"] = False
    assert duel.forecast_counts(track, decisions, ["a", "b"]).sum() < count.sum()


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    work, output = tmp_path / "work", tmp_path / "shared"
    archive = tmp_path / "source.zip"
    archive.write_bytes(b"trusted-fixture-archive")
    monkeypatch.setattr(duel, "ARCHIVE", archive)
    cases = duel.scenarios(3)
    config = {
        "flights": 3,
        "cases": cases,
        "episode_limit_s": 12,
        "parent_archive_sha256": duel.sha256(archive),
        "pilots": {f"pilot_{i:03d}": {"checkpoint_sha256": str(i)} for i in range(1, 11)},
    }
    duel.write(
        work / "tactical_snapshot/training/pilot_001/config.json",
        {"skill_names": ["a", "b", "missing"]},
    )
    return work, output, config


def save_episode(work, case, defeated=False):
    files, initial = {}, {}
    n = 121 if not defeated else 84
    for side, target in (("A01", "B01"), ("B01", "A01")):
        keys = pd.DataFrame(
            {
                "episode_id": case["episode_id"],
                "step": np.arange(n),
                "time_s": np.round(np.arange(n) * 0.1, 8),
            }
        )
        obs = np.zeros((n, 51), np.float32)
        track = keys.copy()
        live = np.ones(n, bool)
        if defeated:
            live[-1] = False
        own = np.array([*case["initial"][side]["position"], 100, 200, 0])
        other = np.array([*case["initial"][target]["position"], -200, 150, 0])
        for role, values in (("observer", own), ("target", other)):
            for name, value in zip(("x", "y", "z", "vx", "vy", "vz"), values):
                track[f"{role}_{name}"] = value
            track[f"{role}_alive"] = live
        scale = np.array([50000, 50000, 10000, 600, 600, 600])
        obs[:, :6] = own / scale
        obs[:, 9:15] = (other - own) / scale
        obs[:, 15] = np.linalg.norm(other[:3] - own[:3]) / 50000
        obs[~live] = 0
        observations = pd.concat(
            [keys, pd.DataFrame(obs, columns=duel.OBSERVATION_COLUMNS)], axis=1
        )
        decisions = keys.iloc[np.arange(0, n - 1, 10)].copy()
        decisions["aircraft_id"] = target
        decisions["skill"] = ["a"] * len(decisions)
        decisions["parameters_json"] = "{}"
        directory = work / "episodes" / case["episode_id"]
        for part, frame in zip(duel.PARTS, (observations, track, decisions)):
            name = f"{side}/{part}.parquet"
            duel.save_table(directory / name, frame)
            files[name] = duel.sha256(directory / name)
        initial[side] = track.iloc[0].to_dict()
    summary = {
        "frames": n,
        "duration_s": round((n - 1) * 0.1, 8),
        "case_sha256": hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest(),
        "files": files,
        "alive": {s: not defeated for s in duel.SIDES},
        "initial_state": initial,
        "termination_reason": "aircraft_defeated" if defeated else "time_limit",
    }
    duel.write(directory / "summary.json", summary)


def test_resume_detects_damage_and_missing_completion_marker(campaign):
    work, _, config = campaign
    case = config["cases"][0]
    assert duel.completed(work, case) is None
    save_episode(work, case)
    assert duel.completed(work, case)["frames"] == 121
    (work / "episodes" / case["episode_id"] / "A01/observations.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Damaged episode"):
        duel.completed(work, case)


def test_partial_then_complete_shards_and_canonical_target(campaign):
    work, output, config = campaign
    for case in config["cases"][:2]:
        save_episode(work, case)
    manifest = duel.package(work, output, config, shard_flights=2)
    assert not manifest["complete"]
    assert manifest["physical_flights"] == 2 and manifest["perspective_episodes"] == 4
    duel.verify(output)
    with pytest.raises(ValueError, match="not been acquired"):
        duel.verify(output, require_complete=True)
    first_shard = duel.sha256(output / "A01/observations/part-00000.parquet")
    save_episode(work, config["cases"][2], defeated=True)
    manifest = duel.package(work, output, config, shard_flights=2)
    assert manifest["complete"] and manifest["physical_flights"] == 3
    duel.verify(output, require_complete=True)
    assert duel.sha256(output / "A01/observations/part-00000.parquet") == first_shard
    obs, tracks, decisions = duel.load_forecasting_view(
        output, config["cases"][0]["episode_id"], "A01"
    )
    assert len(obs) == len(tracks) == 121
    assert set(decisions.aircraft_id) == {"A01"}
    assert set(decisions.actual_aircraft_id) == {"B01"}
    index = pd.read_parquet(output / "episodes.parquet")
    assert (index.groupby("scenario_group").size() == 2).all()
    with pytest.raises(ValueError, match="shard size"):
        duel.package(work, output, config, shard_flights=1)
    path = output / "B01/target_decisions/part-00000.parquet"
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        duel.verify(output)


def test_lock_excludes_another_writer_and_releases_after_error(tmp_path):
    with pytest.raises(RuntimeError, match="example failure"), duel.campaign_lock(tmp_path):
        with pytest.raises(RuntimeError, match="locked"), duel.campaign_lock(tmp_path):
            pytest.fail("Second writer acquired campaign lock")
        raise RuntimeError("example failure")
    assert not (tmp_path / "collection.lock").exists()


def test_metadata_stays_atomic_during_windows_reader_lock(tmp_path, monkeypatch):
    target = tmp_path / "progress.json"
    duel.write(target, {"completed": 1})
    native_replace = duel.os.replace
    attempts = []

    def temporarily_locked(source, destination):
        attempts.append(destination)
        assert duel.read(target) == {"completed": 1}
        if len(attempts) < 3:
            raise PermissionError("Windows reader has the old document open")
        native_replace(source, destination)

    monkeypatch.setattr(duel.os, "replace", temporarily_locked)
    monkeypatch.setattr(duel.time, "sleep", lambda _: None)
    duel.write(target, {"completed": 2})
    assert duel.read(target) == {"completed": 2}
    assert len(attempts) == 3
    assert b"\r" not in target.read_bytes()
    assert not target.with_suffix(".json.tmp").exists()


def test_notebook_is_valid_and_defaults_to_requested_campaign():
    nbformat = pytest.importorskip("nbformat")
    path = Path(__file__).parents[1] / "notebooks/11_nn_vs_nn_training_data.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    sources = "\n".join(c.source for c in notebook.cells)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.source.startswith("%%time\n")
            compile(cell.source.removeprefix("%%time\n"), str(path), "exec")
    assert '"10000"' in sources and '"BVR_NN_SECONDS", "300"' in sources
    assert 'run("verify", full_verify=True)' in sources
    assert "central_cuda" in sources and "estimate_campaign" in sources
    assert 'os.getenv("BVR_NN_PROGRESS_SECONDS", "300")' in sources
    assert "updates.run(args" in sources and "updates.finish()" in sources


def test_verification_receipt_skips_only_unchanged_indexed_payloads(campaign, monkeypatch):
    work, output, config = campaign
    for case in config["cases"]:
        save_episode(work, case)
    duel.package(work, output, config, shard_flights=2)
    cache = work / "receipt.json"
    native_load = duel._load_shard_episode
    calls = []

    def counted_load(*args):
        calls.append(args)
        return native_load(*args)

    monkeypatch.setattr(duel, "_load_shard_episode", counted_load)
    duel.verify(output, cache_path=cache)
    assert len(calls) == 6
    calls.clear()
    duel.verify(output, cache_path=cache)
    assert calls == []
    # A changed episode index invalidates its receipt even with identical payload files.
    index = pd.read_parquet(output / "episodes.parquet")
    index.loc[0, "frames"] += 1
    duel.save_table(output / "episodes.parquet", index)
    manifest = duel.read(output / "manifest.json")
    manifest["files"]["episodes.parquet"] = {
        "bytes": (output / "episodes.parquet").stat().st_size,
        "sha256": duel.sha256(output / "episodes.parquet"),
    }
    duel.write(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Index disagrees"):
        duel.verify(output, cache_path=cache)
    assert len(calls) == 1


def test_runtime_estimate_includes_upload_and_repeated_integrity_work():
    estimate = duel.estimate_campaign(
        1000, 10, 30, 2, 1_000_000, shard_flights=100, upload_mbps=20, scan_mbps=400
    )
    assert estimate["rollout_hours"] == pytest.approx(3000 / 3600)
    assert estimate["upload_hours"] == pytest.approx(400 / 3600)
    assert estimate["repeated_checksum_hours"] > 0
    assert estimate["total_hours"] == pytest.approx(
        sum(
            v
            for k, v in estimate.items()
            if k.endswith("_hours")
            and k not in ("total_hours", "sensitivity_low_hours", "sensitivity_high_hours")
        )
    )
    assert (
        estimate["sensitivity_low_hours"]
        < estimate["total_hours"]
        < estimate["sensitivity_high_hours"]
    )
    with pytest.raises(ValueError):
        duel.estimate_campaign(1000, 0, 1, 1, 1)


def test_central_backend_never_silently_changes_stochastic_sampling(tmp_path):
    with pytest.raises(ValueError, match="deterministic"):
        duel.prepare(tmp_path, flights=3, deterministic=False, inference_backend="central_cuda")


@pytest.mark.parametrize("location", ["root", "notebooks", "external"])
def test_notebook_loads_companions_by_file_path(tmp_path, location):
    import subprocess

    root = Path(__file__).parents[1]
    notebook = json.loads((root / "notebooks/11_nn_vs_nn_training_data.ipynb").read_text())
    cell = "".join(notebook["cells"][1]["source"])
    bootstrap = cell[cell.index("# PyCharm's project index") : cell.index("native_runtime =")]
    bootstrap += (
        'duel = load_companion("generate_nn_duel_corpus")\n'
        'NotebookProgress = load_companion("nn_duel_progress").NotebookProgress\n'
    )
    driver = """
import importlib.util
import os
import sys
import types
from pathlib import Path
root, location, bootstrap = sys.argv[1:]
if location == "external":
    os.environ["BVR_NN_ROOT"] = root
else:
    os.environ.pop("BVR_NN_ROOT", None)
# An already cached, unrelated module must not win over this checkout's file.
sys.modules["generate_nn_duel_corpus"] = types.ModuleType("generate_nn_duel_corpus")
exec(bootstrap)
assert Path(duel.__file__).resolve() == Path(root) / "scripts/generate_nn_duel_corpus.py"
assert sys.modules["generate_nn_duel_corpus"] is duel
assert NotebookProgress.__module__ == "nn_duel_progress"
"""
    cwd = {"root": root, "notebooks": root / "notebooks", "external": tmp_path}[location]
    result = subprocess.run(
        [sys.executable, "-c", driver, str(root), location, bootstrap],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
