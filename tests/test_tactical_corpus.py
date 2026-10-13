"""The export must preserve actual policy input frames and episode boundaries."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "tactical_corpus_schema", Path(__file__).parents[1] / "scripts/tactical_corpus_schema.py"
)
schema = importlib.util.module_from_spec(spec)
spec.loader.exec_module(schema)


def tables():
    keys = pd.DataFrame({"episode_id": "example", "step": np.arange(31),
                         "time_s": np.arange(31) * 0.1})
    observations = keys.copy()
    for column in schema.OBSERVATION_COLUMNS:
        observations[column] = np.zeros(31, dtype=np.float32)
    tracks = keys.copy()
    for column in schema.STATE_COLUMNS:
        tracks[column] = np.zeros(31)
    tracks["observer_alive"] = tracks["target_alive"] = True
    return observations, tracks


def test_history_reconstruction_matches_controller_left_padding():
    frames = np.arange(51 * 40, dtype=np.float32).reshape(40, 51)
    start = schema.reconstruct_policy_history(frames, 0)
    assert np.array_equal(start, np.repeat(frames[:1], 20, axis=0))
    partial = schema.reconstruct_policy_history(frames, 5)
    assert np.array_equal(partial[:15], np.repeat(frames[:1], 15, axis=0))
    assert np.array_equal(partial[-5:], frames[1:6])
    assert np.array_equal(schema.reconstruct_policy_history(frames, 30), frames[11:31])
    frames[31:] = -1000
    assert np.array_equal(schema.reconstruct_policy_history(frames, 30), frames[11:31])


def test_valid_episode_is_unchanged_by_validation():
    observations, tracks = tables()
    original = observations.copy()
    schema.validate_episode(observations, tracks)
    pd.testing.assert_frame_equal(observations, original)
    assert len(schema.OBSERVATION_COLUMNS) == 51


@pytest.mark.parametrize("failure", ["timestamp", "step", "episode", "nan", "resurrection"])
def test_rejects_corrupt_or_misaligned_samples(failure):
    observations, tracks = tables()
    if failure == "timestamp":
        observations.loc[2, "time_s"] += 1
    elif failure == "step":
        observations.loc[2, "step"] = tracks.loc[2, "step"] = 5
    elif failure == "episode":
        observations.loc[2, "episode_id"] = "other"
    elif failure == "nan":
        observations.loc[2, "obs_05"] = np.nan
    else:
        tracks.loc[2, "target_alive"] = False
    with pytest.raises(ValueError):
        schema.validate_episode(observations, tracks)


def test_atomic_metadata_write_retries_temporary_windows_reader_lock(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    import generate_tactical_corpus as collector

    original_replace = Path.replace
    attempts = []

    def temporarily_locked(path, target):
        attempts.append(target)
        if len(attempts) < 3:
            raise PermissionError("reader has the progress file open")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", temporarily_locked)
    monkeypatch.setattr(collector.time, "sleep", lambda _: None)
    target = tmp_path / "progress.json"
    collector.write(target, {"completed": 12})
    assert collector.read(target) == {"completed": 12}
    assert b"\r" not in target.read_bytes()
    assert len(attempts) == 3
    assert not target.with_suffix(".json.tmp").exists()
