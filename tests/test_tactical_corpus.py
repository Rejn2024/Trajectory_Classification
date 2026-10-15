"""The export must preserve actual policy input frames and episode boundaries."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize("observer_id,target_id", [("A01", "B01"), ("B01", "A01")])
def test_observer_frame_uses_correct_aircraft_and_missile_team(observer_id, target_id):
    # Exercise the real compact extractor with distinct aircraft and missiles.
    path = Path(__file__).parents[1] / "bvr_sim_source/bvr_sim/src_py/observation_space.py"
    compact_spec = importlib.util.spec_from_file_location("compact_observation", path)
    compact_module = importlib.util.module_from_spec(compact_spec)
    compact_spec.loader.exec_module(compact_module)

    def aircraft(uid, color, position, velocity, heading):
        return SimpleNamespace(uid=uid, color=color, is_alive=True, partners=[],
                               position=np.array(position, dtype=float),
                               velocity=np.array(velocity, dtype=float),
                               get_speed=lambda: float(np.linalg.norm(velocity)),
                               get_altitude=lambda: position[2], get_heading=lambda: heading)

    learned = aircraft("A01", "Red", [100, 300, 5000], [200, 100, 20], 0.4)
    tactical = aircraft("B01", "Blue", [8000, -6000, 4000], [-300, -20, -10], 2.0)
    learned.enemies, tactical.enemies = [tactical], [learned]
    red_missile = SimpleNamespace(color="Red", is_alive=True, target=tactical,
                                  position=np.array([400, 600, 5100]), velocity=np.array([700, 50, 10]))
    blue_missile = SimpleNamespace(color="Blue", is_alive=True, target=learned,
                                   position=np.array([7000, -5000, 3900]), velocity=np.array([-800, 0, 0]))
    sim = SimpleNamespace(agents={"A01": learned, "B01": tactical},
                          missiles={"red": red_missile, "blue": blue_missile},
                          obs_manager=compact_module.CompactObsSpace())
    observer, target = sim.agents[observer_id], sim.agents[target_id]
    own_missile = red_missile if observer_id == "A01" else blue_missile
    hostile_missile = blue_missile if observer_id == "A01" else red_missile
    frame = schema.observer_frame(sim, observer_id, target_id)
    assert frame.dtype == np.float32
    np.testing.assert_allclose(frame[:3], observer.position / [50000, 50000, 10000])
    np.testing.assert_allclose(frame[9:12], (target.position - observer.position) / [50000, 50000, 10000])
    np.testing.assert_allclose(frame[19:22], (own_missile.position - observer.position) / [50000, 50000, 10000])
    np.testing.assert_allclose(frame[47:], [(observer.get_speed() / 400) ** 2,
                                         observer.get_altitude() / 12000,
                                         (target.get_speed() / 400) ** 2,
                                         target.get_altitude() / 12000])
    assert frame[18] == 1
    hostile_missile.is_alive = False
    assert schema.observer_frame(sim, observer_id, target_id)[18] == 0
    target.is_alive = False
    assert not schema.observer_frame(sim, observer_id, target_id)[9:19].any()
    observer.is_alive = False
    assert not schema.observer_frame(sim, observer_id, target_id)[:47].any()


def test_geometry_validation_rejects_inverted_target_displacement():
    observations, tracks = tables()
    tracks["observer_x"], tracks["target_x"] = 1000.0, 11000.0
    observations["obs_00"], observations["obs_09"], observations["obs_15"] = 0.02, 0.2, 0.2
    schema.validate_observer_geometry(observations, tracks)
    observations["obs_09"] = -0.2
    with pytest.raises(ValueError, match="Observer geometry"):
        schema.validate_observer_geometry(observations, tracks)
