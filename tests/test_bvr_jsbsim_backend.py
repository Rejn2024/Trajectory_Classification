from pathlib import Path
from types import SimpleNamespace

import numpy as np

from bvr_behavior_prediction.rl.bvr_jsbsim_backend import BVRJSBSimBackend


class _Aircraft:
    def __init__(self, altitude, speed):
        self.position = np.array([0.0, 0.0, altitude])
        self.speed = speed
        self.velocity = np.array([speed, 0.0, 0.0])
        self.is_alive = True
        self.enemies_lock = []
        self.under_missiles = []
        self.launched_missiles = []

    def get_speed(self):
        return self.speed

    def get_altitude(self):
        return self.position[2]

    def get_heading(self):
        return 0.0


class _ActionManager:
    def legacy_norm_campus_action(self, action, to_std1=False):
        assert to_std1
        return action


class _Environment:
    def __init__(self, config, logdir):
        self.config = config
        self.logdir = logdir
        self.dt = config["dt"]
        self.current_step = 0
        self.act_manager = _ActionManager()
        self.agents = {"A01": _Aircraft(6000, 250), "B01": _Aircraft(7000, 275)}
        self.render_path = None
        self.render_calls = 0
        self.closed = False

    def enable_render(self, path):
        self.render_path = path

    def reset(self, seed):
        self.seed = seed
        return np.zeros((1, 47), dtype=np.float32), {}

    def render(self):
        self.render_calls += 1

    def step(self, actions):
        self.current_step += 1
        assert set(actions) == {"A01"}
        return np.ones((1, 47), dtype=np.float32), np.array([1.5]), np.array([False]), {}

    def close(self):
        self.closed = True


def test_backend_runs_bvr_flight_and_records_acmi(tmp_path):
    scenario = {
        "range_m": 30_000,
        "blue_altitude_m": 6000,
        "red_altitude_m": 7000,
    }
    replay = tmp_path / "flight.acmi"
    backend = BVRJSBSimBackend(
        scenario,
        recording_path=replay,
        env_factory=_Environment,
        log_dir=tmp_path / "logs",
    )

    assert backend.env.config["initial_separation_nm"] == 30_000 / 1852.0
    assert backend.env.config["blue_opponent_type"] == "simple"
    assert backend.env.render_path == str(replay)
    observation, info = backend.reset(17)
    assert observation.shape == (47,)
    assert info["blue_altitude_m"] == 6000
    assert info["red_speed_mps"] == 275

    observation, reward, terminated, info = backend.step(
        {"skill_name": "maintain_heading", "parameters": {}},
        {"skill_name": "maintain_heading", "parameters": {}},
    )
    assert observation.shape == (47,)
    assert reward == 1.5
    assert not terminated
    assert backend.env.render_calls == 2

    backend.close()
    assert backend.env.closed


def test_backend_declares_expected_compact_observation_and_energy_fields():
    assert BVRJSBSimBackend.OBSERVATION_SIZE == 47
    source = Path("bvr_behavior_prediction/rl/bvr_jsbsim_backend.py").read_text()
    for field in (
        '"blue_speed_mps"',
        '"blue_altitude_m"',
        '"red_speed_mps"',
        '"red_altitude_m"',
    ):
        assert field in source


def _backend(tmp_path):
    return BVRJSBSimBackend(
        {"range_m": 30_000, "blue_altitude_m": 6000, "red_altitude_m": 7000},
        env_factory=_Environment, log_dir=tmp_path,
    )


def _step(backend):
    return backend.step({"skill_name": "maintain_heading", "parameters": {}}, {})[-1]


def test_evasion_tracks_individual_live_missiles_despite_persistent_history(tmp_path):
    backend = _backend(tmp_path)
    blue = backend.env.agents[backend.CONTROLLED_ID]
    backend.reset(7)
    first = SimpleNamespace(uid="first", is_alive=True, is_success=False, target=blue)
    second = SimpleNamespace(uid="second", is_alive=True, is_success=False, target=blue)
    blue.under_missiles.extend([first, second])
    assert _step(backend)["incoming_missiles"] == 2
    first.is_alive = False
    info = _step(backend)
    assert info["incoming_missiles"] == 1
    assert info["missiles_avoided_total"] == 1
    assert info["reward_event_counts"]["missile_avoided"] == 1
    assert not _step(backend)["missile_avoided"]
    second.is_alive = False
    assert _step(backend)["missiles_avoided_total"] == 2
    assert len(blue.under_missiles) == 2
    backend.reset(8)
    assert _step(backend)["missiles_avoided_total"] == 0
    backend.close()


def test_missile_hit_and_aircraft_loss_are_not_evasions(tmp_path):
    backend = _backend(tmp_path)
    blue = backend.env.agents[backend.CONTROLLED_ID]
    backend.reset(7)
    missile = SimpleNamespace(uid="hit", is_alive=True, is_success=False, target=blue)
    blue.under_missiles.append(missile)
    _step(backend)
    missile.is_alive = False
    missile.is_success = True
    blue.is_alive = False
    info = _step(backend)
    assert not info["missile_avoided"]
    assert info["missiles_avoided_total"] == 0
    # Other missiles dying because their target died also must not earn evasion.
    blue.under_missiles.append(SimpleNamespace(
        uid="target_down", is_alive=False, is_success=False, target=blue,
    ))
    assert _step(backend)["missiles_avoided_total"] == 0
    backend.close()


def test_elimination_requires_a_successful_friendly_missile(tmp_path):
    backend = _backend(tmp_path)
    blue, red = backend.env.agents["A01"], backend.env.agents["B01"]
    backend.reset(7)
    red.is_alive = False
    info = _step(backend)
    assert info["opponent_destroyed"]
    assert not info["opponent_eliminated"]
    blue.launched_missiles.append(SimpleNamespace(is_success=True, target=red))
    assert _step(backend)["opponent_eliminated"]
    backend.close()


def test_geometry_measures_actual_flight_path_and_keeps_initial_reference(tmp_path):
    backend = _backend(tmp_path)
    blue, red = backend.env.agents["A01"], backend.env.agents["B01"]
    red.position = blue.position + np.array([30_000.0, 0, 0])
    _, initial = backend.reset(7)
    assert initial["target_alignment"] == 1.0
    red.position = blue.position + np.array([0, 20_000.0, 0])
    info = _step(backend)
    assert info["target_range_m"] == 20_000
    assert info["target_alignment"] == 0
    assert info["initial_target_range_m"] == 30_000
    assert info["initial_target_alignment"] == 1.0
    assert info["red_policy"] == "simple"
    backend.close()


def test_support_requires_a_live_locked_target_and_missile_without_active_seeker(tmp_path):
    backend = _backend(tmp_path)
    blue, red = backend.env.agents["A01"], backend.env.agents["B01"]
    backend.reset(7)
    blue.enemies_lock = [red]
    missile = SimpleNamespace(is_alive=True, is_success=False, radar_on=False, target=red)
    blue.launched_missiles.append(missile)
    assert _step(backend)["supporting_missile"]
    blue.enemies_lock = []
    assert not _step(backend)["supporting_missile"]
    blue.enemies_lock = [red]
    missile.radar_on = True
    assert not _step(backend)["supporting_missile"]
    missile.radar_on = False
    missile.is_alive = False
    assert not _step(backend)["supporting_missile"]
    missile.is_alive = True
    red.is_alive = False
    assert not _step(backend)["supporting_missile"]
    backend.close()
