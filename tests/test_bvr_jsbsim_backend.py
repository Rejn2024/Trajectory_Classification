from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bvr_behavior_prediction.rl.bvr_jsbsim_backend import BVRJSBSimBackend


class _Aircraft:
    def __init__(self, uid, altitude, speed):
        self.uid = uid
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
        result = dict(action)
        result["delta_heading"] = np.clip(float(result["delta_heading"]), -np.deg2rad(45), np.deg2rad(45)) / np.deg2rad(45)
        result["delta_altitude"] = np.clip(float(result["delta_altitude"]), -80, 80) / 80
        result["delta_speed"] = np.clip(float(result["delta_speed"]), -80, 80) / 80
        return result


class _Environment:
    def __init__(self, config, logdir):
        self.config = config
        self.logdir = logdir
        self.dt = config["dt"]
        self.current_step = 0
        self.act_manager = _ActionManager()
        self.agents = {"A01": _Aircraft("A01", 6000, 250), "B01": _Aircraft("B01", 7000, 275)}
        self.render_path = None
        self.render_calls = 0
        self.closed = False
        self.last_actions = None

    def enable_render(self, path):
        self.render_path = path

    def reset(self, seed):
        self.seed = seed
        return np.zeros((1, 47), dtype=np.float32), {}

    def render(self):
        self.render_calls += 1

    def step(self, actions):
        self.current_step += 1
        self.last_actions = actions
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
        {"skill_name": "set_trajectory", "parameters": {}},
        {"skill_name": "set_trajectory", "parameters": {}},
    )
    assert observation.shape == (47,)
    assert reward == 1.5
    assert not terminated
    assert backend.env.render_calls == 2

    backend.close()
    assert backend.env.closed


def test_speed_skill_reaches_simulator_as_controller_command(tmp_path):
    backend = BVRJSBSimBackend(
        {"range_m": 30_000, "blue_altitude_m": 6000, "red_altitude_m": 7000},
        env_factory=_Environment, log_dir=tmp_path,
    )
    backend.reset(17)
    backend.step(
        {"skill_name": "set_trajectory", "parameters": {"speed_mps": 300}},
        {},
    )
    speed_command = backend.env.last_actions["A01"]["delta_speed"]
    assert 0.0 < speed_command < 1.0
    backend.close()


def test_launch_request_without_lock_is_counted_for_notch_reward_contract(tmp_path):
    backend = BVRJSBSimBackend(
        {"range_m": 30_000, "blue_altitude_m": 6000, "red_altitude_m": 7000},
        env_factory=_Environment, log_dir=tmp_path,
    )
    backend.reset(17)
    info = backend._aircraft_info(skill_name="launch", fired=0)
    assert info["fired"] is False
    assert info["reward_event_counts"]["fired_without_lock"] == 1
    backend.close()


def test_guidance_observation_exposes_radar_detected_target(tmp_path):
    backend = BVRJSBSimBackend(
        {"range_m": 30_000, "blue_altitude_m": 6000, "red_altitude_m": 7000},
        env_factory=_Environment, log_dir=tmp_path,
    )
    backend.reset(17)
    controlled = backend.env.agents[backend.CONTROLLED_ID]
    target = SimpleNamespace(
        uid="B01", position=np.array([1000.0, 2000.0, 6000.0]),
        velocity=np.array([250.0, 0.0, 0.0]),
    )
    controlled.enemies_lock.append(target)
    incoming = SimpleNamespace(
        uid="IN01", is_alive=True, target=controlled,
        position=np.array([2500.0, 0.0, 6000.0]), velocity=np.array([-300.0, 0.0, 0.0]),
    )
    controlled.under_missiles.append(incoming)
    controlled.launched_missiles.append(SimpleNamespace(
        uid="M01", is_alive=True, target=target,
        position=np.array([500.0, 0.0, 6000.0]), velocity=np.array([300.0, 0.0, 0.0]),
    ))
    guidance = backend._guidance_observation()
    assert guidance["targets"]["primary"]["id"] == "B01"
    assert guidance["targets"]["primary"]["position"] == [1000.0, 2000.0, 6000.0]
    assert guidance["incoming_missiles"][0]["id"] == "IN01"
    assert guidance["incoming_missiles"][0]["target_id"] == "A01"
    assert guidance["support_missiles"][0]["id"] == "M01"
    assert guidance["support_missiles"][0]["target_id"] == "B01"
    backend.close()


def test_python_environment_does_not_force_throttle_by_default():
    source = Path("bvr_sim_source/bvr_sim/bvr_env.py").read_text()
    assert 'BVR_BYPASS_RL_THROTTLE_CONTROL", "0"' in source


def test_backend_declares_expected_compact_observation_and_energy_fields():
    assert BVRJSBSimBackend.OBSERVATION_SIZE == 47
    assert BVRJSBSimBackend.HOSTILE_MISSILE_OBSERVATION_SIZE == 87
    source = Path("bvr_behavior_prediction/rl/bvr_jsbsim_backend.py").read_text()
    for field in (
        '"blue_speed_mps"',
        '"blue_altitude_m"',
        '"red_speed_mps"',
        '"red_altitude_m"',
    ):
        assert field in source


def test_backend_can_append_hostile_missile_slots_for_forecasting_pilots(tmp_path):
    backend = BVRJSBSimBackend(
        {"range_m": 30_000, "blue_altitude_m": 6000, "red_altitude_m": 7000},
        env_factory=_Environment, log_dir=tmp_path, include_hostile_missiles=True,
    )
    observation, _ = backend.reset(17)
    assert backend.observation_size == 87
    assert observation.shape == (87,)
    backend.close()


def test_backend_can_append_radar_features_after_hostile_missiles(tmp_path):
    backend = BVRJSBSimBackend(
        {"range_m": 30_000, "blue_altitude_m": 6000, "red_altitude_m": 7000},
        env_factory=_Environment, log_dir=tmp_path,
        include_hostile_missiles=True, include_radar_features=True,
    )
    observation, _ = backend.reset(17)
    assert backend.observation_size == 110
    assert observation.shape == (110,)
    assert backend.include_radar_features is True
    backend.close()


def test_tactical_selection_reaches_simulator_and_episode_metadata(tmp_path):
    backend = BVRJSBSimBackend(
        {"range_m": 30_000, "blue_altitude_m": 6000, "red_altitude_m": 7000},
        env_factory=_Environment, log_dir=tmp_path, opponent_type="tactical",
    )
    assert backend.env.config["blue_opponent_type"] == "tactical"
    assert backend.reset(7)[1]["red_policy"] == "tactical"
    assert _step(backend)["red_policy"] == "tactical"
    backend.close()
    with pytest.raises(ValueError, match="opponent_type"):
        BVRJSBSimBackend({}, opponent_type="misspelled")


def _backend(tmp_path):
    return BVRJSBSimBackend(
        {"range_m": 30_000, "blue_altitude_m": 6000, "red_altitude_m": 7000},
        env_factory=_Environment, log_dir=tmp_path,
    )


def _step(backend):
    return backend.step({"skill_name": "set_trajectory", "parameters": {}}, {})[-1]


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
