from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bvr_behavior_prediction.rl.bvr_jsbsim_backend import BVRJSBSimBackend


class _Aircraft:
    def __init__(self, altitude, speed):
        self.position = np.array([0.0, 0.0, altitude])
        self.speed = speed
        self.heading = 0.0
        self.velocity = np.array([speed, 0.0, 0.0])
        self.is_alive = True
        self.enemies_lock = []
        self.under_missiles = []
        self.launched_missiles = []
        self.color = "Red"

    def get_speed(self):
        return self.speed

    def get_altitude(self):
        return self.position[2]

    def get_heading(self):
        return self.heading


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
        self.last_actions = actions
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
        {"skill_name": "change_trajectory", "parameters": {}},
        {"skill_name": "change_trajectory", "parameters": {}},
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


def test_tactical_selection_reaches_simulator_and_episode_metadata(tmp_path):
    backend = BVRJSBSimBackend(
        {"range_m": 30_000, "blue_altitude_m": 6000, "red_altitude_m": 7000},
        env_factory=_Environment, log_dir=tmp_path, opponent_type="tactical",
    )
    assert backend.env.config["blue_opponent_type"] == "tactical"
    assert backend.reset(7)[1]["red_policy"] == "tactical"
    assert _step(backend)["red_policy"] == "tactical"
    backend.close()


@pytest.mark.parametrize("opponent", ["simple", "tactical"])
def test_three_hundred_second_limit_reaches_native_simulator(tmp_path, opponent):
    backend = BVRJSBSimBackend(
        {"range_m": 30_000, "blue_altitude_m": 6000, "red_altitude_m": 7000},
        env_factory=_Environment, log_dir=tmp_path, opponent_type=opponent,
        episode_duration_s=300,
    )
    assert backend.env.config["max_steps"] == 3000
    backend.close()
    with pytest.raises(ValueError, match="opponent_type"):
        BVRJSBSimBackend({}, opponent_type="misspelled")


def _backend(tmp_path):
    return BVRJSBSimBackend(
        {"range_m": 30_000, "blue_altitude_m": 6000, "red_altitude_m": 7000},
        env_factory=_Environment, log_dir=tmp_path,
    )


def _step(backend):
    return backend.step({"skill_name": "change_trajectory", "parameters": {}}, {})[-1]


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


@pytest.mark.parametrize(
    "name,parameters,expected",
    [
        ("change_trajectory", {"delta_heading_deg": 20}, [20 / 45, 0, 0, 0]),
        ("change_trajectory", {"delta_heading_deg": -20}, [-20 / 45, 0, 0, 0]),
        ("change_trajectory", {"delta_altitude_m": -40}, [0, -0.5, 0, 0]),
        ("change_trajectory", {"delta_altitude_m": 40}, [0, 0.5, 0, 0]),
        ("change_trajectory", {"delta_speed_mps": 40}, [0, 0, 0.5, 0]),
        ("change_trajectory", {"delta_speed_mps": -40}, [0, 0, -0.5, 0]),
        ("change_trajectory", {"delta_heading_deg": 20, "delta_altitude_m": -40,
                               "delta_speed_mps": 40}, [20 / 45, -0.5, 0.5, 0]),
        ("launch", {}, [0, 0, 0, 1]),
    ],
)
def test_reduced_skills_reach_the_simulator_with_normalized_physical_controls(
    tmp_path, name, parameters, expected,
):
    pytest.importorskip("gymnasium")
    from bvr_sim.action_space import CampusActionSpace

    backend = _backend(tmp_path)
    backend.env.act_manager = CampusActionSpace()
    backend.reset(7)
    backend.env.agents["A01"].heading = np.deg2rad(350)
    backend.step({"skill_name": name, "parameters": parameters}, {})
    command = backend.env.last_actions["A01"]
    controls = [command[key] for key in ("delta_heading", "delta_altitude", "delta_speed", "shoot")]
    np.testing.assert_allclose(controls, expected)
    backend.close()


def test_backend_holds_relative_delta_and_accepts_new_commands(tmp_path):
    backend = _backend(tmp_path)
    backend.reset(7)
    action = {"skill_name": "change_trajectory", "parameters": {"delta_altitude_m": 40}}
    backend.step(action, {})
    assert backend.env.last_actions["A01"]["delta_altitude"] == 40
    backend.env.agents["A01"].position[2] += 20
    backend.step(action, {})
    assert backend.env.last_actions["A01"]["delta_altitude"] == 40
    backend.step({"skill_name": "change_trajectory", "parameters": {"delta_altitude_m": -40}}, {})
    assert backend.env.last_actions["A01"]["delta_altitude"] == -40
    backend.close()


@pytest.fixture
def missile_backend(tmp_path):
    backend = BVRJSBSimBackend(
        {"range_m": 30_000, "blue_altitude_m": 6000, "red_altitude_m": 7000},
        env_factory=_Environment, log_dir=tmp_path, include_hostile_missiles=True,
    )
    yield backend
    backend.close()


def _incoming(backend, uid, relative_position, relative_velocity=(0, 0, 0), **overrides):
    aircraft = backend.env.agents["A01"]
    missile = SimpleNamespace(**{
        "uid": uid, "color": "Blue", "is_alive": True, "is_success": False,
        "target": aircraft, "radar_on": False,
        "position": aircraft.position + np.asarray(relative_position),
        "velocity": aircraft.velocity + np.asarray(relative_velocity),
        **overrides,
    })
    aircraft.under_missiles.append(missile)
    return missile


def test_hostile_missile_slots_encode_relative_geometry_and_seeker(missile_backend):
    backend = missile_backend
    _incoming(backend, "incoming", [3000, -4000, 12000], [-300, 400, -1200], radar_on=True)
    observation, _ = backend.reset(7)
    assert backend.observation_size == observation.size == 87
    assert observation.dtype == np.float32
    np.testing.assert_array_equal(observation[:47], np.zeros(47))
    np.testing.assert_allclose(
        observation[47:57], [1, .06, -.08, 1.2, -.25, 1 / 3, -1, .26, 13 / 12, 1],
        rtol=1e-6,
    )
    np.testing.assert_array_equal(observation[57:], np.zeros(30))
    # Read fresh state on step, including signed recession and seeker changes.
    missile = backend.env.agents["A01"].under_missiles[0]
    missile.velocity = backend.env.agents["A01"].velocity + [300, -400, 1200]
    missile.radar_on = False
    observation, *_ = backend.step({"skill_name": "launch", "parameters": {}}, {})
    np.testing.assert_array_equal(observation[:47], np.ones(47))
    assert observation[55] == pytest.approx(-13 / 12)
    assert observation[56] == 0


def test_hostile_slots_filter_history_and_keep_nearest_four_with_stable_ties(missile_backend):
    backend = missile_backend
    for uid, offset in (("far", 60), ("b", 10), ("e", 40), ("d", 30), ("c", 20), ("a", -10)):
        _incoming(backend, uid, [offset, 0, 0])
    _incoming(backend, "dead", [1, 0, 0], is_alive=False)
    _incoming(backend, "friendly", [2, 0, 0], color="Red")
    _incoming(backend, "other_target", [3, 0, 0], target=backend.env.agents["B01"])
    observation, _ = backend.reset(7)
    slots = observation[47:].reshape(4, 10)
    np.testing.assert_allclose(slots[:, 1] * 50_000, [-10, 10, 20, 30])
    np.testing.assert_array_equal(slots[:, 0], np.ones(4))
    backend.env.agents["A01"].under_missiles.reverse()
    reordered, _ = backend.reset(7)
    np.testing.assert_array_equal(observation, reordered)


def test_missile_history_is_sampled_and_resolved_slots_do_not_stay_stale(missile_backend):
    from bvr_behavior_prediction.rl.environment import BluePilotEnvironment

    backend = missile_backend
    missile = _incoming(backend, "incoming", [5000, 0, 0], [-600, 0, 0])
    environment = BluePilotEnvironment(lambda *_: backend, SimpleNamespace(as_dict=lambda: {}))
    initial = environment.reset(7)
    assert initial.shape == (20, 91)
    np.testing.assert_array_equal(initial[:, 47], np.ones(20))
    np.testing.assert_allclose(initial[0, 87:], [(250 / 400) ** 2, .5, (275 / 400) ** 2, 7000 / 12000])
    missile.is_alive = False
    history, *_ = environment.step("change_trajectory", {})
    np.testing.assert_array_equal(history[:10, 47], np.ones(10))
    np.testing.assert_array_equal(history[10:, 47:87], np.zeros((10, 40)))
    reset = environment.reset(8)
    np.testing.assert_array_equal(reset[:, 47:87], np.zeros((20, 40)))


def test_missile_zero_range_and_dead_aircraft_have_finite_slots(missile_backend):
    backend = missile_backend
    _incoming(backend, "incoming", [0, 0, 0], [-600, 0, 0])
    observation, _ = backend.reset(7)
    assert np.isfinite(observation).all()
    assert observation[47] == 1
    assert observation[54] == observation[55] == 0
    backend.env.agents["A01"].is_alive = False
    observation, *_ = backend.step({"skill_name": "launch", "parameters": {}}, {})
    np.testing.assert_array_equal(observation[47:], np.zeros(40))


def test_hostile_missile_mode_requires_an_explicit_boolean(tmp_path):
    with pytest.raises(ValueError, match="include_hostile_missiles must be a boolean"):
        BVRJSBSimBackend({}, include_hostile_missiles="false")
