from pathlib import Path

import numpy as np

from bvr_behavior_prediction.rl.bvr_jsbsim_backend import BVRJSBSimBackend


class _Aircraft:
    def __init__(self, altitude, speed):
        self.position = np.array([0.0, 0.0, altitude])
        self.speed = speed
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
