"""Native simulator resources must not wait for Python's cyclic collector."""

import gc
import sys

import numpy as np
import pytest

pytest.importorskip("jsbsim")
pytest.importorskip("gymnasium")
pytest.importorskip("cv2")
pytest.importorskip("commentjson")

from bvr_sim import BVR3DEnv


def test_reset_and_close_release_both_native_handles_without_cyclic_gc(tmp_path):
    env = BVR3DEnv({
        "dt": .1, "max_steps": 10, "obs_type": "compact",
        "red_fighters": {"A01": {"model": "F16", "record": False}},
        "blue_fighters": {"B01": {"model": "F16", "record": False}},
        "blue_opponent_type": "simple",
    }, logdir=str(tmp_path))
    enabled = gc.isenabled()
    gc.disable()
    try:
        expected, _ = env.reset(seed=19)
        old = [a.fdm for a in env.agents.values()]
        assert all(fdm._jsbsim_exec is fdm.jsbsim_exec is not None for fdm in old)
        env.done_missiles["old_flight"] = object()
        actual, _ = env.reset(seed=19)
        np.testing.assert_array_equal(expected, actual)
        assert not env.done_missiles
        assert all(fdm._jsbsim_exec is fdm.jsbsim_exec is None for fdm in old)
        current = [a.fdm for a in env.agents.values()]
        assert all(fdm._initialized for fdm in current)
        env.close()
        env.close()
        assert all(fdm._jsbsim_exec is fdm.jsbsim_exec is None for fdm in current)
        assert all(not fdm._initialized for fdm in current)
    finally:
        env.close()
        if enabled:
            gc.enable()
        gc.collect()


def test_real_simulator_workers_do_not_import_torch_just_for_an_annotation(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("mlflow")
    from bvr_behavior_prediction.rl.bvr_jsbsim_backend import BVRJSBSimBackend
    from bvr_behavior_prediction.rl.config import PilotTrainingConfig
    from bvr_behavior_prediction.rl.environment import BluePilotEnvironment
    from bvr_behavior_prediction.rl.scenarios import ScenarioSampler
    from bvr_behavior_prediction.rl.simulation_workers import ProcessSimulatorPool

    def factory(scenario, path):
        def backend(scenario, recording_path=None):
            return BVRJSBSimBackend(scenario, log_dir=tmp_path, episode_duration_s=1)

        env = BluePilotEnvironment(backend, scenario, episode_duration_s=1)
        step = env.step

        def checked_step(*args):
            obs, terminated, truncated, info = step(*args)
            return obs, terminated, truncated, {**info, "torch_loaded": "torch" in sys.modules}

        env.step = checked_step
        return env

    config = PilotTrainingConfig(output_dir=tmp_path)
    scenarios = ScenarioSampler(config.seed).sample_batch(3)
    pool = ProcessSimulatorPool(factory, 1)
    try:
        pool.reset(scenarios, (1, 2, 3))
        results = pool.step([
            (i, "change_trajectory", {"delta_heading_deg": 0, "delta_altitude_m": 0,
                                      "delta_speed_mps": 0}) for i in range(3)
        ])
        assert all(not result[0][-1]["torch_loaded"] for result in results)
    finally:
        pool.close()
