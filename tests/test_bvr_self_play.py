import numpy as np
import pytest


pytest.importorskip("gymnasium")
from bvr_sim.bvr_env import BVR3DEnv


class _Agent:
    def __init__(self, uid):
        self.uid = uid
        self.is_alive = True


def test_expert_action_diagnostic_is_rectangular_for_self_play():
    environment = BVR3DEnv.__new__(BVR3DEnv)
    environment.ego_ids = ["A01", "B01"]
    environment.agents = {uid: _Agent(uid) for uid in environment.ego_ids}

    def expert_action(agent):
        return np.ones(4) if agent.uid == "A01" else None

    environment._expert_action = expert_action
    result = environment._gen_expert_action()

    assert result.shape == (1, 2, 4)
    np.testing.assert_array_equal(result[0, 0], np.ones(4))
    np.testing.assert_array_equal(result[0, 1], np.zeros(4))
