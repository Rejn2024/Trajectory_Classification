import numpy as np
import pytest

from bvr_behavior_prediction.rl.environment import BluePilotEnvironment
from bvr_behavior_prediction.rl.reward import CombatReward


def test_substep_events_survive_aggregation_without_reclassifying_locked_launches():
    class Backend:
        def __init__(self, *_):
            self.tick = 0

        def reset(self, seed):
            self.tick = 0
            return np.zeros(1), {"incoming_missiles": 0}

        def step(self, *_):
            self.tick += 1
            launched = self.tick in (1, 2, 11)
            evaded = self.tick in (7, 8, 17)
            return np.zeros(1), 0.0, False, {
                "blue_alive": True,
                "blue_altitude_m": 6000,
                "target_locked": self.tick in (1, 2, 11),
                "fired": launched,
                "fired_with_lock": launched,
                "missile_avoided": evaded,
                "incoming_missiles": int(self.tick < 4),
                "supporting_missile": self.tick in (4, 5, 6),
                "red_policy": "simple",
                "reward_event_counts": {
                    "fired_with_lock": int(launched),
                    "fired_without_lock": 0,
                    "missile_avoided": int(evaded),
                },
            }

        def close(self):
            pass

    scenario = type("Scenario", (), {"as_dict": lambda self: {}})()
    env = BluePilotEnvironment(Backend, scenario, episode_duration_s=3)
    env.reset(7)
    reward = CombatReward()
    first = env.step("maintain_heading", {})[-1]
    assert not first["target_locked"]  # Final state remains the final state.
    assert first["fired_with_lock"] and first["missile_avoided"]
    assert first["reward_event_counts"]["fired_with_lock"] == 2
    assert first["reward_event_counts"]["missile_avoided"] == 2
    assert first["threatened_time_s"] == pytest.approx(0.4)
    assert first["missile_support_time_s"] == pytest.approx(0.3)
    assert first["red_policy"] == "simple"
    _, components = reward(first)
    assert components["fired_with_lock"] == 16
    assert components["fired_without_lock"] == 0
    assert components["missile_avoided"] == 40
    # Events in consecutive policy decisions still pay even when both aggregate
    # booleans are true; edge-detecting only those booleans would lose the second.
    second = env.step("maintain_heading", {})[-1]
    _, components = reward(second)
    assert components["fired_with_lock"] == 8
    assert components["missile_avoided"] == 20
    assert second["threatened_time_s"] == first["threatened_time_s"]
    assert second["missile_support_time_s"] == first["missile_support_time_s"]
    env.reset(8)
    reset_info = env.step("maintain_heading", {})[-1]
    assert reset_info["threatened_time_s"] == pytest.approx(0.4)
    assert reset_info["missile_support_time_s"] == pytest.approx(0.3)
    env.close()


def test_persistent_boolean_events_are_counted_once_for_generic_backends():
    class Backend:
        def __init__(self, *_):
            pass

        def reset(self, seed):
            return np.zeros(1), {}

        def step(self, *_):
            return np.zeros(1), 0.0, False, {"opponent_destroyed": True}

        def close(self):
            pass

    scenario = type("Scenario", (), {"as_dict": lambda self: {}})()
    env = BluePilotEnvironment(Backend, scenario)
    env.reset(7)
    assert env.step("maintain_heading", {})[-1]["reward_event_counts"]["opponent_destroyed"] == 1
    assert env.step("maintain_heading", {})[-1]["reward_event_counts"]["opponent_destroyed"] == 0
    env.close()
