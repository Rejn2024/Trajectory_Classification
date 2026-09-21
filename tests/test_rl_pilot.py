import numpy as np
import pytest

from bvr_behavior_prediction.rl.config import PilotTrainingConfig, RewardWeights
from bvr_behavior_prediction.rl.reward import CombatReward
from bvr_behavior_prediction.rl.scenarios import ScenarioSampler


def test_config_enforces_episode_and_scenario_requirements():
    assert PilotTrainingConfig().decisions_per_episode == 90
    with pytest.raises(ValueError):
        PilotTrainingConfig(episode_duration_s=91)
    with pytest.raises(ValueError):
        PilotTrainingConfig(scenarios_per_epoch=2)


def test_scenarios_are_reproducible_diverse_and_safe():
    first = ScenarioSampler(42).sample_batch(3)
    second = ScenarioSampler(42).sample_batch(3)
    assert first == second
    assert len({item.range_m for item in first}) == 3
    assert len({item.bearing_deg for item in first}) == 3
    assert all(item.red_policy == "constant_course" for item in first)
    assert all(item.blue_altitude_m > 2500 and item.red_altitude_m > 2500 for item in first)


def test_reward_is_escalating_and_events_are_edge_triggered():
    reward = CombatReward(RewardWeights())
    safe, _ = reward({"blue_altitude_m": 6000})
    locked, _ = reward({"blue_altitude_m": 6000, "target_locked": True})
    repeated, _ = reward({"blue_altitude_m": 6000, "target_locked": True})
    evaded, _ = reward({"blue_altitude_m": 6000, "missile_avoided": True})
    killed, _ = reward({"blue_altitude_m": 6000, "opponent_destroyed": True})
    assert safe < locked < evaded < killed
    assert repeated == safe


def test_hybrid_policy_exposes_every_skill_and_nn_parameters():
    pytest.importorskip("torch")
    from bvr_behavior_prediction.rl.pilot import HybridSkillPilot

    pilot = HybridSkillPilot(12, hidden_size=16)
    assert len(pilot.skill_names) == 33
    name, params, _, _, raw = pilot.act(np.zeros(12, dtype=np.float32), deterministic=True)
    assert name in pilot.skill_names
    assert len(raw) == len(pilot.parameter_names)
    contract = pilot.manager.get_contract(name)["parameter_schema"]["properties"]
    assert set(params) == set(contract)
    for key, value in params.items():
        if contract[key]["type"] == "number":
            assert contract[key].get("minimum", -np.inf) <= value
            assert value <= contract[key].get("maximum", np.inf)
