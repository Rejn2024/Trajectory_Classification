from dataclasses import asdict

import pytest

from bvr_behavior_prediction.rl.dogfight_reward import (
    DogfightReward,
    DogfightWeights,
    HybridDogfightReward,
    dogfight_population_weights,
    dogfight_reward_definition,
    hybrid_dogfight_reward_definition,
)


def info(**changes):
    return {
        "blue_alive": True,
        "target_range_m": 30_000.0,
        "target_alignment": 1.0,
        "initial_target_range_m": 30_000.0,
        "initial_target_alignment": 1.0,
        "missiles_avoided_total": 0,
        "threatened_time_s": 0.0,
        "opponent_eliminated": False,
        **changes,
    }


@pytest.mark.parametrize("count", [1, 2, 3, 4, 10, 37])
def test_roster_covers_distinct_relative_preferences_for_arbitrary_counts(count):
    roster = dogfight_population_weights(count)
    assert len(roster) == count
    assert roster == dogfight_population_weights(count)
    assert len({tuple(asdict(weights).values()) for weights in roster}) == count
    for weights in roster:
        assert sum(asdict(weights).values()) == pytest.approx(1)
        assert min(asdict(weights).values()) >= 0.1 - 1e-12
    if count >= 4:
        for axis in ("evade", "pursue", "eliminate"):
            assert max(getattr(weights, axis) for weights in roster) == pytest.approx(0.8)


@pytest.mark.parametrize("weights", [(-1, 1, 1), (0, 0, 0), (float("nan"), 1, 1), (float("inf"), 1, 1)])
def test_invalid_weights_are_rejected(weights):
    with pytest.raises(ValueError):
        DogfightWeights(*weights)


def test_weight_ratios_change_which_outcome_is_preferred():
    evasion = info(missiles_avoided_total=4, threatened_time_s=12)
    elimination = info(opponent_eliminated=True)
    for weights, prefer_evasion in [(DogfightWeights(8, 1, 1), True), (DogfightWeights(1, 1, 8), False)]:
        evade_score = DogfightReward(weights)(evasion)[0]
        eliminate_score = DogfightReward(weights)(elimination)[0]
        assert (evade_score > eliminate_score) is prefer_evasion
    assert DogfightWeights(8, 1, 1) == DogfightWeights(80, 10, 10)


def test_pursuit_only_pays_new_progress_and_cannot_be_farmed_by_oscillation():
    reward = DogfightReward()
    assert reward(info())[0] == 0
    first = reward(info(target_range_m=20_000))[1]["pursue"]
    assert first > 0
    assert reward(info(target_range_m=30_000))[1]["pursue"] == 0
    assert reward(info(target_range_m=20_000))[1]["pursue"] == 0
    assert reward(info(target_range_m=10_000, target_alignment=-1))[1]["pursue"] == 0
    assert reward(info(target_range_m=10_000))[1]["pursue"] > 0


def test_evasion_counts_are_capped_and_exposure_is_time_normalized():
    reward = DogfightReward(DogfightWeights(1, 0, 0), episode_duration_s=120, reward_scale=1)
    assert reward(info(missiles_avoided_total=1, threatened_time_s=12))[0] == pytest.approx(0.15)
    assert reward(info(missiles_avoided_total=1, threatened_time_s=24))[0] == pytest.approx(-0.1)
    assert reward(info(missiles_avoided_total=8, threatened_time_s=24))[0] == pytest.approx(0.75)
    assert reward(info(missiles_avoided_total=12, threatened_time_s=24))[0] == 0


def test_only_confirmed_elimination_pays_and_loss_is_common_and_once_only():
    for weights in dogfight_population_weights(10):
        reward = DogfightReward(weights)
        assert reward(info(opponent_destroyed=True))[1]["eliminate"] == 0
        _, components = reward(info(opponent_eliminated=True, blue_alive=False))
        assert components["eliminate"] == pytest.approx(250 * weights.eliminate)
        assert components["loss"] == -250
        assert sum(components.values()) < 0
        assert reward(info(opponent_eliminated=True, blue_alive=False))[0] == 0


def test_skill_names_locks_launches_and_altitude_do_not_pay_extra_rewards():
    reward = DogfightReward()
    assert reward(info(
        blue_altitude_m=6000, target_locked=True, fired=True, fired_with_lock=True,
        skill_name="lead_pursuit", reward_event_counts={"fired_with_lock": 5},
    )) == (0.0, {"evade": 0.0, "pursue": 0.0, "eliminate": 0.0, "loss": 0.0})


def test_missing_signals_fail_loudly_and_factories_keep_independent_state():
    definition = dogfight_reward_definition("style", weights=DogfightWeights(6, 3, 1))
    first, second = definition.factory(), definition.factory()
    with pytest.raises(ValueError, match="requires simulator signals"):
        first({"blue_alive": True})
    assert first(info(missiles_avoided_total=1))[0] > 0
    assert first(info(missiles_avoided_total=1))[0] == 0
    assert second(info(missiles_avoided_total=1))[0] > 0
    first.reset()
    assert first(info(missiles_avoided_total=1))[0] > 0
    assert definition.parameters["weights"] == {"evade": 0.6, "pursue": 0.3, "eliminate": 0.1}


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_invalid_population_count_is_rejected(count):
    with pytest.raises(ValueError):
        dogfight_population_weights(count)


def test_hybrid_guidance_is_shared_bounded_and_cannot_be_farmed_by_lock_cycles():
    for weights in dogfight_population_weights(10):
        reward = HybridDogfightReward(weights)
        common = info(missile_support_time_s=0.0)
        _, first = reward({**common, "reward_event_counts": {"target_locked": 1, "fired_with_lock": 1}})
        assert first["lock_acquired"] == 2
        assert first["locked_launch"] == 8
        _, second = reward({**common, "missile_support_time_s": 10.0,
                            "reward_event_counts": {"target_locked": 20, "fired_with_lock": 20}})
        assert second["lock_acquired"] == second["locked_launch"] == 0
        assert second["missile_support"] == 2.5
        _, third = reward({**common, "missile_support_time_s": 100.0})
        assert third["missile_support"] == 2.5
        assert reward({**common, "missile_support_time_s": 110.0})[1]["missile_support"] == 0


def test_hybrid_does_not_reward_an_unlocked_launch_or_claim_an_unrelated_crash():
    reward = HybridDogfightReward()
    _, components = reward(info(
        missile_support_time_s=0.0, fired=True, target_locked=False,
        opponent_destroyed=True, reward_event_counts={"fired_without_lock": 1},
    ))
    assert sum(components.values()) == 0


def test_hybrid_shared_elimination_and_loss_keep_style_tradeoffs():
    for weights in dogfight_population_weights(10):
        reward = HybridDogfightReward(weights)
        outcome = info(missile_support_time_s=0.0, opponent_eliminated=True, blue_alive=False)
        _, components = reward(outcome)
        assert components["combat_elimination"] == 100
        assert components["eliminate"] == pytest.approx(250 * weights.eliminate)
        assert components["loss"] == -150
        assert reward(outcome)[0] == 0


def test_hybrid_pursuit_stops_paying_for_closure_inside_engagement_range():
    reward = HybridDogfightReward()
    initial = info(initial_target_range_m=45_000, target_range_m=45_000, missile_support_time_s=0.0)
    assert reward(initial)[1]["pursue"] == 0
    assert reward({**initial, "target_range_m": 30_000})[1]["pursue"] > 0
    assert reward({**initial, "target_range_m": 1_000})[1]["pursue"] == 0
    assert reward({**initial, "target_range_m": 35_000})[1]["pursue"] == 0
    assert reward({**initial, "target_range_m": 20_000})[1]["pursue"] == 0


def test_hybrid_recipe_records_all_settings_and_has_independent_resettable_state():
    definition = hybrid_dogfight_reward_definition("hybrid", lock_bonus=3, support_budget_s=10)
    assert definition.parameters["lock_bonus"] == 3
    assert definition.parameters["support_budget_s"] == 10
    assert definition.parameters["loss_penalty"] == 0.6
    first, second = definition.factory(), definition.factory()
    observation = info(target_locked=True, missile_support_time_s=10)
    assert first(observation)[0] == 8
    assert first(observation)[0] == 0
    assert second(observation)[0] == 8
    first.reset()
    assert first(observation)[0] == 8
    with pytest.raises(ValueError, match="missile_support_time_s"):
        first(info())


@pytest.mark.parametrize("settings", [
    {"lock_bonus": -1}, {"launch_bonus": float("nan")},
    {"support_budget_s": 0}, {"elimination_bonus": float("inf")},
])
def test_hybrid_rejects_invalid_common_settings(settings):
    with pytest.raises(ValueError):
        hybrid_dogfight_reward_definition(**settings)
