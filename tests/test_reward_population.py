from dataclasses import asdict

import numpy as np
import pytest

from bvr_behavior_prediction.rl.config import RewardWeights
from bvr_behavior_prediction.rl.reward import CombatReward, combat_reward_definition
from bvr_behavior_prediction.rl.reward_population import (
    DEFAULT_COMBAT_REWARD_RANGES, combat_population_weights, random_combat_population_weights,
)


def test_uniform_ranges_are_actual_coefficients_reproducible_and_extendable():
    population = random_combat_population_weights(10, seed=42)
    assert population == random_combat_population_weights(13, seed=42)[:10]
    assert population != random_combat_population_weights(10, seed=43)
    assert len({tuple(asdict(w).values()) for w in population}) == 10
    for weights in population:
        for name, value in asdict(weights).items():
            low, high = DEFAULT_COMBAT_REWARD_RANGES[name]
            assert low <= value <= high
        assert weights.opponent_destroyed == 250
        assert weights.ground_clearance == 0.02
        assert weights.opponent_destroyed + weights.shot_down < 0
        assert weights.opponent_destroyed + weights.crashed < 0
    # The new sampler does not preserve the previous fixed-budget constraint.
    assert len({sum(abs(v) for v in asdict(w).values()) for w in population}) == 10


def test_uniform_sampler_honours_custom_ranges_without_changing_global_rng():
    ranges = {key: (value, value) for key, value in asdict(RewardWeights()).items()}
    np.random.seed(123)
    expected = np.random.random()
    np.random.seed(123)
    assert random_combat_population_weights(3, 17, ranges) == [RewardWeights()] * 3
    assert np.random.random() == expected
    for bad in ({}, {**ranges, "shot_down": (-100, 1)},
                {**ranges, "opponent_destroyed": (3, 2)},
                {**ranges, "incoming_missile_avoided": (0, float("nan"))}):
        with pytest.raises(ValueError):
            random_combat_population_weights(3, 17, bad)
    for seed in (-1, 2**32, True):
        with pytest.raises(ValueError, match="seed"):
            random_combat_population_weights(3, seed)


@pytest.mark.parametrize("count", [1, 2, 10, 17, 100])
def test_population_preserves_original_anchor_signs_budget_and_prefix(count):
    population = combat_population_weights(count)
    assert len(population) == count
    assert population[0] == RewardWeights()
    assert population == combat_population_weights(count + 3)[:count]
    assert len({tuple(asdict(w).values()) for w in population}) == count
    for weights in population:
        values = asdict(weights)
        assert values.pop("ground_clearance") == 0.02
        assert sum(abs(value) for value in values.values()) == pytest.approx(534)
        assert weights.shot_down / weights.crashed == pytest.approx(1.5)
        assert weights.fired_with_lock / weights.target_lock_acquired == pytest.approx(4)
        for field, value in values.items():
            assert np.isfinite(value)
            assert np.sign(value) == np.sign(getattr(RewardWeights(), field))
        assert isinstance(combat_reward_definition(weights=weights).factory(), CombatReward)


def test_population_varies_relative_preferences_in_both_directions():
    population = combat_population_weights(10)
    for field in ("shot_down", "incoming_missile_avoided", "fired_with_lock", "fired_without_lock"):
        ratios = [abs(getattr(w, field)) / w.opponent_destroyed for w in population]
        assert min(ratios) < ratios[0] < max(ratios)


@pytest.mark.parametrize("count", [0, -1, True, 1.5, "10"])
def test_invalid_population_count_is_rejected(count):
    with pytest.raises(ValueError, match="positive integer"):
        combat_population_weights(count)


def test_first_population_pilot_exactly_matches_original_event_reward():
    original = CombatReward()
    anchor = combat_reward_definition(weights=combat_population_weights(10)[0]).factory()
    events = [
        {"blue_altitude_m": 6000, "target_locked": True},
        {"blue_altitude_m": 6000, "target_locked": True,
         "reward_event_counts": {"fired_with_lock": 2, "missile_avoided": 3}},
        {"blue_altitude_m": 6000, "opponent_destroyed": True},
        {"blue_alive": False, "shot_down": True},
    ]
    for info in events:
        assert anchor(info) == original(info)
