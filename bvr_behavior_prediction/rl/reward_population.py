"""Varied weights for the original event-based combat reward."""

from dataclasses import asdict
import math

import numpy as np

from .config import RewardWeights


# Hypotheses for the next experiment, not empirically calibrated optima. Keeping
# destruction fixed supplies a common training unit; loss penalties make the
# terminal outcome contribution of a mutual destruction negative for every pilot.
DEFAULT_COMBAT_REWARD_RANGES = {
    "crashed": (-450.0, -300.0),
    "shot_down": (-450.0, -300.0),
    "ground_clearance": (0.02, 0.02),
    "target_lock_acquired": (0.0, 2.0),
    "fired_with_lock": (0.0, 2.0),
    "incoming_missile_avoided": (0.0, 15.0),
    "opponent_destroyed": (250.0, 250.0),
    "fired_without_lock": (-12.0, -4.0),
}


def random_combat_population_weights(count: int, seed: int, ranges=None):
    """Sample actual coefficients uniformly, with a separate reproducible RNG.

    No Halton sequence, multiplication of the old weights, or post-normalization.
    Every pilot is sampled. Degenerate intervals hold a coefficient fixed. Sampling
    one pilot at a time preserves existing entries when the population is extended.
    """
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("reward seed must be an integer in [0, 2**32)")
    ranges = dict(DEFAULT_COMBAT_REWARD_RANGES if ranges is None else ranges)
    original = asdict(RewardWeights())
    if set(ranges) != set(original):
        raise ValueError("ranges must specify exactly the RewardWeights fields")
    for name, bounds in ranges.items():
        if len(bounds) != 2 or not all(math.isfinite(v) for v in bounds):
            raise ValueError(f"{name} requires two finite bounds")
        low, high = bounds
        if low > high or (original[name] < 0 and high > 0) or (original[name] > 0 and low < 0):
            raise ValueError(f"{name} bounds must be ordered and preserve the reward sign")
    rng = np.random.default_rng(seed)
    return [RewardWeights(**{
        name: float(rng.uniform(*ranges[name])) for name in original
    }) for _ in range(count)]


def _radical_inverse(index, base):
    value, denominator = 0.0, 1.0
    while index:
        index, digit = divmod(index, base)
        denominator *= base
        value += digit / denominator
    return value


def combat_population_weights(count: int):
    """Return a reproducible, extendable roster beginning with the original weights.

    Five independent Halton coordinates vary loss, destruction, evasion, locked
    weapon use, and unlocked firing costs. Log-spaced multipliers span 0.5 to 2.
    Normalize the sum of absolute event coefficients to the original budget so
    preferences change without simply multiplying the entire reward. This does
    not equalize realized returns, since event frequencies depend on behaviour.
    The small per-decision ground-clearance reward remains fixed at 0.02.
    """
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    original = RewardWeights()
    groups = (
        ("crashed", "shot_down"),
        ("opponent_destroyed",),
        ("incoming_missile_avoided",),
        ("target_lock_acquired", "fired_with_lock"),
        ("fired_without_lock",),
    )
    event_fields = tuple(name for group in groups for name in group)
    budget = sum(abs(getattr(original, name)) for name in event_fields)
    population = [original]
    for index in range(1, count):
        weights = asdict(original)
        for group, prime in zip(groups, (2, 3, 5, 7, 11)):
            multiplier = 2 ** (2 * _radical_inverse(index, prime) - 1)
            for name in group:
                weights[name] *= multiplier
        scale = budget / sum(abs(weights[name]) for name in event_fields)
        for name in event_fields:
            weights[name] *= scale
        population.append(RewardWeights(**weights))
    return population
