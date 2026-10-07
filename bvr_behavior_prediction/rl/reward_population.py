"""Varied weights for the original event-based combat reward."""

from dataclasses import asdict

from .config import RewardWeights


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
