"""Reproducible small architecture and PPO loss-weight variations for fresh pilots."""

import math

import numpy as np

SAMPLER_VERSION = "small-transformer-variations-v1"


def sample_training_variations(
    count, seed=42, widths=(56, 64, 72), layers=(1, 2, 3), transformer_heads=4,
    value_coefficient_range=(0.45, 0.55), entropy_coefficient_range=(0.009, 0.011),
):
    """Return one independent draw per pilot without consuming global RNG state.

    Architecture choices are uniform; coefficients are continuous uniform draws.
    Sampling in pilot order preserves existing draws when count is increased.
    The caller records the seed/options as well as these exact, unrounded values.
    """
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if type(transformer_heads) is not int or transformer_heads < 1:
        raise ValueError("transformer_heads must be a positive integer")
    widths, layers = tuple(widths), tuple(layers)
    for name, choices in (("widths", widths), ("layers", layers)):
        if (not choices or any(type(value) is not int or value < 1 for value in choices)
                or len(set(choices)) != len(choices)):
            raise ValueError(f"{name} must contain distinct positive integers")
    if any(width % transformer_heads for width in widths):
        raise ValueError("Every width must be divisible by transformer_heads")
    bounds = []
    for name, interval in (("value", value_coefficient_range), ("entropy", entropy_coefficient_range)):
        interval = tuple(interval)
        if len(interval) != 2 or any(
            not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or value < 0 for value in interval
        ) or interval[0] > interval[1]:
            raise ValueError(f"{name} coefficient range must be finite, nonnegative and ordered")
        bounds.append(interval)
    rng = np.random.default_rng(seed)
    return [
        {"hidden_size": int(rng.choice(widths)), "transformer_layers": int(rng.choice(layers)),
         "value_coefficient": float(rng.uniform(*bounds[0])),
         "entropy_coefficient": float(rng.uniform(*bounds[1]))}
        for _ in range(count)
    ]
