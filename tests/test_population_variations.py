import pytest

from bvr_behavior_prediction.rl.population_variations import sample_training_variations


def test_training_variations_are_reproducible_and_valid_for_attention():
    first = sample_training_variations(
        10, seed=42, widths=(56, 64, 72, 96), layers=(1, 2, 3), transformer_heads=(4, 8),
        value_coefficient_range=(0.45, 0.55), entropy_coefficient_range=(0.009, 0.011),
    )
    assert first == sample_training_variations(
        10, seed=42, widths=(56, 64, 72, 96), layers=(1, 2, 3), transformer_heads=(4, 8),
        value_coefficient_range=(0.45, 0.55), entropy_coefficient_range=(0.009, 0.011),
    )
    assert all(item["hidden_size"] % item["transformer_heads"] == 0 for item in first)
    assert all(0.45 <= item["value_coefficient"] <= 0.55 for item in first)
    assert all(0.009 <= item["entropy_coefficient"] <= 0.011 for item in first)
    assert len({tuple(item.items()) for item in first}) > 1


def test_training_variations_reject_incompatible_widths_and_heads():
    with pytest.raises(ValueError, match="divisible"):
        sample_training_variations(1, widths=(10,), transformer_heads=(4,))
