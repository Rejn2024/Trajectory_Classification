"""Selection safety checks for the frozen v3-to-v4 comparison."""

import numpy as np
import pandas as pd
import pytest

from bvr_behavior_prediction.skill_forecasting.sequence_search import (
    average_aligned,
    blend_aligned,
    immutable_json,
    passes_guards,
    validate_partition,
)


def member(indices, predicted_class):
    probabilities = np.zeros((len(indices), 3, 33), dtype=np.float32)
    probabilities[:, :, predicted_class] = 1
    return probabilities, np.asarray(indices)


def test_ensemble_requires_exact_origin_order_and_normalized_probabilities():
    left, right = member([2, 4], 0), member([2, 4], 1)
    mean, indices = average_aligned([left, right])
    np.testing.assert_array_equal(indices, [2, 4])
    np.testing.assert_array_equal(mean[:, :, :2], np.full((2, 3, 2), 0.5))
    with pytest.raises(ValueError, match="different selection origins"):
        average_aligned([left, member([4, 2], 1)])
    with pytest.raises(ValueError, match="different origins"):
        blend_aligned(left, member([2, 5], 1), 0.5)
    with pytest.raises(ValueError, match="Invalid"):
        average_aligned([(left[0] * 0.5, left[1])])
    with pytest.raises(ValueError, match="Invalid"):
        average_aligned([member([2, 2], 0)])


def test_fixed_mixture_weights_preserve_endpoints_and_do_not_use_targets():
    left, right = member([2, 4], 0), member([2, 4], 1)
    np.testing.assert_array_equal(blend_aligned(left, right, 0)[0], left[0])
    np.testing.assert_array_equal(blend_aligned(left, right, 1)[0], right[0])
    mixture, _ = blend_aligned(left, right, 0.25)
    np.testing.assert_array_equal(mixture[:, :, 0], np.full((2, 3), 0.75))
    np.testing.assert_array_equal(mixture[:, :, 1], np.full((2, 3), 0.25))


def test_current_baseline_guards_reject_horizon_regression_and_nonpositive_gain():
    baseline = {"nll": 1.0, "mean_macro_f1": 0.4, "accuracy": [0.66, 0.65, 0.63], "mean_accuracy": 0.646}
    candidate = {**baseline, "accuracy": [0.68, 0.66, 0.64], "mean_accuracy": 0.66}
    assert passes_guards(candidate, baseline, require_gain=True)
    assert not passes_guards({**candidate, "nll": 1.021}, baseline)
    assert not passes_guards({**candidate, "mean_macro_f1": 0.389}, baseline)
    assert not passes_guards({**candidate, "accuracy": [0.7, 0.7, 0.62]}, baseline)
    assert not passes_guards(baseline, baseline, require_gain=True)
    assert passes_guards(baseline, baseline)


def test_frozen_decisions_allow_identical_resume_but_refuse_changed_choice(tmp_path):
    path = tmp_path / "selection.json"
    immutable_json(path, {"weight": 0.5, "names": ["a", "b"]})
    original = path.read_bytes()
    immutable_json(path, {"weight": 0.5, "names": ["a", "b"]})
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="Frozen decision changed"):
        immutable_json(path, {"weight": 0.75, "names": ["a", "b"]})
    assert path.read_bytes() == original


def test_partition_rejects_outer_group_even_when_sizes_are_unchanged():
    partition = {f"g{i}": ("fit" if i < 147 else "stop" if i < 168 else "select") for i in range(210)}
    samples = pd.DataFrame({"scenario_group": [*partition, "outer"], "split": ["train"] * 210 + ["validation"]})
    validate_partition(partition, samples)
    contaminated = dict(partition)
    contaminated["outer"] = contaminated.pop("g209")
    with pytest.raises(ValueError, match="original-training"):
        validate_partition(contaminated, samples)
    with pytest.raises(ValueError, match="original-training"):
        validate_partition(partition, samples.assign(split="test"))
