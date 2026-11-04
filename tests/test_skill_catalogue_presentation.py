"""Numerical contracts for new descriptive probability and paired-error metrics."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "skill_catalogue_evidence", Path(__file__).resolve().parents[1] / "scripts/skill_catalogue_evidence.py"
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
diagnostic_metrics = _module.diagnostic_metrics
paired_changes = _module.paired_changes


def test_probability_metrics_have_known_multiclass_values_and_include_confidence_one():
    probabilities = np.array([[1.0, 0.0], [0.5, 0.5], [0.25, 0.75]])
    scores, bins = diagnostic_metrics(probabilities, np.array([0, 1, 0]))
    assert scores["top1_accuracy"] == pytest.approx(1 / 3)
    assert scores["brier"] == pytest.approx((0 + 0.5 + 1.125) / 3)
    assert scores["nll"] == pytest.approx(np.log(8) / 3)
    assert scores["ece_10_bins"] == pytest.approx((0.5 + 0.75) / 3)
    assert sum(row["count"] for row in bins) == 3
    assert bins[9]["count"] == 1 and bins[9]["accuracy"] == 1
    assert bins[0]["count"] == 0 and bins[0]["mean_confidence"] is None


def test_low_calibration_error_does_not_imply_high_accuracy():
    scores, _ = diagnostic_metrics(np.full((2, 2), 0.5), np.array([0, 1]))
    assert scores["ece_10_bins"] == 0
    assert scores["top1_accuracy"] == 0.5
    assert scores["brier"] == 0.5
    with pytest.raises(ValueError, match="Mask missing"):
        diagnostic_metrics(np.full((2, 2), 0.5), np.array([-1, 1]))


def test_paired_error_categories_are_exhaustive_and_net_gain_is_exact():
    target = np.zeros(4, dtype=int)
    old = np.eye(2)[[0, 1, 0, 1]]
    new = np.eye(2)[[0, 0, 1, 1]]
    result = paired_changes(old, new, target)
    assert [result[key] for key in ("both_correct", "corrected", "regressed", "both_wrong")] == [
        1,
        1,
        1,
        1,
    ]
    assert result["net_correct"] == 0 and result["accuracy_gain"] == 0
