"""Protect dependence-aware aggregation and temporal censoring in the review."""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

_scripts = str(Path(__file__).resolve().parents[1] / "scripts")
sys.path.insert(0, _scripts)
try:
    _review = importlib.import_module("skill_catalogue_review")
    _temporal = importlib.import_module("skill_catalogue_temporal")
finally:
    sys.path.remove(_scripts)


def test_long_encounters_do_not_get_equal_episode_weight():
    result = _review.accuracy_weightings(
        [1, 1, 1, 0], ["long", "long", "long", "short"], ["p1", "p1", "p1", "p2"]
    )
    assert result["pooled_windows"] == 0.75
    assert result["equal_episodes"] == result["equal_pilots"] == 0.5
    assert result["episodes"] == result["pilots"] == 2


def test_bootstrap_preserves_whole_encounters_and_paired_zero_gain():
    groups = np.array(["long", "long", "long", "short"])
    correct = np.array([1, 1, 1, 0])
    result = _review.paired_group_intervals(np.zeros(4), correct, groups)
    assert result["gain"] == 0.75
    # Resampling two entire groups includes both-all-wrong and both-all-correct draws.
    assert result["gain_lower"] == 0 and result["gain_upper"] == 1
    same = _review.paired_group_intervals(correct, correct, groups)
    assert same["gain_lower"] == same["gain_upper"] == same["gain"] == 0
    with pytest.raises(ValueError, match="binary"):
        _review.paired_group_intervals([0.3], [1], ["a"])


def test_missing_decisions_do_not_fabricate_a_switch_or_complete_a_duration():
    runs = _temporal.decision_run_records([0, 10, 20, 40, 50, 60], ["a", "a", "b", "c", "c", "d"])
    assert [r["end_reason"] for r in runs] == ["switch", "gap", "switch", "episode_end"]
    assert runs[0]["duration_to_next_switch_s"] == 2
    assert runs[1]["duration_to_next_switch_s"] is None
    assert runs[1]["right_censored"]
    assert runs[2]["left_censored"] and not runs[2]["right_censored"]
    assert runs[-1]["duration_to_next_switch_s"] is None
    assert runs[-1]["observed_span_s"] == 0  # No invented final extra second.


def test_late_recording_start_and_duplicate_times_are_explicit():
    runs = _temporal.decision_run_records([30, 40, 50], ["a", "a", "b"])
    assert runs[0]["left_censored"]
    assert runs[-1]["right_censored"]
    with pytest.raises(ValueError, match="increasing"):
        _temporal.decision_run_records([0, 0], ["a", "b"])
