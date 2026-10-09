from types import SimpleNamespace

import pytest

from bvr_behavior_prediction.rl.evaluation import summarize_test_episodes, wilson_interval


def test_common_outcomes_ignore_reward_totals_and_preserve_paired_cases():
    # The mutual kill earns an enormous shaped return but must still rank below
    # survival and a clean win. An opponent crash is not a confirmed clean win.
    outcomes = [(True, True, True), (True, False, False),
                (False, True, True), (False, False, False), (True, False, True)]
    episodes = [
        (SimpleNamespace(skills=[0, 1]), float(i * 10000), {
            "blue_alive": alive, "opponent_eliminated": killed,
            "opponent_destroyed": destroyed, "missiles_launched_total": i,
            "missiles_avoided_total": 2, "threatened_time_s": 10.0,
            "elapsed_game_s": 30.0,
        })
        for i, (alive, killed, destroyed) in enumerate(outcomes)
    ]
    scenarios = [SimpleNamespace(as_dict=lambda: {"range_m": 18000})] * 5
    summary, records = summarize_test_episodes(episodes, scenarios, list(range(100, 105)), ("a", "b"))
    assert summary["clean_win_rate"] == 0.2
    assert summary["mutual_destruction_rate"] == 0.2
    assert summary["survived_without_elimination_rate"] == 0.4
    assert summary["loss_without_elimination_rate"] == 0.2
    assert summary["outcome_score"] == -10
    assert [row["seed"] for row in records] == list(range(100, 105))
    assert records[2]["outcome_score"] == -50
    assert records[4]["outcome_score"] == 0
    assert summary["mean_missiles_launched"] == 2
    assert summary["skill_decision_fractions"] == {"a": 0.5, "b": 0.5}
    low, high = summary["clean_win_ci95"]
    assert low < 0.2 < high


def test_wilson_interval_has_uncertainty_at_zero_and_full_success():
    assert wilson_interval(0, 100) == pytest.approx([0, 0.0369935], abs=1e-7)
    assert wilson_interval(100, 100) == pytest.approx([0.9630065, 1], abs=1e-7)
    with pytest.raises(ValueError):
        wilson_interval(0, 0)


def test_missing_outcome_signals_are_not_silently_scored_as_failure():
    with pytest.raises(ValueError, match="blue_alive"):
        summarize_test_episodes([(SimpleNamespace(skills=[]), 0.0, {})],
                                [SimpleNamespace(as_dict=lambda: {})], [1], ())
