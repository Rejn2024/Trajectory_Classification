"""Shared outcome and behaviour measurements, independent of training weights."""

import math
from collections import Counter

import numpy as np


OUTCOME_SCORES = {
    "surviving_elimination": 100.0,
    "survived_without_elimination": 0.0,
    "mutual_destruction": -50.0,
    "loss_without_elimination": -100.0,
}


def wilson_interval(successes, count):
    """Approximate 95% binomial interval; uncertainty is across scenarios only."""
    if count < 1 or not 0 <= successes <= count:
        raise ValueError("provide a positive count and valid successes")
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z * z / count
    centre = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count**2)) / denominator
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def summarize_test_episodes(episodes, scenarios, seeds, skill_names):
    """Keep paired per-scenario outcomes as well as aggregate skill usage.

    A clean win requires survival AND a confirmed missile elimination. Both
    aircraft dying is undesirable even if the opponent's death was a crash.
    Surviving an opponent crash is not counted as a confirmed elimination.
    """
    if not episodes or not len(episodes) == len(scenarios) == len(seeds):
        raise ValueError("provide matching nonempty test episodes, scenarios and seeds")
    records = []
    total_skills = Counter()
    for index, ((rollout, reward, info), scenario, seed) in enumerate(zip(episodes, scenarios, seeds)):
        for key in ("blue_alive", "opponent_eliminated", "opponent_destroyed"):
            if key not in info:
                raise ValueError(f"test evaluation requires {key}")
        alive = bool(info["blue_alive"])
        eliminated = bool(info["opponent_eliminated"])
        destroyed = bool(info["opponent_destroyed"])
        if eliminated and not destroyed:
            raise ValueError("a confirmed elimination requires opponent destruction")
        outcome = (
            "surviving_elimination" if alive and eliminated else
            "survived_without_elimination" if alive else
            "mutual_destruction" if destroyed else "loss_without_elimination"
        )
        skills = Counter(skill_names[i] for i in rollout.skills)
        total_skills.update(skills)
        records.append({
            "scenario_index": index, "seed": seed, "scenario": scenario.as_dict(),
            "outcome": outcome, "outcome_score": OUTCOME_SCORES[outcome],
            "survived": alive, "eliminated": eliminated,
            "opponent_destroyed": destroyed, "benchmark_return": float(reward),
            "missiles_launched": info.get("missiles_launched_total"),
            "missiles_avoided": info.get("missiles_avoided_total"),
            "threatened_seconds": info.get("threatened_time_s"),
            "duration_seconds": info.get("elapsed_game_s"),
            "skill_counts": dict(skills),
        })

    def mean(key):
        values = [row[key] for row in records]
        return float(np.mean(values)) if all(v is not None for v in values) else None

    counts = Counter(row["outcome"] for row in records)
    n = len(records)
    summary = {
        "scenarios": n, "seed": seeds[0], "outcome_scores": dict(OUTCOME_SCORES),
        "outcome_score": mean("outcome_score"),
        "clean_win_rate": counts["surviving_elimination"] / n,
        "clean_win_ci95": wilson_interval(counts["surviving_elimination"], n),
        "mutual_destruction_rate": counts["mutual_destruction"] / n,
        "survived_without_elimination_rate": counts["survived_without_elimination"] / n,
        "loss_without_elimination_rate": counts["loss_without_elimination"] / n,
        "survival_rate": mean("survived"), "elimination_rate": mean("eliminated"),
        "benchmark_return": mean("benchmark_return"),
        "mean_missiles_launched": mean("missiles_launched"),
        "mean_missiles_avoided": mean("missiles_avoided"),
        "mean_threatened_seconds": mean("threatened_seconds"),
        "mean_duration_seconds": mean("duration_seconds"),
        "skill_decision_fractions": {
            name: count / sum(total_skills.values()) for name, count in sorted(total_skills.items())
        },
        "role": "post-selection test; never used by training or checkpoint selection",
    }
    return summary, records
