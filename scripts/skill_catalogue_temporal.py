"""Descriptive temporal diagnostics from fixed forecasts and validation decision streams.

No training, inference, model selection or reserved-test evaluation is performed.
Decision Parquet row groups are selected by split before decoding their records.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from skill_catalogue_evidence import read_json, sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def decision_run_records(steps, skills, *, stride_steps=10, dt=0.1):
    """Summarise consecutive identical selections without bridging missing decisions.

    A completed run ends only when the next regularly spaced decision changes its
    label. A gap or episode end right-censors the run: no complete duration is
    fabricated. The last observation supplies a lower bound on its observed span,
    not a claim that the skill lasted another full decision interval. Gap-resumed
    and late-starting first runs are flagged as potentially left-censored too.
    """
    steps, skills = np.asarray(steps), np.asarray(skills)
    if steps.ndim != 1 or skills.shape != steps.shape:
        raise ValueError("Expected aligned one-dimensional decision steps and labels")
    if stride_steps <= 0 or dt <= 0:
        raise ValueError("Cadence and timestep must be positive")
    if not len(steps):
        return []
    if (
        not np.isfinite(steps).all()
        or not np.array_equal(steps, np.rint(steps))
        or np.any(steps < 0)
        or np.any(np.diff(steps) <= 0)
    ):
        raise ValueError("Decision steps must be distinct increasing nonnegative integers")
    regular = np.diff(steps) == stride_steps
    boundaries = np.flatnonzero(~regular | (skills[1:] != skills[:-1])) + 1
    starts, ends = np.r_[0, boundaries], np.r_[boundaries, len(steps)]
    rows = []
    for start, end in zip(starts, ends, strict=True):
        ended = end == len(steps)
        reason = "episode_end" if ended else ("switch" if regular[end - 1] else "gap")
        start_reason = (
            "episode_start" if start == 0 else ("switch" if regular[start - 1] else "gap_resume")
        )
        rows.append(
            {
                "skill": str(skills[start]),
                "start_step": int(steps[start]),
                "last_observed_step": int(steps[end - 1]),
                "start_time_s": float(steps[start] * dt),
                "consecutive_selections": int(end - start),
                "observed_span_s": float((steps[end - 1] - steps[start]) * dt),
                "duration_to_next_switch_s": float((steps[end] - steps[start]) * dt)
                if reason == "switch"
                else None,
                "right_censored": reason != "switch",
                "left_censored": start_reason == "gap_resume" or (start == 0 and steps[0] > 0),
                "end_reason": reason,
                "start_reason": start_reason,
                "episode_initial_run": start == 0,
            }
        )
    return rows


def fixed_prediction_scores(probabilities, targets):
    """Score fixed probabilities with stable class-index ties and explicit support."""
    p, target = np.asarray(probabilities, dtype=float), np.asarray(targets)
    if p.ndim != 2 or target.shape != (len(p),) or not len(p):
        raise ValueError("Expected nonempty aligned predictions and integer targets")
    if (
        not np.isfinite(p).all()
        or np.any(p < 0)
        or np.any(p > 1)
        or not np.allclose(p.sum(1), 1, atol=1e-6)
        or not np.array_equal(target, np.rint(target))
        or np.any(target < 0)
        or np.any(target >= p.shape[1])
    ):
        raise ValueError("Invalid probabilities or targets; mask missing labels first")
    target = target.astype(int)
    ranking = np.argsort(-p, axis=1, kind="stable")
    classes = p.shape[1]
    matrix = np.bincount(target * classes + ranking[:, 0], minlength=classes**2).reshape(
        classes, classes
    )
    support, predicted = matrix.sum(1), matrix.sum(0)
    f1 = np.divide(
        2 * matrix.diagonal(),
        support + predicted,
        out=np.zeros(classes),
        where=(support + predicted) > 0,
    )
    return {
        "count": len(target),
        "supported_skills": int((support > 0).sum()),
        "top1_accuracy": float((ranking[:, 0] == target).mean()),
        "top3_accuracy": float((ranking[:, :3] == target[:, None]).any(1).mean()),
        "macro_f1_catalogue": float(f1.mean()),
        "macro_f1_supported": float(f1[support > 0].mean()),
        "nll": float(-np.log(np.clip(p[np.arange(len(p)), target], 1e-12, 1)).mean()),
    }


def collect_temporal(dest):
    """Write portable CSV/JSON evidence; the caller manages bundle checksums."""
    data = Path(dest) / "data"
    data.mkdir(parents=True, exist_ok=True)
    prepared = ROOT / "artifacts/skill_forecasting/data_v1"
    corpus = ROOT / "shared/corpora/trained-pilot-episodes-v1"
    sources = {}

    def record(path):
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
        return path

    def write_csv(name, rows):
        frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
        frame.to_csv(data / (name + ".csv"), index=False, lineterminator="\n")

    all_samples = pd.read_parquet(record(prepared / "samples.parquet"))
    assert set(all_samples.split) == {"train", "validation"}
    selected = all_samples.split.eq("validation").to_numpy()
    targets = np.load(record(prepared / "targets.npy"))[selected]
    current = np.load(record(prepared / "current.npy"))[selected]
    samples = all_samples[selected].reset_index(drop=True)
    classes = read_json(record(prepared / "classes.json"))
    assert len(samples) == 30802 and len(classes) == 33
    assert samples.scenario_group.nunique() == 45 and samples.episode_id.nunique() == 450
    paths = {
        "transformer": ROOT / "artifacts/skill_forecasting/initial_v1/transformer",
        "ensemble": ROOT / "artifacts/skill_forecasting/improvement_v3/validation/ensemble",
    }
    probabilities = {}
    for model, path in paths.items():
        pd.testing.assert_frame_equal(
            samples, pd.read_parquet(record(path / "validation_samples.parquet"))
        )
        probabilities[model] = np.load(record(path / "validation_probabilities.npy"))
        assert probabilities[model].shape == (len(samples), 3, len(classes))

    common = (targets >= 0).all(axis=1)
    assert common.sum() == 29458
    common_rows, phase_rows, endpoint_rows = [], [], []
    for index, horizon in enumerate((1, 2, 5)):
        for model, p in probabilities.items():
            common_rows.append(
                {
                    "model": model,
                    "horizon_s": horizon,
                    **fixed_prediction_scores(p[common, index], targets[common, index]),
                }
            )
        phases = (
            ("6-20 s", samples.time_s.lt(21)),
            ("21-60 s", samples.time_s.ge(21) & samples.time_s.lt(61)),
            ("61+ s", samples.time_s.ge(61)),
        )
        for label, time_mask in phases:
            mask = common & time_mask.to_numpy()
            truth_counts = np.bincount(targets[mask, index], minlength=len(classes))
            known = mask & (current >= 0)
            for model, p in probabilities.items():
                phase_rows.append(
                    {
                        "model": model,
                        "horizon_s": horizon,
                        "origin_time_bin": label,
                        "episodes": samples.loc[mask, "episode_id"].nunique(),
                        "scenario_groups": samples.loc[mask, "scenario_group"].nunique(),
                        "pilots": samples.loc[mask, "pilot_id"].nunique(),
                        "known_current_count": int(known.sum()),
                        "changed_endpoint_fraction": float(
                            (targets[known, index] != current[known]).mean()
                        ),
                        "dominant_skill": classes[truth_counts.argmax()],
                        "dominant_skill_fraction": float(truth_counts.max() / truth_counts.sum()),
                        **fixed_prediction_scores(p[mask, index], targets[mask, index]),
                    }
                )
        available = targets[:, index] >= 0
        known = available & (current >= 0)
        same = known & (targets[:, index] == current)
        changed = known & (targets[:, index] != current)
        endpoint_rows.append(
            {
                "horizon_s": horizon,
                "available_count": int(available.sum()),
                "known_current_count": int(known.sum()),
                "same_count": int(same.sum()),
                "changed_count": int(changed.sum()),
                "same_endpoint_fraction": float(same.sum() / known.sum()),
                "changed_endpoint_fraction": float(changed.sum() / known.sum()),
            }
        )
    write_csv("temporal_common_origins", common_rows)
    write_csv("temporal_elapsed_phases", phase_rows)
    write_csv("temporal_endpoint_changes", endpoint_rows)

    assignments = pd.read_csv(record(prepared / "split_assignments.csv")).sort_values(
        ["pilot_id", "episode_id"]
    )
    manifest = read_json(record(corpus / "manifest.json"))
    run_rows, transitions, decision_counts = [], [], []
    row_groups_read, gaps, adjacent_pairs = 0, 0, 0
    for pilot, episodes in assignments.groupby("pilot_id", sort=True):
        relative = f"target_decisions/{pilot}.parquet"
        source = corpus / relative
        # Hashes verify provenance; only selected row groups below are decoded.
        assert sha(record(source)) == manifest["files"][relative]["sha256"]
        file = pq.ParquetFile(source)
        assert file.num_row_groups == len(episodes) == 300
        for row_group, episode in enumerate(episodes.itertuples()):
            if episode.split != "validation":
                continue  # Do not decode training or reserved-test decision records.
            frame = (
                file.read_row_group(
                    row_group, columns=["episode_id", "step", "time_s", "skill", "aircraft_id"]
                )
                .to_pandas()
                .sort_values("step")
            )
            assert set(frame.episode_id) == {episode.episode_id}
            assert set(frame.aircraft_id) == {"A01"}
            assert frame.skill.isin(classes).all()
            assert np.allclose(frame.time_s, frame.step * 0.1, atol=1e-7, rtol=0)
            steps, labels = frame.step.to_numpy(), frame.skill.to_numpy()
            records = decision_run_records(steps, labels)
            run_rows.extend(
                {"pilot_id": pilot, "episode_id": episode.episode_id, **row} for row in records
            )
            regular = np.diff(steps) == 10
            gaps += int((~regular).sum())
            adjacent_pairs += int(regular.sum())
            transitions.extend(
                {"pilot_id": pilot, "from_skill": before, "to_skill": after}
                for before, after, contiguous in zip(labels[:-1], labels[1:], regular, strict=True)
                if contiguous
            )
            decision_counts.extend(
                {"pilot_id": pilot, "skill": name, "count": int(count)}
                for name, count in frame.skill.value_counts().items()
            )
            row_groups_read += 1
    assert row_groups_read == 450
    runs, transitions, decisions = map(pd.DataFrame, (run_rows, transitions, decision_counts))
    summaries = []
    for pilot in ["population", *sorted(assignments.pilot_id.unique())]:
        r = runs if pilot == "population" else runs[runs.pilot_id.eq(pilot)]
        t = transitions if pilot == "population" else transitions[transitions.pilot_id.eq(pilot)]
        d = decisions if pilot == "population" else decisions[decisions.pilot_id.eq(pilot)]
        counts = d.groupby("skill")["count"].sum()
        completed = r[~r.right_censored & ~r.left_censored]
        duration = completed.duration_to_next_switch_s
        summaries.append(
            {
                "pilot_id": pilot,
                "decisions": int(counts.sum()),
                "adjacent_pairs": len(t),
                "one_second_repeat_fraction": float(t.from_skill.eq(t.to_skill).mean()),
                "switches": int(t.from_skill.ne(t.to_skill).sum()),
                "runs": len(r),
                "completed_runs": len(completed),
                "right_censored_runs": int(r.right_censored.sum()),
                "left_censored_runs": int(r.left_censored.sum()),
                "completed_run_median_s": float(duration.median()),
                "completed_run_p90_s": float(duration.quantile(0.9)),
                "completed_run_mean_s": float(duration.mean()),
                "dominant_skill": counts.idxmax(),
                "dominant_decision_fraction": float(counts.max() / counts.sum()),
                "supported_skills": len(counts),
            }
        )
    skill_summaries = []
    for skill, frame in runs.groupby("skill", sort=True):
        completed = frame[~frame.right_censored & ~frame.left_censored]
        durations = completed.duration_to_next_switch_s
        skill_summaries.append(
            {
                "skill": skill,
                "runs": len(frame),
                "completed_runs": len(completed),
                "right_censored_runs": int(frame.right_censored.sum()),
                "left_censored_runs": int(frame.left_censored.sum()),
                "completed_run_median_s": float(durations.median()) if len(completed) else None,
                "completed_run_p90_s": float(durations.quantile(0.9)) if len(completed) else None,
            }
        )
    write_csv("temporal_pilot_dynamics", summaries)
    write_csv("temporal_skill_runs", skill_summaries)
    write_csv("temporal_validation_runs", runs)
    write_csv(
        "temporal_transition_counts",
        transitions.groupby(["from_skill", "to_skill"]).size().rename("count").reset_index(),
    )
    write_json(
        data / "temporal_evidence.json",
        {
            "contract": "Descriptive analysis of fixed validation forecasts and validation-only decision records; no fitting, inference, selection or test evaluation.",
            "source_sha256": sources,
            "validation_forecast_origins": len(samples),
            "common_origins_all_horizons": int(common.sum()),
            "decision_row_groups_decoded": {"train": 0, "validation": row_groups_read, "test": 0},
            "regular_adjacent_decision_pairs": adjacent_pairs,
            "irregular_decision_gaps": gaps,
            "phase_boundaries_s": [21, 61],
            "phase_contract": "All phases use the common eligible origins across horizons. Full 20/60-second histories first occur at 21/61-second origins because acceleration needs prior frames and the reset is excluded.",
            "run_contract": "A run comprises consecutive identical catalogue choices at one-second cadence. Only a contiguous next different choice completes it. Gaps and episode ends right-censor; resumed or late first recordings may left-censor. Duration statistics use only fully observed completed runs, not an unbiased estimate of all run durations.",
            "limits": [
                "Full validation decision streams include early/terminal decisions outside eligible forecast origins; supports differ.",
                "Endpoint equality can conceal intervening changes and returns; it is not an event-duration measure.",
                "True-current-label persistence is a privileged diagnostic, not an admissible input baseline.",
                "Elapsed-time groups differ in surviving episodes, skill mix and available history, so their differences do not isolate history length's effect.",
                "Catalogue selections can share physical guidance; runs are not necessarily distinct executed manoeuvres.",
                "Overlapping forecasts are correlated; these descriptive tables do not supply independent-sample significance tests.",
                "The 33-label macro F1 uses a fixed catalogue; supported-label macro F1 may average different label sets across horizons.",
            ],
            "test_evaluated": False,
        },
    )
    print(
        f"Temporal evidence: {int(common.sum()):,} common forecast origins; {row_groups_read} validation decision groups decoded; zero test groups."
    )
