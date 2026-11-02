"""Collect and audit fixed skill predictions; derive descriptive comparisons only."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("frequency", "gru", "transformer", "ensemble")
LABELS = {
    "frequency": "Training frequency",
    "gru": "Original GRU",
    "transformer": "Original transformer",
    "ensemble": "Improved ensemble",
}
HORIZONS = (1, 2, 5)


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )


def diagnostic_metrics(probabilities, target):
    """Multiclass Brier and top-label calibration, using ten fixed equal bins."""
    p, target = np.asarray(probabilities, dtype=float), np.asarray(target, dtype=int)
    if p.ndim != 2 or target.shape != (len(p),) or not len(p):
        raise ValueError("Expected a nonempty probability matrix and aligned targets")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Invalid probabilities")
    np.testing.assert_allclose(p.sum(1), 1, atol=1e-6)
    if (target < 0).any() or (target >= p.shape[1]).any():
        raise ValueError("Mask missing targets before deriving metrics")
    order = np.argsort(-p, axis=1, kind="stable")
    confidence = p.max(1)
    correct = order[:, 0] == target
    bins = np.minimum((confidence * 10).astype(int), 9)
    reliability, ece = [], 0.0
    for b in range(10):
        mask = bins == b
        n = int(mask.sum())
        conf = float(confidence[mask].mean()) if n else None
        acc = float(correct[mask].mean()) if n else None
        if n:
            ece += n / len(p) * abs(conf - acc)
        reliability.append(
            {
                "bin": b,
                "lower": b / 10,
                "upper": (b + 1) / 10,
                "count": n,
                "mean_confidence": conf,
                "accuracy": acc,
            }
        )
    result = {
        "count": len(p),
        "top1_accuracy": float(correct.mean()),
        "top3_accuracy": float((order[:, :3] == target[:, None]).any(1).mean()),
        "nll": float(-np.log(np.clip(p[np.arange(len(p)), target], 1e-12, 1)).mean()),
        "brier": float((np.sum(p * p, axis=1) - 2 * p[np.arange(len(p)), target] + 1).mean()),
        "ece_10_bins": float(ece),
        "mean_confidence": float(confidence.mean()),
    }
    return result, reliability


def paired_changes(old, new, target):
    old_correct = np.asarray(old).argmax(1) == target
    new_correct = np.asarray(new).argmax(1) == target
    result = {
        "count": len(target),
        "both_correct": int((old_correct & new_correct).sum()),
        "corrected": int((~old_correct & new_correct).sum()),
        "regressed": int((old_correct & ~new_correct).sum()),
        "both_wrong": int((~old_correct & ~new_correct).sum()),
    }
    assert sum(
        result[key] for key in ("both_correct", "corrected", "regressed", "both_wrong")
    ) == len(target)
    result["net_correct"] = result["corrected"] - result["regressed"]
    result["accuracy_gain"] = result["net_correct"] / len(target)
    return result


def collect(dest):
    data = dest / "data"
    data.mkdir(parents=True, exist_ok=True)
    base = ROOT / "docs/presentations/forecasting_insights/data"
    for name, expected in read_json(base / "checksums.json").items():
        assert sha(base / name) == expected
    overview = read_json(base / "evidence.json")
    sources = dict(overview["source_sha256"])
    for name, expected in sources.items():
        assert sha(ROOT / name) == expected, f"Changed source evidence: {name}"

    def record(path):
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
        return path

    directory = ROOT / "artifacts/skill_forecasting"
    prepared = directory / "data_v1"
    samples = pd.read_parquet(record(prepared / "samples.parquet"))
    valid_samples = samples[samples.split.eq("validation")].reset_index(drop=True)
    targets = np.load(record(prepared / "targets.npy"))[samples.split.eq("validation")]
    assert len(valid_samples) == 30802 and valid_samples.scenario_group.nunique() == 45
    paths = {
        "frequency": directory / "initial_v1/training_frequency",
        "gru": directory / "initial_v1/gru",
        "transformer": directory / "initial_v1/transformer",
        "ensemble": directory / "improvement_v3/validation/ensemble",
    }
    summary, classes, diagnostics, reliability, confusions = [], [], [], [], {}
    probabilities, run_details = {}, {}
    for model, path in paths.items():
        frame = pd.read_csv(record(path / "validation_metrics.csv"))
        frame["model"] = model
        summary.append(frame)
        per_class = pd.read_csv(record(path / "per_class_metrics.csv"))
        per_class = per_class[per_class.pilot_id.eq("population")].copy()
        per_class["model"] = model
        classes.append(per_class)
        if model == "frequency":
            baseline = read_json(record(path / "baseline.json"))
            probs = np.broadcast_to(np.array(baseline["probabilities"]), (len(targets), 3, 33))
        else:
            probs = np.load(record(path / "validation_probabilities.npy"))
            pd.testing.assert_frame_equal(
                valid_samples, pd.read_parquet(record(path / "validation_samples.parquet"))
            )
        assert probs.shape == (len(targets), 3, 33)
        probabilities[model] = probs
        if model in ("gru", "transformer"):
            run = read_json(record(path / "run.json"))
            run_details[model] = {
                key: run[key] for key in ("model", "training", "parameters", "selection")
            }
            run_details[model]["summary"] = read_json(record(path / "summary.json"))
        for j, h in enumerate(HORIZONS):
            mask = targets[:, j] >= 0
            p, y = probs[mask, j], targets[mask, j]
            scores, bins = diagnostic_metrics(p, y)
            old = frame[
                frame.pilot_id.eq("population") & frame.cohort.eq("all") & frame.horizon_s.eq(h)
            ].iloc[0]
            assert scores["count"] == int(old["count"])
            for field in ("top1_accuracy", "top3_accuracy", "nll"):
                np.testing.assert_allclose(scores[field], old[field], atol=1e-6)
            matrix = np.bincount(y * 33 + p.argmax(1), minlength=33 * 33).reshape(33, 33)
            saved_matrix = pd.read_csv(record(path / f"confusion_{h}s.csv"), index_col=0).to_numpy()
            np.testing.assert_array_equal(matrix, saved_matrix)
            support, predicted = matrix.sum(1), matrix.sum(0)
            f1 = np.divide(
                2 * matrix.diagonal(),
                support + predicted,
                out=np.zeros(33),
                where=(support + predicted) > 0,
            )
            np.testing.assert_allclose(f1.mean(), old.macro_f1_catalogue, atol=1e-12)
            np.testing.assert_allclose(f1[support > 0].mean(), old.macro_f1_supported, atol=1e-12)
            ordered = (
                per_class[per_class.horizon_s.eq(h)].set_index("skill").loc[overview["classes"]]
            )
            np.testing.assert_array_equal(ordered.support, support)
            np.testing.assert_allclose(ordered.f1, f1, atol=1e-12)
            np.testing.assert_allclose(
                ordered.precision,
                np.divide(matrix.diagonal(), predicted, out=np.zeros(33), where=predicted > 0),
                atol=1e-12,
            )
            np.testing.assert_allclose(
                ordered.recall,
                np.divide(matrix.diagonal(), support, out=np.zeros(33), where=support > 0),
                atol=1e-12,
            )
            confusions[f"{model}_{h}"] = matrix.tolist()
            diagnostics.append({"model": model, "horizon_s": h, **scores})
            reliability.extend({"model": model, "horizon_s": h, **row} for row in bins)
    transitions = []
    for j, h in enumerate(HORIZONS):
        mask = targets[:, j] >= 0
        transitions.append(
            {
                "horizon_s": h,
                **paired_changes(
                    probabilities["transformer"][mask, j],
                    probabilities["ensemble"][mask, j],
                    targets[mask, j],
                ),
            }
        )
    for filename, frame in (
        ("summary_metrics.csv", pd.concat(summary)),
        ("class_metrics.csv", pd.concat(classes)),
        ("probability_metrics.csv", pd.DataFrame(diagnostics)),
        ("reliability.csv", pd.DataFrame(reliability)),
        ("paired_changes.csv", pd.DataFrame(transitions)),
    ):
        frame.to_csv(data / filename, index=False, lineterminator="\n")
    for source, name in (
        (base / "scenario_intervals.csv", "scenario_intervals.csv"),
        (base / "internal_confirmation.csv", "internal_confirmation.csv"),
        (directory / "improvement_v2/search_results.csv", "initial_search.csv"),
    ):
        (data / name).write_text(
            record(source).read_text(encoding="utf-8"), encoding="utf-8", newline="\n"
        )
    (data / "confusions.json").write_text(
        json.dumps(confusions, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n"
    )
    evidence = {
        "snapshot_date": overview["snapshot_date"],
        "revision": "Expanded catalogue comparison",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "classes": overview["classes"],
        "counts": overview["skill_counts"],
        "original_runs": run_details,
        "components": overview["components"],
        "runtime_check": overview["runtime_check"],
        "verification": overview["verification"],
        "internal_selection": overview["internal_selection"],
        "probability_metric_contract": "Multiclass Brier is sum over all 33 classes, range 0..2. ECE uses ten fixed equal-width top-label confidence bins; 1.0 belongs to the final bin.",
        "analysis_contract": "Descriptive analysis of saved validation probabilities; no refitting, calibration changes, model selection or test evaluation.",
        "source_sha256": sources,
        "test_evaluated": False,
    }
    write_json(data / "evidence.json", evidence)
    write_json(
        data / "checksums.json",
        {
            path.name: sha(path)
            for path in sorted(data.iterdir())
            if path.is_file() and path.name != "checksums.json"
        },
    )
    print(
        "Audited all 12 model/horizon scorecards against saved probabilities and confusion matrices."
    )


def load(dest):
    data = dest / "data"
    for name, expected in read_json(data / "checksums.json").items():
        assert sha(data / name) == expected, name
    tables = {
        name: pd.read_csv(data / (name + ".csv"))
        for name in (
            "summary_metrics",
            "class_metrics",
            "probability_metrics",
            "reliability",
            "paired_changes",
            "scenario_intervals",
            "internal_confirmation",
            "initial_search",
        )
    }
    evidence = read_json(data / "evidence.json")
    matrices = {key: np.array(value) for key, value in read_json(data / "confusions.json").items()}
    return evidence, tables, matrices


def markdown_table(frame):
    rows = [
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
    ]
    return "\n".join(
        rows
        + [
            "| " + " | ".join(map(str, row)) + " |"
            for row in frame.itertuples(index=False, name=None)
        ]
    )


def write_comparison(dest, evidence, tables):
    summary = tables["summary_metrics"]
    population = summary[summary.pilot_id.eq("population") & summary.cohort.eq("all")]
    probability_table = tables["probability_metrics"].copy()
    probability_table["ECE (points)"] = probability_table.pop("ece_10_bins") * 100
    probability_table["Mean confidence (%)"] = probability_table.pop("mean_confidence") * 100
    text = [
        "# Skills catalogue: original experiments versus improved ensemble\n",
        ("This comparison predicts the **exact A01 catalogue selection at +1, +2 or +5 seconds** from "
        "scripted tactical observer B01's observations. It does not rate pilot competence. All results "
        "are development validation on the same 45 scenario groups and ten known trained pilots.\n"),
        "## Experiments and information budgets\n",
        ("| Experiment | History / features | Parameters | Selection |\n| --- | --- | ---: | --- |\n"
        "| Training frequency | No observation history; per-horizon training priors | No neural model | Training targets only |\n"
        f"| Original GRU | 5 s / 54 | {evidence['original_runs']['gru']['parameters']:,} | Seed 42, best epoch 15 by validation NLL |\n"
        f"| Original transformer | 5 s / 54 | {evidence['original_runs']['transformer']['parameters']:,} | Seed 42, best epoch 20 by validation NLL |\n"
        f"| Improved ensemble | 20/60 s / 91; 12 temporal MLPs | {evidence['runtime_check']['total_parameters']:,} | Two training-only grouped partitions; freeze before outer validation |\n"),
        ("Both initial neural models use width 128, two layers, dropout 0.1 and AdamW. "
        "The improved model combines four variants and three seeds each, with equal probability weights. "
        "This comparison changes architecture, history and compute together; it is not a controlled single-factor ablation.\n"),
        "## Reading the metrics\n",
        ("- **Top-1 / top-3:** correct label is the first choice / among the three highest probabilities. Higher is better.\n"
        "- **Macro-F1 (33):** equal mean of per-label F1 across all catalogue entries; undefined F1 is zero.\n"
        "- **Supported macro-F1 (26):** same calculation over labels with validation truth support. "
        "This changes the averaging set, not the task vocabulary.\n"
        "- **Precision / recall:** fraction of predictions of a label that are correct / fraction of its true occurrences recovered. "
        "F1 is their harmonic mean. Support is the number of true occurrences.\n"
        "- **NLL:** mean negative log probability of the true label (natural logarithm), clipped at 1e-12. Lower is better.\n"
        "- **Brier:** mean sum of squared probability errors over all 33 classes, without dividing by 33; range 0–2. Lower is better.\n"
        "- **ECE:** weighted absolute confidence–accuracy gap in ten fixed equal-width top-label bins. "
        "Reported in percentage points. It is descriptive, bin-dependent and not a complete calibration assessment.\n"
        "- **Percentage-point gain** is new accuracy minus old accuracy; **relative error reduction** divides that gain by old error rate.\n"),
    ]
    for h in HORIZONS:
        frame = population[population.horizon_s.eq(h)].set_index("model").loc[list(MODELS)]
        probs = tables["probability_metrics"].query("horizon_s == @h").set_index("model")
        rows = [
            {
                "Model": LABELS[m],
                "Top-1 %": f"{r.top1_accuracy * 100:.2f}",
                "Top-3 %": f"{r.top3_accuracy * 100:.2f}",
                "F1 (33)": f"{r.macro_f1_catalogue:.4f}",
                "F1 (26)": f"{r.macro_f1_supported:.4f}",
                "NLL": f"{r.nll:.4f}",
                "Brier": f"{probs.loc[m, 'brier']:.4f}",
                "ECE pts": f"{probs.loc[m, 'ece_10_bins'] * 100:.2f}",
            }
            for m, r in frame.iterrows()
        ]
        old, new = frame.loc["transformer"], frame.loc["ensemble"]
        text += [
            f"## +{h}-second scorecard: {int(old['count']):,} valid origins\n",
            markdown_table(pd.DataFrame(rows)),
            (f"\nTop-1 gain: **{(new.top1_accuracy - old.top1_accuracy) * 100:.2f} points**. "
            f"Relative top-1 error reduction: **{(new.top1_accuracy - old.top1_accuracy) / (1 - old.top1_accuracy) * 100:.2f}%**.\n"),
        ]
    text += [
        "## Corrections and regressions on the same origins\n",
        markdown_table(tables["paired_changes"].round(5)),
        ("\nThese are forecast windows, not independent episodes. Net gain equals corrected minus regressed. "
        "This is descriptive reuse of fixed predictions, not a new model-selection step.\n"),
        "## Changed and unchanged endpoint labels\n",
    ]
    cohorts = summary[summary.pilot_id.eq("population") & summary.cohort.isin(["same", "changed"])]
    for h in HORIZONS:
        rows = cohorts[
            cohorts.horizon_s.eq(h) & cohorts.model.isin(["gru", "transformer", "ensemble"])
        ][
            [
                "model",
                "cohort",
                "count",
                "top1_accuracy",
                "top3_accuracy",
                "macro_f1_catalogue",
                "nll",
            ]
        ].copy()
        rows[["top1_accuracy", "top3_accuracy"]] *= 100
        text += [f"### +{h} seconds\n", markdown_table(rows.round(4)), ""]
    text += [
        ("A current true label defines these subsets for diagnosis only. It is not an inference input. "
        "Endpoint equality does not rule out an intermediate change and return. Subsets differ across horizons. "
        "At +5 s, unchanged top-3 and NLL deteriorate slightly even though its top-1 improves.\n"),
        "## Per-pilot +5-second accuracy\n",
    ]
    pilots = summary[
        summary.pilot_id.ne("population") & summary.cohort.eq("all") & summary.horizon_s.eq(5)
    ]
    pilot_table = (
        pilots.pivot(index="pilot_id", columns="model", values="top1_accuracy")[list(MODELS)] * 100
    )
    pilot_table["gain_points"] = pilot_table.ensemble - pilot_table.transformer
    text += [
        markdown_table(pilot_table.round(2).reset_index()),
        ("\nAll 30 pilot/horizon top-1 comparisons improve. This does not demonstrate unseen-pilot generalisation "
        "or establish significance for every pilot individually. Complete per-pilot/horizon values are in data/summary_metrics.csv.\n"),
        "## Every catalogue entry at +5 seconds\n",
    ]
    classes = tables["class_metrics"].query("horizon_s == 5")
    old = classes[classes.model.eq("transformer")].set_index("skill")
    new = (
        classes[classes.model.eq("ensemble")]
        .set_index("skill")
        .sort_values("support", ascending=False)
    )
    frame = pd.DataFrame(
        {
            "Label": new.index,
            "Support": new.support.to_numpy(),
            "Old precision": old.loc[new.index].precision.to_numpy(),
            "New precision": new.precision.to_numpy(),
            "Old recall": old.loc[new.index].recall.to_numpy(),
            "New recall": new.recall.to_numpy(),
            "Old F1": old.loc[new.index].f1.to_numpy(),
            "New F1": new.f1.to_numpy(),
        }
    )
    text += [
        markdown_table(frame.round(4)),
        ("\nSeven entries have zero truth support. Small-support estimates "
        "are unstable, and zero-support performance is not measured. The slide class groups are chosen by "
        "support (at least 500 versus fewer than 500), not by improvement size. All 33 labels remain scored. "
        "Class metrics for all four methods and all three horizons are in data/class_metrics.csv.\n"),
        "## Probability quality\n",
        markdown_table(probability_table.round(5)),
        ("\nThe ensemble improves Brier, NLL and ECE relative to both original neural models. "
        "The frequency baseline has very low ECE despite poor accuracy: near-constant low confidence can "
        "match an aggregate hit rate without distinguishing examples. ECE must be read with accuracy and "
        "proper probability scores. Reliability-bin counts and the underlying 10-bin definition are published.\n"),
        "## Development search: evidence about what helped\n",
        ("Internal screen values must not be compared directly with outer-validation scorecards. "
        "They use smaller fitting partitions and different selection origins. The first screen includes "
        "unsuccessful geometry, regularization and long-history transformer candidates.\n"),
        markdown_table(tables["initial_search"].round(4)),
        "\nFixed-mixture confirmation:\n",
        markdown_table(tables["internal_confirmation"].round(4)),
        ("\nThe final four-variant mixture averaged 60.98% internal accuracy versus 60.96% for the "
        "wide/long pair. That small gap is not decisive evidence that twelve final models are necessary. "
        "Experiments support the combined procedure, not an isolated causal attribution to each feature.\n"),
        "## Uncertainty and evidence limits\n",
        markdown_table(tables["scenario_intervals"].round(5)),
        ("\nThe paired 95% bootstrap intervals use 5,000 resamples of 45 complete scenario groups. "
        "All pilots and overlapping windows in a group stay together. The original validation set informed "
        "the initial experiments, and the internal search was adaptive. These are development intervals, "
        "not a correction for all historical selection or a final test. The reserved test is untouched. "
        "No new model, temperature or mixture weight was fitted for this expanded analysis.\n"),
        ("The original GRU training process exited with code 1 after writing results; independent fresh-process "
        "reloads subsequently reproduced all saved probabilities exactly. This documented runtime event is "
        "retained in the initial experiment guide. All experiments use idealised simulator tracks from the "
        "same ten known pilots; sensor noise and unseen policies remain untested.\n"),
    ]
    (dest / "comparison.md").write_text("\n".join(text), encoding="utf-8", newline="\n")
