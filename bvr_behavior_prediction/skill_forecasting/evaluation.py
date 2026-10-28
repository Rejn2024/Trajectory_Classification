"""Validation metrics, training-frequency baseline and a portable local report."""

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from bvr_behavior_prediction.forecasting.data import HORIZONS, read_json, write_json

from .data import verify_dataset


def frequency_probabilities(training_targets, num_classes):
    result = []
    for h in range(3):
        values = training_targets[:, h]
        counts = np.bincount(values[values >= 0], minlength=num_classes)
        if not counts.sum():
            raise ValueError("No training labels for horizon")
        result.append(counts / counts.sum())
    return np.asarray(result, dtype=np.float32)


def classification_metrics(probabilities, targets, current, samples, classes, model):
    """Window-weighted metrics; both fixed-catalogue and supported-class macro F1."""
    probabilities, targets = np.asarray(probabilities), np.asarray(targets)
    n_classes = len(classes)
    if (probabilities.shape != (*targets.shape, n_classes) or targets.shape != (len(samples), 3)
            or not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(-1), 1, atol=1e-5)):
        raise ValueError("Invalid class probabilities or target shape")
    # Stable catalogue-order tie breaking is shared by top-1 and top-3.
    ranking = np.argsort(-probabilities, axis=-1, kind="stable")
    summaries, per_class, matrices = [], [], {}
    for pilot in ["population", *sorted(samples.pilot_id.unique())]:
        pilot_mask = np.ones(len(samples), dtype=bool)
        if pilot != "population":
            pilot_mask = samples.pilot_id.eq(pilot).to_numpy()
        for h, horizon in enumerate(HORIZONS):
            available = pilot_mask & (targets[:, h] >= 0)
            known_current = current >= 0
            cohorts = {
                "all": available,
                "same": available & known_current & (targets[:, h] == current),
                "changed": available & known_current & (targets[:, h] != current),
                "unknown_current": available & ~known_current,
            }
            for cohort, mask in cohorts.items():
                actual, predicted = targets[mask, h], ranking[mask, h, 0]
                matrix = np.bincount(actual * n_classes + predicted,
                                     minlength=n_classes ** 2).reshape(n_classes, n_classes)
                support, predicted_count = matrix.sum(1), matrix.sum(0)
                correct = matrix.diagonal()
                precision = np.divide(correct, predicted_count, out=np.zeros(n_classes), where=predicted_count > 0)
                recall = np.divide(correct, support, out=np.zeros(n_classes), where=support > 0)
                f1 = np.divide(2 * correct, support + predicted_count, out=np.zeros(n_classes),
                               where=(support + predicted_count) > 0)
                row = {"model": model, "pilot_id": pilot, "horizon_s": float(horizon),
                       "cohort": cohort, "count": int(mask.sum()), "supported_classes": int((support > 0).sum()),
                       "top1_accuracy": float(correct.sum() / len(actual)) if len(actual) else None,
                       "top3_accuracy": float((ranking[mask, h, :3] == actual[:, None]).any(1).mean()) if len(actual) else None,
                       "macro_f1_catalogue": float(f1.mean()) if len(actual) else None,
                       "macro_f1_supported": float(f1[support > 0].mean()) if len(actual) else None,
                       "nll": float(-np.log(np.clip(probabilities[mask, h][np.arange(len(actual)), actual],
                                                   1e-12, 1)).mean()) if len(actual) else None}
                summaries.append(row)
                if cohort == "all":
                    matrices[f"{pilot}_{int(horizon)}s"] = matrix
                    per_class.extend({"model": model, "pilot_id": pilot, "horizon_s": float(horizon),
                                      "skill": name, "support": int(support[c]),
                                      "predicted_count": int(predicted_count[c]),
                                      "precision": float(precision[c]), "recall": float(recall[c]),
                                      "f1": float(f1[c])} for c, name in enumerate(classes))
    return pd.DataFrame(summaries), pd.DataFrame(per_class), matrices


def save_evaluation(output, probabilities, targets, current, samples, classes, model):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary, per_class, matrices = classification_metrics(
        probabilities, targets, current, samples, classes, model)
    summary.to_csv(output / "validation_metrics.csv", index=False)
    per_class.to_csv(output / "per_class_metrics.csv", index=False)
    np.savez_compressed(output / "confusion_matrices.npz", classes=np.asarray(classes), **matrices)
    for h in HORIZONS:
        pd.DataFrame(matrices[f"population_{int(h)}s"], index=classes, columns=classes).to_csv(
            output / f"confusion_{int(h)}s.csv", index_label="true_skill")
    return summary


def evaluate_baseline(data_directory, output):
    data_directory, output = Path(data_directory), Path(output)
    verify_dataset(data_directory)
    samples = pd.read_parquet(data_directory / "samples.parquet")
    targets = np.load(data_directory / "targets.npy")
    current = np.load(data_directory / "current.npy")
    classes = read_json(data_directory / "classes.json")
    priors = frequency_probabilities(targets[samples.split.eq("train")], len(classes))
    validation = samples.split.eq("validation").to_numpy()
    metrics = save_evaluation(output, np.broadcast_to(priors, (int(validation.sum()), *priors.shape)),
                              targets[validation], current[validation],
                              samples.loc[validation].reset_index(drop=True), classes, "training_frequency")
    write_json(output / "baseline.json", {
        "name": "training_frequency", "fitted_on": "train only", "privileged": False,
        "classes": classes, "horizons_s": HORIZONS.tolist(), "probabilities": priors.tolist(),
        "test_evaluated": False,
    })
    return metrics


def build_report(output, data_directory):
    output, data_directory = Path(output), Path(data_directory)
    tables = []
    details = []
    for path in sorted(output.glob("*/validation_metrics.csv")):
        frame = pd.read_csv(path)
        tables.append(frame)
        per_class = pd.read_csv(path.parent / "per_class_metrics.csv")
        details.append(f"<details><summary>{escape(path.parent.name)}: per-class validation metrics</summary>"
                       + per_class.to_html(index=False, float_format=lambda x: f"{x:.4f}") + "</details>")
        for matrix in sorted(path.parent.glob("confusion_*s.csv")):
            details.append(f"<details><summary>{escape(path.parent.name)} / {escape(matrix.stem)}: "
                           "true rows, predicted columns</summary>"
                           + pd.read_csv(matrix).to_html(index=False) + "</details>")
    if not tables:
        raise ValueError("No validation results to report")
    metrics = pd.concat(tables, ignore_index=True)
    population = metrics.query("pilot_id == 'population' and cohort == 'all'")
    changes = metrics.query("pilot_id == 'population' and cohort in ['same', 'changed']")
    pilots = metrics.query("pilot_id != 'population' and cohort == 'all'")
    support = pd.read_csv(data_directory / "label_support.csv").query("pilot_id == 'population'")
    text = """<!doctype html><html><head><meta charset="utf-8"><title>Skill catalogue forecasting</title>
<style>body{font:15px system-ui;margin:32px;color:#162335}table{border-collapse:collapse;font-size:13px}
td,th{padding:6px 10px;border:1px solid #ddd;text-align:right}th{background:#edf2f7}
details{margin:18px 0;overflow:auto}p{max-width:1000px}.scroll{overflow:auto}</style></head><body>
<h1>Forecasting trained-pilot catalogue selections</h1>
<p>B01 observation histories predict A01 selections at +1, +2 and +5 seconds. These labels do
not necessarily describe distinct executed manoeuvres. The reserved test set is not evaluated.</p>
<p>All metrics weight valid forecast windows equally. Macro-F1 catalogue averages all 33 labels,
assigning zero to undefined F1; macro-F1 supported averages labels with truth support in that subset.
Rare-class estimates are unstable. Overlapping windows are correlated. NLL clips probabilities at 1e-12.
The same/changed analysis compares the future selection to the privileged true selection at the origin;
that origin label is never a model input. Ties use alphabetical catalogue order.</p>"""
    for title, frame in [("Population validation", population), ("Changed versus unchanged selections", changes),
                         ("Per-pilot validation", pilots), ("Aligned training and validation support", support)]:
        text += f"<h2>{title}</h2><div class='scroll'>" + frame.to_html(
            index=False, float_format=lambda x: f"{x:.4f}") + "</div>"
    text += "".join(details) + "</body></html>"
    (output / "report.html").write_text(text, encoding="utf-8")
    population.to_csv(output / "comparison.csv", index=False)
    return population
