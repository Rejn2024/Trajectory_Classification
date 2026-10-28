"""Physical-unit validation metrics and portable reports; no implicit test evaluation."""

import base64
from pathlib import Path

import numpy as np
import pandas as pd

from .data import HORIZONS, read_json


def motion_metrics(prediction, target, valid, samples, model):
    rows = []
    for h, seconds in enumerate(HORIZONS):
        selected = samples.loc[valid[:, h], ["episode_id", "pilot_id"]].reset_index(drop=True)
        for quantity, start, units in (("position", 0, "m"), ("velocity", 3, "m/s")):
            error = prediction[valid[:, h], h, start:start + 3] - target[valid[:, h], h, start:start + 3]
            selected["error"] = np.linalg.norm(error.astype(np.float64), axis=-1)
            groups = [("population", selected)] + list(selected.groupby("pilot_id", sort=True))
            for pilot_id, frame in groups:
                values = frame.error.to_numpy()
                rows.append({"model": model, "pilot_id": pilot_id, "horizon_s": float(seconds),
                             "quantity": quantity, "units": units, "samples": len(frame),
                             "episodes": frame.episode_id.nunique(),
                             "episode_mean_error": float(frame.groupby("episode_id").error.mean().mean()),
                             "mean_error": float(values.mean()), "rmse": float(np.sqrt(np.mean(values ** 2))),
                             "median_error": float(np.median(values)), "p90_error": float(np.quantile(values, 0.9))})
    return pd.DataFrame(rows)


def selection_score(metrics, reference):
    keys = ["horizon_s", "quantity"]
    values = metrics[metrics.pilot_id == "population"].set_index(keys).episode_mean_error
    baseline = reference[reference.pilot_id == "population"].set_index(keys).episode_mean_error
    return float((values / baseline.clip(lower=1e-9)).mean())


def build_report(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root)
    tables, histories, configurations = [], {}, {}
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or not (directory / "validation_metrics.csv").exists():
            continue
        tables.append(pd.read_csv(directory / "validation_metrics.csv"))
        if not any((t.model == "constant_velocity").any() for t in tables):
            tables.append(pd.read_csv(directory / "baseline_metrics.csv"))
        if (directory / "history.csv").exists():
            histories[directory.name] = pd.read_csv(directory / "history.csv")
        configurations[directory.name] = read_json(directory / "run.json")
    if not tables:
        raise ValueError("No completed validation metrics to report")
    metrics = pd.concat(tables, ignore_index=True).drop_duplicates(["model", "pilot_id", "horizon_s", "quantity"])
    metrics.to_csv(root / "validation_comparison.csv", index=False)
    population = metrics[metrics.pilot_id == "population"]
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for axis, quantity, units in zip(axes, ("position", "velocity"), ("metres", "m/s"), strict=True):
        for model, frame in population[population.quantity == quantity].groupby("model"):
            axis.plot(frame.horizon_s, frame.episode_mean_error, marker="o", label=model.replace("_", " "))
        axis.set(xlabel="Forecast horizon (seconds)", ylabel=f"Mean 3D error ({units})",
                 title=f"Validation {quantity} error", xticks=[1, 2, 5])
        axis.legend(fontsize=8)
    fig.savefig(root / "validation_errors.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for name, frame in histories.items():
        axes[0].plot(frame.epoch, frame.train_loss, label=name)
        axes[1].plot(frame.epoch, frame.validation_score, label=name)
    axes[0].set(xlabel="Epoch", ylabel="Scaled Huber loss", title="Training loss")
    axes[1].set(xlabel="Epoch", ylabel="Error relative to constant velocity",
                title="Validation score (constant velocity = 1)",
                ylim=(0, 1.1 * max(frame.validation_score.max() for frame in histories.values())))
    for axis in axes:
        axis.legend()
    fig.savefig(root / "learning_curves.png", dpi=160)
    plt.close(fig)
    comparison = population.pivot(index="model", columns=["quantity", "horizon_s"], values="episode_mean_error")
    images = "".join(f'<img alt="{name}" src="data:image/png;base64,{base64.b64encode((root / name).read_bytes()).decode()}">' for name in ("validation_errors.png", "learning_curves.png"))
    html = """<!doctype html><html><head><meta charset="utf-8"><title>Trained pilot motion forecasting</title>
<style>body{font:16px system-ui;max-width:1150px;margin:40px auto;padding:0 20px;color:#172535}img{max-width:100%}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border:1px solid #d6dce4;text-align:right}h1,h2{color:#173f65}</style></head><body>
<h1>Trained pilot motion forecasting</h1><p>Observer: scripted tactical B01. Prediction target: trained A01.
Inputs use only current and past observations. A five-second neural sequence predicts position and velocity at 1, 2 and 5 seconds.
Backward acceleration features need an additional 0.5 seconds of raw history (56 raw frames for 51 model tokens).</p>
<p>These are development validation results, not final test results. The test scenarios remain locked.
Splits hold out complete matched scenario groups across all ten pilots. Metrics average 3D error within each episode, then weight episodes equally.</p>
<h2>Validation error by horizon</h2>""" + comparison.to_html(float_format=lambda x: f"{x:.3f}") + images
    html += "<h2>Per-pilot validation results</h2>" + metrics[metrics.pilot_id != "population"].to_html(index=False, float_format=lambda x: f"{x:.3f}")
    html += "<h2>Scope</h2><p>Idealised exact simulator tracks; known trained pilot population, held-out initial scenarios. Overlapping windows are correlated. No claim of unseen-pilot or real-sensor generalisation is made. Internal target actions and identities are excluded from inputs.</p></body></html>"
    (root / "report.html").write_text(html, encoding="utf-8")
    return comparison
