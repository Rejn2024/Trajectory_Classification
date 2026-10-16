"""Portable forecasting evidence and slides; reads saved results without inference."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

BG, FG, MUTED, TEAL, BLUE, RED = (
    "#101e30", "#f2f5f9", "#aebdd0", "#62ddbc", "#8baeff", "#ff958a"
)


def collect_forecasting(root, dest):
    """Copy small evidence files and one validation example into the portable pack."""
    import pandas as pd
    import pyarrow.parquet as pq

    root, dest = Path(root), Path(dest) / "data/forecasting"
    run_root = root / "artifacts/motion_forecasting/initial_v1"
    prepared = root / "artifacts/motion_forecasting/data_v1"
    corpus = root / "shared/corpora/trained-pilot-episodes-v1"
    hashes = {}

    def record(path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[path.relative_to(root).as_posix()] = digest
        return digest

    def read(path):
        record(path)
        return json.loads(path.read_text(encoding="utf-8"))

    manifest = read(prepared / "data_manifest.json")
    manifest_hash = hashes[(prepared / "data_manifest.json").relative_to(root).as_posix()]
    corpus_config = read(corpus / "config.json")
    assert corpus_config["observer_id"] == "B01" and corpus_config["target_id"] == "A01"
    assert record(corpus / "manifest.json") == manifest["corpus_manifest_sha256"]
    assignments = pd.read_csv(prepared / "split_assignments.csv")
    record(prepared / "split_assignments.csv")
    assert assignments.groupby("scenario_group").split.nunique().eq(1).all()
    assert assignments.groupby(["split", "pilot_id"]).size().to_dict() == {
        (split, f"pilot_{pilot:03d}"): count
        for split, count in (("train", 210), ("validation", 45), ("test", 45))
        for pilot in range(1, 11)
    }
    metric_file = run_root / "validation_comparison.csv"
    record(metric_file)
    metrics = pd.read_csv(metric_file)
    runs, arrays, samples, thresholds = {}, {}, {}, []
    for model in ("transformer", "gru"):
        run = read(run_root / model / "run.json")
        summary = read(run_root / model / "summary.json")
        assert not run["test_evaluated"] and not summary["test_evaluated"]
        assert run["data_manifest_sha256"] == manifest_hash
        runs[model] = {
            "model": run["model"], "training": run["training"],
            "parameters": run["parameters"], "selection": run["selection"],
            "summary": summary,
        }
        array_path = run_root / model / "validation_predictions.npz"
        sample_path = run_root / model / "validation_samples.parquet"
        record(array_path)
        record(sample_path)
        with np.load(array_path) as stored:
            arrays[model] = {key: stored[key] for key in stored.files}
        samples[model] = pd.read_parquet(sample_path)
        assert samples[model].split.eq("validation").all()
        assert samples[model].episode_id.nunique() == 450
        a, sample = arrays[model], samples[model]
        errors = {}
        for quantity, start in (("position", 0), ("velocity", 3)):
            errors[quantity] = np.linalg.norm(
                a["residual_prediction"][:, :, start:start + 3].astype(np.float64)
                - a["residual_target"][:, :, start:start + 3].astype(np.float64), axis=-1
            )
            for h, seconds in enumerate((1, 2, 5)):
                valid = a["valid"][:, h]
                frame = sample.loc[valid, ["episode_id", "pilot_id"]].copy()
                frame["error"] = errors[quantity][valid, h]
                for pilot, group in [("population", frame), *frame.groupby("pilot_id")]:
                    row = metrics[(metrics.model == model) & (metrics.pilot_id == pilot)
                                  & (metrics.quantity == quantity) & (metrics.horizon_s == seconds)]
                    assert len(row) == 1
                    np.testing.assert_allclose(group.groupby("episode_id").error.mean().mean(),
                                               row.iloc[0].episode_mean_error, rtol=1e-6)
        for h, limit in ((0, 1), (1, 5), (2, 10), (2, 25), (2, 50)):
            valid = a["valid"][:, h]
            frame = sample.loc[valid, ["episode_id"]].copy()
            frame["within"] = errors["position"][valid, h] <= limit
            thresholds.append({"model": model, "horizon_s": (1, 2, 5)[h],
                               "position_tolerance_m": limit,
                               "episode_mean_fraction": float(frame.groupby("episode_id").within.mean().mean())})
    assert samples["transformer"].equals(samples["gru"])
    np.testing.assert_array_equal(arrays["transformer"]["valid"], arrays["gru"]["valid"])
    np.testing.assert_array_equal(arrays["transformer"]["residual_target"], arrays["gru"]["residual_target"])

    # A deterministic median-error illustration, rather than a selected best case.
    a = arrays["transformer"]
    errors = np.linalg.norm(a["residual_prediction"][:, 2, :3].astype(float)
                            - a["residual_target"][:, 2, :3].astype(float), axis=-1)
    candidates = np.flatnonzero(a["valid"][:, 2])
    chosen = candidates[np.argmin(abs(errors[candidates] - np.median(errors[candidates])))]
    sample = samples["transformer"].iloc[chosen]
    # Read a single physical track for display only. It never becomes a model input.
    episodes = pd.read_parquet(corpus / "episodes.parquet")
    pilot_episodes = episodes[episodes.pilot_id == sample.pilot_id].sort_values("episode_id")
    row_group = list(pilot_episodes.episode_id).index(sample.episode_id)
    track_path = corpus / "tracks" / f"{sample.pilot_id}.parquet"
    record(track_path)
    track = pq.ParquetFile(track_path).read_row_group(row_group).to_pandas()
    assert set(track.episode_id) == {sample.episode_id}
    meta = pd.read_parquet(prepared / "samples.parquet")
    indexes = np.flatnonzero((meta.split == "validation").to_numpy())
    assert meta.iloc[indexes].reset_index(drop=True).equals(samples["transformer"])
    global_index = indexes[chosen]
    end = np.load(prepared / "anchors.npy", mmap_mode="r")[global_index]
    obs = np.load(prepared / "features.npy", mmap_mode="r")[end].astype(float)
    position = (obs[:3] + obs[9:12]) * [50000, 50000, 10000]
    velocity = (obs[3:6] + obs[12:15]) * 600
    horizon = np.array([1, 2, 5])
    baseline = np.concatenate((position + horizon[:, None] * velocity,
                               np.broadcast_to(velocity, (3, 3))), axis=1)
    future = track.iloc[int(sample.step) + horizon * 10][
        [f"target_{name}" for name in ("x", "y", "z", "vx", "vy", "vz")]
    ].to_numpy()
    np.testing.assert_allclose(baseline + a["residual_target"][chosen], future, atol=0.002, rtol=0)
    example = {
        "selection": "Valid five-second window closest to the pooled median transformer position error; illustration only",
        "episode_id": sample.episode_id, "pilot_id": sample.pilot_id,
        "time_s": float(sample.time_s), "horizons_s": horizon.tolist(),
        "position_nwu_m": position.tolist(), "velocity_nwu_mps": velocity.tolist(),
        "track_relative_time_s": (np.arange(-50, 51) / 10).tolist(),
        "track_position_nwu_m": track.iloc[int(sample.step) - 50:int(sample.step) + 51][
            ["target_x", "target_y", "target_z"]].to_numpy().tolist(),
        "true_future_state": future.tolist(),
        "predicted_future_state": {
            model: (baseline + arrays[model]["residual_prediction"][chosen]).tolist()
            for model in ("transformer", "gru")
        },
        "constant_velocity_state": baseline.tolist(),
    }
    bundle = {
        "corpus": "shared/corpora/trained-pilot-episodes-v1",
        "observer": "B01: scripted tactical F-16", "target": "A01: trained F-16",
        "counts": manifest["counts"], "split_seed": manifest["split_seed"],
        "test_evaluated": False, "metric": "Mean 3D error within episodes, then equal episode weight",
        "tolerance_metric": "Fraction within the stated 3D radius per episode, then equal episode weight",
        "tolerances": thresholds, "runs": runs, "source_sha256": hashes,
        "prepared_file_sha256": manifest["files"],
    }
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "metrics.csv").write_bytes(metric_file.read_bytes())
    for name, value in (("summary.json", bundle), ("example.json", example)):
        (dest / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Packaged forecasting validation evidence in {dest}")


def add_forecasting_slides(dest, page, finish, chart, table):
    """Render from the small portable evidence bundle, without local training files."""
    evidence = Path(dest) / "data/forecasting"
    summary = json.loads((evidence / "summary.json").read_text(encoding="utf-8"))
    example = json.loads((evidence / "example.json").read_text(encoding="utf-8"))
    with (evidence / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        metrics = list(csv.DictReader(stream))

    def value(model, quantity, horizon, pilot="population"):
        return next(float(row["episode_mean_error"]) for row in metrics if
                    row["model"] == model and row["quantity"] == quantity
                    and float(row["horizon_s"]) == horizon and row["pilot_id"] == pilot)

    def within(model, horizon, tolerance):
        return next(100 * row["episode_mean_fraction"] for row in summary["tolerances"]
                    if row["model"] == model and row["horizon_s"] == horizon
                    and row["position_tolerance_m"] == tolerance)

    f = page("From trained pilots to forecasting data",
             "Observer: scripted tactical F-16 (B01) • prediction target: trained pilot F-16 (A01)")
    f.text(.065, .70, "3,000 RECORDED ENCOUNTERS", size=20, color=TEAL, weight="bold")
    f.text(.065, .64, "300 matched starting scenarios × 10 frozen pilot policies\n"
           "Five seconds of past observations → movement at +1, +2 and +5 seconds\n"
           "Each horizon predicts a 3D position and a 3D velocity vector",
           size=17, va="top", linespacing=1.6)
    table(f, [.065, .25, .49, .20],
          [["Training", "210", "2,100"], ["Validation", "45", "450"], ["Reserved test", "45", "450"]],
          ["Partition", "Scenario groups", "Episodes"], [.36, .34, .30], fontsize=13)
    f.text(.60, .44, "TWO SHARED FORECASTERS", color=BLUE, size=18, weight="bold")
    f.text(.60, .38, "Transformer and GRU\nEach learns across all ten pilots\nPilot identity is excluded\nFuture states supply training targets",
           size=15, va="top", linespacing=1.55)
    f.text(.065, .14, "Whole scenario groups stay together across all ten pilots when splitting the dataset.", size=16, color=TEAL)
    f.text(.065, .087, "Forecast training reuses recorded flights; it does not rerun the simulator or update the pilots.", size=14, color=MUTED)
    finish(f, "Population PPO and tactical adaptation are upstream data-generation preparation. The 3,000 corpus episodes are new encounters generated with selected frozen tactical-adapted checkpoints, not training epochs. Only 2,100 episodes fit the supervised forecasters; 450 select checkpoints and support the present validation report; 450 remain unevaluated. All ten encounters from each starting scenario remain in one partition. The neural input has 51 tokens at 10 Hz spanning five seconds, with 51 observation values and three causal acceleration features. The oldest acceleration feature requires another half-second of raw history: 56 raw frames in total. Target actions, target identity and future observations are not inputs. Predictions use world North/West/Up coordinates. Both models output all three horizons directly from the same history; later horizons do not receive earlier future observations. There is no PPO critic or reward function in the supervised forecasting loss.")

    f = page("Location forecast and prediction error",
             "One saved validation example • observed history ends at the forecast origin")
    origin = np.asarray(example["position_nwu_m"])
    track = np.asarray(example["track_position_nwu_m"]) - origin
    truth = np.asarray(example["true_future_state"])[:, :3] - origin
    pred = np.asarray(example["predicted_future_state"]["transformer"])[:, :3] - origin
    cv = np.asarray(example["constant_velocity_state"])[:, :3] - origin
    ax = chart(f, [.085, .25, .34, .46])
    ax.plot(track[:51, 1], track[:51, 0], color=MUTED, label="Past 5 s")
    ax.plot(track[50:, 1], track[50:, 0], color=FG, label="Actual next 5 s")
    ax.plot([0, cv[2, 1]], [0, cv[2, 0]], "--", color=RED, label="Constant velocity")
    ax.scatter(pred[:, 1], pred[:, 0], color=TEAL, marker="x", s=60, label="Transformer")
    ax.scatter([0], [0], color=BLUE, s=30, zorder=5)
    ax.annotate("Now", (0, 0), xytext=(8, -5), textcoords="offset points", color=BLUE, size=10)
    for i, seconds in enumerate((1, 2, 5)):
        ax.annotate(f"+{seconds} s", (pred[i, 1], pred[i, 0]), xytext=(8, -5), textcoords="offset points", color=TEAL, size=10)
    ax.set(xlabel="West displacement (m)", ylabel="North displacement (m)", title="Horizontal flight path")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=10, frameon=False, labelcolor=FG, loc="best")
    zoom = chart(f, [.50, .40, .20, .27])
    diff = pred[2] - truth[2]
    zoom.scatter([0], [0], color=FG, s=70, label="Actual")
    zoom.scatter([diff[1]], [diff[0]], color=TEAL, marker="x", s=90, label="Predicted")
    zoom.plot([0, diff[1]], [0, diff[0]], ":", color=TEAL)
    zoom.set(xlabel="West error (m)", ylabel="North error (m)", title="5 s endpoint detail")
    lim = max(5, np.max(np.abs(diff[:2])) * 1.5)
    zoom.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal")
    zoom.legend(fontsize=9, frameon=False, labelcolor=FG, loc="best")
    error = np.linalg.norm(diff)
    f.text(.755, .64, f"{error:.1f} m", color=TEAL, size=35, weight="bold")
    f.text(.755, .565, "3D position error\nin this example", size=15, linespacing=1.5)
    f.text(.755, .44, f"{abs(diff[2]):.1f} m", color=BLUE, size=25, weight="bold")
    f.text(.755, .38, "Vertical component\n(abs. height error)", size=13, linespacing=1.5)
    f.text(.49, .23, "Position error = distance from predicted to actual location.\n"
           "Velocity error = difference in movement speed and direction.", size=14, linespacing=1.6)
    f.text(.065, .14, f"Pilot {example['pilot_id'][-3:]} • forecast at {example['time_s']:.0f} s • example chosen at the median 5 s error", size=15, color=TEAL)
    f.text(.065, .087, "The enlarged view shows horizontal error; the reported distance also includes height error.", size=13, color=MUTED)
    finish(f, f"This is an actual saved validation prediction, not an illustrative invented trajectory. Source: {example['episode_id']}, forecast origin {example['time_s']} seconds. Selection is deterministic: the valid five-second window nearest the pooled median Transformer position error. It illustrates a typical window and is not the episode-weighted population mean of 19.09 m. Only the history is input; future physical tracks are shown for assessment. The dots are direct forecasts, not a continuously predicted path. In general, a 19 m mean position error does not mean 19 m on each axis, a 19 m flight distance, a guaranteed error bound or a classification accuracy. Velocity error is the length of the vector difference: it can be nonzero even when speeds match but headings differ. Neither error has a natural conversion into a percentage accuracy.")

    f = page("Initial forecasting performance in physical units",
             "450 validation encounters • mean 3D error with equal episode weighting • lower is better")
    models = [("constant_velocity", "Constant velocity"), ("constant_acceleration", "Constant acceleration"),
              ("gru", "GRU"), ("transformer", "Transformer")]
    f.text(.065, .70, "POSITION ERROR (METRES)", color=TEAL, size=18, weight="bold")
    f.text(.54, .70, "VELOCITY ERROR (METRES / SECOND)", color=BLUE, size=18, weight="bold")
    for quantity, x in (("position", .065), ("velocity", .54)):
        rows = [[name, *[f"{value(key, quantity, h):.2f}" for h in (1, 2, 5)]] for key, name in models]
        table(f, [x, .43, .40, .23], rows, ["Method", "+1 s", "+2 s", "+5 s"], [.49, .17, .17, .17], fontsize=12)
    gain = 100 * (1 - value("transformer", "position", 5) / value("constant_acceleration", "position", 5))
    f.text(.065, .36, f"At 5 s: Transformer reduces mean location error by {gain:.0f}% versus constant acceleration.", size=17, color=TEAL)
    f.text(.065, .29, "HOW OFTEN IS THE TRANSFORMER CLOSE TO THE ACTUAL LOCATION?", size=15, color=BLUE, weight="bold")
    for x, h, tolerance in ((.065, 1, 1), (.365, 2, 5), (.665, 5, 25)):
        f.text(x, .225, f"{within('transformer', h, tolerance):.1f}%", size=28, color=TEAL, weight="bold")
        f.text(x, .175, f"within {tolerance} m at +{h} s", size=16)
    f.text(.065, .10, "These are measured tolerance rates, not guarantees. Final test scenarios remain unevaluated.", size=14, color=MUTED)
    finish(f, "Constant velocity assumes the observed velocity stays fixed. Constant acceleration extends the backward estimate of acceleration; it is the stronger elementary reference. Both neural models learn corrections to constant-velocity predictions. GRU has slightly lower short-horizon position error, while Transformer has lower five-second position and velocity error. Both outperform constant acceleration for each pilot, horizon and quantity in this validation report. Position tolerance rates count eligible forecasts within the specified 3D radius within each episode, then average episodes equally; the radii are explanatory choices, not operational acceptance requirements. Transformer has 48.7% within 10 m, 80.0% within 25 m and 92.1% within 50 m at five seconds. This makes the tail visible: the 19.09 m mean is not an upper bound. Missing or dead future targets are masked per horizon. These are validation results used during development and model selection, not independent final test estimates.")

    f = page("Forecasting difficulty across the ten pilots",
             "Same 45 held-out starting scenarios per pilot • five-second horizon • lower is better")
    pilots = [f"pilot_{i:03d}" for i in range(1, 11)]
    y = np.arange(10)
    for quantity, bounds, title, unit in (
        ("position", [.095, .27, .36, .41], "Future location", "Mean 3D position error (m)"),
        ("velocity", [.565, .27, .36, .41], "Future movement", "Mean 3D velocity error (m/s)"),
    ):
        ax = chart(f, bounds)
        for model, shift, colour in (("transformer", -.15, TEAL), ("gru", .15, BLUE)):
            vals = [value(model, quantity, 5, p) for p in pilots]
            ax.barh(y + shift, vals, height=.27, color=colour, label=model.title())
            for i, v in enumerate(vals):
                ax.text(v + .25, i + shift, f"{v:.1f}", va="center", size=9, color=colour)
        ax.set(yticks=y, yticklabels=[p[-3:] for p in pilots], xlabel=unit, title=title,
               ylim=(9.65, -.65), xlim=(0, 31 if quantity == "position" else 19))
        ax.set_ylabel("Target pilot")
        if quantity == "position":
            handles, labels = ax.get_legend_handles_labels()
            f.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .765),
                     ncol=2, frameon=False, labelcolor=FG, fontsize=12)
    f.text(.065, .17, "Pilot 001 is easiest to predict. Pilots 002, 005, 006 and 009 have larger five-second errors.", size=16, color=TEAL)
    f.text(.065, .115, "Transformer improves both five-second errors for all ten pilots relative to GRU.", size=15)
    f.text(.065, .075, "Predictability is a different property from combat effectiveness or aggression.", size=13, color=MUTED)
    finish(f, "Two shared forecasters were trained across the ten-pilot population; these bars break their validation errors down by target pilot. There are not ten separate forecasting networks. The metric first averages within each episode and then across its 45 episodes. All pilots share the held-out starting scenarios, but their interactions produce different trajectories and episode lengths. Larger errors alone do not establish greater aggressiveness or deliberate unpredictability. Full per-pilot position and velocity errors at all horizons, including constant-motion references and error distribution summaries, are in data/forecasting/metrics.csv. Comparisons use a single training seed and partition and should not be presented as statistical proof of architecture superiority.")

    f = page("From motion forecasts to interpretable manoeuvres",
             "Current outputs describe physical movement; behavioural labels are a subsequent interpretation layer")
    f.text(.065, .70, "CURRENT CAPABILITY", color=TEAL, size=19, weight="bold")
    f.text(.065, .635, "Predict future position and velocity\n"
           "Assess errors in metres and m/s\n"
           "Compare error across horizons and pilots\n"
           "Display actual and predicted trajectories",
           size=17, va="top", linespacing=1.65)
    f.text(.535, .70, "PROPOSED INTERPRETATION", color=BLUE, size=19, weight="bold")
    f.text(.535, .635, "Turn direction from horizontal velocity\n"
           "Climb / descend from vertical motion\n"
           "Acceleration from change in speed\n"
           "Evaluate these labels against actual motion",
           size=17, va="top", linespacing=1.65)
    f.text(.065, .33, "A named catalogue skill expresses intent; several skills can produce similar observed motion.", size=16, color=TEAL)
    f.text(.065, .25, "Predicting one of the 33 internal skills would be a separate supervised classification task.", size=16)
    f.text(.065, .16, "Evidence scope: known pilot population • idealised tracks • one split and training seed", size=15, color=MUTED)
    f.text(.065, .10, "Next validation questions: sensor noise, unseen pilots, difficult manoeuvres and the reserved test set.", size=14, color=MUTED)
    finish(f, "The purpose of the first forecast is physical state prediction from one-sided observations. Kinematic labels such as turning left, climbing or accelerating can be derived using defined thresholds and compared with actual future motion. This interpretation has not yet been implemented or measured. Catalogue labels describe the controller's selected intent and need not be uniquely identifiable from the trajectory; classification requires a distinct target definition and validation. Recorded private actions could supply labels for that later task, but must never become prediction inputs. Present evidence uses idealised exact tracks, previously seen pilot identities and held-out starting scenarios. It does not establish realistic radar performance or generalisation to an unseen pilot. All forecasts are point estimates without calibrated uncertainty. The untouched test split should be evaluated after development choices are fixed.")
