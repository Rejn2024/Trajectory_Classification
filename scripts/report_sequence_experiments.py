"""Publish a portable five-page v4 report from completed, fixed experiment outputs.

No fitting, inference, calibration or model selection is performed here. Run with
--collect after the frozen recipe has been evaluated (or v3 has been retained).
Without --collect, only the portable CSV/JSON bundle is needed to rebuild the PDF.
"""

import argparse
import json
import shutil
import textwrap
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from skill_catalogue_evidence import diagnostic_metrics, read_json, sha, write_json
from skill_catalogue_review import accuracy_weightings, paired_group_intervals

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "docs/experiments/skill_forecasting_v4"
BG, FG, MUTED = "#101c2c", "#edf2f7", "#b5c2d0"
TEAL, BLUE, GOLD, RED = "#4bd7b1", "#69aafa", "#f1c16d", "#f58c8c"
LABELS = {"baseline_v3": "Retained v3", "candidate_v4": "Frozen v4"}


def csv(path, frame):
    frame.to_csv(path, index=False, lineterminator="\n")


def calibration_note(metrics):
    population = metrics[metrics.pilot_id.eq("population")]
    old = population[population.model.eq("baseline_v3")].sort_values("horizon_s")
    new = population[population.model.eq("candidate_v4")].sort_values("horizon_s")
    if len(old) != 3 or len(new) != 3:
        return ""
    delta = new.ece_10_bins.to_numpy() - old.ece_10_bins.to_numpy()
    direction = "worsened at all three horizons" if (delta > 0).all() else (
        "improved at all three horizons" if (delta < 0).all() else "changed across horizons")
    old_values = "/".join(f"{value * 100:.2f}" for value in old.ece_10_bins)
    new_values = "/".join(f"{value * 100:.2f}" for value in new.ece_10_bins)
    probability_note = ("; NLL and Brier improved." if (new.nll.to_numpy() < old.nll.to_numpy()).all()
                        and (new.brier.to_numpy() < old.brier.to_numpy()).all() else ".")
    return f"ECE {direction} ({old_values} → {new_values} pp at +1/+2/+5 s){probability_note}"


def mean_horizon_interval(old, new, targets, groups, draws=5000, seed=1729):
    """Bootstrap the mean of three horizon accuracies using shared group draws."""
    old, new, targets, groups = np.asarray(old), np.asarray(new), np.asarray(targets), np.asarray(groups)
    if old.shape != new.shape or old.shape[:2] != targets.shape or targets.shape != (len(groups), 3):
        raise ValueError("Aligned probabilities, three targets and scenario groups are required")
    unique = np.unique(groups)
    counts, deltas = [], []
    change = (new.argmax(-1) == targets).astype(float) - (old.argmax(-1) == targets).astype(float)
    for group in unique:
        mask = (groups == group)[:, None] & (targets >= 0)
        counts.append(mask.sum(0))
        deltas.append((change * mask).sum(0))
    counts, deltas = np.asarray(counts), np.asarray(deltas)
    if not len(unique) or np.any(counts.sum(0) == 0):
        raise ValueError("Every horizon must have supported targets")
    selections = np.random.default_rng(seed).integers(0, len(unique), size=(draws, len(unique)))
    denominators = counts[selections].sum(1)
    if np.any(denominators == 0):
        raise ValueError("A bootstrap draw has no targets at one horizon")
    # Average within each shared replicate, then take quantiles; never average CI endpoints.
    replicates = (deltas[selections].sum(1) / denominators).mean(1)
    lower, upper = np.quantile(replicates, [0.025, 0.975])
    return {"mean_accuracy_gain": float((deltas.sum(0) / counts.sum(0)).mean()),
            "lower_95": float(lower), "upper_95": float(upper), "scenario_groups": len(unique),
            "draws": draws, "seed": seed,
            "interpretation": "Conditional on fixed models and development data; not selection-adjusted"}


def export_source_archive(experiment, destination, source_hashes):
    """Package the exact frozen source bytes with stable ZIP paths and timestamps."""
    experiment, destination = Path(experiment), Path(destination)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, expected in sorted(source_hashes.items()):
            source = experiment / "source" / name
            if sha(source) != expected:
                raise ValueError(f"Frozen source snapshot changed: {name}")
            member = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.create_system = 0
            member.external_attr = 0o644 << 16
            archive.writestr(member, source.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return {"file": destination.name, "sha256": sha(destination), "source_files": len(source_hashes),
            "contents": "Registered training source/config/CLI bytes; no model checkpoints or datasets",
            "timestamps": "Fixed 1980-01-01; sorted repository-relative paths"}


def collect(experiment, dest):
    """Read completed saved results, audit preservation, and export small evidence."""
    experiment, dest = Path(experiment).resolve(), Path(dest).resolve()
    protocol = read_json(experiment / "protocol.json")
    selection = read_json(experiment / "frozen_selection.json")
    if selection["protocol_sha256"] != sha(experiment / "protocol.json"):
        raise ValueError("Frozen selection no longer matches its protocol")
    evaluated = not selection["retain_baseline"]
    evaluation = read_json(experiment / "validation/evaluation.json") if evaluated else None
    if evaluated and evaluation["selection_sha256"] != sha(experiment / "frozen_selection.json"):
        raise ValueError("Completed evaluation does not match the frozen selection")
    if ((dest / "data/evidence.json").exists()
            and read_json(dest / "data/evidence.json")["protocol_sha256"] != sha(experiment / "protocol.json")):
        raise ValueError("Preserve the existing report for its original experiment")
    preserved = read_json(experiment / "preserved_artifacts.json")
    if sha(experiment / "preserved_artifacts.json") != protocol["preserved_artifacts_sha256"]:
        raise ValueError("The original preservation manifest no longer matches the protocol")
    for name, expected in preserved.items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Preserved source artifact changed: {name}")
    for name, expected in protocol["source_hashes"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Frozen experiment source changed: {name}")
    for name, expected in protocol["partition_hashes"].items():
        if sha(experiment / name) != expected:
            raise ValueError(f"Frozen partition changed: {name}")
    if evaluated:
        for name, expected in read_json(experiment / "validation/output_hashes.json").items():
            if sha(experiment / "validation" / name) != expected:
                raise ValueError(f"Completed evaluation output changed: {name}")
    data = dest / "data"
    data.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(experiment / "protocol.json", data / "frozen_protocol.json")
    shutil.copyfile(experiment / "preserved_artifacts.json", data / "preserved_artifacts.json")
    archive = export_source_archive(experiment, dest / "source_snapshot.zip", protocol["source_hashes"])
    sources = {}

    def record(path):
        path = Path(path)
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
        return path

    baseline = Path(protocol["baseline_directory"])
    prepared = Path(protocol["data_directory"])
    all_samples = pd.read_parquet(record(prepared / "samples.parquet"))
    if not all_samples.split.isin(["train", "validation"]).all():
        raise ValueError("Prepared development data contains a reserved partition")
    validation = all_samples.split.eq("validation").to_numpy()
    samples = all_samples.loc[validation].reset_index(drop=True)
    targets = np.load(record(prepared / "targets.npy"))[validation]
    classes = read_json(record(prepared / "classes.json"))
    if len(classes) != 33 or len(samples) != 30802 or samples.scenario_group.nunique() != 45:
        raise ValueError("The fixed outer-validation comparison population changed")
    paths = {"baseline_v3": baseline / "validation/ensemble"}
    if evaluated:
        paths["candidate_v4"] = experiment / "validation/ensemble"
    probabilities, summary_frames, class_frames = {}, [], []
    for model, path in paths.items():
        pd.testing.assert_frame_equal(samples, pd.read_parquet(record(path / "validation_samples.parquet")))
        probabilities[model] = np.load(record(path / "validation_probabilities.npy"))
        summary_frames.append(pd.read_csv(record(path / "validation_metrics.csv")).assign(model=model))
        class_frames.append(pd.read_csv(record(path / "per_class_metrics.csv")).assign(model=model))
    summaries, skill_metrics = pd.concat(summary_frames), pd.concat(class_frames)
    primary_interval = None
    if evaluated:
        primary_interval = mean_horizon_interval(probabilities["baseline_v3"], probabilities["candidate_v4"],
                                                  targets, samples.scenario_group.to_numpy())
        if not np.isclose(primary_interval["mean_accuracy_gain"], evaluation["mean_accuracy_gain"], atol=1e-10):
            raise ValueError("Primary comparison no longer matches the frozen evaluation")
        write_json(data / "mean_horizon_interval.json", primary_interval)
        write_json(experiment / "report_mean_horizon_interval.json", primary_interval)
    csv(data / "saved_summary_metrics.csv", summaries)
    csv(data / "pilot_skill_metrics.csv", skill_metrics)
    diagnostic_rows, reliability_rows, weighting_rows = [], [], []
    for model, probability in probabilities.items():
        for j, horizon in enumerate((1, 2, 5)):
            for pilot in ("population", *sorted(samples.pilot_id.unique())):
                mask = targets[:, j] >= 0
                if pilot != "population":
                    mask &= samples.pilot_id.eq(pilot).to_numpy()
                metric, reliability = diagnostic_metrics(probability[mask, j], targets[mask, j])
                saved = summaries[(summaries.model == model) & (summaries.pilot_id == pilot)
                                  & summaries.horizon_s.eq(horizon) & summaries.cohort.eq("all")].iloc[0]
                for key in ("top1_accuracy", "top3_accuracy", "nll"):
                    if not np.isclose(metric[key], saved[key], atol=2e-6):
                        raise ValueError(f"Saved and recomputed metric differ: {model}/{pilot}/{horizon}/{key}")
                diagnostic_rows.append({"model": model, "pilot_id": pilot, "horizon_s": horizon,
                                        "macro_f1_catalogue": saved.macro_f1_catalogue,
                                        "macro_f1_supported": saved.macro_f1_supported,
                                        "supported_classes": int(saved.supported_classes), **metric})
                reliability_rows.extend({"model": model, "pilot_id": pilot, "horizon_s": horizon, **row}
                                        for row in reliability)
            mask = targets[:, j] >= 0
            correct = probability[mask, j].argmax(-1) == targets[mask, j]
            weighting_rows.append({"model": model, "horizon_s": horizon,
                                   **accuracy_weightings(correct, samples.loc[mask, "episode_id"],
                                                         samples.loc[mask, "pilot_id"])})
    diagnostics = pd.DataFrame(diagnostic_rows)
    csv(data / "metrics.csv", diagnostics)
    csv(data / "reliability.csv", pd.DataFrame(reliability_rows))
    csv(data / "accuracy_weightings.csv", pd.DataFrame(weighting_rows))
    pilot_rows, support_rows, temporal_rows = [], [], []
    for j, horizon in enumerate((1, 2, 5)):
        for pilot in sorted(samples.pilot_id.unique()):
            mask = (targets[:, j] >= 0) & samples.pilot_id.eq(pilot).to_numpy()
            old = probabilities["baseline_v3"][mask, j].argmax(-1) == targets[mask, j]
            row = {"pilot_id": pilot, "horizon_s": horizon, "count": int(mask.sum()),
                   "episodes": int(samples.loc[mask, "episode_id"].nunique()),
                   "scenario_groups": int(samples.loc[mask, "scenario_group"].nunique()),
                   "baseline_top1": float(old.mean())}
            if evaluated:
                new = probabilities["candidate_v4"][mask, j].argmax(-1) == targets[mask, j]
                row.update({"candidate_top1": float(new.mean()),
                            **paired_group_intervals(old, new, samples.loc[mask, "scenario_group"].to_numpy())})
            pilot_rows.append(row)
        for label, skill in enumerate(classes):
            mask = targets[:, j] == label
            support_rows.append({"horizon_s": horizon, "skill": skill, "count": int(mask.sum()),
                                 "episodes": int(samples.loc[mask, "episode_id"].nunique()),
                                 "scenario_groups": int(samples.loc[mask, "scenario_group"].nunique()),
                                 "pilots": int(samples.loc[mask, "pilot_id"].nunique())})
        for lower, upper, label in ((6, 20, "6-20 s"), (21, 60, "21-60 s"), (61, np.inf, "61+ s")):
            mask = (targets >= 0).all(1) & samples.time_s.between(lower, upper).to_numpy()
            for model, probability in probabilities.items():
                metric, _ = diagnostic_metrics(probability[mask, j], targets[mask, j])
                temporal_rows.append({"model": model, "horizon_s": horizon, "elapsed_phase": label,
                                      "episodes": int(samples.loc[mask, "episode_id"].nunique()),
                                      "scenario_groups": int(samples.loc[mask, "scenario_group"].nunique()), **metric})
    csv(data / "pilot_comparison.csv", pd.DataFrame(pilot_rows))
    csv(data / "skill_support.csv", pd.DataFrame(support_rows))
    csv(data / "temporal_common_origin_metrics.csv", pd.DataFrame(temporal_rows))
    runs, history_frames = [], []
    run_paths = [(path, False) for path in sorted(experiment.glob("inner_*/*/run.json"))]
    run_paths += [(path, False) for path in sorted(experiment.glob("refit/*/run.json"))]
    run_paths += [(baseline / f"inner_{fold}/{name}_s42/run.json", True)
                 for fold in (0, 1) for name in protocol["baseline_components"]]
    for path, reused in run_paths:
        run = read_json(record(path))
        summary_path = path.parent / "summary.json"
        result = read_json(record(summary_path)) if summary_path.exists() else {}
        candidate = run["candidate"]
        row = {"candidate": result.get("name", path.parent.name.rsplit("_s", 1)[0]),
               "phase": "refit" if run["final"] else "internal", "fold": run["fold"],
               "seed": run["seed"], "reused_v3": reused, "status": "complete" if result else "incomplete",
               "architecture": candidate["architecture"], "history_seconds": candidate["history_seconds"],
               "width": candidate["width"], "layers": candidate["layers"], "parameters": run["parameters"],
               "best_epoch": result.get("best_epoch"), "epochs_completed": result.get("epochs_completed"),
               "elapsed_seconds": result.get("elapsed_seconds"), "temperature": result.get("temperature"),
               "source_directory": path.parent.relative_to(ROOT).as_posix()}
        if "selection_metrics" in result:
            row.update({"mean_accuracy": result["selection_metrics"]["mean_accuracy"],
                        "nll": result["selection_metrics"]["nll"],
                        "mean_macro_f1": result["selection_metrics"]["mean_macro_f1"]})
        runs.append(row)
        if (path.parent / "history.csv").exists():
            history_frames.append(pd.read_csv(record(path.parent / "history.csv")).assign(
                candidate=row["candidate"], phase=row["phase"], fold=row["fold"], seed=row["seed"], reused_v3=reused))
    inventory = pd.DataFrame(runs)
    csv(data / "run_inventory.csv", inventory)
    csv(data / "training_histories.csv", pd.concat(history_frames, ignore_index=True))
    for name in ("sequence_screen.csv", "sequence_confirmation.csv", "paired_scenario_bootstrap.csv"):
        path = experiment / name
        if path.exists():
            # Empty confirmation is a valid no-shortlist outcome.
            try:
                frame = pd.read_csv(record(path))
            except pd.errors.EmptyDataError:
                frame = pd.DataFrame(columns=["recipe", "new_weight", "fold", "mean_accuracy", "eligible"])
            csv(data / name, frame)
    for name in ("protocol.json", "shortlist.json", "frozen_selection.json", "internal_baseline_metrics.json",
                 "environment.json", "execution_notes.json"):
        if (experiment / name).exists():
            value = read_json(record(experiment / name))
            # The protocol has local data locations; use repo-relative paths in the portable copy.
            if name == "protocol.json":
                value = dict(value)
                for key in ("data_directory", "baseline_directory"):
                    value[key] = Path(value[key]).relative_to(ROOT).as_posix()
            write_json(data / name, value)
    if evaluated:
        write_json(data / "evaluation.json", read_json(record(experiment / "validation/evaluation.json")))
    provenance = {"frozen_protocol_file": "frozen_protocol.json",
                  "frozen_protocol_sha256": sha(data / "frozen_protocol.json"),
                  "portable_protocol_file": "protocol.json",
                  "portable_protocol_sha256": sha(data / "protocol.json"),
                  "portable_protocol_note": "Only data_directory and baseline_directory are rewritten to repository-relative paths; this copy is not the byte-exact registered protocol.",
                  "preservation_manifest_file": "preserved_artifacts.json",
                  "preservation_manifest_sha256": sha(data / "preserved_artifacts.json"),
                  "source_archive": archive}
    if provenance["frozen_protocol_sha256"] != selection["protocol_sha256"]:
        raise ValueError("The exact exported protocol does not match the frozen selection")
    write_json(data / "provenance.json", provenance)
    verification = {"preserved_artifacts_checked": len(preserved), "preserved_artifacts_unchanged": True,
                    "frozen_sources_checked": len(protocol["source_hashes"]), "frozen_sources_unchanged": True,
                    "archived_source_bytes_verified": True,
                    "frozen_protocol_export_matches_selection": True,
                    "internal_partitions_unchanged": True, "validation_origins_matched": True,
                    "new_outer_validation_evaluated": evaluated, "reserved_test_evaluated": False,
                    "report_performed_training_or_inference": False,
                    "protocol_sha256": sha(experiment / "protocol.json"),
                    "frozen_selection_sha256": sha(experiment / "frozen_selection.json")}
    write_json(experiment / "report_verification.json", verification)
    write_json(data / "verification.json", verification)
    for helper in (Path(__file__), ROOT / "scripts/skill_catalogue_evidence.py", ROOT / "scripts/skill_catalogue_review.py"):
        record(helper)
    evidence = {"protocol_sha256": sha(experiment / "protocol.json"), "evaluated": evaluated,
                "baseline": "Preserved v3 twelve-member MLP ensemble", "selection": selection,
                "evaluation": evaluation, "primary_interval": primary_interval,
                "calibration_note": calibration_note(diagnostics),
                "classes": classes, "horizons_s": [1, 2, 5],
                "validation_origins": len(samples), "validation_groups": 45, "validation_episodes": 450,
                "valid_targets": (targets >= 0).sum(0).tolist(),
                "new_completed_runs": int((~inventory.reused_v3 & inventory.status.eq("complete")).sum()),
                "new_training_seconds": float(inventory.loc[~inventory.reused_v3, "elapsed_seconds"].sum()),
                "reused_internal_runs": int(inventory.reused_v3.sum()), "source_sha256": sources,
                "execution_notes": read_json(experiment / "execution_notes.json")
                if (experiment / "execution_notes.json").exists() else None,
                "notes": ["No reserved test evaluation. Outer validation was used in previous development.",
                          "Internal partitions overlap and have guided previous development; not independent tests.",
                          "Intervals resample whole shared scenario groups; pointwise, without multiplicity correction.",
                          "Comparisons preserve labels, origins, pilots, exact horizons and metric definitions.",
                          "Expanded mixtures change model budget; gains do not isolate architecture causally.",
                          "Training timers exclude data/features/model setup; include training and stop-side work.",
                          "Some confirmation fits ran concurrently on the shared GPU; timers include contention and do not measure isolated throughput."]}
    write_json(data / "evidence.json", evidence)
    write_json(data / "checksums.json", {path.name: sha(path) for path in sorted(data.iterdir())
                                        if path.is_file() and path.name != "checksums.json"})
    return evidence


def short_name(name):
    return (name.replace("history20_", "20 s ").replace("history60_", "60 s ")
            .replace("multiscale", "multiscale MLP").replace("mlp", "MLP")
            .replace("tcn", "TCN").replace("gru", "GRU").replace("_", " "))


def render(dest, preview=None):
    dest = Path(dest)
    data = dest / "data"
    for name, expected in read_json(data / "checksums.json").items():
        if sha(data / name) != expected:
            raise ValueError(f"Portable evidence changed: {name}")
    provenance = read_json(data / "provenance.json")
    if (sha(data / provenance["frozen_protocol_file"]) != provenance["frozen_protocol_sha256"]
            or provenance["frozen_protocol_sha256"] != read_json(data / "frozen_selection.json")["protocol_sha256"]
            or sha(data / provenance["portable_protocol_file"]) != provenance["portable_protocol_sha256"]
            or sha(dest / provenance["source_archive"]["file"]) != provenance["source_archive"]["sha256"]):
        raise ValueError("Portable provenance or exact source archive no longer matches")
    evidence = read_json(data / "evidence.json")
    selection, evaluated = evidence["selection"], evidence["evaluated"]
    metrics = pd.read_csv(data / "metrics.csv")
    population = metrics[metrics.pilot_id.eq("population")]
    screen = pd.read_csv(data / "sequence_screen.csv")
    confirmation = pd.read_csv(data / "sequence_confirmation.csv")
    inventory = pd.read_csv(data / "run_inventory.csv")
    pilots = pd.read_csv(data / "pilot_comparison.csv")
    skills = pd.read_csv(data / "pilot_skill_metrics.csv")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "text.color": FG,
                         "axes.labelcolor": FG, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.edgecolor": MUTED, "pdf.fonttype": 42, "savefig.facecolor": BG})
    figures, notes = [], []

    def page(title, subtitle):
        fig = plt.figure(figsize=(16, 9), facecolor=BG)
        fig.text(0.05, 0.95, "SKILL FORECASTING  /  V4 ARCHITECTURE EXPERIMENT", color=TEAL, fontsize=11, weight="bold")
        fig.text(0.05, 0.875, title, fontsize=27, weight="bold")
        fig.text(0.05, 0.818, subtitle, fontsize=13, color=MUTED)
        fig.text(0.05, 0.035, "B01 tactical observer → ten frozen trained pilots  |  Development evidence; reserved test untouched",
                 color=MUTED, fontsize=9)
        fig.text(0.95, 0.035, f"{len(figures) + 1:02}", ha="right", color=TEAL, fontsize=11)
        figures.append(fig)
        return fig

    def table(fig, bounds, headers, rows, widths=None, size=12):
        ax = fig.add_axes(bounds)
        ax.axis("off")
        cells = ax.table(cellText=rows, colLabels=headers, colWidths=widths, cellLoc="left", colLoc="left", bbox=[0, 0, 1, 1])
        cells.auto_set_font_size(False)
        cells.set_fontsize(size)
        for (row, _), cell in cells.get_celld().items():
            cell.set_facecolor("#21354d" if row == 0 else BG)
            cell.set_edgecolor("#38516a")
            cell.set_linewidth(0.4)
            cell.get_text().set_color(TEAL if row == 0 else FG)
            if row == 0:
                cell.get_text().set_weight("bold")
        return cells

    def words(fig, x, y, text, width=90, size=13, color=FG):
        wrapped = "\n".join(textwrap.fill(line, width=width) for line in text.split("\n"))
        return fig.text(x, y, wrapped, va="top", fontsize=size, color=color, linespacing=1.45)

    gain = evidence["evaluation"]["mean_accuracy_gain"] if evaluated else None
    title = (f"Mean accuracy: {gain * 100:+.2f} points against v3" if evaluated
             else "The architecture search retains the v3 baseline")
    fig = page(title, "Fixed comparison: 33 labels, +1/+2/+5 seconds, 45 shared validation scenario groups")
    rows = []
    for horizon in (1, 2, 5):
        for model in ("baseline_v3", "candidate_v4") if evaluated else ("baseline_v3",):
            row = population[(population.model == model) & population.horizon_s.eq(horizon)].iloc[0]
            rows.append([f"+{horizon} s", LABELS[model], f"{row.top1_accuracy * 100:.2f}%", f"{row.top3_accuracy * 100:.2f}%",
                         f"{row.macro_f1_catalogue:.4f}", f"{row.nll:.4f}", f"{row.brier:.4f}", f"{row.ece_10_bins * 100:.2f} pp"])
    table(fig, [0.05, 0.41, 0.90, 0.34], ["Horizon", "Model", "Top-1", "Top-3", "F1 / 33", "NLL", "Brier", "ECE"],
          rows, [0.09, 0.17, 0.12, 0.12, 0.12, 0.12, 0.12, 0.14], 12)
    if evaluated:
        intervals = pd.read_csv(data / "paired_scenario_bootstrap.csv")
        ci = "   |   ".join(f"+{int(r.horizon_s)} s: {r.accuracy_gain * 100:+.2f} pp [{r.lower_95 * 100:+.2f}, {r.upper_95 * 100:+.2f}]"
                          for r in intervals.itertuples())
        words(fig, 0.05, 0.36, "Paired scenario-bootstrap gains (95% intervals):\n" + ci, width=135, size=13)
        target_met = evidence["evaluation"]["practical_gain_target_met"]
        primary = evidence["primary_interval"]
        words(fig, 0.05, 0.24, f"Mean gain 95% interval: [{primary['lower_95'] * 100:+.2f}, {primary['upper_95'] * 100:+.2f}] pp. "
              f"Predeclared +1.00 pp target: {'met' if target_met else 'not met'}. "
              f"Active budget: {selection['active_members']} models / {selection['active_parameters'] / 1e6:.2f}M parameters; "
              f"v3: 12 / {selection['baseline_parameters'] / 1e6:.2f}M.", width=125, size=14, color=TEAL if target_met else GOLD)
    else:
        words(fig, 0.05, 0.35, "No candidate recipe qualified under the registered selection rules. No new outer-validation "
              "evaluation was performed. The numbers above are the preserved baseline, not a new result.", width=120, size=16, color=GOLD)
    if evaluated:
        words(fig, 0.05, 0.15, calibration_note(population), width=145, size=10.5, color=GOLD)
    words(fig, 0.05, 0.105, "F1 averages all 33 labels. Validation has been inspected before: these are development findings, "
          "not final test results.", width=145, size=10.5, color=MUTED)
    notes.append("The comparator is the current twelve-model v3 ensemble, never the older Transformer. All saved metric checks use exactly matched origins and horizon-specific target masks. The mean-accuracy interval uses the same 5,000 shared scenario-group draws across all three horizons, averages the three gains within each replicate, and then takes quantiles. It does not average interval endpoints. Intervals are conditional on fixed models and this development set, without selection adjustment. " + calibration_note(population))

    fig = page("Eight preregistered candidates, one frozen decision", "Screening and confirmation remain entirely within the original 210 training scenario groups")
    screen_rows = []
    for name, part in screen.groupby("recipe", sort=False):
        eligible = part[part.eligible]
        best = (eligible if len(eligible) else part).sort_values("mean_accuracy", ascending=False).iloc[0]
        screen_rows.append([short_name(name), f"{best.new_weight:.2f}", f"{best.mean_accuracy * 100:.2f}%",
                            f"{best.gain * 100:+.2f}", f"{best.nll:.3f}", "yes" if best.eligible else "no"])
    table(fig, [0.05, 0.34, 0.53, 0.41], ["Candidate", "Weight", "Top-1", "Δ pp", "NLL", "Guard"],
          screen_rows, [0.43, 0.10, 0.13, 0.11, 0.12, 0.11], 11)
    words(fig, 0.05, 0.29, "Screen uses a four-member seed-42 reference. Each row shows its best eligible registered blend "
          "(or best unguarded result if none passed). These are not validation scores.", width=73, size=11, color=MUTED)
    if len(confirmation):
        ranked = confirmation.groupby(["recipe", "new_weight"], sort=False).agg(
            mean_accuracy=("mean_accuracy", "mean"), gain=("gain", "mean"), eligible=("eligible", "all")).reset_index()
        ranked = ranked.sort_values("mean_accuracy", ascending=False).head(6)
        confirmation_rows = [[short_name(row.recipe), f"{row.new_weight:.2f}", f"{row.gain * 100:+.2f}",
                              "yes" if row.eligible else "no"] for row in ranked.itertuples()]
        table(fig, [0.62, 0.44, 0.33, 0.31], ["Confirmed blend", "w", "Δ pp", "Both"],
              confirmation_rows, [0.55, 0.10, 0.19, 0.16], 11)
    else:
        words(fig, 0.62, 0.72, "No shortlist survived screening; confirmation was not run.", width=42, size=15, color=GOLD)
    decision = (f"Frozen: {short_name(selection['recipe'])}; new-member weight {selection['new_weight']:.2f}."
                if evaluated else "Frozen decision: retain the existing v3 ensemble.")
    words(fig, 0.62, 0.37, decision, width=42, size=14, color=TEAL)
    words(fig, 0.62, 0.26, "Confirmation uses all 12 internal v3 members. Each fold must show positive mean accuracy gain, "
          "NLL ≤ baseline +0.02, F1 ≥ baseline −0.01 and each horizon ≥ baseline −0.003.", width=48, size=11)
    words(fig, 0.05, 0.12, "147 fit / 21 stop / 42 select groups per internal partition. Normalization uses fit groups; epochs and "
          "temperature use stop groups. The same two partitions overlap and have informed earlier experiments.", width=140, size=11, color=MUTED)
    notes.append("The screen compares predefined weights 0.25, 0.5, 0.75 and 1.0. The two shortlisted component families receive three seeds each on both internal partitions. The equal pair is also a preregistered recipe. No architecture is added after seeing the screen.")

    p5 = pilots[pilots.horizon_s.eq(5)].sort_values("pilot_id")
    pilot_title = (f"+5 s improves for {int((p5.gain > 0).sum())} of 10 pilots" if evaluated
                   else "Pilot-specific validation remains the known v3 result")
    fig = page(pilot_title, "Same ten trained policies; paired intervals keep shared scenario groups together")
    pilot_table = []
    for row in p5.itertuples():
        values = [row.pilot_id.replace("pilot_", ""), f"{row.count:,}", f"{row.baseline_top1 * 100:.2f}%"]
        if evaluated:
            values += [f"{row.candidate_top1 * 100:.2f}%", f"{row.gain * 100:+.2f}",
                       f"[{row.gain_lower * 100:+.2f}, {row.gain_upper * 100:+.2f}]"]
        pilot_table.append(values)
    headers = ["Pilot", "Origins", "v3 top-1"] + (["v4 top-1", "Gain pp", "95% interval"] if evaluated else [])
    table(fig, [0.05, 0.25, 0.59, 0.50], headers, pilot_table,
          [0.10, 0.16, 0.17, 0.17, 0.15, 0.25] if evaluated else [0.20, 0.30, 0.50], 12)
    ax = fig.add_axes([0.70, 0.29, 0.25, 0.45], facecolor=BG)
    y = np.arange(len(p5))
    if evaluated:
        ax.hlines(y, p5.gain_lower * 100, p5.gain_upper * 100, color=MUTED)
        ax.scatter(p5.gain.to_numpy() * 100, y, color=np.where(p5.gain > 0, TEAL, RED), zorder=3)
        ax.axvline(0, color=MUTED, linewidth=1)
        ax.set_xlabel("+5 s accuracy gain (pp)")
    else:
        ax.barh(y, p5.baseline_top1 * 100, color=BLUE)
        ax.set_xlabel("Known v3 +5 s accuracy (%)")
    ax.set_yticks(y, p5.pilot_id.str.replace("pilot_", "", regex=False))
    ax.invert_yaxis()
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    words(fig, 0.05, 0.18, "Pilot difficulty differs because behavior and label prevalence differ. These results do not test unseen "
          "pilots. Pointwise intervals are descriptive and are not corrected for examining ten pilots and three horizons.", width=138, size=12)
    notes.append("Portable pilot_comparison.csv includes all 30 pilot/horizon comparisons with support and paired intervals. metrics.csv also includes pilot-specific NLL, Brier, ECE and F1. Equal-pilot and equal-episode accuracy are exported separately from the primary pooled-origin estimand.")

    fig = page("Per-skill F1 preserves the full 33-label catalogue", "+5 seconds; support counts eligible forecast origins, which are correlated within flights")
    old = skills[skills.model.eq("baseline_v3") & skills.pilot_id.eq("population") & skills.horizon_s.eq(5)].set_index("skill")
    new = skills[skills.model.eq("candidate_v4") & skills.pilot_id.eq("population") & skills.horizon_s.eq(5)].set_index("skill") if evaluated else None
    rows = []
    for skill in evidence["classes"]:
        row = old.loc[skill]
        values = [skill, f"{int(row.support):,}", f"{row.f1:.3f}" if row.support else "—"]
        if evaluated:
            values.extend([f"{new.loc[skill].f1:.3f}" if row.support else "—",
                           f"{new.loc[skill].f1 - row.f1:+.3f}" if row.support else "—"])
        rows.append(values)
    headers = ["Skill", "n", "v3 F1"] + (["v4 F1", "Δ F1"] if evaluated else [])
    widths = [0.52, 0.12, 0.12, 0.12, 0.12] if evaluated else [0.64, 0.18, 0.18]
    table(fig, [0.05, 0.22, 0.435, 0.53], headers, rows[:17], widths, 9)
    table(fig, [0.515, 0.22, 0.435, 0.53], headers, rows[17:], widths, 9)
    words(fig, 0.05, 0.17, "Dash = no true validation examples at this horizon; it is not evidence of perfect or failed recall. "
          "Catalogue macro-F1 retains all 33 labels, including absent labels. Rare-skill changes need support-aware interpretation.", width=142, size=11)
    words(fig, 0.05, 0.095, "Complete precision, recall, F1 and predicted counts for every pilot × horizon × skill are in the evidence CSVs; "
          "skill_support.csv adds flight, scenario-group and pilot coverage.", width=145, size=10, color=MUTED)
    notes.append("F1 is the harmonic mean of precision and recall. It describes prediction of simulator-selected catalogue labels, not pilot competence or a unique physical maneuver. The primary macro-F1 always averages all 33 catalogue classes. Displayed dashes only prevent unsupported labels from looking like measured recall performance.")

    fig = page("Training cost and the evidence needed to reproduce this run", "Complete runs and unsuccessful candidates are retained; the previous slide pack and baseline remain unchanged")
    fresh = inventory[~inventory.reused_v3]
    costs = fresh.groupby("phase").agg(runs=("candidate", "size"), seconds=("elapsed_seconds", "sum"), epochs=("epochs_completed", "sum"))
    cost_rows = [[phase, f"{int(row.runs)}", f"{row.seconds / 60:.2f} min", f"{int(row.epochs)}"] for phase, row in costs.iterrows()]
    table(fig, [0.05, 0.57, 0.43, 0.18], ["Phase", "Runs", "Logged time", "Epochs"], cost_rows, [0.29, 0.16, 0.31, 0.24], 13)
    words(fig, 0.05, 0.51, f"{evidence['new_completed_runs']} completed new fits; {evidence['reused_internal_runs']} cached internal v3 runs. "
          f"Total logged training time: {evidence['new_training_seconds'] / 60:.2f} minutes. Timers exclude data loading, "
          "features and setup. Some confirmation fits shared the GPU concurrently: summed timers include contention, "
          "not end-to-end wall time or isolated-model throughput.", width=61, size=13)
    parameters = selection["active_parameters"] / 1e6
    words(fig, 0.05, 0.30, f"Selected budget: {selection['active_members']} models, {parameters:.2f}M parameters. "
          "Final models use fixed epochs and temperatures chosen internally. Final refits have training-objective logs only; "
          "outer-validation curves are deliberately absent.", width=61, size=13, color=TEAL)
    histories = pd.read_csv(data / "training_histories.csv")
    representative = selection["components"][0]["candidate"] if selection["components"] else read_json(data / "shortlist.json")["names"]
    if isinstance(representative, list):
        representative = representative[0] if representative else screen.iloc[0].recipe
    ax = fig.add_axes([0.56, 0.40, 0.39, 0.33], facecolor=BG)
    for fold, color in ((0, TEAL), (1, BLUE)):
        part = histories[histories.candidate.eq(representative) & histories.phase.eq("internal")
                         & histories.fold.eq(fold) & histories.seed.eq(42) & ~histories.reused_v3]
        if len(part):
            ax.plot(part.epoch, part.stop_mean_accuracy * 100, color=color, label=f"Partition {fold}, seed 42")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Mean +1/+2/+5 s top-1 (%)")
    ax.set_title(short_name(representative) + " — seed 42 stop accuracy", color=FG, fontsize=13)
    ax.legend(fontsize=9, facecolor=BG, edgecolor=MUTED, labelcolor=FG, loc="lower right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    words(fig, 0.56, 0.32, "Shown: seed 42 in each partition, not an average over seeds. Stop curves choose epochs within "
          "training-only partitions. Complete objectives, NLL, seed variation and timing are exported in training_histories.csv "
          "and run_inventory.csv.", width=61, size=12, color=MUTED)
    verification = read_json(data / "verification.json")
    words(fig, 0.05, 0.13, f"Verified: all {verification['preserved_artifacts_checked']} pre-existing artifact files unchanged; frozen sources "
          "and partitions match; validation origins align. Rebuild commands and source hashes accompany the PDF. No reserved test was evaluated.", width=142, size=11)
    notes.append("All displayed curves use seed 42 separately in the two internal partitions; they are not three-seed averages. They come from internal stop groups. Training uses label-smoothed cross entropy, while NLL evaluates unsmoothed predicted probability of the true label. These values therefore need not match numerically. Some confirmation fits ran concurrently on the same GPU; individual timers include shared-resource contention. Their sum is not end-to-end elapsed time or an isolated-model throughput comparison. Execution notes are included when recorded. The portable evidence supports a PDF rebuild without model checkpoints or simulator data.")

    dest.mkdir(parents=True, exist_ok=True)
    pdf_path = dest / "skill_forecasting_v4.pdf"
    with PdfPages(pdf_path, metadata={"Title": "Skill forecasting v4: architecture experiment", "Author": "Trajectory Classification",
                                     "CreationDate": None, "ModDate": None}) as pdf:
        for i, fig in enumerate(figures, 1):
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            # Check every text artist, including table cells and axis labels.
            for text in fig.findobj(matplotlib.text.Text):
                if not text.get_visible() or not text.get_text().strip():
                    continue
                bounds = text.get_window_extent(renderer)
                if bounds.x0 < -2 or bounds.y0 < -2 or bounds.x1 > fig.bbox.x1 + 2 or bounds.y1 > fig.bbox.y1 + 2:
                    raise ValueError(f"Text outside page {i}: {text.get_text()!r}")
            pdf.savefig(fig)
            if preview:
                Path(preview).mkdir(parents=True, exist_ok=True)
                fig.savefig(Path(preview) / f"slide_{i:02}.png", dpi=110)
            plt.close(fig)
    (dest / "speaker_notes.md").write_text("# v4 experiment notes\n\n" + "\n\n".join(
        f"## Slide {i}\n\n{note}" for i, note in enumerate(notes, 1)) + "\n", encoding="utf-8", newline="\n")
    write_readme(dest, evidence, population)
    write_json(dest / "checksums.json", {path.relative_to(dest).as_posix(): sha(path)
                                        for path in sorted(dest.rglob("*"))
                                        if path.is_file() and path != dest / "checksums.json"})
    return {"pdf": str(pdf_path), "pages": len(figures), "new_outer_validation_evaluated": evaluated}


def write_readme(dest, evidence, population):
    evaluated, selection = evidence["evaluated"], evidence["selection"]
    if evaluated:
        gain = evidence["evaluation"]["mean_accuracy_gain"] * 100
        interval = evidence["primary_interval"]
        outcome = (f"The frozen v4 recipe changes mean accuracy by **{gain:+.2f} percentage points** against the retained v3 ensemble "
                   f"(paired scenario-bootstrap 95% interval **[{interval['lower_95'] * 100:+.2f}, {interval['upper_95'] * 100:+.2f}] pp**).")
    else:
        outcome = "The registered search retains the existing v3 ensemble. No new outer-validation evaluation was performed."
    lines = ["# Skill forecasting v4: architecture experiment", "", outcome, "",
             "[Open the five-page PDF](skill_forecasting_v4.pdf) · [Speaker notes](speaker_notes.md)", "",
             ("The comparator is the current **12-member v3 MLP ensemble**. Both versions use the same ten trained pilots, "
             "33 catalogue labels, exact +1/+2/+5-second targets, forecast origins and scenario partitions. "
             "B01 is the tactical observer of the frozen trained pilots. Labels describe selected catalogue behavior, not competence."), "",
             "| Horizon | Model | Top-1 | Top-3 | Macro-F1 / 33 | NLL | Brier | ECE |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in population.itertuples():
        lines.append(f"| +{int(row.horizon_s)} s | {LABELS[row.model]} | {row.top1_accuracy * 100:.2f}% | "
                     f"{row.top3_accuracy * 100:.2f}% | {row.macro_f1_catalogue:.4f} | {row.nll:.4f} | "
                     f"{row.brier:.4f} | {row.ece_10_bins * 100:.2f} pp |")
    if evaluated:
        lines.extend(["", calibration_note(population)])
    lines.extend(["", (f"Selected budget: **{selection['active_members']} models / {selection['active_parameters'] / 1e6:.2f}M parameters**. "
                  f"This experiment completed {evidence['new_completed_runs']} new fits, with {evidence['new_training_seconds'] / 60:.2f} "
                  "minutes of summed logged training time; loading and setup are excluded. Some confirmation fits ran concurrently "
                  "on one shared GPU: their timers include resource contention and do not measure end-to-end elapsed time or "
                  "isolated-model throughput."), "",
                  ("Selection used two grouped partitions entirely within original training. Epochs and temperature came from stop groups; "
                  "recipe selection used selection groups. The final recipe was frozen before any new outer-validation evaluation. "
                  "Those internal partitions overlap and have guided previous development. Outer validation has also been inspected previously. "
                  "The reserved test remains untouched; these are development results for new scenarios with known pilots."), "",
                  ("Expanded blends increase model count. Standalone and blended scores are retained, but this comparison does not isolate "
                  "an architecture-only causal gain. Bootstrap intervals resample whole shared scenario groups and are pointwise, without "
                  "multiple-comparison correction."), "", "## Portable evidence", "",
                  "- [Full metrics](data/metrics.csv): population and pilot metrics at each horizon, including NLL, Brier and ten-bin ECE.",
                  "- [Pilot comparisons](data/pilot_comparison.csv): support and paired scenario intervals.",
                  "- [Pilot × skill metrics](data/pilot_skill_metrics.csv) and [skill support](data/skill_support.csv): P/R/F1, counts and coverage.",
                  "- [Internal screen](data/sequence_screen.csv), [confirmation](data/sequence_confirmation.csv) and [frozen selection](data/frozen_selection.json).",
                  "- [Run inventory](data/run_inventory.csv) and [training histories](data/training_histories.csv): completed/failed runs, seed, epoch, timing and parameters.",
                  "- [Common-origin temporal results](data/temporal_common_origin_metrics.csv) and [accuracy weighting comparisons](data/accuracy_weightings.csv).",
                  "- [Verification](data/verification.json), [provenance](data/evidence.json) and [bundle checksums](checksums.json).", "",
                  "## Exact experiment provenance", "",
                  ("[frozen_protocol.json](data/frozen_protocol.json) preserves the original protocol bytes and its SHA-256 matches "
                  "the hash recorded in the frozen selection. [protocol.json](data/protocol.json) is a convenience copy with only "
                  "the data and baseline directory paths rewritten as repository-relative paths; its bytes intentionally have a different hash. "
                  "[provenance.json](data/provenance.json) records both hashes and the source-archive hash."), "",
                  ("The [preservation manifest](data/preserved_artifacts.json) contains every original dataset/experiment file hash. "
                  "The [source snapshot ZIP](source_snapshot.zip) contains the exact registered training source, configuration and CLI bytes "
                  "at repository-relative paths, with fixed ZIP timestamps. It includes no checkpoints or datasets. Extract it into an "
                  "isolated repository checkout when exact source replay is required; avoid line-ending conversion, which changes registered "
                  "hashes even when Python behavior is unchanged. The original source hashes remain inside both protocol copies."), "",
                  "## Rebuild", "", "From the repository root, the portable PDF can be rebuilt without simulator data or checkpoints:", "",
                  "```powershell", ".venv\\Scripts\\python.exe scripts/report_sequence_experiments.py", "```", "",
                  "Collect the completed local experiment and regenerate the report:", "", "```powershell",
                  ".venv\\Scripts\\python.exe scripts/report_sequence_experiments.py --collect", "```", "",
                  "Reproduce training only in a **new, empty output directory**; preserve existing runs:", "", "```powershell",
                  ".venv\\Scripts\\python.exe scripts/sequence_skill_forecaster.py initialize --output artifacts/skill_forecasting/v4_reproduction",
                  ".venv\\Scripts\\python.exe scripts/sequence_skill_forecaster.py screen --output artifacts/skill_forecasting/v4_reproduction",
                  ".venv\\Scripts\\python.exe scripts/sequence_skill_forecaster.py confirm --output artifacts/skill_forecasting/v4_reproduction",
                  ".venv\\Scripts\\python.exe scripts/sequence_skill_forecaster.py refit --output artifacts/skill_forecasting/v4_reproduction",
                  ".venv\\Scripts\\python.exe scripts/sequence_skill_forecaster.py evaluate --output artifacts/skill_forecasting/v4_reproduction",
                  "```", "", ("The final evaluation command refuses a second evaluation in that output directory. Reproduction requires "
                  "the preserved local dataset and baseline artifacts, listed by hash in the protocol. GPU/software differences may "
                  "change numerical results. The earlier 41-slide catalogue presentation is preserved separately."), ""])
    (dest / "README.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--experiment", type=Path, default=ROOT / "artifacts/skill_forecasting/improvement_v4")
    parser.add_argument("--output", type=Path, default=DEST)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    if args.collect:
        collect(args.experiment, args.output)
    print(json.dumps(render(args.output, args.preview_dir), indent=2))


if __name__ == "__main__":
    main()
