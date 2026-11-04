"""Publish the frozen Transformer challenge without fitting or model inference.

--collect reads completed, fixed predictions. The default rebuild needs only the
portable evidence bundle. The primary all-Transformer result is always shown;
an optional incumbent blend remains a separately named secondary comparison.
"""

import argparse
import json
import shutil
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from report_sequence_experiments import export_source_archive, mean_horizon_interval
from skill_catalogue_evidence import diagnostic_metrics, read_json, sha, write_json
from skill_catalogue_review import accuracy_weightings, paired_group_intervals

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "docs/experiments/skill_forecasting_v5_transformers"
BG, FG, MUTED = "#101c2c", "#edf2f7", "#b5c2d0"
TEAL, BLUE, GOLD, RED = "#4bd7b1", "#69aafa", "#f1c16d", "#f58c8c"
LABELS = {"baseline_v4": "v4 incumbent", "primary_transformer": "Transformer only",
          "secondary_blend": "Secondary blend"}
HORIZONS = (1, 2, 5)


def csv(path, frame):
    frame.to_csv(path, index=False, lineterminator="\n")


def output_roles(selection):
    """The reporting order follows the frozen question, never validation scores."""
    primary = selection["primary_transformer"]
    if primary is None or primary["kind"] != "primary_transformer" or primary["new_weight"] != 1:
        raise ValueError("An all-Transformer primary challenger is required")
    roles = ["baseline_v4", "primary_transformer"]
    secondary = selection.get("secondary_blend")
    if secondary is not None:
        if secondary["kind"] != "secondary_blend" or not 0 < secondary["new_weight"] < 1:
            raise ValueError("The secondary result must be an explicitly labelled incumbent blend")
        roles.append("secondary_blend")
    return roles


def metric_direction(values):
    values = np.asarray(values)
    if np.all(values == 0):
        return "unchanged at all horizons"
    return "improved at all horizons" if np.all(values < 0) else (
        "worsened at all horizons" if np.all(values > 0) else "changed across horizons")


def quality_notes(metrics, role):
    """Describe each probability metric separately, including regressions."""
    population = metrics[metrics.pilot_id.eq("population")]
    old = population[population.model.eq("baseline_v4")].sort_values("horizon_s")
    new = population[population.model.eq(role)].sort_values("horizon_s")
    if (len(old) != 3 or len(new) != 3
            or old.horizon_s.tolist() != list(HORIZONS) or new.horizon_s.tolist() != list(HORIZONS)):
        raise ValueError("Probability comparisons require all three fixed horizons")
    return "; ".join(f"{label} {metric_direction(new[column].to_numpy() - old[column].to_numpy())}"
                     for column, label in [("nll", "NLL"), ("brier", "Brier"), ("ece_10_bins", "ECE")]) + "."


def comparison_statement(label, result):
    interval = result["mean_accuracy_interval"]
    return (f"{label}: mean accuracy gain {result['mean_accuracy_gain'] * 100:+.2f} pp; "
            f"95% scenario interval [{interval['lower_95'] * 100:+.2f}, "
            f"{interval['upper_95'] * 100:+.2f}] pp.")


def guard_summary(protocol):
    guards = protocol["guards"]
    nll = guards["maximum_nll_increase"]
    f1 = guards["maximum_macro_f1_decrease"]
    accuracy = guards["maximum_horizon_accuracy_decrease"]
    return (f"Each partition versus v4: NLL <= v4 +{nll:.2f}; mean F1/33 >= v4 -{f1:.2f}; "
            f"each horizon accuracy >= v4 -{accuracy:.3f} (-{accuracy * 100:.2f} pp). "
            "Secondary: mean accuracy gain >0 in both partitions.")


def screen_reference(data):
    references = read_json(Path(data) / "internal_baseline_metrics.json")
    selected = [row for row in references if row["fold"] == 0]
    if len(selected) != 1 or selected[0]["members"] != 15:
        raise ValueError("Screening requires the cached fifteen-member v4 fold-0 reference")
    return selected[0]


def collect(experiment, dest):
    experiment, dest = Path(experiment).resolve(), Path(dest).resolve()
    protocol = read_json(experiment / "protocol.json")
    selection = read_json(experiment / "frozen_selection.json")
    evaluation = read_json(experiment / "validation/evaluation.json")
    roles = output_roles(selection)
    if (selection["protocol_sha256"] != sha(experiment / "protocol.json")
            or evaluation["selection_sha256"] != sha(experiment / "frozen_selection.json")
            or set(evaluation["recipes"]) != set(roles[1:]) or evaluation.get("test_evaluated", True)):
        raise ValueError("Completed evaluation does not match the jointly frozen outputs")
    if not set(selection["primary_transformer"]["components"]).issubset(protocol["pure_candidates"]):
        raise ValueError("The primary result contains a non-Transformer forecasting member")
    existing = dest / "data/evidence.json"
    if existing.exists() and read_json(existing)["protocol_sha256"] != sha(experiment / "protocol.json"):
        raise ValueError("Preserve the existing report for its own experiment")
    preserved = read_json(experiment / "preserved_artifacts.json")
    if sha(experiment / "preserved_artifacts.json") != protocol["preserved_artifacts_sha256"]:
        raise ValueError("Preservation manifest changed")
    for name, expected in preserved.items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Historical artifact changed: {name}")
    for name, expected in protocol["source_hashes"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    for name, expected in protocol["partition_hashes"].items():
        if sha(experiment / name) != expected:
            raise ValueError(f"Frozen partition changed: {name}")
    for name, expected in read_json(experiment / "validation/output_hashes.json").items():
        if sha(experiment / "validation" / name) != expected:
            raise ValueError(f"Frozen evaluation output changed: {name}")
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

    prepared = Path(protocol["data_directory"])
    baseline_v3, baseline_v4 = Path(protocol["baseline_v3_directory"]), Path(protocol["baseline_v4_directory"])
    all_samples = pd.read_parquet(record(prepared / "samples.parquet"))
    if not all_samples.split.isin(["train", "validation"]).all():
        raise ValueError("Prepared data contains a reserved partition")
    mask = all_samples.split.eq("validation").to_numpy()
    samples = all_samples.loc[mask].reset_index(drop=True)
    targets = np.load(record(prepared / "targets.npy"))[mask]
    classes = read_json(record(prepared / "classes.json"))
    if (len(classes) != 33 or len(samples) != 30802 or samples.scenario_group.nunique() != 45
            or (targets >= 0).sum(0).tolist() != [30802, 30466, 29458]):
        raise ValueError("The comparison population or target masks changed")
    directories = {"baseline_v4": baseline_v4 / "validation/ensemble"}
    directories.update({role: experiment / "validation" / role for role in roles[1:]})
    predictions, summary_frames, skill_frames = {}, [], []
    for role, directory in directories.items():
        pd.testing.assert_frame_equal(samples, pd.read_parquet(record(directory / "validation_samples.parquet")))
        probability = np.load(record(directory / "validation_probabilities.npy"))
        if (probability.shape != (len(samples), 3, 33) or not np.isfinite(probability).all()
                or np.any(probability < 0) or not np.allclose(probability.sum(-1), 1, atol=1e-5)):
            raise ValueError("Invalid saved probability array")
        predictions[role] = probability
        summary_frames.append(pd.read_csv(record(directory / "validation_metrics.csv")).assign(model=role))
        skill_frames.append(pd.read_csv(record(directory / "per_class_metrics.csv")).assign(model=role))
    summaries, skills = pd.concat(summary_frames), pd.concat(skill_frames)
    csv(data / "saved_summary_metrics.csv", summaries)
    csv(data / "pilot_skill_metrics.csv", skills)
    intervals = {}
    for role in roles[1:]:
        recipe = selection[role]
        members = [np.load(record(experiment / "validation/members" / f"{name}_s{seed}"
                                  / "validation_probabilities.npy"))
                   for name in recipe["components"] for seed in protocol["training_seeds"]]
        proposed = np.mean(members, axis=0)
        expected = (1 - recipe["new_weight"]) * predictions["baseline_v4"] + recipe["new_weight"] * proposed
        np.testing.assert_array_equal(expected, predictions[role])
        interval = mean_horizon_interval(predictions["baseline_v4"], predictions[role], targets,
                                         samples.scenario_group.to_numpy())
        for key in ("mean_accuracy_gain", "lower_95", "upper_95"):
            if not np.isclose(interval[key], evaluation["recipes"][role]["mean_accuracy_interval"][key], atol=1e-12, rtol=0):
                raise ValueError(f"Joint scenario interval changed: {role}/{key}")
        intervals[role] = interval
    write_json(data / "mean_horizon_intervals.json", intervals)
    diagnostics, reliability_rows, weighting_rows, pilot_rows, support_rows, temporal_rows = [], [], [], [], [], []
    for role, probability in predictions.items():
        for h, horizon in enumerate(HORIZONS):
            for pilot in ("population", *sorted(samples.pilot_id.unique())):
                valid = targets[:, h] >= 0
                if pilot != "population":
                    valid &= samples.pilot_id.eq(pilot).to_numpy()
                metrics, reliability = diagnostic_metrics(probability[valid, h], targets[valid, h])
                saved = summaries[summaries.model.eq(role) & summaries.pilot_id.eq(pilot)
                                  & summaries.horizon_s.eq(horizon) & summaries.cohort.eq("all")].iloc[0]
                for key in ("top1_accuracy", "top3_accuracy", "nll"):
                    if not np.isclose(metrics[key], saved[key], atol=2e-6):
                        raise ValueError(f"Saved metric mismatch: {role}/{pilot}/{horizon}/{key}")
                diagnostics.append({"model": role, "pilot_id": pilot, "horizon_s": horizon,
                                    "macro_f1_catalogue": saved.macro_f1_catalogue,
                                    "macro_f1_supported": saved.macro_f1_supported,
                                    "supported_classes": int(saved.supported_classes), **metrics})
                reliability_rows.extend({"model": role, "pilot_id": pilot, "horizon_s": horizon, **r}
                                        for r in reliability)
                if pilot != "population" and role != "baseline_v4":
                    old = predictions["baseline_v4"][valid, h].argmax(-1) == targets[valid, h]
                    new = probability[valid, h].argmax(-1) == targets[valid, h]
                    pilot_rows.append({"model": role, "pilot_id": pilot, "horizon_s": horizon,
                                       "count": int(valid.sum()), "episodes": samples.loc[valid, "episode_id"].nunique(),
                                       "scenario_groups": samples.loc[valid, "scenario_group"].nunique(),
                                       "baseline_top1": float(old.mean()), "candidate_top1": float(new.mean()),
                                       **paired_group_intervals(old, new, samples.loc[valid, "scenario_group"].to_numpy())})
            valid = targets[:, h] >= 0
            correct = probability[valid, h].argmax(-1) == targets[valid, h]
            weighting_rows.append({"model": role, "horizon_s": horizon,
                                   **accuracy_weightings(correct, samples.loc[valid, "episode_id"],
                                                         samples.loc[valid, "pilot_id"])})
            for lower, upper, phase in ((6, 20, "6-20 s"), (21, 60, "21-60 s"), (61, np.inf, "61+ s")):
                valid = (targets >= 0).all(1) & samples.time_s.between(lower, upper).to_numpy()
                metrics, _ = diagnostic_metrics(probability[valid, h], targets[valid, h])
                temporal_rows.append({"model": role, "horizon_s": horizon, "elapsed_phase": phase,
                                      "episodes": samples.loc[valid, "episode_id"].nunique(),
                                      "scenario_groups": samples.loc[valid, "scenario_group"].nunique(), **metrics})
    for h, horizon in enumerate(HORIZONS):
        for label, skill in enumerate(classes):
            valid = targets[:, h] == label
            support_rows.append({"horizon_s": horizon, "skill": skill, "count": int(valid.sum()),
                                 "episodes": samples.loc[valid, "episode_id"].nunique(),
                                 "scenario_groups": samples.loc[valid, "scenario_group"].nunique(),
                                 "pilots": samples.loc[valid, "pilot_id"].nunique()})
    metrics = pd.DataFrame(diagnostics)
    for name, frame in [("metrics", metrics), ("reliability", pd.DataFrame(reliability_rows)),
                        ("pilot_comparison", pd.DataFrame(pilot_rows)), ("skill_support", pd.DataFrame(support_rows)),
                        ("accuracy_weightings", pd.DataFrame(weighting_rows)),
                        ("temporal_common_origin_metrics", pd.DataFrame(temporal_rows))]:
        csv(data / f"{name}.csv", frame)
    runs, histories = [], []
    paths = [(path, False) for phase in ("inner_*", "refit") for path in sorted(experiment.glob(f"{phase}/*/run.json"))]
    for fold in (0, 1):
        for name in protocol["baseline_mlp_components"]:
            for seed in protocol["training_seeds"]:
                directory = baseline_v3 if seed == 42 else baseline_v4
                paths.append((directory / f"inner_{fold}/{name}_s{seed}/run.json", True))
        for seed in protocol["training_seeds"]:
            paths.append((baseline_v4 / f"inner_{fold}/history60_tcn_s{seed}/run.json", True))
    for path, reused in paths:
        run = read_json(record(path))
        summary = read_json(record(path.parent / "summary.json")) if (path.parent / "summary.json").exists() else {}
        candidate = run["candidate"]
        row = {"candidate": summary.get("name", path.parent.name.rsplit("_s", 1)[0]),
               "phase": "refit" if run["final"] else "internal", "fold": run["fold"], "seed": run["seed"],
               "reused_baseline": reused, "status": "complete" if summary else "incomplete",
               "architecture": candidate["architecture"], "history_seconds": candidate["history_seconds"],
               "width": candidate["width"], "layers": candidate["layers"], "parameters": run["parameters"],
               "epochs_completed": summary.get("epochs_completed"), "best_epoch": summary.get("best_epoch"),
               "temperature": summary.get("temperature"), "elapsed_seconds": summary.get("elapsed_seconds"),
               "source_directory": path.parent.relative_to(ROOT).as_posix()}
        if "selection_metrics" in summary:
            row.update({key: summary["selection_metrics"][key] for key in ("mean_accuracy", "nll", "mean_macro_f1")})
        runs.append(row)
        if (path.parent / "history.csv").exists():
            histories.append(pd.read_csv(record(path.parent / "history.csv")).assign(
                candidate=row["candidate"], phase=row["phase"], fold=row["fold"], seed=row["seed"], reused_baseline=reused))
    inventory = pd.DataFrame(runs)
    csv(data / "run_inventory.csv", inventory)
    csv(data / "training_histories.csv", pd.concat(histories, ignore_index=True))
    for name in ("transformer_screen.csv", "transformer_confirmation.csv", "paired_scenario_bootstrap.csv"):
        csv(data / name, pd.read_csv(record(experiment / name)))
    for name in ("protocol.json", "shortlist.json", "frozen_selection.json", "selection_receipt.json",
                 "internal_baseline_metrics.json", "environment.json", "execution_notes.json"):
        if (experiment / name).exists():
            value = read_json(record(experiment / name))
            if name == "protocol.json":
                value = dict(value)
                for key in ("data_directory", "baseline_v3_directory", "baseline_v4_directory"):
                    value[key] = Path(value[key]).relative_to(ROOT).as_posix()
            write_json(data / name, value)
    write_json(data / "evaluation.json", read_json(record(experiment / "validation/evaluation.json")))
    provenance = {"frozen_protocol_sha256": sha(data / "frozen_protocol.json"),
                  "portable_protocol_sha256": sha(data / "protocol.json"),
                  "portable_protocol_note": "Only data/baseline directory paths are repository-relative; use frozen_protocol.json for registered bytes.",
                  "preservation_manifest_sha256": sha(data / "preserved_artifacts.json"), "source_archive": archive}
    if provenance["frozen_protocol_sha256"] != selection["protocol_sha256"]:
        raise ValueError("Exact exported protocol differs from frozen selection")
    write_json(data / "provenance.json", provenance)
    verification = {"preserved_artifacts_checked": len(preserved), "preserved_artifacts_unchanged": True,
                    "old_pdfs_checked": [name for name in preserved if name.endswith(".pdf")],
                    "frozen_source_files_checked": len(protocol["source_hashes"]), "frozen_sources_unchanged": True,
                    "frozen_partitions_unchanged": True, "forecast_origins_and_masks_match": True,
                    "saved_recipe_probabilities_exact": True, "joint_scenario_intervals_reproduced": True,
                    "primary_contains_only_registered_pure_transformers": True,
                    "report_performed_training_or_model_inference": False, "reserved_test_evaluated": False,
                    "protocol_sha256": sha(experiment / "protocol.json"),
                    "selection_sha256": sha(experiment / "frozen_selection.json")}
    write_json(data / "verification.json", verification)
    write_json(experiment / "report_verification.json", verification)
    for name in (Path(__file__).name, "report_sequence_experiments.py", "skill_catalogue_evidence.py",
                 "skill_catalogue_review.py", "build_forecasting_insights.py"):
        record(ROOT / "scripts" / name)
    evidence = {"protocol_sha256": sha(experiment / "protocol.json"), "roles": roles,
                "selection": selection, "evaluation": evaluation, "mean_horizon_intervals": intervals,
                "classes": classes, "horizons_s": list(HORIZONS), "validation_origins": len(samples),
                "validation_groups": 45, "validation_episodes": 450,
                "valid_targets": (targets >= 0).sum(0).tolist(),
                "new_completed_runs": int((~inventory.reused_baseline & inventory.status.eq("complete")).sum()),
                "new_training_seconds": float(inventory.loc[~inventory.reused_baseline, "elapsed_seconds"].sum()),
                "reused_internal_runs": int(inventory.reused_baseline.sum()), "source_sha256": sources,
                "quality_notes": {role: quality_notes(metrics, role) for role in roles[1:]},
                "execution_notes": read_json(experiment / "execution_notes.json")
                if (experiment / "execution_notes.json").exists() else None}
    write_json(data / "evidence.json", evidence)
    write_json(data / "checksums.json", {path.name: sha(path) for path in sorted(data.iterdir())
                                        if path.is_file() and path.name != "checksums.json"})
    return evidence


def short_name(name):
    return (name.replace("history20_", "20 s ").replace("history60_", "60 s ")
            .replace("_transformer", " T").replace("pure_pair", "Pure pair").replace("_", " "))


def recipe_text(selection, role):
    recipe = selection.get(role)
    if recipe is None:
        return "No secondary incumbent blend qualified internally."
    components = " + ".join(short_name(name) for name in recipe["components"])
    if role == "primary_transformer":
        return f"Primary: {components}; {recipe['active_members']} Transformer members, no incumbent members."
    return (f"Secondary: {recipe['new_weight']:.0%} new ({components}) + "
            f"{1 - recipe['new_weight']:.0%} preserved v4; {recipe['active_members']} total models.")


def render(dest, preview=None):
    dest = Path(dest)
    data = dest / "data"
    for name, expected in read_json(data / "checksums.json").items():
        if sha(data / name) != expected:
            raise ValueError(f"Portable evidence changed: {name}")
    provenance, selection = read_json(data / "provenance.json"), read_json(data / "frozen_selection.json")
    if (sha(data / "frozen_protocol.json") != selection["protocol_sha256"]
            or sha(data / "frozen_protocol.json") != provenance["frozen_protocol_sha256"]
            or sha(dest / "source_snapshot.zip") != provenance["source_archive"]["sha256"]):
        raise ValueError("Exact protocol or frozen-source archive changed")
    evidence, protocol = read_json(data / "evidence.json"), read_json(data / "protocol.json")
    roles = output_roles(selection)
    metrics = pd.read_csv(data / "metrics.csv")
    population = metrics[metrics.pilot_id.eq("population")]
    pilots = pd.read_csv(data / "pilot_comparison.csv")
    skills = pd.read_csv(data / "pilot_skill_metrics.csv")
    inventory = pd.read_csv(data / "run_inventory.csv")
    histories = pd.read_csv(data / "training_histories.csv")
    screen = pd.read_csv(data / "transformer_screen.csv")
    confirmation = pd.read_csv(data / "transformer_confirmation.csv")
    reference = screen_reference(data)
    guards = guard_summary(protocol)
    figures, notes = [], []
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "text.color": FG,
                         "axes.labelcolor": FG, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.edgecolor": MUTED, "pdf.fonttype": 42})

    def page(title, subtitle):
        fig = plt.figure(figsize=(16, 9), facecolor=BG)
        fig.text(0.05, 0.95, "SKILL FORECASTING / V5 TRANSFORMER CHALLENGE", color=TEAL, fontsize=11, weight="bold")
        fig.text(0.05, 0.875, title, fontsize=25, weight="bold")
        fig.text(0.05, 0.82, subtitle, fontsize=12, color=MUTED)
        fig.text(0.05, 0.035, "B01 observes ten frozen trained pilots | Development validation; reserved test untouched", fontsize=9, color=MUTED)
        fig.text(0.95, 0.035, f"{len(figures) + 1:02}", ha="right", color=TEAL, fontsize=10)
        figures.append(fig)
        return fig

    def words(fig, x, y, text, width=125, size=12, color=FG):
        wrapped = "\n".join(textwrap.fill(line, width=width) for line in text.split("\n"))
        return fig.text(x, y, wrapped, va="top", fontsize=size, color=color, linespacing=1.35)

    def table(fig, bounds, columns, rows, widths, size=11):
        ax = fig.add_axes(bounds)
        ax.axis("off")
        cells = ax.table(cellText=rows, colLabels=columns, colWidths=widths, cellLoc="left", colLoc="left", bbox=[0, 0, 1, 1])
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

    primary = evidence["evaluation"]["recipes"]["primary_transformer"]
    gain = primary["mean_accuracy_gain"] * 100
    fig = page(f"Best tested Transformer-only challenger: {gain:+.2f} points", "Primary question: can an all-Transformer ensemble match the current 15-model v4 incumbent?")
    rows = []
    for horizon in HORIZONS:
        for role in roles:
            row = population[population.model.eq(role) & population.horizon_s.eq(horizon)].iloc[0]
            rows.append([f"+{horizon} s", LABELS[role], f"{row.top1_accuracy * 100:.2f}%", f"{row.top3_accuracy * 100:.2f}%",
                         f"{row.macro_f1_catalogue:.4f}", f"{row.nll:.4f}", f"{row.brier:.4f}", f"{row.ece_10_bins * 100:.2f} pp"])
    table(fig, [0.05, 0.36, 0.90, 0.40], ["Horizon", "Model", "Top-1", "Top-3", "F1 / 33", "NLL", "Brier", "ECE"],
          rows, [0.08, 0.20, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12], 11)
    statements = [comparison_statement(LABELS[role], evidence["evaluation"]["recipes"][role]) for role in roles[1:]]
    words(fig, 0.05, 0.32, "\n".join(statements), width=140, size=13, color=TEAL if gain >= 0 else GOLD)
    quality = "\n".join(f"{LABELS[role]}: {quality_notes(metrics, role)}" for role in roles[1:])
    words(fig, 0.05, 0.22, quality, width=145, size=11)
    words(fig, 0.05, 0.11, "Best tested is not an established optimum. Intervals are conditional on frozen models and previously inspected "
          "development data; an interval crossing zero does not prove equivalence. The secondary blend is a different model family.", width=145, size=10, color=MUTED)
    notes.append("The primary all-Transformer challenger is reported even when worse. A positive secondary blend does not answer the primary question by itself: it still retains the incumbent MLP/TCN models. Reported outputs were frozen before one validation invocation. The labels, origins, masks and horizons are unchanged. Intervals use 5,000 shared scenario resamples, averaging the three horizon gains within each draw. These are not selection-adjusted intervals or an untouched final test.")

    fig = page("What the Transformer variants actually change", "Same observation information: 91 features and 20/60-second sparse histories, with all earlier origins retained")
    architecture_rows = [
        ["Pooled Transformer", "26 / 66 tokens", "Final state + masked mean + current state", "Transformer"],
        ["Temporal flattened", "26 / 66 tokens", "Keep all contextual token positions in readout", "Transformer"],
        ["Multiscale patch", "9 / 17 patches", "Five older tokens / two recent / current separate", "Transformer"],
        ["Attention + MLP", "26 / 66 tokens", "Attention plus a separate full-history MLP path", "Hybrid"],
    ]
    table(fig, [0.05, 0.49, 0.90, 0.26], ["Family", "20 s / 60 s", "Representation", "Classification"], architecture_rows,
          [0.20, 0.15, 0.49, 0.16], 11)
    words(fig, 0.05, 0.43, "New attention variants encode known relative times and gaps, mask missing history, and use independently "
          "initialized pre-norm blocks. Bidirectional attention stays inside observations available at the forecast origin. "
          "The pooled comparison retains its original learned-position representation.", width=140, size=12)
    words(fig, 0.05, 0.31, recipe_text(selection, "primary_transformer"), width=135, size=13, color=TEAL)
    words(fig, 0.05, 0.24, recipe_text(selection, "secondary_blend"), width=135, size=13, color=BLUE)
    words(fig, 0.05, 0.13, "Transformer-only means no separate MLP/TCN forecasting members: standard feed-forward blocks and current-state "
          "readouts remain part of the network. Width, regularization and representation are recipe choices, not a proof that attention "
          "alone caused any gain. Full settings and parameter counts accompany the PDF.", width=144, size=11, color=MUTED)
    notes.append("Sparse history uses older one-second observations and recent half-second observations. Patches never cross that sampling boundary and keep current state separate; all original input tokens are represented. Fixed time/gap metadata adds no new sensor information. Attention can connect all observed historical positions without seeing future observations. The explicitly named hybrid adds a parallel full-history residual MLP and cannot be presented as an all-Transformer result. Primary and secondary recipes have independently reported member counts and parameters.")

    fig = page("A bounded screen and confirmation inside original training", "147 fit / 21 stop / 42 selection scenario groups per partition; all ten versions of each scenario stay together")
    screen_rows = [["v4 reference", "15 models", f"{reference['mean_accuracy'] * 100:.2f}%",
                    f"{reference['nll']:.3f}", f"{reference['mean_macro_f1']:.3f}"]]
    screen_rows += [[short_name(row.recipe), "Hybrid" if row.kind == "architectural_hybrid" else "Transformer",
                    f"{row.mean_accuracy * 100:.2f}%", f"{row.nll:.3f}", f"{row.mean_macro_f1:.3f}"]
                   for row in screen.itertuples()]
    table(fig, [0.05, 0.34, 0.51, 0.42], ["Screen / reference", "Family", "Top-1", "NLL", "F1 / 33"], screen_rows,
          [0.38, 0.21, 0.15, 0.13, 0.13], 10)
    words(fig, 0.05, 0.30, "Preliminary fold-0 screen: one model per candidate; v4 combines 15.", width=75, size=10, color=MUTED)
    primary_rows = confirmation[confirmation.kind.eq("primary_transformer")].groupby("recipe").agg(
        mean_accuracy=("mean_accuracy", "mean"), gain=("gain", "mean"), guards=("eligible", "all"))
    confirmed = [[short_name(name), f"{row.mean_accuracy * 100:.2f}%", f"{row.gain * 100:+.2f}", "yes" if row.guards else "no"]
                 for name, row in primary_rows.sort_values("mean_accuracy", ascending=False).iterrows()]
    table(fig, [0.60, 0.53, 0.35, 0.23], ["Pure recipe", "Top-1", "Gain pp", "Guards"], confirmed,
          [0.42, 0.22, 0.20, 0.16], 10)
    words(fig, 0.60, 0.47, "Primary selection: " + ("quality guards passed on both partitions." if selection["primary_transformer"]["quality_guards_passed_both_folds"]
          else "best tested pure challenger; quality guards did not pass on both partitions."), width=49, size=11, color=TEAL)
    words(fig, 0.60, 0.36, guards, width=58, size=10, color=BLUE)
    words(fig, 0.05, 0.23, "The final primary is a multi-seed ensemble: three seeds per shortlisted architecture, confirmed on both "
          "internal partitions against the cached 15-member v4 reference. These are internal selection scores.", width=140, size=12)
    words(fig, 0.05, 0.12, "Normalization uses fit groups; epochs and temperatures use stop groups; recipe ranking uses selection groups. "
          "The partitions overlap and have informed earlier studies. Final epochs, temperatures and reported outputs are frozen before validation.", width=145, size=11, color=MUTED)
    notes.append(f"The preliminary screen compares single fitted candidates with the cached 15-member v4 fold-0 reference, whose "
                 f"mean accuracy is {reference['mean_accuracy'] * 100:.4f}%. This value comes from internal_baseline_metrics.json. "
                 "The final primary is a multi-seed ensemble selected from single-architecture three-seed ensembles and their "
                 f"predefined pair. {guards} F1 here is the mean of the three all-33-label macro-F1 scores. Quality guards are "
                 "reported even if no pure model passed them and the best pure challenger had to be retained as a negative experiment. "
                 "The secondary never replaces the primary result. Exported screening and confirmation tables include all candidates "
                 "and mixtures, including rejected results. All internal references use historical internal fits, never incumbent "
                 "final weights trained on selection scenarios.")

    p5 = pilots[pilots.horizon_s.eq(5)]
    positive = {role: int((p5[p5.model.eq(role)].gain > 0).sum()) for role in roles[1:]}
    title = f"+5 s: primary gains for {positive['primary_transformer']} of ten pilots"
    fig = page(title, "Pilot point estimates and paired scenario uncertainty; the model is still evaluated on known trained policies")
    table_rows = []
    for pilot in sorted(p5.pilot_id.unique()):
        base = p5[p5.pilot_id.eq(pilot) & p5.model.eq("primary_transformer")].iloc[0]
        row = [pilot.replace("pilot_", ""), f"{int(base['count']):,}", f"{base.baseline_top1 * 100:.2f}%",
               f"{base.candidate_top1 * 100:.2f}%", f"{base.gain * 100:+.2f}"]
        if "secondary_blend" in roles:
            mixed = p5[p5.pilot_id.eq(pilot) & p5.model.eq("secondary_blend")].iloc[0]
            row += [f"{mixed.candidate_top1 * 100:.2f}%", f"{mixed.gain * 100:+.2f}"]
        table_rows.append(row)
    headers = ["Pilot", "Origins", "v4 top-1", "Primary", "Gain pp"]
    widths = [0.10, 0.15, 0.25, 0.25, 0.25]
    if "secondary_blend" in roles:
        headers += ["Secondary", "Gain pp"]
        widths = [0.08, 0.13, 0.17, 0.17, 0.14, 0.17, 0.14]
    table(fig, [0.05, 0.38, 0.90, 0.38], headers, table_rows, widths, 11)
    ax = fig.add_axes([0.10, 0.15, 0.85, 0.18], facecolor=BG)
    for j, role in enumerate(roles[1:]):
        part = p5[p5.model.eq(role)].sort_values("pilot_id")
        x = np.arange(len(part)) + (j - (len(roles) - 2) / 2) * 0.22
        color = TEAL if j == 0 else BLUE
        ax.vlines(x, part.gain_lower * 100, part.gain_upper * 100, color=color, linewidth=1.5)
        ax.plot(x, part.gain * 100, "o", color=color, label=LABELS[role], markersize=4)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xticks(np.arange(10), [pilot.replace("pilot_", "") for pilot in sorted(p5.pilot_id.unique())])
    ax.set_ylabel("Gain pp", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9, labelcolor=FG, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False)
    words(fig, 0.05, 0.09, "Intervals are pointwise and conditional, without correction for checking ten pilots and three horizons. "
          "All 30 comparisons per model, including regressions, are in pilot_comparison.csv.", width=150, size=10, color=MUTED)
    notes.append("Per-pilot differences must not be read as independent experiments: the same scenario groups are shared across pilots. Intervals resample complete shared scenario groups. Pilot prevalence and encounter difficulty differ, so aggregate improvement need not hold for every pilot. The complete CSV contains all horizons, counts and intervals for primary and optional secondary, without a post-validation choice between them.")

    fig = page("Per-skill F1: all 33 catalogue labels remain in scope", "+5 seconds; support counts correlated eligible origins, not independent flights")
    skill_tables = {role: skills[skills.model.eq(role) & skills.pilot_id.eq("population") & skills.horizon_s.eq(5)].set_index("skill") for role in roles}
    rows = []
    for skill in evidence["classes"]:
        old = skill_tables["baseline_v4"].loc[skill]
        row = [skill, f"{int(old.support):,}"]
        row.extend(f"{skill_tables[role].loc[skill].f1:.3f}" if old.support else "—" for role in roles)
        rows.append(row)
    headers = ["Skill", "n", "v4 F1", "Primary"] + (["Secondary"] if "secondary_blend" in roles else [])
    widths = [0.48, 0.12, 0.12, 0.14, 0.14] if len(roles) == 3 else [0.52, 0.14, 0.16, 0.18]
    table(fig, [0.05, 0.22, 0.435, 0.54], headers, rows[:17], widths, 9)
    table(fig, [0.515, 0.22, 0.435, 0.54], headers, rows[17:], widths, 9)
    words(fig, 0.05, 0.17, "Dash = no true examples at this horizon. Catalogue macro-F1 still averages all 33 labels; unsupported labels "
          "are not removed from the primary metric. Small-support changes need cautious interpretation.", width=145, size=11)
    words(fig, 0.05, 0.095, "Full precision, recall, F1 and predicted counts for every pilot × horizon × skill, plus flight/scenario/pilot support, "
          "are exported. These labels measure selected catalogue behavior, not pilot competence.", width=150, size=10, color=MUTED)
    notes.append("The displayed skill metrics are population-level +5-second F1. Full CSVs retain every pilot, horizon and class. Rare-skill fluctuations may be large even when only a handful of examples exist, so scenario/flight coverage accompanies raw origin counts. F1 combines precision and recall; its catalogue macro-average does not weight common labels more heavily and retains absent labels with zero F1 in the unchanged metric.")

    fig = page("Training cost, learning curves and reproducibility", "All completed fits and rejected candidates are retained; reported curves are internal stop curves, not final-test curves")
    fresh = inventory[~inventory.reused_baseline]
    costs = fresh.groupby("phase").agg(runs=("candidate", "size"), seconds=("elapsed_seconds", "sum"), epochs=("epochs_completed", "sum"))
    cost_rows = [[phase, str(int(row.runs)), f"{row.seconds / 60:.2f} min", str(int(row.epochs))] for phase, row in costs.iterrows()]
    table(fig, [0.05, 0.57, 0.43, 0.18], ["Phase", "Runs", "Logged time", "Epochs"], cost_rows, [0.29, 0.16, 0.31, 0.24], 12)
    words(fig, 0.05, 0.51, f"{evidence['new_completed_runs']} new completed fits; {evidence['reused_internal_runs']} cached internal "
          f"reference fits. Summed logged time: {evidence['new_training_seconds'] / 60:.2f} minutes. Loading/setup are excluded; "
          "shared-GPU execution, when used, adds contention. This is not elapsed wall time or isolated throughput.", width=61, size=12)
    budgets = [f"v4 incumbent: {protocol['baseline_members']} models / {protocol['baseline_parameters'] / 1e6:.2f}M parameters."]
    budgets += [f"{LABELS[role]}: {selection[role]['active_members']} models / {selection[role]['active_parameters'] / 1e6:.2f}M parameters." for role in roles[1:]]
    words(fig, 0.05, 0.32, "\n".join(budgets), width=65, size=12, color=TEAL)
    representative = selection["primary_transformer"]["components"][0]
    ax = fig.add_axes([0.57, 0.40, 0.38, 0.34], facecolor=BG)
    for fold, color in ((0, TEAL), (1, BLUE)):
        part = histories[histories.candidate.eq(representative) & histories.phase.eq("internal")
                         & histories.seed.eq(42) & histories.fold.eq(fold) & ~histories.reused_baseline]
        if len(part):
            ax.plot(part.epoch, part.stop_mean_accuracy * 100, color=color, label=f"Partition {fold}, seed 42")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Stop mean top-1 (%)", fontsize=10)
    ax.set_title(short_name(representative), fontsize=12, color=FG)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=9, facecolor=BG, edgecolor=MUTED, labelcolor=FG)
    words(fig, 0.57, 0.32, "Shown: seed 42 separately in each partition, not a seed average. Final refits use fixed epochs and "
          "temperatures, without outer-validation callbacks. Training objectives, stop NLL and all seeds are exported.", width=60, size=11, color=MUTED)
    verified = read_json(data / "verification.json")
    words(fig, 0.05, 0.13, f"Verified {verified['preserved_artifacts_checked']} preserved historical files, exact frozen source/protocol bytes, "
          "unchanged validation origins/masks and saved recipe probabilities. Existing PDFs remain separate. No reserved test evaluation.", width=145, size=11)
    notes.append("Timers describe the recorded fit loop and stop-side operations, excluding loading and setup. Concurrent use of one GPU can affect those durations. Final refits have training objectives only because their epoch budgets were frozen internally. Stop accuracy, training cross entropy and stop NLL are different measurements, especially with label smoothing. Source snapshots, exact protocol bytes and complete evidence support independent review and a portable PDF rebuild without model checkpoints or raw observations.")

    pdf_path = dest / "skill_forecasting_v5_transformers.pdf"
    with PdfPages(pdf_path, metadata={"Title": "Skill forecasting v5: Transformer challenge", "Author": "Trajectory Classification",
                                     "CreationDate": None, "ModDate": None}) as pdf:
        for index, fig in enumerate(figures, 1):
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            # Matplotlib retains undrawn tick labels outside the axes limits;
            # validate page text, titles and table cells without those artifacts.
            page_text = list(fig.texts)
            for ax in fig.axes:
                page_text.extend((ax.title, ax.xaxis.label, ax.yaxis.label))
                for cells in ax.tables:
                    page_text.extend(cell.get_text() for cell in cells.get_celld().values())
            for text in page_text:
                if text.get_visible() and text.get_text().strip():
                    bounds = text.get_window_extent(renderer)
                    if bounds.x0 < -2 or bounds.y0 < -2 or bounds.x1 > fig.bbox.x1 + 2 or bounds.y1 > fig.bbox.y1 + 2:
                        raise ValueError(f"Text outside page {index}: {text.get_text()!r}")
            pdf.savefig(fig)
            if preview:
                Path(preview).mkdir(parents=True, exist_ok=True)
                fig.savefig(Path(preview) / f"slide_{index:02}.png", dpi=110)
            plt.close(fig)
    (dest / "speaker_notes.md").write_text("# Transformer experiment notes\n\n" + "\n\n".join(
        f"## Slide {index}\n\n{note}" for index, note in enumerate(notes, 1)) + "\n", encoding="utf-8", newline="\n")
    write_readme(dest, evidence, population)
    write_json(dest / "checksums.json", {path.relative_to(dest).as_posix(): sha(path)
                                        for path in sorted(dest.rglob("*")) if path.is_file() and path != dest / "checksums.json"})
    return {"pdf": str(pdf_path), "pages": len(figures), "primary_transformer_reported": True,
            "secondary_blend_reported": "secondary_blend" in roles}


def write_readme(dest, evidence, population):
    selection, roles = evidence["selection"], evidence["roles"]
    reference = screen_reference(dest / "data")
    guards = guard_summary(read_json(dest / "data/protocol.json"))
    lines = ["# Skill forecasting v5: Transformer challenge", "",
             "[Open the six-page PDF](skill_forecasting_v5_transformers.pdf) · [Speaker notes](speaker_notes.md)", "",
             ("The primary question is whether an all-Transformer challenger can match the retained **15-model v4 MLP/TCN ensemble**. "
             "The primary result is reported even if worse. Any secondary incumbent blend is a separate model family and does not establish a Transformer-only replacement."), ""]
    for role in roles[1:]:
        lines += [comparison_statement(LABELS[role], evidence["evaluation"]["recipes"][role]), ""]
    lines += [("These are the **best tested recipes**, not a demonstrated optimum. The same 33 labels, ten trained pilots, forecast origins, "
              "target masks and +1/+2/+5-second horizons are retained. No equivalence margin was registered: a confidence interval crossing zero does not prove equal performance."), "",
              "| Horizon | Model | Top-1 | Top-3 | Macro-F1 / 33 | NLL | Brier | ECE |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for horizon in HORIZONS:
        for role in roles:
            row = population[population.model.eq(role) & population.horizon_s.eq(horizon)].iloc[0]
            lines.append(f"| +{horizon} s | {LABELS[role]} | {row.top1_accuracy * 100:.2f}% | {row.top3_accuracy * 100:.2f}% | "
                         f"{row.macro_f1_catalogue:.4f} | {row.nll:.4f} | {row.brier:.4f} | {row.ece_10_bins * 100:.2f} pp |")
    for role in roles[1:]:
        lines += ["", f"{LABELS[role]}: {evidence['quality_notes'][role]}", "", recipe_text(selection, role)]
    lines += ["", (f"The **preliminary screen compares individual candidate models** with the cached **15-member v4 fold-0 "
              f"reference**, which has mean accuracy **{reference['mean_accuracy'] * 100:.4f}%**, read from "
              "[internal_baseline_metrics.json](data/internal_baseline_metrics.json). The final primary is a **multi-seed ensemble**: "
              "three seeds per shortlisted architecture, confirmed on both internal partitions."), "",
              guards + " F1/33 means the mean of the three all-33-label macro-F1 scores. The primary is retained as a labelled "
              "diagnostic challenger if no pure recipe passes all guards; a secondary blend must pass the guards and the positive-gain requirement.", ""]
    lines += ["", ("Transformer-family models retain standard feed-forward blocks and current-state readouts. The explicitly labelled "
              "architectural hybrid adds a parallel full-history MLP. Attention connects only observations supplied at or before the forecast origin; "
              "fixed time/gap coordinates describe the existing sampling layout."), "",
              (f"The study completed **{evidence['new_completed_runs']} new fits**, reusing {evidence['reused_internal_runs']} internal baseline fits. "
              f"Summed recorded fit time was **{evidence['new_training_seconds'] / 60:.2f} minutes**, excluding loading/setup. "
              "Any concurrent GPU work introduces contention; this is not end-to-end elapsed time or an isolated throughput benchmark."), "",
              ("Reported outputs were frozen before one outer-validation invocation. All selection used grouped partitions inside original training. "
              "Those partitions overlap and have guided earlier studies; outer validation has also been inspected before. These are development results "
              "for new scenarios involving known pilots, not an untouched final test or unseen-pilot evaluation. The reserved test remains untouched."), "",
              "## Complete evidence", "",
              "- [All population and pilot metrics](data/metrics.csv), [pilot comparisons and intervals](data/pilot_comparison.csv), [probability reliability](data/reliability.csv).",
              "- [All pilot/skill precision, recall and F1](data/pilot_skill_metrics.csv), [skill support and coverage](data/skill_support.csv).",
              "- [Screen](data/transformer_screen.csv), [confirmation](data/transformer_confirmation.csv), [frozen recipes](data/frozen_selection.json).",
              "- [Run inventory](data/run_inventory.csv), [training histories](data/training_histories.csv), [temporal comparisons](data/temporal_common_origin_metrics.csv).",
              "- [Joint mean intervals](data/mean_horizon_intervals.json), [per-horizon intervals](data/paired_scenario_bootstrap.csv), [alternative accuracy weightings](data/accuracy_weightings.csv).",
              "- [Verification](data/verification.json), [source provenance](data/evidence.json), [bundle checksums](checksums.json).", "",
              "## Exact provenance and reproduction", "",
              ("[frozen_protocol.json](data/frozen_protocol.json) contains the byte-exact registered protocol. Its hash matches the frozen selection. "
              "[protocol.json](data/protocol.json) only normalizes data/baseline paths for readability and has a different hash. "
              "[provenance.json](data/provenance.json) records both hashes. Keep the frozen protocol and preservation manifest without line-ending conversion."), "",
              ("The [preservation manifest](data/preserved_artifacts.json) records all earlier artifacts. The deterministic "
              "[source archive](source_snapshot.zip) contains exact registered code/configuration/CLI bytes, without datasets or checkpoints. "
              "Extract it into an isolated checkout when reproducing the frozen source. Earlier PDFs and results remain unchanged."), "",
              "Rebuild the portable PDF without raw observations or model checkpoints:", "", "```powershell",
              ".venv\\Scripts\\python.exe scripts/report_transformer_experiments.py", "```", "",
              "Collect a completed local experiment and rebuild:", "", "```powershell",
              ".venv\\Scripts\\python.exe scripts/report_transformer_experiments.py --collect", "```", "",
              ("A new training reproduction requires preserved prepared data and baseline artifacts, and a fresh output directory. "
              "Use `scripts/transformer_skill_forecaster.py` commands `initialize`, `screen`, `confirm`, `refit`, and `evaluate` with that output. "
              "Do not overwrite this experiment or select another recipe using its validation scores. Hardware/software differences may alter numerical results."), ""]
    (dest / "README.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--experiment", type=Path, default=ROOT / "artifacts/skill_forecasting/improvement_v5_transformers")
    parser.add_argument("--output", type=Path, default=DEST)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    if args.collect:
        collect(args.experiment, args.output)
    print(json.dumps(render(args.output, args.preview_dir), indent=2))


if __name__ == "__main__":
    main()
