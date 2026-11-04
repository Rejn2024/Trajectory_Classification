"""Small, portable revision evidence assembled only from committed reports.

``collect(root, dest)`` writes a new data directory; ``load(dest)`` returns
``(evidence, tables)`` with tables keyed by CSV stem. No model or raw-data access.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

OLD = Path("docs/presentations/skills_catalogue/data")
V4 = Path("docs/experiments/skill_forecasting_v4/data")
V5 = Path("docs/experiments/skill_forecasting_v5_transformers/data")
MODELS = ("original", "v3", "v4", "v5")
HORIZONS = (1, 2, 5)
COUNTS = (30802, 30466, 29458)
METRICS = ("top1_accuracy", "top3_accuracy", "macro_f1_catalogue", "nll")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8", newline="\n")


def _same_metrics(left, right, left_model, right_model):
    columns = ["pilot_id", "horizon_s", "count", *METRICS]
    left = left[left.model.eq(left_model)][columns].sort_values(columns[:2]).reset_index(drop=True)
    right = right[right.model.eq(right_model)][columns].sort_values(columns[:2]).reset_index(drop=True)
    if not left[columns[:3]].equals(right[columns[:3]]):
        raise ValueError("Shared model populations or horizon counts differ across reports")
    if not np.allclose(left[list(METRICS)], right[list(METRICS)], atol=2e-6, rtol=0):
        raise ValueError("Shared model metrics differ across reports")


def _validate(metrics, skills, classes):
    if len(classes) != 33 or len(set(classes)) != 33:
        raise ValueError("The fixed catalogue must contain 33 unique labels")
    if metrics.duplicated(["model", "pilot_id", "horizon_s"]).any():
        raise ValueError("Duplicate model/pilot/horizon score")
    population = metrics[metrics.pilot_id.eq("population")]
    pilots = set(metrics.pilot_id) - {"population"}
    if len(pilots) != 10 or set(metrics.model) != set(MODELS):
        raise ValueError("Expected exactly four main models and ten pilots")
    for model in MODELS:
        rows = population[population.model.eq(model)].sort_values("horizon_s")
        if tuple(rows.horizon_s) != HORIZONS or tuple(rows["count"]) != COUNTS:
            raise ValueError("The fixed horizon population changed")
        group = metrics[metrics.model.eq(model)]
        if len(group) != 33 or set(group.pilot_id) != pilots | {"population"}:
            raise ValueError("Each model must retain population plus all ten pilots at every horizon")
        for horizon, count in zip(HORIZONS, COUNTS, strict=True):
            subset = skills[skills.model.eq(model) & skills.horizon_s.eq(horizon)]
            if len(subset) != 33 or set(subset.skill) != set(classes) or subset.support.sum() != count:
                raise ValueError("Skill labels or support do not match the unchanged task")


def collect(root, dest):
    """Collect compact teaching tables from immutable published CSV/JSON evidence."""
    root, dest = Path(root).resolve(), Path(dest).resolve()
    if dest in [(root / path).resolve() for path in (OLD, V4, V5)]:
        raise ValueError("The archived evidence directories must remain unchanged")
    sources = {}

    def read(relative):
        path = root / relative
        sources[path.relative_to(root).as_posix()] = sha(path)
        if path.suffix == ".csv":
            return pd.read_csv(path)
        return json.loads(path.read_text(encoding="utf-8"))

    old = read(OLD / "evidence.json")
    architecture = read(OLD / "architecture_review.json")
    old_summary = read(OLD / "summary_metrics.csv")
    if set(old_summary.cohort) != {"all", "same", "changed", "unknown_current"}:
        raise ValueError("Unexpected historical cohort schema")
    old_summary = old_summary[old_summary.cohort.eq("all")].copy()
    old_summary["horizon_s"] = old_summary.horizon_s.astype(int)
    old_probability = read(OLD / "probability_metrics.csv")
    current = read(V4 / "metrics.csv")
    newest = read(V5 / "metrics.csv")
    _same_metrics(old_summary, current, "ensemble", "baseline_v3")
    _same_metrics(current, newest, "candidate_v4", "baseline_v4")
    original = old_summary[old_summary.model.eq("transformer")].drop(columns="cohort")
    probability = old_probability[old_probability.model.eq("transformer")].copy()
    probability["pilot_id"] = "population"
    extra = ["brier", "ece_10_bins", "mean_confidence"]
    original = original.merge(probability[["model", "pilot_id", "horizon_s", *extra]],
                              on=["model", "pilot_id", "horizon_s"], how="left", validate="one_to_one")
    metrics = pd.concat([
        original.assign(model="original"),
        current[current.model.eq("baseline_v3")].assign(model="v3"),
        current[current.model.eq("candidate_v4")].assign(model="v4"),
        newest[newest.model.eq("primary_transformer")].assign(model="v5"),
    ], ignore_index=True)
    metrics = metrics.sort_values(["model", "pilot_id", "horizon_s"])

    skills = []
    for location, filename, mapping in (
        (OLD, "class_metrics.csv", {"transformer": "original"}),
        (V4, "pilot_skill_metrics.csv", {"baseline_v3": "v3", "candidate_v4": "v4"}),
        (V5, "pilot_skill_metrics.csv", {"primary_transformer": "v5"}),
    ):
        frame = read(location / filename)
        frame = frame[frame.pilot_id.eq("population") & frame.model.isin(mapping)].copy()
        frame["model"] = frame.model.map(mapping)
        frame["horizon_s"] = frame.horizon_s.astype(int)
        skills.append(frame)
    skills = pd.concat(skills, ignore_index=True).sort_values(["model", "horizon_s", "skill"])
    e4, e5 = read(V4 / "evidence.json"), read(V5 / "evidence.json")
    if old["classes"] != e4["classes"] or old["classes"] != e5["classes"]:
        raise ValueError("The archived class orders differ")
    _validate(metrics, skills, old["classes"])
    if old["counts"]["test"]["samples"] != 0 or any(
        entry.get("test_evaluated", False) for entry in (old, read(V4 / "evaluation.json"), read(V5 / "evaluation.json"))
    ):
        raise ValueError("Expected unevaluated reserved test")

    pilot = read(V5 / "pilot_comparison.csv")
    pilot = pilot[pilot.model.eq("primary_transformer")].assign(model="v5", baseline_model="v4")
    if len(pilot) != 30:
        raise ValueError("Expected all ten pilots at all three horizons")
    curves, curve_selection = [], {}
    original_curves = read(OLD / "forecast_learning_curves.csv")
    old_inventory = read(OLD / "training_run_inventory.csv")
    inventories = {"v4": read(V4 / "run_inventory.csv"), "v5": read(V5 / "run_inventory.csv")}
    for model, location, candidate in (
        ("v3", OLD, "history20_mlp_accuracy"),
        ("v4", V4, "history60_tcn"),
        ("v5", V5, "history60_patch_transformer"),
    ):
        if model == "v3":
            frame = original_curves.rename(columns={"validation_nll": "stop_nll", "variant": "candidate"})
            inventory = old_inventory[old_inventory.version.eq("improvement_v3")
                                      & old_inventory.phase.eq("inner_0") & old_inventory.variant.eq(candidate)
                                      & old_inventory.seed.eq(42)].iloc[0]
            best_epoch = int(inventory.selected_epoch)
        else:
            frame = read(location / "training_histories.csv")
            inventory = inventories[model]
            inventory = inventory[inventory.phase.eq("internal") & inventory.fold.eq(0)
                                  & inventory.seed.eq(42) & inventory.candidate.eq(candidate)].iloc[0]
            best_epoch = int(inventory.best_epoch)
        frame = frame[frame.phase.eq("internal") & frame.fold.eq(0) & frame.seed.eq(42)
                      & frame.candidate.eq(candidate)].copy()
        frame = frame.assign(model=model).rename(columns={"candidate": "variant"})
        columns = ["model", "variant", "fold", "seed", "epoch", "training_objective", "stop_nll", "stop_mean_accuracy"]
        frame = frame[columns].sort_values("epoch")
        if frame.empty or frame[["training_objective", "stop_nll", "stop_mean_accuracy"]].isna().any().any():
            raise ValueError("Representative internal learning history is incomplete")
        curves.append(frame)
        curve_selection[model] = {"variant": candidate, "fold": 0, "seed": 42, "selected_epoch": best_epoch,
                                  "epochs_completed": int(frame.epoch.max()), "scope": "Internal stop groups"}
    curves = pd.concat(curves, ignore_index=True)

    temporal = read(V5 / "temporal_common_origin_metrics.csv")
    temporal = temporal[temporal.model.isin(["baseline_v4", "primary_transformer"])].copy()
    temporal["model"] = temporal.model.map({"baseline_v4": "v4", "primary_transformer": "v5"})
    if len(temporal) != 18 or not temporal.groupby(["model", "horizon_s"])["count"].sum().eq(COUNTS[-1]).all():
        raise ValueError("Temporal tables must share the same eligible origins across horizons")
    phase_context = read(OLD / "temporal_elapsed_phases.csv")
    phase_context = phase_context[phase_context.model.eq("ensemble") & phase_context.horizon_s.eq(5)]
    temporal = temporal.merge(phase_context[["origin_time_bin", "changed_endpoint_fraction", "dominant_skill", "dominant_skill_fraction"]],
                               left_on="elapsed_phase", right_on="origin_time_bin", validate="many_to_one").drop(columns="origin_time_bin")

    costs = []
    subset = old_inventory[old_inventory.version.eq("improvement_v3")].copy()
    subset["phase"] = np.where(subset.phase.eq("refit"), "refit", "internal")
    for model, frame, expected in (("v3", subset, 23), ("v4", inventories["v4"], 37), ("v5", inventories["v5"], 29)):
        if model != "v3":
            flag = "reused_v3" if model == "v4" else "reused_baseline"
            frame = frame[~frame[flag] & frame.status.eq("complete")]
        if len(frame) != expected:
            raise ValueError("The completed new-fit inventory changed")
        for phase, group in frame.groupby("phase"):
            costs.append({"model": model, "phase": phase, "runs": len(group),
                          "seconds": float(group.elapsed_seconds.sum()), "epochs": int(group.epochs_completed.sum())})
    costs = pd.DataFrame(costs)

    comparisons = []
    for model, baseline, location in (("v4", "v3", V4), ("v5", "v4", V5)):
        frame = read(location / "paired_scenario_bootstrap.csv")
        if model == "v5":
            frame = frame[frame.recipe.eq("primary_transformer")].drop(columns="recipe")
            interval = read(location / "mean_horizon_intervals.json")["primary_transformer"]
        else:
            interval = read(location / "mean_horizon_interval.json")
        comparisons.extend(frame.assign(model=model, baseline_model=baseline, scope="horizon").to_dict("records"))
        comparisons.append({"model": model, "baseline_model": baseline, "scope": "mean", "horizon_s": 0,
                            "accuracy_gain": interval["mean_accuracy_gain"], "lower_95": interval["lower_95"],
                            "upper_95": interval["upper_95"], "scenario_groups": interval["scenario_groups"],
                            "bootstrap_draws": interval["draws"]})
    comparisons = pd.DataFrame(comparisons)
    population = metrics[metrics.pilot_id.eq("population")]
    for row in comparisons.itertuples():
        values = population if row.scope == "mean" else population[population.horizon_s.eq(row.horizon_s)]
        difference = values[values.model.eq(row.model)].top1_accuracy.mean() - values[values.model.eq(row.baseline_model)].top1_accuracy.mean()
        if not np.isclose(difference, row.accuracy_gain, atol=1e-12, rtol=0):
            raise ValueError("Saved paired comparison no longer matches headline accuracy")

    s4, s5 = read(V4 / "frozen_selection.json"), read(V5 / "frozen_selection.json")
    p4, p5 = read(V4 / "protocol.json"), read(V5 / "protocol.json")
    original_run = old["original_runs"]["transformer"]
    models = {
        "original": {"label": "Original Transformer", "members": 1, "parameters": original_run["parameters"],
                     "features": original_run["model"]["input_size"], "history_seconds": [5],
                     "history_tokens": [original_run["model"]["history_steps"]], "settings": original_run["model"]},
        "v3": {"label": "12 temporal MLPs", "members": sum(len(c["seeds"]) for c in architecture["components"]),
               "parameters": architecture["total_parameters"], "features": architecture["components"][0]["features_per_timestamp"],
               "history_seconds": sorted({c["history_seconds"] for c in architecture["components"]}),
               "history_tokens": sorted({c["history_tokens"] for c in architecture["components"]}),
               "components": architecture["components"], "ensemble": "Equal mean of twelve calibrated probability arrays"},
        "v4": {"label": "12 MLPs + 3 TCNs", "members": s4["active_members"], "parameters": s4["active_parameters"],
               "features": 91, "history_seconds": [20, 60], "history_tokens": [26, 66],
               "components": s4["components"], "tcn_settings": p4["candidates"]["history60_tcn"],
               "ensemble": "50% v3 probabilities + 50% equal mean of three TCNs", "new_weight": s4["new_weight"]},
        "v5": {"label": "6 Transformers", "members": s5["primary_transformer"]["active_members"],
               "parameters": s5["primary_transformer"]["active_parameters"], "features": 91,
               "history_seconds": [60], "history_tokens": [66], "components": s5["components"],
               "settings": {c["candidate"]: p5["candidates"][c["candidate"]] for c in s5["components"]},
               "ensemble": "Equal mean of three patch and three temporal Transformers"},
    }
    for model, row in models.items():
        selected = population[population.model.eq(model)]
        row["mean_metrics"] = {metric: float(selected[metric].mean()) for metric in METRICS}
        row["history_text"] = "/".join(str(value) for value in row["history_seconds"]) + " s"
        row["tokens_text"] = "/".join(str(value) for value in row["history_tokens"])
    temporal_contract = read(OLD / "temporal_evidence.json")
    dynamics = read(OLD / "temporal_pilot_dynamics.csv")
    dynamics = dynamics[dynamics.pilot_id.eq("population")].iloc[0]
    secondary = e5["evaluation"]["recipes"]["secondary_blend"]
    horizon_comparisons = comparisons[comparisons.scope.eq("horizon")].copy()
    comparisons = comparisons[comparisons.scope.eq("mean")].copy()
    comparisons["comparison"] = comparisons.model + "_vs_" + comparisons.baseline_model
    comparisons["label"] = comparisons.model + " versus " + comparisons.baseline_model
    comparisons["gain"] = comparisons.accuracy_gain
    evidence = {
        "schema_version": 1, "revision_date": "2026-11-04", "model_order": list(MODELS), "models": models,
        "classes": old["classes"], "horizons_s": list(HORIZONS), "counts": old["counts"],
        "task": {"observer": "B01", "target": "A01", "known_pilots": 10,
                 "classes": len(old["classes"]), "pilots": 10,
                 "scenario_groups": sum(v["scenario_groups"] for v in old["counts"].values()),
                 "episodes": sum(v["episodes"] for v in old["counts"].values()),
                 "train_groups": old["counts"]["train"]["scenario_groups"],
                 "validation_groups": old["counts"]["validation"]["scenario_groups"],
                 "test_groups": old["counts"]["test"]["scenario_groups"],
                 "validation_origins": old["counts"]["validation"]["samples"],
                 "valid_targets": old["counts"]["validation"]["valid_by_horizon"],
                 "meaning": "Exact future catalogue selection; not pilot competence"},
        "selection": {"fit_groups": 147, "stop_groups": 21, "selection_groups": 42,
                      "training_seeds": p5["training_seeds"], "partition_seeds": p5["split_seeds"]},
        "learning_curves": curve_selection,
        "learning_contract": "Representative population-wide internal fold-0 seed-42 curves; training uses smoothing/dropout, stop NLL is uncalibrated; no per-pilot forecast epoch curves were logged.",
        "cost_contract": "New fits per study only. v3 excludes initial_v1 and v2; v4/v5 exclude reused fits. Summed recorded timers exclude setup and include shared-GPU contention; not wall time or isolated throughput.",
        "prior_cost_context": {"initial_and_v2_fits": int((~old_inventory.version.eq("improvement_v3")).sum()),
                               "initial_and_v2_seconds": float(old_inventory.loc[~old_inventory.version.eq("improvement_v3"), "elapsed_seconds"].sum()),
                               "old_all_stages_fits": len(old_inventory), "old_all_stages_seconds": float(old_inventory.elapsed_seconds.sum())},
        "temporal_contract": temporal_contract["phase_contract"], "temporal_limits": temporal_contract["limits"],
        "persistence": {"fraction_adjacent_equal": float(dynamics.one_second_repeat_fraction),
                        "median_completed_run_s": float(dynamics.completed_run_median_s),
                        "p90_completed_run_s": float(dynamics.completed_run_p90_s),
                        "censored_runs": int(dynamics.right_censored_runs), "completed_runs": int(dynamics.completed_runs)},
        "secondary": {"mean_accuracy": secondary["metrics"]["mean_accuracy"], "gain": secondary["mean_accuracy_gain"],
                      "lower_95": secondary["mean_accuracy_interval"]["lower_95"],
                      "upper_95": secondary["mean_accuracy_interval"]["upper_95"],
                      "members": s5["secondary_blend"]["active_members"],
                      "parameters": s5["secondary_blend"]["active_parameters"]},
        "secondary_summary": {"selection": s5["secondary_blend"], "evaluation": e5["evaluation"]["recipes"]["secondary_blend"]},
        "comparison_contract": "Previously inspected development validation; shared whole-scenario intervals conditional on frozen models, not selection-adjusted. Crossing zero does not establish equivalence.",
        "collection_contract": "Committed portable CSV/JSON evidence only; no raw observations, inference, fitting, calibration or reserved-test access.",
        "source_sha256": sources, "test_evaluated": False,
    }
    tables = {"metrics": metrics, "skills": skills, "pilot_intervals": pilot, "learning_curves": curves,
              "temporal": temporal, "costs": costs, "comparisons": comparisons, "horizon_comparisons": horizon_comparisons}
    dest.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(dest / f"{name}.csv", index=False, lineterminator="\n")
    write_json(dest / "evidence.json", evidence)
    write_json(dest / "checksums.json", {p.name: sha(p) for p in sorted(dest.iterdir()) if p.is_file() and p.name != "checksums.json"})
    return load(dest)


def load(dest):
    """Verify and load the small evidence bundle without its historical sources."""
    dest = Path(dest)
    checksums = json.loads((dest / "checksums.json").read_text(encoding="utf-8"))
    for name, expected in checksums.items():
        if sha(dest / name) != expected:
            raise ValueError(f"Revision evidence checksum changed: {name}")
    evidence = json.loads((dest / "evidence.json").read_text(encoding="utf-8"))
    tables = {path.stem: pd.read_csv(path) for path in sorted(dest.glob("*.csv"))}
    _validate(tables["metrics"], tables["skills"], evidence["classes"])
    return evidence, tables
