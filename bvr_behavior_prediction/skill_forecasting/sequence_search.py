"""Bounded sequence-model search against the preserved v3 ensemble.

Every selection operation uses original-training scenario groups. The outer
validation command is separate, refuses a second invocation, and only evaluates
the already frozen recipe. Baseline seed-42 internal runs are reused; final v3
checkpoints must never be used to predict internal selection groups.
"""

import gc
import json
import platform
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bvr_behavior_prediction.forecasting.data import read_json, sha256, write_json

from .evaluation import save_evaluation
from .improved_models import ImprovedForecaster
from .improvement import (
    DevelopmentData,
    check_protocol,
    fit_run,
    paired_group_intervals,
    predict,
    quick_metrics,
)
from .improvement import (
    initialize as initialize_base,
)


def initialize(root, data_directory, experiment, design_path, baseline_directory=None):
    """Register immutable data, source and comparison provenance before any fit."""
    root, experiment = Path(root).resolve(), Path(experiment).resolve()
    design_path = Path(design_path).resolve()
    baseline_directory = (Path(baseline_directory).resolve() if baseline_directory else
                          root / "artifacts/skill_forecasting/improvement_v3")
    if experiment.exists() and any(experiment.iterdir()):
        if (experiment / "protocol.json").exists() and "baseline_hashes" in read_json(experiment / "protocol.json"):
            return check_runner(root, experiment)
        raise ValueError("Initialization requires an empty experiment; preserve incomplete outputs")
    design = read_json(design_path)
    baseline_protocol = read_json(baseline_directory / "protocol.json")
    for name in design["baseline_components"]:
        if design["candidates"][name] != baseline_protocol["candidates"][name]:
            raise ValueError(f"Retained baseline candidate definition changed: {name}")
    preserved_path = root / "artifacts/checks/sequence_search_v4/preserved_artifacts.json"
    if preserved_path.exists():
        preserved = read_json(preserved_path)
        for name, expected in preserved.items():
            if sha256(root / name) != expected:
                raise ValueError(f"Previously preserved artifact changed: {name}")
    else:
        preserved = {}
        for directory in (Path(data_directory), root / "artifacts/skill_forecasting/initial_v1",
                          root / "artifacts/skill_forecasting/improvement_v2", baseline_directory):
            preserved.update({path.relative_to(root).as_posix(): sha256(path)
                              for path in sorted(directory.rglob("*")) if path.is_file()})
    protocol = initialize_base(root, data_directory, experiment, design_path=design_path)
    for key, value in design.items():
        if key not in ("description", "candidates"):
            protocol[key] = value
    protocol["baseline_directory"] = str(baseline_directory)
    protocol["partition_hashes"] = {}
    samples = pd.read_parquet(Path(data_directory) / "samples.parquet")
    for fold in (0, 1):
        name = f"inner_split_{fold}.json"
        if read_json(experiment / name) != read_json(baseline_directory / name):
            raise ValueError("New internal partitions differ from the established baseline")
        validate_partition(read_json(experiment / name), samples)
        protocol["partition_hashes"][name] = sha256(experiment / name)
    baseline_files = [baseline_directory / name for name in (
        "protocol.json", "frozen_selection.json", "inner_split_0.json", "inner_split_1.json",
        "validation/ensemble/validation_probabilities.npy", "validation/ensemble/validation_samples.parquet")]
    for fold in (0, 1):
        for component in protocol["baseline_components"]:
            member = baseline_directory / f"inner_{fold}" / f"{component}_s42"
            baseline_files.extend(member / name for name in (
                "best.pt", "run.json", "summary.json", "selection_probabilities.npy", "selection_indices.npy"))
    protocol["baseline_hashes"] = {path.relative_to(root).as_posix(): sha256(path) for path in baseline_files}
    protocol["baseline_parameters"] = 3 * sum(
        read_json(baseline_directory / "inner_0" / f"{name}_s42/run.json")["parameters"]
        for name in protocol["baseline_components"])
    targets = np.load(Path(data_directory) / "targets.npy")
    protocol["fixed_origin_counts"] = {
        split: {"origins": int(samples.split.eq(split).sum()),
                "valid_targets": (targets[samples.split.eq(split)] >= 0).sum(0).tolist()}
        for split in ("train", "validation")}
    for source in (root / "scripts/sequence_skill_forecaster.py", design_path):
        relative = source.relative_to(root)
        destination = experiment / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        protocol["source_hashes"][relative.as_posix()] = sha256(source)
    protocol["selection"] = "Finite mixture grid; both train-only folds must improve on 12-member v3 with guards"
    protocol["confirmation"] = "Top two screen candidates; all three seeds per variant on both internal partitions"
    protocol["final_fit"] = "New components only; epochs/temperature median of six internal fits; old v3 unchanged"
    write_json(experiment / "preserved_artifacts.json", preserved)
    protocol["preserved_artifacts_sha256"] = sha256(experiment / "preserved_artifacts.json")
    write_json(experiment / "environment.json", {
        "python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
        "numpy": np.__version__, "pandas": pd.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    write_json(experiment / "protocol.json", protocol)
    return check_runner(root, experiment)


def validate_partition(partition, samples):
    """A partition is precisely the established 210 training scenario groups."""
    if (not samples.split.isin(["train", "validation"]).all()
            or set(partition) != set(samples.loc[samples.split.eq("train"), "scenario_group"])
            or set(partition).intersection(samples.loc[samples.split.eq("validation"), "scenario_group"])
            or pd.Series(partition).value_counts().to_dict() != {"fit": 147, "select": 42, "stop": 21}):
        raise ValueError("Internal partition must contain only 147/21/42 original-training groups")


def immutable_json(path, value):
    """Permit an identical resume, never replace a previously frozen decision."""
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Frozen decision changed: {path}")
        return value
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return value


def check_runner(root, experiment):
    root, experiment = Path(root), Path(experiment)
    protocol = check_protocol(root, experiment)
    if sha256(experiment / "preserved_artifacts.json") != protocol["preserved_artifacts_sha256"]:
        raise ValueError("Preserved artifact manifest changed")
    for fold in (0, 1):
        name = f"inner_split_{fold}.json"
        if sha256(experiment / name) != protocol["partition_hashes"][name]:
            raise ValueError(f"Frozen internal partition changed: {name}")
    for name, expected in protocol["baseline_hashes"].items():
        if sha256(root / name) != expected:
            raise ValueError(f"Preserved baseline changed: {name}")
    if protocol["training_seeds"] != [42, 43, 44]:
        raise ValueError("The registered comparison requires seeds 42, 43 and 44")
    if len(protocol["baseline_components"]) != 4:
        raise ValueError("The retained baseline has four variants and three seeds")
    weights = protocol.get("sequence_weights", [0.25, 0.5, 0.75, 1.0])
    if list(weights) != [0.25, 0.5, 0.75, 1.0]:
        raise ValueError("Mixture weights must match the preregistered finite grid")
    return protocol


def _probability_array(probabilities, indices):
    values, indices = np.asarray(probabilities), np.asarray(indices)
    if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
            or len(np.unique(indices)) != len(indices) or np.any(indices < 0)
            or values.shape != (len(indices), 3, 33)
            or not np.isfinite(values).all() or np.any(values < 0)
            or not np.allclose(values.sum(-1), 1, atol=1e-5)):
        raise ValueError("Invalid selection probabilities or origin indices")
    return values, indices


def average_aligned(members):
    """Average only exactly aligned probability arrays, retaining origin order."""
    reference, values = None, []
    for probabilities, indices in members:
        probabilities, indices = _probability_array(probabilities, indices)
        if reference is not None and not np.array_equal(reference, indices):
            raise ValueError("Ensemble members have different selection origins")
        reference = indices
        values.append(probabilities)
    if not values:
        raise ValueError("Cannot average an empty ensemble")
    return np.mean(values, axis=0), reference


def blend_aligned(baseline, proposed, weight):
    old, old_indices = _probability_array(*baseline)
    new, new_indices = _probability_array(*proposed)
    if not np.array_equal(old_indices, new_indices):
        raise ValueError("Baseline and proposed models have different origins")
    if not 0 <= weight <= 1:
        raise ValueError("Mixture weight must lie in [0, 1]")
    return (1 - weight) * old + weight * new, old_indices


def passes_guards(metrics, baseline, guards=None, require_gain=False):
    """Guard the current ensemble, including each unchanged forecast horizon."""
    guards = guards or {}
    return bool(
        metrics["nll"] <= baseline["nll"] + guards.get("maximum_nll_increase", 0.02)
        and metrics["mean_macro_f1"] >= baseline["mean_macro_f1"]
        - guards.get("maximum_macro_f1_decrease", 0.01)
        and np.all(np.asarray(metrics["accuracy"]) >= np.asarray(baseline["accuracy"])
                   - guards.get("maximum_horizon_accuracy_decrease", 0.003))
        and (not require_gain or metrics["mean_accuracy"] > baseline["mean_accuracy"])
    )


def _baseline_directory(root, protocol):
    return Path(root) / protocol["baseline_directory"]


def _member_path(root, experiment, protocol, name, fold, seed):
    directory = (_baseline_directory(root, protocol)
                 if name in protocol["baseline_components"] and seed == 42 else Path(experiment))
    return directory / f"inner_{fold}" / f"{name}_s{seed}"


def _read_member(root, experiment, protocol, name, fold, seed):
    path = _member_path(root, experiment, protocol, name, fold, seed)
    run, summary = read_json(path / "run.json"), read_json(path / "summary.json")
    partition = read_json(Path(experiment) / f"inner_split_{fold}.json")
    expected_groups = sorted(group for group, role in partition.items() if role == "fit")
    if (run["candidate"] != protocol["candidates"][name] or run["seed"] != seed
            or run["fold"] != fold or run["final"] or run["fit_groups"] != expected_groups
            or summary["name"] != name or summary["seed"] != seed
            or summary["fold"] != fold or summary["final"]
            or summary.get("test_evaluated", True) or summary.get("outer_validation_evaluated", True)
            or sha256(path / "best.pt") != summary["checkpoint_sha256"]):
        raise ValueError(f"Internal run provenance changed: {path}")
    source_protocol = read_json(path.parent.parent / "protocol.json")
    if (run["protocol_sha256"] != sha256(path.parent.parent / "protocol.json")
            or source_protocol["data_manifest_sha256"] != protocol["data_manifest_sha256"]):
        raise ValueError(f"Internal run does not match its protocol/data: {path}")
    checkpoint = torch.load(path / "best.pt", map_location="cpu", weights_only=False)
    if (checkpoint["candidate"] != run["candidate"] or checkpoint["seed"] != seed
            or checkpoint["epoch"] != summary["best_epoch"]
            or checkpoint["temperature"] != summary["temperature"]
            or checkpoint["scaling"]["fit_groups"] != expected_groups
            or checkpoint["data_manifest_sha256"] != protocol["data_manifest_sha256"]):
        raise ValueError(f"Internal checkpoint provenance changed: {path}")
    if path.parent.parent.resolve() == Path(experiment).resolve():
        for filename, field in (("selection_probabilities.npy", "selection_probability_sha256"),
                                ("selection_indices.npy", "selection_indices_sha256")):
            if sha256(path / filename) != summary[field]:
                raise ValueError(f"Cached internal predictions changed: {path / filename}")
    return (np.load(path / "selection_probabilities.npy"),
            np.load(path / "selection_indices.npy"))


def _internal_ensemble(root, experiment, protocol, names, fold, seeds):
    probabilities, indices = average_aligned(
        _read_member(root, experiment, protocol, name, fold, seed)
        for name in names for seed in seeds)
    samples = pd.read_parquet(Path(protocol["data_directory"]) / "samples.parquet")
    partition = read_json(Path(experiment) / f"inner_split_{fold}.json")
    expected = np.flatnonzero(samples.scenario_group.map(partition).eq("select").to_numpy())
    if (not np.array_equal(indices, expected)
            or not samples.iloc[indices].split.eq("train").all()):
        raise ValueError("Ranking must use exactly the internal selection origins, in dataset order")
    return probabilities, indices


def _score_row(name, names, weight, fold, probabilities, indices, targets, baseline,
               baseline_members, seeds, guards, require_gain):
    scores = quick_metrics(probabilities, targets[indices])
    return {"recipe": name, "components": list(names), "new_weight": weight, "fold": fold,
            "active_members": (baseline_members if weight < 1 else 0) + len(names) * len(seeds),
            "mean_accuracy": scores["mean_accuracy"], "nll": scores["nll"],
            "mean_macro_f1": scores["mean_macro_f1"],
            "gain": scores["mean_accuracy"] - baseline["mean_accuracy"],
            "eligible": passes_guards(scores, baseline, guards, require_gain),
            **{f"accuracy_{h}s": value for h, value in zip((1, 2, 5), scores["accuracy"], strict=True)}}


def screen(root, experiment):
    experiment = Path(experiment)
    protocol = check_runner(root, experiment)
    if (experiment / "frozen_selection.json").exists():
        raise ValueError("The recipe is frozen; screen cannot be rerun")
    if (experiment / "shortlist.json").exists():
        return read_json(experiment / "shortlist.json")
    targets = np.load(Path(protocol["data_directory"]) / "targets.npy")
    baseline = _internal_ensemble(root, experiment, protocol, protocol["baseline_components"], 0, [42])
    baseline_scores = quick_metrics(baseline[0], targets[baseline[1]])
    rows = []
    for name in protocol["sequence_candidates"]:
        fit_run(root, experiment, name, fold=0, seed=42)
        proposed = _internal_ensemble(root, experiment, protocol, [name], 0, [42])
        for weight in protocol.get("sequence_weights", [0.25, 0.5, 0.75, 1.0]):
            probabilities, indices = blend_aligned(baseline, proposed, weight)
            rows.append(_score_row(name, [name], weight, 0, probabilities, indices, targets,
                                   baseline_scores, 4, [42], protocol.get("guards"), False))
    frame = pd.DataFrame(rows)
    frame.to_csv(experiment / "sequence_screen.csv", index=False, lineterminator="\n")
    eligible = frame[frame.eligible].sort_values(["mean_accuracy", "recipe", "new_weight"],
                                                ascending=[False, True, True])
    names = eligible.drop_duplicates("recipe").head(2).recipe.tolist()
    return immutable_json(experiment / "shortlist.json", {
        "names": names, "selected_on": "Fold 0 original-training select groups only",
        "screening_reference_members": 4, "final_reference_members": 12,
        "baseline_metrics": baseline_scores, "protocol_sha256": sha256(experiment / "protocol.json"),
        "outer_validation_evaluated": False, "test_evaluated": False})


def confirm(root, experiment):
    experiment = Path(experiment)
    protocol = check_runner(root, experiment)
    if (experiment / "frozen_selection.json").exists():
        return read_json(experiment / "frozen_selection.json")
    shortlist = read_json(experiment / "shortlist.json")
    if shortlist["protocol_sha256"] != sha256(experiment / "protocol.json"):
        raise ValueError("Shortlist protocol changed")
    names = shortlist["names"]
    if len(names) > 2 or not set(names).issubset(protocol["sequence_candidates"]):
        raise ValueError("Invalid frozen shortlist")
    targets = np.load(Path(protocol["data_directory"]) / "targets.npy")
    seeds, rows, baseline_scores = protocol["training_seeds"], [], []
    recipes = {name: [name] for name in names}
    if len(names) == 2:
        recipes["equal_pair"] = names
    for fold in (0, 1):
        if not names:
            break
        for name in protocol["baseline_components"]:
            for seed in seeds:
                if seed != 42:
                    fit_run(root, experiment, name, fold=fold, seed=seed)
        for name in names:
            for seed in seeds:
                fit_run(root, experiment, name, fold=fold, seed=seed)
        baseline = _internal_ensemble(root, experiment, protocol, protocol["baseline_components"], fold, seeds)
        scores = quick_metrics(baseline[0], targets[baseline[1]])
        baseline_scores.append({"fold": fold, "members": 12, **scores})
        for recipe, components in recipes.items():
            proposed = _internal_ensemble(root, experiment, protocol, components, fold, seeds)
            for weight in protocol.get("sequence_weights", [0.25, 0.5, 0.75, 1.0]):
                probabilities, indices = blend_aligned(baseline, proposed, weight)
                rows.append(_score_row(recipe, components, weight, fold, probabilities, indices, targets,
                                       scores, 12, seeds, protocol.get("guards"), True))
    frame = pd.DataFrame(rows)
    frame.to_csv(experiment / "sequence_confirmation.csv", index=False, lineterminator="\n")
    immutable_json(experiment / "internal_baseline_metrics.json", baseline_scores)
    selected = None
    if len(frame):
        ranked = frame.groupby(["recipe", "new_weight"], sort=True).agg(
            mean_accuracy=("mean_accuracy", "mean"), eligible=("eligible", "all"),
            folds=("fold", "nunique"), mean_gain=("gain", "mean"), nll=("nll", "mean"),
            mean_macro_f1=("mean_macro_f1", "mean"))
        ranked = ranked[ranked.eligible & ranked.folds.eq(2)].reset_index().sort_values(
            ["mean_accuracy", "recipe", "new_weight"], ascending=[False, True, True])
        if len(ranked):
            selected = ranked.iloc[0].to_dict()
    components = []
    if selected is not None:
        for name in recipes[selected["recipe"]]:
            summaries = [read_json(experiment / f"inner_{fold}" / f"{name}_s{seed}" / "summary.json")
                         for fold in (0, 1) for seed in seeds]
            components.append({"candidate": name, "seeds": seeds,
                               "epochs": int(np.rint(np.median([r["best_epoch"] for r in summaries]))),
                               "temperature": float(np.median([r["temperature"] for r in summaries])),
                               "parameters_per_member": read_json(experiment / "inner_0"
                                                                    / f"{name}_s42/run.json")["parameters"]})
    selection = {
        "retain_baseline": selected is None, "components": components,
        "new_weight": float(selected["new_weight"]) if selected else 0.0,
        "recipe": selected["recipe"] if selected else "baseline_v3",
        "active_members": ((12 if selected["new_weight"] < 1 else 0) + 3 * len(components)) if selected else 12,
        "baseline_parameters": protocol["baseline_parameters"],
        "active_parameters": ((protocol["baseline_parameters"] if selected["new_weight"] < 1 else 0)
                              + 3 * sum(c["parameters_per_member"] for c in components))
        if selected else protocol["baseline_parameters"],
        "selection_metrics": selected,
        "practical_gain_target": protocol.get("practical_target_mean_accuracy_gain", 0.01),
        "internal_practical_target_met": bool(selected and selected["mean_gain"]
                                               >= protocol.get("practical_target_mean_accuracy_gain", 0.01)),
        "ensemble": "Fixed-weight blend of preserved v3 baseline and equal mean of new components/seeds",
        "comparison_scope": "Same targets/origins; expanded model budget when baseline and new members are blended",
        "selected_using": "Both original-training partitions; guards and positive mean accuracy gain in each",
        "epochs_temperature_rule": "Median of six internal fits per selected component; no outer callbacks",
        "protocol_sha256": sha256(experiment / "protocol.json"),
        "shortlist_sha256": sha256(experiment / "shortlist.json"),
        "outer_validation_evaluated": False, "test_evaluated": False}
    return immutable_json(experiment / "frozen_selection.json", selection)


def _selection(experiment):
    experiment = Path(experiment)
    selection = read_json(experiment / "frozen_selection.json")
    if (selection["protocol_sha256"] != sha256(experiment / "protocol.json")
            or selection["shortlist_sha256"] != sha256(experiment / "shortlist.json")):
        raise ValueError("Frozen selection provenance changed")
    return selection


def refit(root, experiment):
    check_runner(root, experiment)
    selection = _selection(experiment)
    for component in selection["components"]:
        for seed in component["seeds"]:
            fit_run(root, experiment, component["candidate"], seed=seed,
                    final_epochs=component["epochs"], final_temperature=component["temperature"])
    return selection


def evaluate(root, experiment):
    experiment = Path(experiment)
    protocol = check_runner(root, experiment)
    selection = _selection(experiment)
    if selection["retain_baseline"]:
        return {"retain_baseline": True, "outer_validation_evaluated": False, "test_evaluated": False}
    output = experiment / "validation"
    if output.exists():
        raise ValueError("Outer validation was already started; preserve it and do not tune against its scores")
    paths = []
    partition = read_json(experiment / "inner_split_0.json")
    classes = read_json(Path(protocol["data_directory"]) / "classes.json")
    for component in selection["components"]:
        for seed in component["seeds"]:
            path = experiment / "refit" / f"{component['candidate']}_s{seed}"
            summary, run = read_json(path / "summary.json"), read_json(path / "run.json")
            if (not summary["final"] or not run["final"] or summary["seed"] != seed
                    or summary["best_epoch"] != component["epochs"]
                    or summary["epochs_completed"] != component["epochs"]
                    or summary["temperature"] != component["temperature"]
                    or run["candidate"] != protocol["candidates"][component["candidate"]]
                    or run["fit_groups"] != sorted(partition)
                    or run["protocol_sha256"] != sha256(experiment / "protocol.json")
                    or sha256(path / "best.pt") != summary["checkpoint_sha256"]):
                raise ValueError(f"Refit does not match the frozen selection: {path}")
            saved = torch.load(path / "best.pt", map_location="cpu", weights_only=False)
            scaling = saved["scaling"]
            if (saved["candidate"] != run["candidate"] or saved["model_config"] != run["model"]
                    or saved["seed"] != seed or saved["epoch"] != component["epochs"]
                    or saved["temperature"] != component["temperature"]
                    or saved["classes"] != classes or saved["horizons_s"] != [1, 2, 5]
                    or saved["data_manifest_sha256"] != protocol["data_manifest_sha256"]
                    or scaling["fit_groups"] != sorted(partition)
                    or len(saved["feature_names"]) != run["model"]["input_size"]
                    or len(saved["offsets"]) != run["model"]["tokens"]
                    or saved["offsets"][-1] != 0 or np.any(np.asarray(saved["offsets"]) > 0)
                    or len(scaling["mean"]) != run["model"]["input_size"]
                    or len(scaling["std"]) != run["model"]["input_size"]
                    or not np.isfinite(scaling["mean"]).all() or not np.isfinite(scaling["std"]).all()
                    or np.any(np.asarray(scaling["std"]) <= 0)):
                raise ValueError(f"Refit checkpoint metadata does not match the frozen selection: {path}")
            del saved
            paths.append(path)
    baseline_directory = _baseline_directory(root, protocol) / "validation/ensemble"
    samples = pd.read_parquet(baseline_directory / "validation_samples.parquet")
    all_samples = pd.read_parquet(Path(protocol["data_directory"]) / "samples.parquet")
    validation_indices = np.flatnonzero(all_samples.split.eq("validation").to_numpy())
    pd.testing.assert_frame_equal(samples, all_samples.iloc[validation_indices].reset_index(drop=True))
    baseline = np.load(baseline_directory / "validation_probabilities.npy")
    targets = np.load(Path(protocol["data_directory"]) / "targets.npy")[validation_indices]
    current = np.load(Path(protocol["data_directory"]) / "current.npy")[validation_indices]
    _probability_array(baseline, validation_indices)
    output.mkdir()
    immutable_json(output / "evaluation_started.json", {
        "frozen_selection_sha256": sha256(experiment / "frozen_selection.json"),
        "policy": "One outer-validation evaluation; no tuning follows", "test_evaluated": False})
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    predictions, checkpoint_hashes = [], {}
    for path in paths:
        saved = torch.load(path / "best.pt", map_location="cuda", weights_only=False)
        data = DevelopmentData(protocol["data_directory"], saved["candidate"], [], scaling=saved["scaling"])
        indices = torch.as_tensor(validation_indices, device=data.device)
        model = ImprovedForecaster(**saved["model_config"]).to(data.device)
        model.load_state_dict(saved["model_state"])
        probabilities = predict(model, data, indices, temperature=saved["temperature"])
        member = output / path.name
        member.mkdir()
        np.save(member / "validation_probabilities.npy", probabilities)
        predictions.append((probabilities, validation_indices))
        checkpoint_hashes[str(path.relative_to(experiment))] = sha256(path / "best.pt")
        del model, data
        gc.collect()
        torch.cuda.empty_cache()
    proposed = average_aligned(predictions)
    probabilities, _ = blend_aligned((baseline, validation_indices), proposed, selection["new_weight"])
    save_evaluation(output / "ensemble", probabilities, targets, current, samples, classes, "sequence_v4")
    np.save(output / "ensemble/validation_probabilities.npy", probabilities)
    samples.to_parquet(output / "ensemble/validation_samples.parquet", index=False)
    # Standalone new members are descriptive ablations, never another selection.
    save_evaluation(output / "new_members_only", proposed[0], targets, current, samples, classes, "new_members_only")
    intervals = paired_group_intervals(probabilities, baseline, targets, samples)
    intervals.to_csv(experiment / "paired_scenario_bootstrap.csv", index=False, lineterminator="\n")
    baseline_metrics, metrics = quick_metrics(baseline, targets), quick_metrics(probabilities, targets)
    result = {
        "selection_sha256": sha256(experiment / "frozen_selection.json"), "metrics": metrics,
        "baseline_metrics": baseline_metrics, "new_members_only_metrics": quick_metrics(proposed[0], targets),
        "mean_accuracy_gain": metrics["mean_accuracy"] - baseline_metrics["mean_accuracy"],
        "practical_gain_target_met": metrics["mean_accuracy"] - baseline_metrics["mean_accuracy"]
        >= selection["practical_gain_target"],
        "paired_intervals": intervals.to_dict(orient="records"), "checkpoint_hashes": checkpoint_hashes,
        "matched_validation_origins": True, "outer_validation_evaluated": True, "test_evaluated": False,
        "validation_role": "Previously inspected development validation; not an untouched final test"}
    write_json(output / "evaluation.json", result)
    write_json(output / "output_hashes.json", {
        path.relative_to(output).as_posix(): sha256(path) for path in sorted(output.rglob("*")) if path.is_file()})
    return result
