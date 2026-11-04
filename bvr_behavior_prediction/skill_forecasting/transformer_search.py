"""Train-only Transformer challenge against the frozen fifteen-model v4 recipe.

The primary result is always a Transformer-only challenger. A secondary blend
with the incumbent is a separately named, conditional result. Both are frozen
together before a single validation invocation; validation never chooses a winner.
"""

import gc
import platform
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bvr_behavior_prediction.forecasting.data import read_json, sha256, write_json

from .data import verify_dataset
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
from .sequence_search import (
    _probability_array,
    average_aligned,
    blend_aligned,
    immutable_json,
    passes_guards,
    validate_partition,
)


def verify_historical_source(directory):
    """Old runs belong to their saved sources, not the subsequently extended code."""
    directory = Path(directory)
    for filename in ("protocol.json", "ensemble_protocol.json"):
        path = directory / filename
        if path.exists():
            for name, expected in read_json(path).get("source_hashes", {}).items():
                if sha256(directory / "source" / name) != expected:
                    raise ValueError(f"Historical source snapshot changed: {directory}/{name}")


def initialize(root, data_directory, experiment, design_path, baseline_v3=None, baseline_v4=None):
    root, experiment = Path(root).resolve(), Path(experiment).resolve()
    data_directory, design_path = Path(data_directory).resolve(), Path(design_path).resolve()
    baseline_v3 = Path(baseline_v3).resolve() if baseline_v3 else root / "artifacts/skill_forecasting/improvement_v3"
    baseline_v4 = Path(baseline_v4).resolve() if baseline_v4 else root / "artifacts/skill_forecasting/improvement_v4"
    if experiment.exists() and any(experiment.iterdir()):
        if (experiment / "protocol.json").exists() and "preserved_artifacts_sha256" in read_json(experiment / "protocol.json"):
            return check_runner(root, experiment)
        raise ValueError("Initialization requires an empty experiment; preserve incomplete outputs")
    for directory in (baseline_v3, baseline_v4):
        verify_historical_source(directory)
    v3, v4 = read_json(baseline_v3 / "frozen_selection.json"), read_json(baseline_v4 / "frozen_selection.json")
    if (len(v3["components"]) != 4 or v4["new_weight"] != 0.5 or v4["active_members"] != 15
            or len(v4["components"]) != 1 or v4["components"][0]["candidate"] != "history60_tcn"
            or any(c["seeds"] != [42, 43, 44] for c in [*v3["components"], *v4["components"]])):
        raise ValueError("The retained reference must be the registered v4 fifteen-model recipe")
    preserved = {}
    for directory in (data_directory, root / "artifacts/skill_forecasting/initial_v1",
                      root / "artifacts/skill_forecasting/improvement_v2", baseline_v3, baseline_v4,
                      root / "docs/experiments/skill_forecasting_v4",
                      root / "docs/presentations/skills_catalogue"):
        preserved.update({path.relative_to(root).as_posix(): sha256(path)
                          for path in sorted(directory.rglob("*")) if path.is_file()})
    for path in sorted((root / "docs/presentations").rglob("*.pdf")):
        preserved[path.relative_to(root).as_posix()] = sha256(path)
    design = read_json(design_path)
    protocol = initialize_base(root, data_directory, experiment, design_path=design_path)
    for key, value in design.items():
        if key not in ("description", "candidates"):
            protocol[key] = value
    protocol.update({"baseline_v3_directory": str(baseline_v3), "baseline_v4_directory": str(baseline_v4),
                     "baseline_mlp_components": [c["candidate"] for c in v3["components"]],
                     "baseline_tcn_component": "history60_tcn", "baseline_members": 15,
                     "baseline_parameters": v4["active_parameters"], "baseline_new_weight": 0.5,
                     "secondary_weights": design.get("transformer_weights", [0.25, 0.5, 0.75])})
    protocol["selection"] = "Screen by standalone accuracy; confirm primary Transformer-only and optional incumbent blend separately"
    protocol["confirmation"] = "Two pure Transformers and one architectural hybrid, each with three seeds in both training-only partitions"
    protocol["final_fit"] = "Union of frozen recipes; epochs/temperature median of six internal fits, three seeds, all original train groups"
    protocol["validation_policy"] = "One invocation evaluates the jointly frozen primary and optional secondary; no validation-based winner selection"
    protocol["partition_hashes"] = {}
    samples = pd.read_parquet(data_directory / "samples.parquet")
    for fold in (0, 1):
        name = f"inner_split_{fold}.json"
        partition = read_json(experiment / name)
        if any(partition != read_json(directory / name) for directory in (baseline_v3, baseline_v4)):
            raise ValueError("Internal partitions differ from the preserved reference")
        validate_partition(partition, samples)
        protocol["partition_hashes"][name] = sha256(experiment / name)
    targets = np.load(data_directory / "targets.npy")
    protocol["fixed_origin_counts"] = {
        split: {"origins": int(samples.split.eq(split).sum()),
                "valid_targets": (targets[samples.split.eq(split)] >= 0).sum(0).tolist()}
        for split in ("train", "validation")}
    for source in (root / "scripts/transformer_skill_forecaster.py", design_path):
        relative = source.relative_to(root)
        destination = experiment / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        protocol["source_hashes"][relative.as_posix()] = sha256(source)
    write_json(experiment / "preserved_artifacts.json", preserved)
    protocol["preserved_artifacts_sha256"] = sha256(experiment / "preserved_artifacts.json")
    write_json(experiment / "environment.json", {
        "python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
        "numpy": np.__version__, "pandas": pd.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    write_json(experiment / "protocol.json", protocol)
    return check_runner(root, experiment)


def check_runner(root, experiment):
    root, experiment = Path(root), Path(experiment)
    protocol = check_protocol(root, experiment)
    manifest = experiment / "preserved_artifacts.json"
    if sha256(manifest) != protocol["preserved_artifacts_sha256"]:
        raise ValueError("Historical preservation manifest changed")
    for name, expected in read_json(manifest).items():
        if sha256(root / name) != expected:
            raise ValueError(f"Preserved historical artifact changed: {name}")
    verify_dataset(protocol["data_directory"])
    samples = pd.read_parquet(Path(protocol["data_directory"]) / "samples.parquet")
    for filename, expected in protocol["partition_hashes"].items():
        if sha256(experiment / filename) != expected:
            raise ValueError("Frozen internal partition changed")
        validate_partition(read_json(experiment / filename), samples)
    pure, hybrid = protocol["pure_candidates"], protocol["hybrid_candidates"]
    if (len(pure) < 2 or not hybrid or set(pure).intersection(hybrid)
            or len(set(pure + hybrid)) != len(pure + hybrid)
            or set(pure + hybrid) != set(protocol["candidates"])
            or protocol["training_seeds"] != [42, 43, 44]
            or protocol["secondary_weights"] != [0.25, 0.5, 0.75]
            or protocol["baseline_members"] != 15 or protocol["baseline_new_weight"] != 0.5):
        raise ValueError("Candidate families or the registered comparison changed")
    if (any(protocol["candidates"][name]["architecture"] not in
            ("legacy", "transformer", "temporal_transformer", "patch_transformer") for name in pure)
            or any(protocol["candidates"][name]["architecture"] != "hybrid_transformer" for name in hybrid)):
        raise ValueError("Primary Transformer and architectural-hybrid families must remain distinct")
    return protocol


def read_internal_member(path, name, fold, seed, protocol, partition, current=False):
    """Check data, split, checkpoint and saved-prediction provenance before reuse."""
    path = Path(path)
    run, summary = read_json(path / "run.json"), read_json(path / "summary.json")
    source = read_json(path.parent.parent / "protocol.json")
    expected_groups = sorted(group for group, role in partition.items() if role == "fit")
    expected_candidate = protocol["candidates"][name] if current else source["candidates"][name]
    if (run["candidate"] != expected_candidate or run["seed"] != seed or run["fold"] != fold
            or run["final"] or run["fit_groups"] != expected_groups or summary["name"] != name
            or summary["seed"] != seed or summary["fold"] != fold or summary["final"]
            or summary.get("test_evaluated", True) or summary.get("outer_validation_evaluated", True)
            or run["protocol_sha256"] != sha256(path.parent.parent / "protocol.json")
            or source["data_manifest_sha256"] != protocol["data_manifest_sha256"]
            or sha256(path / "best.pt") != summary["checkpoint_sha256"]):
        raise ValueError(f"Internal run provenance changed: {path}")
    saved = torch.load(path / "best.pt", map_location="cpu", weights_only=False)
    if (saved["candidate"] != expected_candidate or saved["model_config"] != run["model"]
            or saved["seed"] != seed or saved["epoch"] != summary["best_epoch"]
            or saved["temperature"] != summary["temperature"]
            or saved["scaling"]["fit_groups"] != expected_groups
            or saved["data_manifest_sha256"] != protocol["data_manifest_sha256"]):
        raise ValueError(f"Internal checkpoint provenance changed: {path}")
    del saved
    for filename, field in (("selection_probabilities.npy", "selection_probability_sha256"),
                            ("selection_indices.npy", "selection_indices_sha256")):
        if (current or field in summary) and sha256(path / filename) != summary[field]:
            raise ValueError(f"Cached internal prediction changed: {path / filename}")
    return _probability_array(np.load(path / "selection_probabilities.npy"),
                              np.load(path / "selection_indices.npy"))


def _check_selection_origins(values, experiment, protocol, fold):
    samples = pd.read_parquet(Path(protocol["data_directory"]) / "samples.parquet")
    partition = read_json(Path(experiment) / f"inner_split_{fold}.json")
    expected = np.flatnonzero(samples.scenario_group.map(partition).eq("select").to_numpy())
    if not np.array_equal(values[1], expected) or not samples.iloc[values[1]].split.eq("train").all():
        raise ValueError("Internal metrics require the exact original-training selection origins")
    return values


def incumbent_internal(experiment, protocol, fold):
    """The current reference: half the twelve MLPs, half the three TCNs."""
    partition = read_json(Path(experiment) / f"inner_split_{fold}.json")
    mlps, tcns = [], []
    for name in protocol["baseline_mlp_components"]:
        for seed in protocol["training_seeds"]:
            directory = Path(protocol["baseline_v3_directory"] if seed == 42 else protocol["baseline_v4_directory"])
            mlps.append(read_internal_member(directory / f"inner_{fold}/{name}_s{seed}",
                                             name, fold, seed, protocol, partition))
    name = protocol["baseline_tcn_component"]
    for seed in protocol["training_seeds"]:
        path = Path(protocol["baseline_v4_directory"]) / f"inner_{fold}/{name}_s{seed}"
        tcns.append(read_internal_member(path, name, fold, seed, protocol, partition))
    if len(mlps) != 12 or len(tcns) != 3:
        raise ValueError("The complete fifteen-member incumbent is required")
    return _check_selection_origins(blend_aligned(average_aligned(mlps), average_aligned(tcns), 0.5),
                                    experiment, protocol, fold)


def candidate_internal(experiment, protocol, names, fold, seeds):
    partition = read_json(Path(experiment) / f"inner_split_{fold}.json")
    values = average_aligned(read_internal_member(
        Path(experiment) / f"inner_{fold}/{name}_s{seed}", name, fold, seed, protocol, partition, current=True)
        for name in names for seed in seeds)
    return _check_selection_origins(values, experiment, protocol, fold)


def metric_row(recipe, names, weight, fold, values, targets, baseline, guards,
               kind, seeds, require_gain=False):
    probability, indices = values
    metric = quick_metrics(probability, targets[indices])
    return {"kind": kind, "recipe": recipe, "components": list(names), "new_weight": weight,
            "fold": fold, "active_members": (15 if weight < 1 else 0) + len(names) * len(seeds),
            "mean_accuracy": metric["mean_accuracy"], "nll": metric["nll"],
            "mean_macro_f1": metric["mean_macro_f1"],
            "gain": metric["mean_accuracy"] - baseline["mean_accuracy"],
            "eligible": passes_guards(metric, baseline, guards, require_gain),
            **{f"accuracy_{h}s": value for h, value in zip((1, 2, 5), metric["accuracy"], strict=True)}}


def shortlist_names(frame, pure, hybrid):
    """Single-member guard flags are descriptive; screen by standalone accuracy."""
    ranked = frame.sort_values(["mean_accuracy", "recipe"], ascending=[False, True])
    return (ranked[ranked.recipe.isin(pure)].head(2).recipe.tolist(),
            ranked[ranked.recipe.isin(hybrid)].head(1).recipe.tolist())


def screen(root, experiment):
    experiment = Path(experiment)
    protocol = check_runner(root, experiment)
    if (experiment / "frozen_selection.json").exists():
        raise ValueError("The recipes are already frozen")
    if (experiment / "shortlist.json").exists():
        return read_json(experiment / "shortlist.json")
    targets = np.load(Path(protocol["data_directory"]) / "targets.npy")
    reference = incumbent_internal(experiment, protocol, 0)
    reference_metrics = quick_metrics(reference[0], targets[reference[1]])
    rows = []
    for name in [*protocol["pure_candidates"], *protocol["hybrid_candidates"]]:
        fit_run(root, experiment, name, fold=0, seed=42)
        proposed = candidate_internal(experiment, protocol, [name], 0, [42])
        family = "pure_transformer" if name in protocol["pure_candidates"] else "architectural_hybrid"
        rows.append(metric_row(name, [name], 1.0, 0, proposed, targets, reference_metrics,
                               protocol.get("guards"), family, [42]))
    frame = pd.DataFrame(rows)
    frame.to_csv(experiment / "transformer_screen.csv", index=False, lineterminator="\n")
    pure, hybrid = shortlist_names(frame, protocol["pure_candidates"], protocol["hybrid_candidates"])
    return immutable_json(experiment / "shortlist.json", {
        "pure_names": pure, "hybrid_names": hybrid, "names": pure + hybrid,
        "selected_on": "Fold 0 standalone mean accuracy; single-seed guard flags are descriptive",
        "baseline_metrics": reference_metrics, "baseline_members": 15,
        "protocol_sha256": sha256(experiment / "protocol.json"),
        "outer_validation_evaluated": False, "test_evaluated": False})


def rank_confirmed(frame, kind, allow_fallback):
    """Rank exactly two-fold recipes; an unqualified primary remains a challenger."""
    selected = frame[frame.kind.eq(kind)]
    if len(selected) and set(selected.fold) != {0, 1}:
        raise ValueError("Each confirmed recipe requires both registered folds 0 and 1")
    if selected.duplicated(["recipe", "new_weight", "fold"]).any():
        raise ValueError("Duplicate confirmation rows would alter fold weighting")
    ranked = selected.groupby(["recipe", "new_weight"], sort=True).agg(
        mean_accuracy=("mean_accuracy", "mean"), eligible=("eligible", "all"),
        folds=("fold", "nunique"), mean_gain=("gain", "mean"), nll=("nll", "mean"),
        mean_macro_f1=("mean_macro_f1", "mean")).reset_index()
    if len(ranked) and not ranked.folds.eq(2).all():
        raise ValueError("Each confirmed recipe requires both registered folds")
    eligible = ranked[ranked.eligible]
    candidates = eligible if len(eligible) else ranked if allow_fallback else ranked.iloc[:0]
    if not len(candidates):
        return None
    return candidates.sort_values(["mean_accuracy", "recipe", "new_weight"],
                                  ascending=[False, True, True]).iloc[0].to_dict()


def _recipe_record(selected, definitions, parameter_counts, kind, baseline_parameters):
    if selected is None:
        return None
    names = definitions[selected["recipe"]]
    weight = float(selected["new_weight"])
    return {"kind": kind, "recipe": selected["recipe"], "components": names,
            "new_weight": weight, "selection_metrics": selected,
            "quality_guards_passed_both_folds": bool(selected["eligible"]),
            "active_members": (15 if weight < 1 else 0) + 3 * len(names),
            "active_parameters": (baseline_parameters if weight < 1 else 0)
            + 3 * sum(parameter_counts[name] for name in names)}


def confirm(root, experiment):
    experiment = Path(experiment)
    protocol = check_runner(root, experiment)
    if (experiment / "frozen_selection.json").exists():
        return selection_for(experiment)
    shortlist = read_json(experiment / "shortlist.json")
    pure, hybrid = shortlist["pure_names"], shortlist["hybrid_names"]
    names = pure + hybrid
    if (shortlist["protocol_sha256"] != sha256(experiment / "protocol.json")
            or len(pure) != 2 or len(hybrid) != 1 or len(set(names)) != 3 or shortlist["names"] != names
            or not set(pure).issubset(protocol["pure_candidates"])
            or not set(hybrid).issubset(protocol["hybrid_candidates"])):
        raise ValueError("Frozen shortlist provenance/families changed")
    primary_definitions = {name: [name] for name in pure}
    primary_definitions["pure_pair"] = pure
    secondary_definitions = {**primary_definitions, hybrid[0]: hybrid}
    targets = np.load(Path(protocol["data_directory"]) / "targets.npy")
    seeds, rows, references = protocol["training_seeds"], [], []
    for fold in (0, 1):
        for name in names:
            for seed in seeds:
                fit_run(root, experiment, name, fold=fold, seed=seed)
        reference = incumbent_internal(experiment, protocol, fold)
        metric = quick_metrics(reference[0], targets[reference[1]])
        references.append({"fold": fold, "members": 15, **metric})
        for recipe, components in secondary_definitions.items():
            proposed = candidate_internal(experiment, protocol, components, fold, seeds)
            if recipe in primary_definitions:
                rows.append(metric_row(recipe, components, 1.0, fold, proposed, targets, metric,
                                       protocol.get("guards"), "primary_transformer", seeds))
            else:
                rows.append(metric_row(recipe, components, 1.0, fold, proposed, targets, metric,
                                       protocol.get("guards"), "architectural_hybrid_only", seeds))
            for weight in protocol["secondary_weights"]:
                rows.append(metric_row(recipe, components, weight, fold,
                                       blend_aligned(reference, proposed, weight), targets, metric,
                                       protocol.get("guards"), "secondary_blend", seeds, require_gain=True))
    frame = pd.DataFrame(rows)
    frame.to_csv(experiment / "transformer_confirmation.csv", index=False, lineterminator="\n")
    immutable_json(experiment / "internal_baseline_metrics.json", references)
    primary = rank_confirmed(frame, "primary_transformer", allow_fallback=True)
    secondary = rank_confirmed(frame, "secondary_blend", allow_fallback=False)
    if primary is None:
        raise ValueError("A primary Transformer-only challenger must be frozen")
    selected_names = sorted(set(primary_definitions[primary["recipe"]]) |
                            (set(secondary_definitions[secondary["recipe"]]) if secondary else set()))
    components, parameter_counts = [], {}
    for name in selected_names:
        summaries = [read_json(experiment / f"inner_{fold}/{name}_s{seed}/summary.json")
                     for fold in (0, 1) for seed in seeds]
        parameters = read_json(experiment / f"inner_0/{name}_s42/run.json")["parameters"]
        parameter_counts[name] = parameters
        components.append({"candidate": name, "seeds": seeds,
                           "family": "pure_transformer" if name in pure else "architectural_hybrid",
                           "epochs": int(np.rint(np.median([row["best_epoch"] for row in summaries]))),
                           "temperature": float(np.median([row["temperature"] for row in summaries])),
                           "parameters_per_member": parameters})
    result = {"components": components,
              "primary_transformer": _recipe_record(primary, primary_definitions, parameter_counts,
                                                       "primary_transformer", protocol["baseline_parameters"]),
              "secondary_blend": _recipe_record(secondary, secondary_definitions, parameter_counts,
                                                 "secondary_blend", protocol["baseline_parameters"]),
              "baseline_members": 15, "baseline_parameters": protocol["baseline_parameters"],
              "epochs_temperature_rule": "Median of six internal fits, then fixed three-seed refit",
              "primary_policy": "Always evaluate the frozen Transformer-only challenger; guard failure is not promotion",
              "secondary_policy": "Optional incumbent blend requires positive gain and quality guards in both folds",
              "validation_policy": "Both named outputs frozen jointly; no selection or tuning from their validation scores",
              "protocol_sha256": sha256(experiment / "protocol.json"),
              "shortlist_sha256": sha256(experiment / "shortlist.json"),
              "outer_validation_evaluated": False, "test_evaluated": False}
    immutable_json(experiment / "frozen_selection.json", result)
    immutable_json(experiment / "selection_receipt.json", {
        "frozen_selection_sha256": sha256(experiment / "frozen_selection.json")})
    return result


def selection_for(experiment):
    experiment = Path(experiment)
    result = read_json(experiment / "frozen_selection.json")
    if (result["protocol_sha256"] != sha256(experiment / "protocol.json")
            or result["shortlist_sha256"] != sha256(experiment / "shortlist.json")
            or read_json(experiment / "selection_receipt.json")["frozen_selection_sha256"]
            != sha256(experiment / "frozen_selection.json")):
        raise ValueError("Joint frozen selection provenance changed")
    return result


def refit(root, experiment):
    check_runner(root, experiment)
    selection = selection_for(experiment)
    for component in selection["components"]:
        for seed in component["seeds"]:
            fit_run(root, experiment, component["candidate"], seed=seed,
                    final_epochs=component["epochs"], final_temperature=component["temperature"])
    return selection


def final_paths(experiment, protocol, selection):
    """Validate every frozen refit before opening outer-validation predictions."""
    experiment = Path(experiment)
    groups = sorted(read_json(experiment / "inner_split_0.json"))
    classes = read_json(Path(protocol["data_directory"]) / "classes.json")
    paths = []
    for component in selection["components"]:
        for seed in component["seeds"]:
            path = experiment / f"refit/{component['candidate']}_s{seed}"
            run, summary = read_json(path / "run.json"), read_json(path / "summary.json")
            if (not run["final"] or not summary["final"] or run["seed"] != seed or summary["seed"] != seed
                    or run["fit_groups"] != groups or run["candidate"] != protocol["candidates"][component["candidate"]]
                    or run["protocol_sha256"] != sha256(experiment / "protocol.json")
                    or summary["best_epoch"] != component["epochs"] or summary["epochs_completed"] != component["epochs"]
                    or run["epochs_budget"] != component["epochs"]
                    or summary["temperature"] != component["temperature"]
                    or sha256(path / "best.pt") != summary["checkpoint_sha256"]):
                raise ValueError(f"Final refit differs from the frozen selection: {path}")
            saved = torch.load(path / "best.pt", map_location="cpu", weights_only=False)
            scaling = saved["scaling"]
            if (saved["candidate"] != run["candidate"] or saved["model_config"] != run["model"]
                    or saved["seed"] != seed or saved["epoch"] != component["epochs"]
                    or saved["temperature"] != component["temperature"] or saved["classes"] != classes
                    or saved["horizons_s"] != [1, 2, 5] or scaling["fit_groups"] != groups
                    or saved["data_manifest_sha256"] != protocol["data_manifest_sha256"]
                    or len(saved["feature_names"]) != run["model"]["input_size"]
                    or len(saved["offsets"]) != run["model"]["tokens"] or saved["offsets"][-1] != 0
                    or np.any(np.asarray(saved["offsets"]) > 0)
                    or len(scaling["mean"]) != run["model"]["input_size"]
                    or len(scaling["std"]) != run["model"]["input_size"]
                    or not np.isfinite(scaling["mean"]).all() or not np.isfinite(scaling["std"]).all()
                    or np.any(np.asarray(scaling["std"]) <= 0)):
                raise ValueError(f"Final checkpoint metadata differs from the frozen selection: {path}")
            history = pd.read_csv(path / "history.csv")
            if (history.epoch.tolist() != list(range(1, component["epochs"] + 1))
                    or any(c.startswith(("stop_", "validation_")) for c in history)):
                raise ValueError("Final fitting must use fixed epochs without stop/validation callbacks")
            del saved
            paths.append((component["candidate"], seed, path))
    return paths


def mean_accuracy_interval(new, old, targets, samples, draws=5000, seed=1729):
    """Average horizon gains inside each shared-scenario bootstrap replicate."""
    new, old, targets = np.asarray(new), np.asarray(old), np.asarray(targets)
    if (new.ndim != 3 or new.shape != old.shape or targets.shape != (len(samples), 3)
            or new.shape[:2] != targets.shape or not len(samples)):
        raise ValueError("Bootstrap requires aligned probabilities, samples and three target horizons")
    groups = sorted(samples.scenario_group.unique())
    selection = np.random.default_rng(seed).integers(0, len(groups), size=(draws, len(groups)))
    differences = (new.argmax(-1) == targets).astype(float) - (old.argmax(-1) == targets).astype(float)
    counts, deltas = [], []
    for group in groups:
        mask = samples.scenario_group.eq(group).to_numpy()[:, None] & (targets >= 0)
        counts.append(mask.sum(0))
        deltas.append((differences * mask).sum(0))
    counts, deltas = np.asarray(counts), np.asarray(deltas)
    denominators = counts[selection].sum(1)
    if np.any(denominators <= 0):
        raise ValueError("Every bootstrap replicate must support each horizon")
    replicates = (deltas[selection].sum(1) / denominators).mean(1)
    lower, upper = np.quantile(replicates, [0.025, 0.975])
    return {"mean_accuracy_gain": float((deltas.sum(0) / counts.sum(0)).mean()),
            "lower_95": float(lower), "upper_95": float(upper), "scenario_groups": len(groups),
            "bootstrap_draws": draws, "seed": seed,
            "interpretation": "Conditional on frozen models and development data, not selection-adjusted; crossing zero is not proof of equivalence"}


def evaluate(root, experiment):
    experiment = Path(experiment)
    protocol = check_runner(root, experiment)
    selection = selection_for(experiment)
    output = experiment / "validation"
    if output.exists():
        raise ValueError("Frozen validation evaluation already started; preserve it and do not retune")
    paths = final_paths(experiment, protocol, selection)
    baseline_directory = Path(protocol["baseline_v4_directory"]) / "validation/ensemble"
    samples = pd.read_parquet(baseline_directory / "validation_samples.parquet")
    prepared = Path(protocol["data_directory"])
    all_samples = pd.read_parquet(prepared / "samples.parquet")
    indices = np.flatnonzero(all_samples.split.eq("validation").to_numpy())
    pd.testing.assert_frame_equal(samples, all_samples.iloc[indices].reset_index(drop=True))
    baseline = _probability_array(np.load(baseline_directory / "validation_probabilities.npy"), indices)
    targets, current = np.load(prepared / "targets.npy")[indices], np.load(prepared / "current.npy")[indices]
    classes = read_json(prepared / "classes.json")
    output.mkdir()
    immutable_json(output / "evaluation_started.json", {
        "selection_sha256": sha256(experiment / "frozen_selection.json"),
        "policy": "One invocation for jointly frozen outputs; no validation-based selection", "test_evaluated": False})
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    predictions, checkpoint_hashes = {}, {}
    for name, seed, path in paths:
        saved = torch.load(path / "best.pt", map_location="cuda", weights_only=False)
        data = DevelopmentData(prepared, saved["candidate"], [], scaling=saved["scaling"])
        model = ImprovedForecaster(**saved["model_config"]).to(data.device)
        model.load_state_dict(saved["model_state"])
        probabilities = predict(model, data, torch.as_tensor(indices, device=data.device), temperature=saved["temperature"])
        directory = output / "members" / path.name
        directory.mkdir(parents=True)
        np.save(directory / "validation_probabilities.npy", probabilities)
        predictions[(name, seed)] = (probabilities, indices)
        checkpoint_hashes[path.relative_to(experiment).as_posix()] = sha256(path / "best.pt")
        del model, data
        gc.collect()
        torch.cuda.empty_cache()
    baseline_metrics = quick_metrics(baseline[0], targets)
    results, interval_frames = {}, []
    for label in ("primary_transformer", "secondary_blend"):
        recipe = selection[label]
        if recipe is None:
            continue
        proposed = average_aligned(predictions[(name, seed)] for name in recipe["components"]
                                   for seed in protocol["training_seeds"])
        probability, _ = blend_aligned(baseline, proposed, recipe["new_weight"])
        directory = output / label
        save_evaluation(directory, probability, targets, current, samples, classes, label)
        np.save(directory / "validation_probabilities.npy", probability)
        samples.to_parquet(directory / "validation_samples.parquet", index=False)
        metrics = quick_metrics(probability, targets)
        intervals = paired_group_intervals(probability, baseline[0], targets, samples)
        interval_frames.append(intervals.assign(recipe=label))
        results[label] = {"recipe": recipe, "metrics": metrics,
                          "mean_accuracy_gain": metrics["mean_accuracy"] - baseline_metrics["mean_accuracy"],
                          "paired_intervals": intervals.to_dict(orient="records"),
                          "mean_accuracy_interval": mean_accuracy_interval(probability, baseline[0], targets, samples),
                          "standalone_new_components_metrics": quick_metrics(proposed[0], targets)}
    pd.concat(interval_frames, ignore_index=True).to_csv(experiment / "paired_scenario_bootstrap.csv", index=False, lineterminator="\n")
    result = {"selection_sha256": sha256(experiment / "frozen_selection.json"),
              "baseline": "Preserved v4 fifteen-model ensemble", "baseline_metrics": baseline_metrics,
              "recipes": results, "checkpoint_hashes": checkpoint_hashes,
              "matched_validation_origins": True, "outer_validation_evaluated": True, "test_evaluated": False,
              "validation_role": "Previously inspected development validation, not an untouched final test",
              "decision_policy": "Primary and optional secondary were frozen jointly; validation does not select a winner"}
    write_json(output / "evaluation.json", result)
    write_json(output / "output_hashes.json", {
        path.relative_to(output).as_posix(): sha256(path) for path in sorted(output.rglob("*")) if path.is_file()})
    return result
