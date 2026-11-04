"""Train-only confirmation of fixed heterogeneous probability averages."""

import gc
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bvr_behavior_prediction.forecasting.data import read_json, sha256, write_json

from .evaluation import build_report, save_evaluation
from .improved_models import ImprovedForecaster
from .improvement import (
    DevelopmentData,
    check_protocol,
    fit_run,
    paired_group_intervals,
    predict,
    quick_metrics,
    summarize_search,
)


def mixture_definitions():
    a, b, c, d = ("history20_mlp_accuracy", "history20_mlp_wide_accuracy",
                  "history20_mlp_smooth_accuracy", "history60_mlp_accuracy")
    return {"control": ["control"], "nll_mlp": ["history20_mlp_nll"],
            "accuracy_pair": [a, b], "accuracy_triple": [a, b, c],
            "long_pair": [b, d], "accuracy_all": [a, b, c, d]}


def register(root, experiment):
    root, experiment = Path(root), Path(experiment)
    check_protocol(root, experiment)
    path = experiment / "ensemble_protocol.json"
    if path.exists():
        return read_json(path)
    if (experiment / "validation").exists() or (experiment / "frozen_selection.json").exists():
        raise ValueError("Register the ensemble extension before freezing/evaluating validation")
    protocol = {
        "reason": "Internal individual models gained accuracy but lost NLL; exploratory narrow/wide averaging improved both versus control",
        "definitions": mixture_definitions(), "weights": "Uniform; no fitted mixture weights",
        "confirmation": "Top two eligible mixtures and control on internal split 1",
        "guards": {"maximum_nll_increase": 0.02, "maximum_macro_f1_decrease": 0.01},
        "final_seeds_per_component": [42, 43, 44],
        "outer_validation_evaluated": False, "test_evaluated": False,
    }
    files = [Path(__file__), root / "scripts/ensemble_skill_forecaster.py"]
    protocol["source_hashes"] = {}
    for source in files:
        relative = source.relative_to(root)
        destination = experiment / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        protocol["source_hashes"][relative.as_posix()] = sha256(source)
    write_json(path, protocol)
    return protocol


def check_extension(root, experiment):
    protocol = check_protocol(root, experiment)
    extension = read_json(Path(experiment) / "ensemble_protocol.json")
    for name, expected in extension["source_hashes"].items():
        if sha256(Path(root) / name) != expected:
            raise ValueError(f"Ensemble extension source changed: {name}")
    return protocol, extension


def combine_predictions(paths):
    indices, predictions = None, []
    for path in paths:
        current = np.load(Path(path) / "selection_indices.npy")
        if indices is not None and not np.array_equal(indices, current):
            raise ValueError("Ensemble members have different selection origins")
        indices = current
        predictions.append(np.load(Path(path) / "selection_probabilities.npy"))
    return np.mean(predictions, axis=0), indices


def mixture_scores(experiment, fold, definitions):
    experiment = Path(experiment)
    protocol = read_json(experiment / "protocol.json")
    targets = np.load(Path(protocol["data_directory"]) / "targets.npy")
    samples = pd.read_parquet(Path(protocol["data_directory"]) / "samples.parquet")
    partition = read_json(experiment / f"inner_split_{fold}.json")
    rows = []
    for name, components in definitions.items():
        paths = [experiment / f"inner_{fold}" / f"{component}_s42" for component in components]
        if not all((path / "summary.json").exists() for path in paths):
            continue
        probabilities, indices = combine_predictions(paths)
        selected = samples.iloc[indices]
        if not selected.split.eq("train").all() or not selected.scenario_group.map(partition).eq("select").all():
            raise ValueError("Mixture ranking must use internal selection groups only")
        metrics = quick_metrics(probabilities, targets[indices])
        rows.append({"name": name, "fold": fold, "nll": metrics["nll"],
                     "mean_accuracy": metrics["mean_accuracy"], "macro_f1": metrics["mean_macro_f1"],
                     **{f"accuracy_{h}s": value for h, value in zip((1, 2, 5), metrics["accuracy"], strict=True)}})
    return pd.DataFrame(rows)


def eligible(frame):
    control = frame.loc["control"]
    return frame[(frame.nll <= control.nll + 0.02) & (frame.macro_f1 >= control.macro_f1 - 0.01)]


def confirm_mixtures(root, experiment):
    experiment = Path(experiment)
    _, extension = check_extension(root, experiment)
    definitions = extension["definitions"]
    first = mixture_scores(experiment, 0, definitions)
    first.to_csv(experiment / "ensemble_screen.csv", index=False)
    shortlisted = eligible(first.set_index("name")).sort_values("mean_accuracy", ascending=False).head(2).index.tolist()
    names = list(dict.fromkeys(["control", *shortlisted]))
    write_json(experiment / "ensemble_shortlist.json", {"names": names, "selected_on": "internal split 0 only"})
    components = sorted({component for name in names for component in definitions[name]})
    for component in components:
        fit_run(root, experiment, component, fold=1)
    second = mixture_scores(experiment, 1, {name: definitions[name] for name in names})
    paired = pd.concat([first[first.name.isin(names)], second], ignore_index=True)
    paired.to_csv(experiment / "ensemble_confirmation.csv", index=False)
    average = paired.groupby("name")[["mean_accuracy", "nll", "macro_f1"]].mean()
    chosen = eligible(average).sort_values("mean_accuracy", ascending=False).index[0]
    results = summarize_search(experiment)
    components = []
    for component in definitions[chosen]:
        rows = results[(results.name == component) & results.seed.eq(42)]
        if set(rows.fold) != {0, 1}:
            raise ValueError("Selected component lacks two-fold confirmation")
        components.append({"candidate": component, "epochs": int(np.rint(rows.best_epoch.median())),
                           "temperature": float(rows.temperature.median()), "seeds": [42, 43, 44]})
    selection = {"candidate": chosen, "components": components,
                 "ensemble": "Uniform arithmetic average across all selected components and three fixed seeds each",
                 "selected_using": "Two internal grouped partitions; no new outer-validation scores",
                 "paired_internal_results": average.reset_index().to_dict(orient="records"),
                 "outer_validation_evaluated": False, "test_evaluated": False}
    frozen = experiment / "frozen_selection.json"
    if frozen.exists() and read_json(frozen) != selection:
        raise ValueError("Final selection is already frozen")
    if not frozen.exists():
        write_json(frozen, selection)
    return selection


def refit_mixture(root, experiment):
    check_extension(root, experiment)
    selection = read_json(Path(experiment) / "frozen_selection.json")
    for component in selection["components"]:
        for seed in component["seeds"]:
            fit_run(root, experiment, component["candidate"], seed=seed,
                    final_epochs=component["epochs"], final_temperature=component["temperature"])


def evaluate_mixture(root, experiment, verify=False):
    root, experiment = Path(root), Path(experiment)
    protocol, _ = check_extension(root, experiment)
    selection = read_json(experiment / "frozen_selection.json")
    output = experiment / "validation"
    if output.exists() and not verify:
        raise ValueError("Validation already evaluated; no further tuning against these scores")
    paths = [experiment / "refit" / f"{component['candidate']}_s{seed}"
             for component in selection["components"] for seed in component["seeds"]]
    if not all((path / "summary.json").exists() for path in paths):
        raise ValueError("All frozen final fits must complete before validation")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    predictions, verifications = [], []
    for path in paths:
        summary = read_json(path / "summary.json")
        if sha256(path / "best.pt") != summary["checkpoint_sha256"]:
            raise ValueError("Checkpoint hash changed")
        saved = torch.load(path / "best.pt", map_location="cuda", weights_only=False)
        data = DevelopmentData(protocol["data_directory"], saved["candidate"], [], scaling=saved["scaling"])
        indices = torch.tensor(np.flatnonzero(data.samples.split.eq("validation")), device=data.device)
        model = ImprovedForecaster(**saved["model_config"]).to(data.device)
        model.load_state_dict(saved["model_state"])
        probability = predict(model, data, indices, temperature=saved.get("temperature", 1.0))
        targets, current = data.targets[indices].cpu().numpy(), data.current[indices].cpu().numpy()
        samples = data.samples.iloc[indices.cpu().numpy()].reset_index(drop=True)
        if verify:
            np.testing.assert_array_equal(probability, np.load(output / path.name / "validation_probabilities.npy"))
            verifications.append({"member": path.name, "probabilities_exact": True, "checkpoint_hash_matches": True})
        else:
            save_evaluation(output / path.name, probability, targets, current, samples, data.classes, path.name)
            np.save(output / path.name / "validation_probabilities.npy", probability)
        predictions.append(probability)
        del model, data
        gc.collect()
        torch.cuda.empty_cache()
    probability = np.mean(predictions, axis=0)
    classes = read_json(Path(protocol["data_directory"]) / "classes.json")
    if verify:
        np.testing.assert_array_equal(probability, np.load(output / "ensemble/validation_probabilities.npy"))
    else:
        save_evaluation(output / "ensemble", probability, targets, current, samples, classes, "improved_ensemble")
        np.save(output / "ensemble/validation_probabilities.npy", probability)
        samples.to_parquet(output / "ensemble/validation_samples.parquet", index=False)
        write_json(output / "evaluation.json", {"selection_sha256": sha256(experiment / "frozen_selection.json"),
                                                 "metrics": quick_metrics(probability, targets), "test_evaluated": False})
        return quick_metrics(probability, targets)
    preserved = read_json(experiment / "preserved_initial_hashes.json")
    for name, expected in preserved.items():
        if sha256(root / name) != expected:
            raise ValueError(f"Original artifact changed: {name}")
    original = root / "artifacts/skill_forecasting/initial_v1"
    pd.testing.assert_frame_equal(samples, pd.read_parquet(original / "transformer/validation_samples.parquet"))
    intervals = paired_group_intervals(probability, np.load(original / "transformer/validation_probabilities.npy"),
                                       targets, samples)
    intervals.to_csv(experiment / "paired_scenario_bootstrap.csv", index=False)
    for name in ("transformer", "gru", "training_frequency"):
        destination = output / f"original_{name}"
        destination.mkdir(exist_ok=True)
        for path in (original / name).glob("*.csv"):
            shutil.copy2(path, destination / path.name)
    build_report(output, protocol["data_directory"]).to_csv(experiment / "comparison.csv", index=False)
    result = {"members": verifications, "ensemble_probabilities_exact": True,
              "original_artifacts_unchanged": True, "matched_validation_origins": True, "test_evaluated": False}
    write_json(experiment / "verification.json", result)
    return {**result, "paired_intervals": intervals.to_dict(orient="records")}
