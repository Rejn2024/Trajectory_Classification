"""Training-only model development, frozen refits and one validation comparison."""

import gc
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bvr_behavior_prediction.forecasting.data import read_json, sha256, write_json

from .data import verify_dataset
from .evaluation import build_report, save_evaluation
from .improved_features import (
    geometry_features,
    history_offsets,
    raw_history_features,
    window_indices,
)
from .improved_models import ImprovedForecaster, supervised_loss
from .temperature import fit_temperature
from .train import atomic_checkpoint, source_snapshot, validation_nll


def inner_partition(scenario_splits, seed):
    """147 fit / 21 epoch-stop / 42 candidate-selection groups within original train."""
    rng = np.random.default_rng(seed)
    result = {}
    ranges = sorted({v["requested_range_m"] for v in scenario_splits.values()})
    for distance in ranges:
        groups = np.array(sorted(k for k, v in scenario_splits.items()
                                if v["split"] == "train" and v["requested_range_m"] == distance))
        rng.shuffle(groups)
        n_fit, n_stop = round(len(groups) * 0.7), round(len(groups) * 0.1)
        for role, names in zip(("fit", "stop", "select"),
                               np.split(groups, [n_fit, n_fit + n_stop]), strict=True):
            result.update({str(name): role for name in names})
    return result


def candidates():
    base = {"architecture": "transformer", "features": "geometry", "history_seconds": 5,
            "recent_stride": 1, "width": 128, "layers": 2, "dropout": 0.2,
            "smoothing": 0.02, "weight_decay": 0.03, "auxiliary_weight": 0.0}
    return {
        "control": {**base, "architecture": "legacy", "features": "raw", "dropout": 0.1,
                    "smoothing": 0.0, "weight_decay": 0.01},
        "regularized": {**base, "architecture": "legacy", "features": "raw", "dropout": 0.25},
        "geometry": base,
        "geometry_mlp": {**base, "architecture": "mlp", "width": 256, "layers": 3, "recent_stride": 5},
        "history20": {**base, "history_seconds": 20},
        "history20_aux": {**base, "history_seconds": 20, "auxiliary_weight": 0.25},
        "history20_mlp": {**base, "history_seconds": 20, "architecture": "mlp", "width": 256,
                          "layers": 3, "recent_stride": 5},
        "history20_wide": {**base, "history_seconds": 20, "width": 192, "layers": 3, "dropout": 0.3,
                           "smoothing": 0.03, "weight_decay": 0.05},
        "history20_mlp_wide": {**base, "history_seconds": 20, "architecture": "mlp", "width": 512,
                               "layers": 3, "recent_stride": 5, "dropout": 0.3,
                               "smoothing": 0.03, "weight_decay": 0.05},
    }


class DevelopmentData:
    def __init__(self, directory, candidate, fit_groups, scaling=None, device="cuda"):
        self.directory, self.device = Path(directory), torch.device(device)
        self.manifest = verify_dataset(directory)
        self.samples = pd.read_parquet(self.directory / "samples.parquet")
        if not self.samples.split.isin(["train", "validation"]).all():
            raise ValueError("Reserved test rows must not be present")
        self.classes = self.manifest["classes"]
        self.offsets = history_offsets(candidate["history_seconds"], candidate["recent_stride"])
        anchors = np.load(self.directory / "anchors.npy")
        self.frame_indices, self.valid_history = window_indices(anchors, self.samples.step.to_numpy(), self.offsets)
        frames, self.feature_names = geometry_features(np.load(self.directory / "features.npy"), candidate["features"])
        if scaling is None:
            fit = self.samples.scenario_group.isin(fit_groups).to_numpy()
            if not fit.any() or not self.samples.loc[fit, "split"].eq("train").all():
                raise ValueError("Scaling must use nonempty original-training groups only")
            used = np.zeros(len(frames), dtype=bool)
            used[self.frame_indices[fit][self.valid_history[fit]]] = True
            values = frames[used].astype(np.float64)
            scaling = {"mean": values.mean(0).tolist(), "std": np.maximum(values.std(0), 0.01).tolist(),
                       "fit_groups": sorted(fit_groups), "frames": int(used.sum()),
                       "fitted_on": "fit groups only"}
        self.scaling = scaling
        self.features = torch.as_tensor(frames, device=self.device)
        self.features.sub_(torch.tensor(scaling["mean"], dtype=torch.float32, device=self.device))
        self.features.div_(torch.tensor(scaling["std"], dtype=torch.float32, device=self.device))
        self.indices_tensor = torch.as_tensor(self.frame_indices, device=self.device)
        self.valid_tensor = torch.as_tensor(self.valid_history, device=self.device)
        self.targets = torch.as_tensor(np.load(self.directory / "targets.npy"), device=self.device)
        self.current = torch.as_tensor(np.load(self.directory / "current.npy"), device=self.device)

    def windows(self, indices):
        return self.features[self.indices_tensor[indices]], self.valid_tensor[indices]

    def indices_for(self, groups):
        return torch.tensor(np.flatnonzero(self.samples.scenario_group.isin(groups)), device=self.device)


def quick_metrics(probabilities, targets):
    scores, f1s = [], []
    for h in range(3):
        valid = targets[:, h] >= 0
        actual, predicted = targets[valid, h], probabilities[valid, h].argmax(-1)
        scores.append(float(np.mean(actual == predicted)))
        classes = probabilities.shape[-1]
        confusion = np.bincount(actual * classes + predicted, minlength=classes ** 2).reshape(classes, classes)
        denominator = confusion.sum(0) + confusion.sum(1)
        f1s.append(float(np.divide(2 * confusion.diagonal(), denominator,
                                   out=np.zeros(classes), where=denominator > 0).mean()))
    return {"accuracy": scores, "mean_accuracy": float(np.mean(scores)),
            "macro_f1": f1s, "mean_macro_f1": float(np.mean(f1s)),
            "nll": validation_nll(probabilities, targets)}


@torch.inference_mode()
def predict(model, data, indices, batch_size=2048, temperature=1.0):
    model.eval()
    predictions = []
    for batch in indices.split(batch_size):
        with torch.autocast(device_type=data.device.type, dtype=torch.bfloat16, enabled=data.device.type == "cuda"):
            logits = model(*data.windows(batch))[:, :3]
        predictions.append((logits.float() / temperature).softmax(-1).cpu().numpy())
    return np.concatenate(predictions)


def initialize(root, data_directory, experiment, design_path=None):
    root, experiment = Path(root), Path(experiment)
    if (experiment / "protocol.json").exists():
        return read_json(experiment / "protocol.json")
    experiment.mkdir(parents=True, exist_ok=True)
    splits = read_json(Path(data_directory) / "scenario_splits.json")
    protocol = {
        "data_directory": str(Path(data_directory).resolve()),
        "data_manifest_sha256": sha256(Path(data_directory) / "data_manifest.json"),
        "candidates": candidates(), "split_seeds": [271828, 314159], "training_seeds": [42, 43, 44],
        "max_epochs": 60, "patience": 10, "min_delta": 0.001,
        "learning_rate": 0.0003, "batch_size": 1024, "evaluation_batch_size": 2048,
        "selection": "Epochs by stop-group NLL; candidates by select-group mean top-1, with NLL/F1 guards",
        "confirmation": "Top two eligible candidates plus control on a second internal partition",
        "final_fit": "All original training groups; fixed epochs from median of two internal best epochs; three seeds",
        "validation_policy": "Existing validation is used only after candidate, epochs and uniform ensemble are frozen",
        "test_policy": "Reserved test is neither prepared nor evaluated",
    }
    if design_path is not None:
        design = read_json(design_path)
        protocol["candidates"] = design["candidates"]
        protocol["design_description"] = design["description"]
        protocol["selection"] = "Candidate-specific stop-group NLL or accuracy; optional stop-only temperature; select-group accuracy with NLL/F1 guards"
        protocol["final_fit"] += "; temperature fixed to median of internal stop-fitted values"
        write_json(experiment / "design.json", design)
    protocol["source_hashes"] = source_snapshot(root, experiment)
    entry = root / "scripts/improve_skill_forecaster.py"
    destination = experiment / "source/scripts/improve_skill_forecaster.py"
    shutil.copy2(entry, destination)
    protocol["source_hashes"][entry.relative_to(root).as_posix()] = sha256(entry)
    for fold, seed in enumerate(protocol["split_seeds"]):
        write_json(experiment / f"inner_split_{fold}.json", inner_partition(splits, seed))
    write_json(experiment / "protocol.json", protocol)
    return protocol


def check_protocol(root, experiment):
    protocol = read_json(Path(experiment) / "protocol.json")
    for name, expected in protocol["source_hashes"].items():
        if sha256(Path(root) / name) != expected:
            raise ValueError(f"Frozen experiment source changed: {name}; use its source snapshot or a new experiment")
    if sha256(Path(protocol["data_directory"]) / "data_manifest.json") != protocol["data_manifest_sha256"]:
        raise ValueError("Prepared data manifest changed")
    return protocol


def fit_run(root, experiment, name, fold=0, seed=42, final_epochs=None, final_temperature=1.0):
    experiment = Path(experiment)
    protocol = check_protocol(root, experiment)
    candidate = protocol["candidates"][name]
    final = final_epochs is not None
    output = experiment / ("refit" if final else f"inner_{fold}") / f"{name}_s{seed}"
    if (output / "summary.json").exists():
        return read_json(output / "summary.json")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Incomplete run exists: {output}; preserve it and choose another experiment")
    partition = read_json(experiment / f"inner_split_{fold}.json")
    fit_groups = [group for group, role in partition.items() if final or role == "fit"]
    stop_groups = [group for group, role in partition.items() if role == "stop"]
    select_groups = [group for group, role in partition.items() if role == "select"]
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    data = DevelopmentData(protocol["data_directory"], candidate, fit_groups)
    fit_indices = data.indices_for(fit_groups)
    stop_indices = data.indices_for(stop_groups)
    model_config = {"architecture": candidate["architecture"], "input_size": len(data.feature_names),
                    "tokens": len(data.offsets), "num_classes": len(data.classes),
                    "width": candidate["width"], "layers": candidate["layers"], "dropout": candidate["dropout"],
                    "auxiliary": candidate["auxiliary_weight"] > 0}
    model = ImprovedForecaster(**model_config).to(data.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["learning_rate"],
                                  weight_decay=candidate["weight_decay"], fused=True)
    # Fixed cosine schedule in every phase; final fits do not depend on outer-validation feedback.
    epochs = final_epochs if final else protocol["max_epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=protocol["max_epochs"], eta_min=3e-5)
    generator = torch.Generator(device=data.device).manual_seed(seed)
    output.mkdir(parents=True)
    write_json(output / "run.json", {"candidate": candidate, "model": model_config, "seed": seed,
                                      "fold": fold, "final": final, "epochs_budget": epochs,
                                      "fit_groups": sorted(fit_groups), "test_evaluated": False,
                                      "protocol_sha256": sha256(experiment / "protocol.json"),
                                      "parameters": sum(p.numel() for p in model.parameters())})
    best, stale, history = float("inf"), 0, []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = fit_indices[torch.randperm(len(fit_indices), device=data.device, generator=generator)]
        total_loss = torch.zeros((), device=data.device)
        for batch in order.split(protocol["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(*data.windows(batch))
                loss = supervised_loss(logits, data.targets[batch], data.current[batch],
                                       smoothing=candidate["smoothing"], auxiliary_weight=candidate["auxiliary_weight"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training objective")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total_loss += loss.detach() * len(batch)
        scheduler.step()
        row = {"epoch": epoch, "training_objective": float(total_loss.item() / len(order))}
        if final:
            improved = epoch == epochs
        else:
            probabilities = predict(model, data, stop_indices)
            metrics = quick_metrics(probabilities, data.targets[stop_indices].cpu().numpy())
            row.update({f"stop_{key}": value for key, value in metrics.items() if not isinstance(value, list)})
            score = -metrics["mean_accuracy"] if candidate.get("checkpoint_metric") == "accuracy" else metrics["nll"]
            improved = score < best - protocol["min_delta"]
            if improved:
                best, stale = score, 0
            else:
                stale += 1
        if improved:
            atomic_checkpoint(output / "best.pt", {
                "model_state": model.state_dict(), "model_config": model_config, "candidate": candidate,
                "scaling": data.scaling, "feature_names": data.feature_names, "offsets": data.offsets.tolist(),
                "classes": data.classes, "horizons_s": [1, 2, 5], "epoch": epoch,
                "data_manifest_sha256": protocol["data_manifest_sha256"], "seed": seed})
        history.append(row)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        write_json(output / "progress.json", {"status": "running", "name": name, "seed": seed,
                                              "fold": fold, "final": final, **row})
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps({"name": name, "fold": fold, "seed": seed, "final": final, **row}), flush=True)
        if not final and stale >= protocol["patience"]:
            break
    saved = torch.load(output / "best.pt", map_location=data.device, weights_only=False)
    model.load_state_dict(saved["model_state"])
    temperature = final_temperature if final else 1.0
    if not final and candidate.get("calibrate", False):
        probabilities = predict(model, data, stop_indices)
        temperature = fit_temperature(probabilities, data.targets[stop_indices].cpu().numpy())
    saved["temperature"] = temperature
    atomic_checkpoint(output / "best.pt", saved)
    summary = {"name": name, "seed": seed, "fold": fold, "final": final,
               "temperature": temperature,
               "epochs_completed": len(history), "best_epoch": saved["epoch"],
               "elapsed_seconds": time.perf_counter() - started,
               "checkpoint_sha256": sha256(output / "best.pt"), "test_evaluated": False,
               "outer_validation_evaluated": False}
    if not final:
        indices = data.indices_for(select_groups)
        probabilities = predict(model, data, indices, temperature=temperature)
        summary["selection_metrics"] = quick_metrics(probabilities, data.targets[indices].cpu().numpy())
        np.save(output / "selection_probabilities.npy", probabilities)
        np.save(output / "selection_indices.npy", indices.cpu().numpy())
    write_json(output / "summary.json", summary)
    write_json(output / "progress.json", {"status": "completed", **summary})
    print(json.dumps(summary), flush=True)
    del model, optimizer, data
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def screen(root, experiment):
    protocol = check_protocol(root, experiment)
    for name in protocol["candidates"]:
        fit_run(root, experiment, name)
    summarize_search(experiment)


def summarize_search(experiment):
    rows = []
    for path in sorted(Path(experiment).glob("inner_*/*/summary.json")):
        result = read_json(path)
        metrics = result["selection_metrics"]
        rows.append({"name": result["name"], "fold": result["fold"], "seed": result["seed"],
                     "best_epoch": result["best_epoch"], "temperature": result.get("temperature", 1.0), "nll": metrics["nll"],
                     "mean_accuracy": metrics["mean_accuracy"], "macro_f1": metrics["mean_macro_f1"],
                     **{f"accuracy_{h}s": value for h, value in zip((1, 2, 5), metrics["accuracy"], strict=True)}})
    frame = pd.DataFrame(rows)
    frame.to_csv(Path(experiment) / "search_results.csv", index=False)
    return frame


def confirm(root, experiment):
    experiment = Path(experiment)
    check_protocol(root, experiment)
    results = summarize_search(experiment).query("fold == 0 and seed == 42")
    control = results.set_index("name").loc["control"]
    eligible = results[(results.nll <= control.nll + 0.02) & (results.macro_f1 >= control.macro_f1 - 0.01)]
    names = eligible.sort_values("mean_accuracy", ascending=False).head(2).name.tolist()
    names = list(dict.fromkeys(["control", *names]))
    write_json(experiment / "confirmation_candidates.json", {"names": names, "chosen_from": "inner fold 0 only"})
    for name in names:
        fit_run(root, experiment, name, fold=1)
    results = summarize_search(experiment)
    paired = results[results.name.isin(names)].groupby("name").agg(
        mean_accuracy=("mean_accuracy", "mean"), nll=("nll", "mean"), macro_f1=("macro_f1", "mean"))
    control = paired.loc["control"]
    eligible = paired[(paired.nll <= control.nll + 0.02) & (paired.macro_f1 >= control.macro_f1 - 0.01)]
    chosen = eligible.sort_values("mean_accuracy", ascending=False).index[0]
    epochs = int(np.rint(results[(results.name == chosen) & (results.seed == 42)].best_epoch.median()))
    temperature = float(results[(results.name == chosen) & (results.seed == 42)].temperature.median())
    selection = {"candidate": chosen, "epochs": epochs, "seeds": [42, 43, 44],
                 "temperature": temperature,
                 "ensemble": "Uniform arithmetic mean of the three seed probabilities",
                 "selected_using": "Two grouped partitions entirely within original training",
                 "outer_validation_evaluated": False, "test_evaluated": False,
                 "paired_internal_results": paired.reset_index().to_dict(orient="records")}
    if (experiment / "frozen_selection.json").exists():
        if read_json(experiment / "frozen_selection.json") != selection:
            raise ValueError("Selection is already frozen")
    else:
        write_json(experiment / "frozen_selection.json", selection)
    return selection


def refit(root, experiment):
    experiment = Path(experiment)
    selection = read_json(experiment / "frozen_selection.json")
    for seed in selection["seeds"]:
        fit_run(root, experiment, selection["candidate"], seed=seed, final_epochs=selection["epochs"],
                final_temperature=selection.get("temperature", 1.0))


def evaluate(root, experiment):
    experiment = Path(experiment)
    protocol = check_protocol(root, experiment)
    selection = read_json(experiment / "frozen_selection.json")
    output = experiment / "validation"
    if output.exists():
        raise ValueError("Validation comparison already exists; it must not drive further tuning")
    paths = [experiment / "refit" / f"{selection['candidate']}_s{seed}" for seed in selection["seeds"]]
    if not all((path / "summary.json").exists() for path in paths):
        raise ValueError("All frozen refits must complete before validation")
    output.mkdir()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    ensemble = []
    for path in paths:
        saved = torch.load(path / "best.pt", map_location="cuda", weights_only=False)
        data = DevelopmentData(protocol["data_directory"], saved["candidate"], [], scaling=saved["scaling"])
        indices = torch.tensor(np.flatnonzero(data.samples.split.eq("validation")), device=data.device)
        model = ImprovedForecaster(**saved["model_config"]).to(data.device)
        model.load_state_dict(saved["model_state"])
        probabilities = predict(model, data, indices, temperature=saved.get("temperature", 1.0))
        target = data.targets[indices].cpu().numpy()
        current = data.current[indices].cpu().numpy()
        samples = data.samples.iloc[indices.cpu().numpy()].reset_index(drop=True)
        save_evaluation(output / path.name, probabilities, target, current, samples, data.classes, path.name)
        np.save(output / path.name / "validation_probabilities.npy", probabilities)
        ensemble.append(probabilities)
        del model, data
        gc.collect()
        torch.cuda.empty_cache()
    probabilities = np.mean(ensemble, axis=0)
    classes = read_json(Path(protocol["data_directory"]) / "classes.json")
    save_evaluation(output / "ensemble", probabilities, target, current, samples, classes, "improved_ensemble")
    np.save(output / "ensemble/validation_probabilities.npy", probabilities)
    samples.to_parquet(output / "ensemble/validation_samples.parquet", index=False)
    result = {"selection_sha256": sha256(experiment / "frozen_selection.json"),
              "metrics": quick_metrics(probabilities, target), "test_evaluated": False}
    write_json(output / "evaluation.json", result)
    return result


def paired_group_intervals(new_probabilities, old_probabilities, targets, samples, draws=5000, seed=1729):
    """Resample whole matched scenario groups, keeping pilots/windows together."""
    if new_probabilities.shape != old_probabilities.shape:
        raise ValueError("Prediction arrays must cover the same origins")
    groups = sorted(samples.scenario_group.unique())
    rng = np.random.default_rng(seed)
    selections = rng.integers(0, len(groups), size=(draws, len(groups)))
    rows = []
    for h, seconds in enumerate((1, 2, 5)):
        valid = targets[:, h] >= 0
        new_correct = new_probabilities[:, h].argmax(-1) == targets[:, h]
        old_correct = old_probabilities[:, h].argmax(-1) == targets[:, h]
        deltas, counts = [], []
        for group in groups:
            mask = valid & samples.scenario_group.eq(group).to_numpy()
            deltas.append((new_correct[mask].astype(float) - old_correct[mask]).sum())
            counts.append(mask.sum())
        deltas, counts = np.asarray(deltas), np.asarray(counts)
        replicates = deltas[selections].sum(1) / counts[selections].sum(1)
        lower, upper = np.quantile(replicates, [0.025, 0.975])
        rows.append({"horizon_s": seconds, "accuracy_gain": float(deltas.sum() / counts.sum()),
                     "lower_95": float(lower), "upper_95": float(upper),
                     "scenario_groups": len(groups), "bootstrap_draws": draws})
    return pd.DataFrame(rows)


def verify_and_report(root, experiment):
    experiment = Path(experiment)
    protocol = check_protocol(root, experiment)
    selection = read_json(experiment / "frozen_selection.json")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    results = []
    for seed in selection["seeds"]:
        name = f"{selection['candidate']}_s{seed}"
        path = experiment / "refit" / name
        summary = read_json(path / "summary.json")
        if sha256(path / "best.pt") != summary["checkpoint_sha256"]:
            raise ValueError("Final checkpoint changed")
        saved = torch.load(path / "best.pt", map_location="cuda", weights_only=False)
        data = DevelopmentData(protocol["data_directory"], saved["candidate"], [], scaling=saved["scaling"])
        indices = torch.tensor(np.flatnonzero(data.samples.split.eq("validation")), device=data.device)
        model = ImprovedForecaster(**saved["model_config"]).to(data.device)
        model.load_state_dict(saved["model_state"])
        actual = predict(model, data, indices, temperature=saved.get("temperature", 1.0))
        expected = np.load(experiment / "validation" / name / "validation_probabilities.npy")
        np.testing.assert_array_equal(actual, expected)
        results.append({"member": name, "checkpoint_hash_matches": True, "probabilities_exact": True})
        target = data.targets[indices].cpu().numpy()
        samples = data.samples.iloc[indices.cpu().numpy()].reset_index(drop=True)
        del model, data
        gc.collect()
        torch.cuda.empty_cache()
    preserved = read_json(experiment / "preserved_initial_hashes.json")
    for name, expected in preserved.items():
        if sha256(Path(root) / name) != expected:
            raise ValueError(f"Original artifact changed: {name}")
    old = Path(root) / "artifacts/skill_forecasting/initial_v1"
    old_samples = pd.read_parquet(old / "transformer/validation_samples.parquet")
    pd.testing.assert_frame_equal(samples, old_samples)
    probabilities = np.load(experiment / "validation/ensemble/validation_probabilities.npy")
    old_probability = np.load(old / "transformer/validation_probabilities.npy")
    intervals = paired_group_intervals(probabilities, old_probability, target, samples)
    intervals.to_csv(experiment / "paired_scenario_bootstrap.csv", index=False)
    for name in ("transformer", "gru", "training_frequency"):
        destination = experiment / "validation" / f"original_{name}"
        destination.mkdir(exist_ok=True)
        for path in (old / name).glob("*.csv"):
            shutil.copy2(path, destination / path.name)
    comparison = build_report(experiment / "validation", protocol["data_directory"])
    comparison.to_csv(experiment / "comparison.csv", index=False)
    write_json(experiment / "verification.json", {"members": results, "original_artifacts_unchanged": True,
                                                  "matched_validation_origins": True, "test_evaluated": False})
    return {"members": results, "paired_intervals": intervals.to_dict(orient="records"),
            "original_artifacts_unchanged": True, "test_evaluated": False}


class ImprovedPredictor:
    def __init__(self, checkpoints, device="cpu"):
        self.device = torch.device(device)
        self.members = []
        for path in checkpoints:
            saved = torch.load(path, map_location=self.device, weights_only=False)
            model = ImprovedForecaster(**saved["model_config"]).to(self.device).eval()
            model.load_state_dict(saved["model_state"])
            self.members.append((model, saved))
        if not self.members:
            raise ValueError("Supply at least one checkpoint")
        self.classes = np.asarray(self.members[0][1]["classes"])
        if any(saved["classes"] != self.classes.tolist() for _, saved in self.members):
            raise ValueError("Ensemble class vocabularies differ")

    @torch.inference_mode()
    def predict(self, observations):
        predictions = []
        for model, saved in self.members:
            frames, valid = raw_history_features(observations, saved["offsets"], saved["candidate"]["features"])
            frames = (frames - np.asarray(saved["scaling"]["mean"], dtype=np.float32)) / np.asarray(
                saved["scaling"]["std"], dtype=np.float32)
            features = torch.tensor(frames[None], device=self.device)
            mask = torch.tensor(valid[None], device=self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                logits = model(features, mask)[0, :3]
            predictions.append((logits.float() / saved.get("temperature", 1.0)).softmax(-1).cpu().numpy())
        probability = np.mean(predictions, axis=0)
        order = np.argsort(-probability, axis=-1, kind="stable")
        return {"horizons_s": np.array([1, 2, 5]), "classes": self.classes.copy(),
                "probabilities": probability, "top3_skills": self.classes[order[:, :3]]}
