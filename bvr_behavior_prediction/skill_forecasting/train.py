"""Reproducible supervised training with validation-only checkpoint selection."""

import json
import random
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bvr_behavior_prediction.forecasting.data import read_json, sha256, write_json

from .data import verify_dataset
from .evaluation import evaluate_baseline, save_evaluation
from .models import SkillForecaster, class_weights, masked_cross_entropy


class SkillData:
    def __init__(self, directory, device):
        self.directory, self.device = Path(directory), torch.device(device)
        self.manifest = verify_dataset(self.directory)
        self.classes = self.manifest["classes"]
        self.scaling = read_json(self.directory / "scaling.json")
        self.samples = pd.read_parquet(self.directory / "samples.parquet")
        if not self.samples.split.isin(["train", "validation"]).all():
            raise ValueError("Development data must not contain test samples")
        self.features = torch.as_tensor(np.load(self.directory / "features.npy"), device=self.device)
        mean = torch.tensor(self.scaling["feature_mean"], device=self.device, dtype=torch.float32)
        std = torch.tensor(self.scaling["feature_std"], device=self.device, dtype=torch.float32)
        self.features.sub_(mean).div_(std)
        self.anchors = torch.as_tensor(np.load(self.directory / "anchors.npy"), device=self.device)
        self.targets = torch.as_tensor(np.load(self.directory / "targets.npy"), device=self.device)
        self.current = np.load(self.directory / "current.npy")  # Evaluation only.
        self.offsets = torch.arange(1 - self.manifest["history_steps"], 1, device=self.device)
        self.indices = {split: torch.tensor(np.flatnonzero(self.samples.split.eq(split)), device=self.device)
                        for split in ("train", "validation")}

    def windows(self, indices):
        return self.features[self.anchors[indices, None] + self.offsets[None]]

    def validation_arrays(self):
        mask = self.samples.split.eq("validation").to_numpy()
        return (self.targets[self.indices["validation"]].cpu().numpy(), self.current[mask],
                self.samples.loc[mask].reset_index(drop=True))


def atomic_checkpoint(path, data):
    temporary = Path(path).with_suffix(".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


@torch.inference_mode()
def predict_validation(model, data, batch_size):
    model.eval()
    probabilities = []
    for indices in data.indices["validation"].split(batch_size):
        with torch.autocast(device_type=data.device.type, dtype=torch.bfloat16,
                            enabled=data.device.type == "cuda"):
            logits = model(data.windows(indices))
        probabilities.append(logits.float().softmax(-1).cpu().numpy())
    return np.concatenate(probabilities)


def validation_nll(probabilities, targets):
    losses = []
    for h in range(3):
        mask = targets[:, h] >= 0
        if not mask.any():
            raise ValueError("No validation labels for horizon")
        values = probabilities[mask, h][np.arange(mask.sum()), targets[mask, h]]
        losses.append(-np.log(np.clip(values, 1e-12, 1)).mean())
    return float(np.mean(losses))


def source_snapshot(root, output):
    files = [root / "bvr_behavior_prediction/__init__.py", root / "scripts/train_skill_forecaster.py",
             root / "configs/training/skill_forecasting.json"]
    for package in ("forecasting", "skill_forecasting"):
        files += sorted((root / "bvr_behavior_prediction" / package).glob("*.py"))
    hashes = {}
    for path in files:
        relative = path.relative_to(root)
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        hashes[relative.as_posix()] = sha256(path)
    return hashes


def train_model(root, data_directory, output, architecture, config, resume=False):
    root, data_directory, output = Path(root), Path(data_directory), Path(output)
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError("Run output must be empty; use --resume or a new directory")
    if resume and not (output / "last.pt").exists():
        raise ValueError("No last checkpoint to resume")
    torch.set_num_threads(config["cpu_threads"])
    torch.set_float32_matmul_precision("high")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
    print(f"Loading skill data for {architecture} onto {device}", flush=True)
    data = SkillData(data_directory, device)
    model_config = {"architecture": architecture, "num_classes": len(data.classes),
                    "history_steps": data.manifest["history_steps"], "input_size": len(data.manifest["features"]),
                    **config["model"]}
    model = SkillForecaster(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"], fused=device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, min_lr=1e-5)
    generator = torch.Generator(device=device).manual_seed(config["seed"])
    weights = class_weights(data.targets[data.indices["train"]], len(data.classes),
                            mode=config["class_weighting"], max_weight=config["max_class_weight"])
    truth, current, samples = data.validation_arrays()
    dataset_hash = sha256(data_directory / "data_manifest.json")
    best_score, best_epoch, start_epoch, stale, history = float("inf"), 0, 1, 0, []
    run = {"model": model_config, "training": config, "data_directory": str(data_directory.resolve()),
           "data_manifest_sha256": dataset_hash, "classes": data.classes,
           "class_weights": None if weights is None else weights.cpu().tolist(),
           "selection": "Mean unweighted validation negative log-likelihood across horizons (lower is better)",
           "test_evaluated": False, "parameters": sum(p.numel() for p in model.parameters()),
           "created_utc": datetime.now(timezone.utc).isoformat(), "torch_version": str(torch.__version__),
           "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"}
    if resume:
        previous = read_json(output / "run.json")
        if (previous["training"] != config or previous["model"] != model_config
                or previous["data_manifest_sha256"] != dataset_hash):
            raise ValueError("Resume configuration or data differs from recorded run")
        for name, expected in previous["source_hashes"].items():
            if sha256(root / name) != expected:
                raise ValueError("Source changed; resume using the frozen source snapshot")
        run = previous
        saved = torch.load(output / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        generator.set_state(saved["generator_state"].cpu())
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
        best_score, best_epoch, stale = saved["best_score"], saved["best_epoch"], saved["stale"]
        start_epoch, history = saved["epoch"] + 1, saved["history"]
    else:
        output.mkdir(parents=True, exist_ok=True)
        run["source_hashes"] = source_snapshot(root, output)
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
        run["repository_commit"] = revision.stdout.strip() if revision.returncode == 0 else None
        write_json(output / "run.json", run)
    evaluate_baseline(data_directory, output.parent / "training_frequency")
    start_time = time.perf_counter()
    status = "completed"
    try:
        for epoch in range(start_epoch, config["epochs"] + 1):
            if stale >= config["patience"]:
                status = "early_stopped"
                break
            epoch_start = time.perf_counter()
            model.train()
            indices = data.indices["train"]
            order = indices[torch.randperm(len(indices), generator=generator, device=device)]
            loss_sum = torch.zeros((), device=device)
            for batch in order.split(config["batch_size"]):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    logits = model(data.windows(batch))
                    loss = masked_cross_entropy(logits, data.targets[batch], weights)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                loss_sum += loss.detach() * len(batch)
            probabilities = predict_validation(model, data, config["evaluation_batch_size"])
            score = validation_nll(probabilities, truth)
            if not np.isfinite(score):
                raise FloatingPointError("Nonfinite validation score")
            if score < best_score - config["min_delta"]:
                best_score, best_epoch, stale = score, epoch, 0
                atomic_checkpoint(output / "best.pt", {
                    "model_state": model.state_dict(), "model_config": model_config, "scaling": data.scaling,
                    "classes": data.classes, "horizons_s": data.manifest["horizons_s"],
                    "epoch": epoch, "score": score, "data_manifest_sha256": dataset_hash})
            else:
                stale += 1
            scheduler.step(score)
            row = {"epoch": epoch, "train_loss": float(loss_sum.item() / len(indices)),
                   "validation_nll": score, "best_nll": best_score, "best_epoch": best_epoch,
                   "learning_rate": optimizer.param_groups[0]["lr"],
                   "epoch_seconds": time.perf_counter() - epoch_start}
            history.append(row)
            pd.DataFrame(history).to_csv(output / "history.csv", index=False)
            atomic_checkpoint(output / "last.pt", {
                "model_state": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": epoch, "history": history,
                "best_score": best_score, "best_epoch": best_epoch, "stale": stale,
                "generator_state": generator.get_state(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []})
            write_json(output / "progress.json", {"status": "running", **row, "max_epochs": config["epochs"]})
            print(json.dumps({"model": architecture, **row}), flush=True)
    except (Exception, KeyboardInterrupt) as error:
        write_json(output / "progress.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                              "last_saved_epoch": len(history), "error": str(error)})
        raise
    best = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    probabilities = predict_validation(model, data, config["evaluation_batch_size"])
    save_evaluation(output, probabilities, truth, current, samples, data.classes, architecture)
    np.save(output / "validation_probabilities.npy", probabilities)
    samples.to_parquet(output / "validation_samples.parquet", index=False)
    final = {"status": status, "epochs_completed": len(history), "best_epoch": best["epoch"],
             "validation_nll": validation_nll(probabilities, truth), "test_evaluated": False,
             "elapsed_seconds": time.perf_counter() - start_time,
             "best_checkpoint_sha256": sha256(output / "best.pt")}
    write_json(output / "summary.json", final)
    write_json(output / "progress.json", final)
    print(json.dumps(final), flush=True)
    return final


def verify_run(data_directory, output):
    """Reload in a fresh process and reproduce every saved validation probability."""
    output = Path(output)
    run = read_json(output / "run.json")
    summary = read_json(output / "summary.json")
    if sha256(output / "best.pt") != summary["best_checkpoint_sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    if sha256(Path(data_directory) / "data_manifest.json") != run["data_manifest_sha256"]:
        raise ValueError("Prepared data manifest changed")
    torch.set_num_threads(run["training"]["cpu_threads"])
    torch.set_float32_matmul_precision("high")
    data = SkillData(data_directory, run["training"]["device"])
    saved = torch.load(output / "best.pt", map_location=data.device, weights_only=False)
    model = SkillForecaster(**saved["model_config"]).to(data.device)
    model.load_state_dict(saved["model_state"])
    actual = predict_validation(model, data, run["training"]["evaluation_batch_size"])
    expected = np.load(output / "validation_probabilities.npy")
    np.testing.assert_array_equal(actual, expected)
    result = {"checkpoint_checksum_matches": True, "validation_probabilities_exact": True,
              "validation_windows": len(actual), "test_evaluated": False}
    write_json(output / "verification.json", result)
    return result
