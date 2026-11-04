"""GPU-resident indexed histories, validation selection, and resumable supervised training."""

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

from .data import read_json, sha256, write_json
from .evaluation import motion_metrics, selection_score
from .models import MotionForecaster, masked_loss


class ForecastData:
    def __init__(self, directory, device):
        self.directory, self.device = Path(directory), device
        self.manifest = read_json(self.directory / "data_manifest.json")
        self.scaling = read_json(self.directory / "scaling.json")
        for name, expected in self.manifest["files"].items():
            if sha256(self.directory / name) != expected:
                raise ValueError(f"Prepared data changed: {name}")
        self.samples = pd.read_parquet(self.directory / "samples.parquet")
        self.features = torch.as_tensor(np.load(self.directory / "features.npy"), device=device)
        mean = torch.tensor(self.scaling["feature_mean"], device=device, dtype=torch.float32)
        std = torch.tensor(self.scaling["feature_std"], device=device, dtype=torch.float32)
        self.features.sub_(mean).div_(std)
        self.anchors = torch.as_tensor(np.load(self.directory / "anchors.npy"), device=device)
        self.residuals = torch.as_tensor(np.load(self.directory / "residuals.npy"), device=device)
        self.valid = torch.as_tensor(np.load(self.directory / "valid.npy"), device=device)
        self.scale = torch.tensor(self.scaling["residual_scale"], device=device, dtype=torch.float32)
        self.offsets = torch.arange(1 - self.manifest["history_steps"], 1, device=device)
        self.indices = {split: torch.tensor(np.flatnonzero(self.samples.split == split), device=device)
                        for split in ("train", "validation")}

    def windows(self, indices):
        return self.features[self.anchors[indices, None] + self.offsets[None]]

    def validation_arrays(self):
        indices = self.indices["validation"]
        return (self.residuals[indices].cpu().numpy(), self.valid[indices].cpu().numpy(),
                self.samples[self.samples.split == "validation"].reset_index(drop=True))


def atomic_checkpoint(path, data):
    temporary = Path(path).with_suffix(".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


@torch.inference_mode()
def predict_validation(model, data, batch_size):
    model.eval()
    predictions = []
    for indices in data.indices["validation"].split(batch_size):
        with torch.autocast(device_type=data.device.type, dtype=torch.bfloat16,
                            enabled=data.device.type == "cuda"):
            prediction = model(data.windows(indices))
        predictions.append((prediction.float() * data.scale).cpu().numpy())
    return np.concatenate(predictions)


def source_snapshot(root, output):
    files = sorted((root / "bvr_behavior_prediction/forecasting").glob("*.py"))
    files += [root / "bvr_behavior_prediction/__init__.py", root / "scripts/train_motion_forecaster.py",
              root / "configs/training/motion_forecasting.json"]
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
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(config["cpu_threads"])
    torch.set_float32_matmul_precision("high")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
    output.mkdir(parents=True, exist_ok=True)
    exists = (output / "run.json").exists()
    if exists and not resume:
        raise ValueError("Run exists; use --resume or a new output directory")
    if resume and not (output / "last.pt").exists():
        raise ValueError("No last checkpoint is available to resume")
    print(f"Loading prepared data for {architecture} onto {device}", flush=True)
    data = ForecastData(data_directory, device)
    model_config = {"architecture": architecture, "input_size": len(data.manifest["features"]),
                    "history_steps": data.manifest["history_steps"], **config["model"]}
    model = MotionForecaster(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"], fused=device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, min_lr=1e-5)
    truth, valid, samples = data.validation_arrays()
    cv = motion_metrics(np.zeros_like(truth), truth, valid, samples, "constant_velocity")
    ca = np.load(data_directory / "constant_acceleration.npy")[data.samples.split == "validation"]
    baselines = pd.concat([cv, motion_metrics(ca, truth, valid, samples, "constant_acceleration")])
    baselines.to_csv(output / "baseline_metrics.csv", index=False)
    dataset_hash = sha256(data_directory / "data_manifest.json")
    generator = torch.Generator(device=device).manual_seed(config["seed"])
    best_score, best_epoch, start_epoch, stale, history = 1.0, 0, 1, 0, []
    run = {"architecture": architecture, "model": model_config, "training": config,
           "data_directory": str(data_directory.resolve()), "data_manifest_sha256": dataset_hash,
           "corpus_manifest_sha256": data.manifest["corpus_manifest_sha256"],
           "selection": "Mean validation episode-macro error ratio to constant velocity across 3 horizons and position/velocity",
           "test_evaluated": False, "parameters": sum(p.numel() for p in model.parameters()),
           "created_utc": datetime.now(timezone.utc).isoformat(), "torch_version": str(torch.__version__)}
    if resume:
        previous = read_json(output / "run.json")
        if (previous["training"] != config or previous["model"] != model_config or
                previous["data_manifest_sha256"] != dataset_hash):
            raise ValueError("Resume configuration or dataset differs from the recorded run")
        for name, expected in previous["source_hashes"].items():
            if sha256(root / name) != expected:
                raise ValueError("Source changed; resume from the frozen source snapshot")
        run = previous
        saved = torch.load(output / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        generator.set_state(saved["generator_state"].cpu())
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng"]])
        best_score, best_epoch, stale = saved["best_score"], saved["best_epoch"], saved["stale"]
        start_epoch, history = saved["epoch"] + 1, saved["history"]
    else:
        run["source_hashes"] = source_snapshot(root, output)
        run["repository_commit"] = subprocess.check_output(
            ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        write_json(output / "run.json", run)
        atomic_checkpoint(output / "best.pt", {"model_state": model.state_dict(), "model_config": model_config,
                                               "scaling": data.scaling, "epoch": 0, "score": 1.0,
                                               "data_manifest_sha256": dataset_hash})
    start_time = time.perf_counter()
    status = "completed"
    try:
        for epoch in range(start_epoch, config["epochs"] + 1):
            if stale >= config["patience"]:
                status = "early_stopped"
                break
            epoch_start = time.perf_counter()
            model.train()
            train_indices = data.indices["train"]
            order = train_indices[torch.randperm(len(train_indices), generator=generator, device=device)]
            losses = torch.zeros((), device=device)
            for indices in order.split(config["batch_size"]):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    prediction = model(data.windows(indices))
                    loss = masked_loss(prediction, data.residuals[indices] / data.scale, data.valid[indices])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                losses += loss.detach() * len(indices)
            prediction = predict_validation(model, data, config["evaluation_batch_size"])
            metrics = motion_metrics(prediction, truth, valid, samples, architecture)
            score = selection_score(metrics, cv)
            if not np.isfinite(score):
                raise FloatingPointError("Nonfinite validation score")
            improved = score < best_score - config["min_delta"]
            if improved:
                best_score, best_epoch, stale = score, epoch, 0
                atomic_checkpoint(output / "best.pt", {"model_state": model.state_dict(), "model_config": model_config,
                                                       "scaling": data.scaling, "epoch": epoch, "score": score,
                                                       "data_manifest_sha256": dataset_hash})
            else:
                stale += 1
            scheduler.step(score)
            population = metrics[metrics.pilot_id == "population"]
            row = {"epoch": epoch, "train_loss": float(losses.item() / len(train_indices)),
                   "validation_score": score, "best_score": best_score, "best_epoch": best_epoch,
                   "learning_rate": optimizer.param_groups[0]["lr"],
                   "epoch_seconds": time.perf_counter() - epoch_start}
            for metric in population.itertuples():
                row[f"validation_{metric.quantity}_{int(metric.horizon_s)}s"] = metric.episode_mean_error
            history.append(row)
            pd.DataFrame(history).to_csv(output / "history.csv", index=False)
            atomic_checkpoint(output / "last.pt", {"model_state": model.state_dict(), "optimizer": optimizer.state_dict(),
                                                   "scheduler": scheduler.state_dict(), "epoch": epoch, "history": history,
                                                   "best_score": best_score, "best_epoch": best_epoch, "stale": stale,
                                                   "generator_state": generator.get_state(), "torch_rng": torch.get_rng_state(),
                                                   "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []})
            write_json(output / "progress.json", {"status": "running", **row, "max_epochs": config["epochs"],
                                                  "elapsed_seconds": time.perf_counter() - start_time})
            print(json.dumps({"model": architecture, **row}), flush=True)
    except KeyboardInterrupt:
        write_json(output / "progress.json", {"status": "interrupted", "last_saved_epoch": len(history),
                                              "best_epoch": best_epoch, "best_score": best_score})
        raise
    best = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    predictions = predict_validation(model, data, config["evaluation_batch_size"])
    metrics = motion_metrics(predictions, truth, valid, samples, architecture)
    metrics.to_csv(output / "validation_metrics.csv", index=False)
    np.savez_compressed(output / "validation_predictions.npz", residual_prediction=predictions,
                        residual_target=truth, valid=valid)
    samples.to_parquet(output / "validation_samples.parquet", index=False)
    final = {"status": status, "epochs_completed": len(history), "best_epoch": best_epoch,
             "validation_score": selection_score(metrics, cv), "test_evaluated": False,
             "elapsed_seconds": time.perf_counter() - start_time,
             "best_checkpoint_sha256": sha256(output / "best.pt")}
    write_json(output / "summary.json", final)
    write_json(output / "progress.json", final)
    print(json.dumps(final), flush=True)
    return final
