"""Restartable notebook-30 continuation, tactical adaptation and 91-input corpus.

Run in a fresh subprocess, including from Jupyter. Stage boundaries use selected
policy/critic weights and fresh Adam; interruptions restore the full optimizer.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
import traceback
import zipfile
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path

from nn_duel_runtime import bootstrap_native_runtime

bootstrap_native_runtime()
# isort: off
import torch  # Torch must initialize before the other native libraries on Windows.
import numpy as np
# isort: on

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "bvr_sim_source")]

import generate_nn_duel_corpus as duel
from weekend_training_progress import EpochProgressReporter, metric_rows

from bvr_behavior_prediction.rl.config import PilotTrainingConfig, RewardWeights
from bvr_behavior_prediction.rl.dogfight_reward import (
    DogfightWeights,
    hybrid_dogfight_reward_definition,
)
from bvr_behavior_prediction.rl.evaluation import summarize_test_episodes
from bvr_behavior_prediction.rl.multi_pilot import (
    initialize_pilot_from_checkpoint,
    load_trained_pilot,
)
from bvr_behavior_prediction.rl.scenarios import ScenarioSampler
from bvr_behavior_prediction.rl.trainer import PPOTrainer

read, write, sha256 = duel.read, duel.write, duel.sha256
PILOTS = [f"pilot_{i:03d}" for i in range(1, 11)]


def now():
    return datetime.now(timezone.utc).isoformat()


def reward_definition(saved):
    parameters = dict(saved["parameters"])
    weights = DogfightWeights(**parameters.pop("weights"))
    result = hybrid_dogfight_reward_definition(saved["name"], weights=weights, **parameters)
    if result.as_dict() != saved:
        raise ValueError("Source must use notebook 30's unchanged hybrid dogfight reward")
    return result


def source_inventory():
    """Pin code AND simulator assets for all stages, including later resumption."""
    paths = [
        ROOT / "scripts" / name
        for name in (
            "run_varied_pilot_weekend.py",
            "weekend_training_progress.py",
            "generate_nn_duel_corpus.py",
            "tactical_corpus_schema.py",
            "nn_duel_inference.py",
            "nn_duel_source.py",
            "nn_duel_runtime.py",
        )
    ]
    for directory in (ROOT / "bvr_behavior_prediction", ROOT / "bvr_sim_source/bvr_sim"):
        paths.extend(
            p
            for p in directory.rglob("*")
            if p.is_file()
            and not set(p.relative_to(directory).parts) & {"__pycache__", "build", "install"}
            and p.suffix not in {".pyc", ".pyo"}
        )
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(paths)}


def population_inventory(source):
    batch = read(source / "pilots.json")
    records = {p["pilot_id"]: p for p in batch["pilots"]}
    observation = batch.get("population_metadata", {}).get("observation", {})
    if (
        set(records) != set(PILOTS)
        or observation.get("schema") != "compact_hostile_missiles_energy_v1"
        or observation.get("frame_size") != 91
    ):
        raise ValueError("Select the completed ten-pilot, 91-feature notebook-30 population")
    inventory = {"pilots.json": sha256(source / "pilots.json")}
    for pid in PILOTS:
        record = records[pid]
        directory = source / record["output_dir"]
        metadata = read(directory / "config.json")
        cfg = metadata["training"]
        if (
            record["status"] != "completed"
            or metadata["observation_size"] != 91
            or cfg["sample_interval_s"] != 0.1
            or cfg["history_duration_s"] != 2
            or cfg["planning_horizon_s"] != 1
            or not cfg["selected_skill_parameters"]
        ):
            raise ValueError(f"Incomplete or incompatible source pilot: {pid}")
        if metadata["reward"] != record["reward"]:
            raise ValueError(f"Source reward metadata differs: {pid}")
        reward_definition(metadata["reward"])
        for filename in ("config.json", "manifest.json", "best_model.pt"):
            path = directory / filename
            inventory[path.relative_to(source).as_posix()] = sha256(path)
        checksum = sha256(directory / "best_model.pt")
        if (
            checksum != record["checkpoint_sha256"]
            or checksum != read(directory / "manifest.json")["checkpoint_sha256"]
        ):
            raise ValueError(f"Source checkpoint hash differs: {pid}")
        # Validate the versioned skill catalogue and every saved architecture now.
        del_model = load_trained_pilot(directory)
        del del_model
    return inventory


def discover_source():
    candidates = sorted(
        (ROOT / "artifacts/rl_pilot_simplified/varied_transformers").glob("*/pilots.json"),
        key=lambda p: p.parent.name,
        reverse=True,
    )
    for file in candidates:
        batch = read(file)
        pilots = batch.get("pilots", [])
        if (
            len(pilots) == 10
            and all(p.get("status") == "completed" for p in pilots)
            and batch.get("population_metadata", {}).get("observation", {}).get("frame_size") == 91
        ):
            return file.parent.resolve()
    raise FileNotFoundError(
        "No completed notebook-30 population found. Set BVR_WEEKEND_SOURCE to the "
        "directory containing its pilots.json and ten pilot directories."
    )


def make_environment(config, opponent):
    from bvr_behavior_prediction.rl.bvr_jsbsim_backend import BVRJSBSimBackend
    from bvr_behavior_prediction.rl.environment import BluePilotEnvironment

    def backend(scenario, recording_path=None):
        return BVRJSBSimBackend(
            scenario,
            recording_path=recording_path,
            log_dir=config.output_dir / "simulator_logs",
            opponent_type=opponent,
            include_hostile_missiles=True,
            episode_duration_s=config.episode_duration_s,
        )

    def factory(scenario, recording_path=None):
        return BluePilotEnvironment(
            backend,
            scenario,
            config.planning_horizon_s,
            config.episode_duration_s,
            recording_path,
            config.history_duration_s,
            config.sample_interval_s,
        )

    return factory


def stage_config(source, output, settings, stage):
    metadata = read(source / "config.json")
    saved = metadata["training"]
    known = {f.name for f in fields(PilotTrainingConfig)}
    config = {k: v for k, v in saved.items() if k in known}
    config["reward"] = RewardWeights(**config["reward"])
    if settings.get("evaluation_interval") not in (None, 1):
        raise ValueError("Notebook 31 validates every epoch; evaluation_interval must be 1")
    # Per-pilot architecture, PPO coefficients and reward scales come from disk.
    config.update(
        output_dir=output,
        epochs=settings[f"{stage}_epochs"],
        seed=metadata.get("continuation_origin_seed", saved["seed"])
        + (1_000_000 if stage == "simple" else 2_000_000),
        evaluation_seed=400_000_000 if stage == "simple" else 500_000_000,
        simulator_workers=settings["workers"],
        simulator_executor="process",
        test_scenarios_per_pilot=0,
        test_seed=None,
        episode_duration_s=300.0,
        evaluation_interval=1,
    )
    # Explicit small-run overrides; absent values preserve notebook 30's settings.
    for name in (
        "episode_duration_s",
        "scenarios_per_epoch",
        "evaluation_scenarios_per_epoch",
        "update_epochs",
    ):
        if settings.get(name) is not None:
            config[name] = settings[name]
    return PilotTrainingConfig(**config)


def save_state(trainer, epoch, metrics):
    state = {
        "epoch": epoch,
        "pilot": trainer.pilot.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "scaler": trainer.grad_scaler.state_dict(),
        "best_score": trainer.best_score,
        "best_pilot": trainer.best_pilot,
        "best_epoch": trainer.weekend_best_epoch,
        "metrics": metrics,
        "training_batch_index": trainer._training_batch_index,
        "evaluation_scores": trainer.weekend_evaluation_scores,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if trainer.device.type == "cuda" else None,
    }
    path = trainer.output / "resume.pt"
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    duel.durable_replace(temporary, path)


def save_best_weights(trainer):
    temporary = trainer.output / "best_model.pt.tmp"
    torch.save(trainer.best_pilot, temporary)
    duel.durable_replace(temporary, trainer.output / "best_model.pt")


def train_one(source, output, config, opponent, device, progress=None):
    """Use the existing PPO implementation; commit a full state each epoch."""
    metadata = read(source / "config.json")
    reward = reward_definition(metadata["reward"])
    trainer = PPOTrainer(make_environment(config, opponent), 91, config, device, reward)
    trainer._training_active = True  # Reuse CPU simulation workers across epochs.
    trainer.score_only_evaluation = True
    report = progress or (lambda phase, epoch, metrics: None)
    try:
        metadata.setdefault("continuation_origin_seed", metadata["training"]["seed"])
        metadata.update(training=config.as_dict(), torch_version=str(torch.__version__))
        write(output / "config.json", metadata)
        resume = output / "resume.pt"
        if resume.exists():
            # Trusted local training state includes Python/NumPy RNG objects.
            state = torch.load(resume, map_location=trainer.device, weights_only=False)
            trainer.pilot.load_state_dict(state["pilot"])
            trainer.optimizer.load_state_dict(state["optimizer"])
            trainer.grad_scaler.load_state_dict(state["scaler"])
            trainer.best_score = state["best_score"]
            trainer.best_pilot = {k: v.cpu() for k, v in state["best_pilot"].items()}
            trainer.weekend_best_epoch = state.get("best_epoch", 0)
            trainer._training_batch_index = state["training_batch_index"]
            random.setstate(state["python_rng"])
            np.random.set_state(state["numpy_rng"])
            torch.set_rng_state(state["torch_rng"].cpu())
            if trainer.device.type == "cuda":
                torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
            start, metrics = state["epoch"], state["metrics"]
            trainer.weekend_evaluation_scores = state["evaluation_scores"]
            del state  # Release duplicate loaded model tensors, especially on CUDA.
            # Repair a selected-weight file left stale by a crash after resume.pt.
            save_best_weights(trainer)
        else:
            report("initial validation", 0, [])
            initialize_pilot_from_checkpoint(trainer, source)
            trainer.best_pilot = {
                k: v.detach().cpu().clone() for k, v in trainer.pilot.state_dict().items()
            }
            start, metrics = 0, []
            trainer.weekend_evaluation_scores = None
            trainer.weekend_best_epoch = 0
            save_state(trainer, start, metrics)
        print(
            json.dumps(
                {
                    "pilot": output.name,
                    "opponent": opponent,
                    "resumed_epoch": start,
                    "target_epochs": config.epochs,
                }
            ),
            flush=True,
        )
        # Repair derived files if a previous process died after committing resume.pt.
        trainer.diagnostics_path.write_text(
            "".join(json.dumps(m) + "\n" for m in metrics), encoding="utf8"
        )
        last_scores = trainer.weekend_evaluation_scores
        for epoch in range(start + 1, config.epochs + 1):
            started = time.perf_counter()
            report("collecting flights", epoch - 1, metrics)
            episodes = trainer._training_episodes()
            rollout_seconds = time.perf_counter() - started
            collection_profile = dict(trainer._last_collection_profile)
            report("PPO update", epoch - 1, metrics)
            update_start = time.perf_counter()
            loss = trainer._update([e[0] for e in episodes])
            update_seconds = time.perf_counter() - update_start
            training_mean_return = float(np.mean([e[1] for e in episodes]))
            del episodes  # Training histories are no longer needed during validation.
            report("validation flights", epoch - 1, metrics)
            evaluation_start = time.perf_counter()
            last_scores = trainer._evaluate()
            evaluation_seconds = time.perf_counter() - evaluation_start
            trainer.weekend_evaluation_scores = last_scores
            score = float(np.mean(last_scores))
            if not np.isfinite(last_scores).all():
                raise ValueError("Nonfinite validation returns")
            improved = score > trainer.best_score
            if improved:
                trainer.best_score = score
                trainer.weekend_best_epoch = epoch
                trainer.best_pilot = {
                    k: v.detach().cpu().clone() for k, v in trainer.pilot.state_dict().items()
                }
            row = {
                "epoch": epoch,
                "loss": float(loss),
                "training_mean_return": training_mean_return,
                "mean_return": float(np.mean(last_scores)),
                "best_return": trainer.best_score,
                "evaluation_performed": 1,
                "best_model_improved": int(improved),
                "best_epoch": trainer.weekend_best_epoch,
                "rollout_seconds": rollout_seconds,
                "update_seconds": update_seconds,
                "evaluation_seconds": evaluation_seconds,
                "epoch_seconds": time.perf_counter() - started,
                **trainer._last_update_profile,
                **{f"training_{k}": v for k, v in collection_profile.items()},
            }
            if not all(np.isfinite(v) for v in row.values()):
                raise ValueError("Nonfinite training metrics")
            metrics.append(row)
            report("saving checkpoint", epoch - 1, metrics[:-1])
            save_state(trainer, epoch, metrics)
            if improved:
                save_best_weights(trainer)
            with trainer.diagnostics_path.open("a", encoding="utf8") as stream:
                stream.write(json.dumps(row) + "\n")
            report("epoch complete", epoch, metrics)
            print(json.dumps({"pilot": output.name, "opponent": opponent, **row}), flush=True)
        manifest = {
            "pilot_id": output.name,
            "status": "completed",
            "seed": config.seed,
            "output_dir": output.name,
            "checkpoint_sha256": sha256(output / "best_model.pt"),
            "reward": reward.as_dict(),
            "selection_return": trainer.best_score,
            "best_epoch": trainer.weekend_best_epoch,
            "additional_epochs": config.epochs,
            "opponent_type": opponent,
            "parent_run": str(source),
            "parent_checkpoint_sha256": sha256(source / "best_model.pt"),
            "training_overrides": {
                k: getattr(config, k)
                for k in (
                    "hidden_size",
                    "transformer_heads",
                    "transformer_layers",
                    "value_coefficient",
                    "entropy_coefficient",
                )
            },
        }
        write(output / "manifest.json", manifest)
        return manifest
    finally:
        trainer.close()


def train_population(job, stage, settings):
    source = Path(settings["source"]) if stage == "simple" else job / "simple"
    source_records = {p["pilot_id"]: p for p in read(source / "pilots.json")["pilots"]}
    output = job / stage
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for pid in PILOTS:
        parent = source / source_records[pid]["output_dir"]
        directory = output / pid
        config = stage_config(parent, directory, settings, stage)
        progress = EpochProgressReporter(job, settings, stage, pid, write)
        if (directory / "manifest.json").exists():
            result = read(directory / "manifest.json")
            load_trained_pilot(directory)
            progress(
                "pilot complete", config.epochs, metric_rows(directory / "training_metrics.jsonl")
            )
        else:
            result = train_one(
                parent, directory, config, stage, settings["device"], progress=progress
            )
        records.append(result)
        write(
            output / "pilots.json",
            {
                "pilots": records,
                "population_metadata": read(Path(settings["source"]) / "pilots.json")[
                    "population_metadata"
                ],
            },
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return output


def assess(job, population, label, settings):
    """Shared tactical assessment, separate from validation/checkpoint selection."""
    suite = ScenarioSampler(600_000_000).sample_batch(settings["assessment_scenarios"])
    seeds = tuple(range(600_000_000, 600_000_000 + len(suite)))
    for pid in PILOTS:
        target = job / label / pid / "evaluation.json"
        if target.exists():
            continue
        source = population / pid
        config = stage_config(source, target.parent, settings, "tactical")
        trainer = PPOTrainer(
            make_environment(config, "tactical"),
            91,
            config,
            settings["device"],
            reward_definition(read(source / "config.json")["reward"]),
        )
        try:
            trainer.pilot.load_state_dict(load_trained_pilot(source).state_dict())
            episodes = trainer.evaluate_scenarios(suite, seeds)
            summary, records = summarize_test_episodes(
                episodes, suite, seeds, trainer.pilot.skill_names
            )
            summary["role"] = (
                "Matched tactical assessment; never used for training or checkpoint selection"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            write(
                target,
                {
                    "opponent_type": "tactical",
                    "checkpoint_sha256": sha256(source / "best_model.pt"),
                    "summary": summary,
                    "episodes": records,
                },
            )
            print(json.dumps({"assessment": label, "pilot": pid, **summary}), flush=True)
        finally:
            trainer.close()


def freeze_population(job, settings):
    destination = job / "frozen"
    destination.mkdir(exist_ok=True)
    archive = destination / "reproduction.zip"
    if archive.exists() and (destination / "manifest.json").exists():
        if (
            sha256(archive)
            != read(destination / "manifest.json")["files"]["reproduction.zip"]["sha256"]
        ):
            raise ValueError("Frozen archive changed")
        return archive
    temp = archive.with_suffix(".tmp")
    if any(sha256(ROOT / p) != checksum for p, checksum in settings["source_hashes"].items()):
        raise ValueError(
            "Source changed while training; restore the campaign's recorded version before resuming"
        )
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as z:
        source_hashes = settings["source_hashes"]
        for relative in source_hashes:
            z.write(ROOT / relative, f"source/{relative}")
        for pid in PILOTS:
            for filename in ("config.json", "manifest.json", "best_model.pt"):
                z.write(job / "tactical" / pid / filename, f"training/{pid}/{filename}")
        z.writestr(
            "experiment.json",
            json.dumps(
                {
                    "population_kind": "varied_transformers_v1",
                    "source_experiment": str(job),
                    "source_hashes": source_hashes,
                    "campaign": settings,
                },
                indent=2,
            ),
        )
    duel.durable_replace(temp, archive)
    write(
        destination / "manifest.json", {"files": {"reproduction.zip": {"sha256": sha256(archive)}}}
    )
    return archive


def collect_corpus(job, settings):
    archive = freeze_population(job, settings)
    work, output = job / "collection", Path(settings["output"])
    backend = settings["inference_backend"]
    candidates = (
        [backend]
        if backend != "auto"
        else (["local_cpu", "central_cuda"] if torch.cuda.is_available() else ["local_cpu"])
    )
    selection = job / "collection_backend.json"
    if selection.exists():
        backend = read(selection)["backend"]
    elif len(candidates) == 1:
        backend = candidates[0]
        write(selection, {"backend": backend})
    else:
        timings = []
        for candidate in candidates:
            bench_work = job / f"benchmark_{candidate}"
            config = duel.prepare(
                bench_work,
                flights=settings["benchmark_flights"],
                seed=710_000_000,
                seconds=settings["flight_seconds"],
                inference_backend=candidate,
                population_archive=archive,
            )
            try:
                duel.collect(bench_work, config, workers=settings["workers"])
                progress = read(bench_work / "progress.json")
                timings.append(
                    {"backend": candidate, "seconds": progress["cumulative_wall_seconds"]}
                )
            except Exception:  # noqa: BLE001 -- record failed candidate and try the other backend
                write(bench_work / "benchmark_failure.json", {"traceback": traceback.format_exc()})
        if not timings:
            raise RuntimeError("All inference benchmarks failed; inspect benchmark_failure.json")
        backend = min(timings, key=lambda x: x["seconds"])["backend"]
        write(selection, {"backend": backend, "timings": timings})
    config = duel.prepare(
        work,
        flights=settings["flights"],
        seed=settings["corpus_seed"],
        seconds=settings["flight_seconds"],
        inference_backend=backend,
        population_archive=archive,
    )
    # Keep simulator/inference processes warm across the campaign. Each flight
    # commits its own checksummed files before its completion marker. Package
    # once to avoid repeatedly scanning an ever-growing corpus between batches.
    retries = 0
    while any(duel.completed(work, c, verify=False) is None for c in config["cases"]):
        try:
            duel.collect(work, config, workers=settings["workers"])
            retries = 0
        except RuntimeError:
            retries += 1
            if retries > 2:
                raise
            print(f"Retrying saved pending flights ({retries}/2)", flush=True)
    duel.package(work, output, config, shard_flights=settings["shard_flights"])
    duel.verify(output, require_complete=True)


def check_seeds(source, settings):
    reserved, planned = [], []
    for pid in PILOTS:
        batch = read(source / "pilots.json")
        record = next(p for p in batch["pilots"] if p["pilot_id"] == pid)
        parent = source / record["output_dir"]
        saved = read(parent / "config.json")["training"]
        reserved.extend(
            [
                (
                    saved["seed"],
                    saved["scenarios_per_epoch"]
                    * (saved["epochs"] if saved.get("resample_training_scenarios") else 1),
                ),
                (
                    saved["seed"]
                    if saved.get("evaluation_seed") is None
                    else saved["evaluation_seed"],
                    saved["evaluation_scenarios_per_epoch"],
                ),
            ]
        )
        if saved.get("test_seed") is not None:
            reserved.append((saved["test_seed"], saved["test_scenarios_per_pilot"]))
        for stage in ("simple", "tactical"):
            config = stage_config(parent, source, settings, stage)
            planned.append((stage + " training", config.seed, config.training_seed_count))
            planned.append(
                (
                    stage + " validation",
                    config.resolved_evaluation_seed,
                    config.evaluation_scenarios_per_epoch,
                )
            )
    planned.extend(
        [
            ("assessment", 600_000_000, settings["assessment_scenarios"]),
            ("benchmark", 710_000_000, settings["benchmark_flights"]),
            ("corpus", settings["corpus_seed"], settings["flights"]),
        ]
    )
    for i, (name, start, size) in enumerate(planned):
        if not 0 <= start < start + size <= 2**32:
            raise ValueError(f"Invalid seed range: {name}")
        for other, lo, count in planned[:i]:
            # Pilots within one stage intentionally share conditions.
            if other != name and max(start, lo) < min(start + size, lo + count):
                raise ValueError(f"Seed ranges overlap: {name}, {other}")
        if any(max(start, lo) < min(start + size, lo + count) for lo, count in reserved):
            raise ValueError(f"Seed range overlaps source training/validation/test: {name}")


def run(args):
    job = args.job.resolve()
    plan = job / "campaign.json"
    source = (
        args.source.resolve()
        if args.source
        else Path(read(plan)["source"])
        if plan.exists()
        else discover_source()
    )
    job.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.torch_threads)
    settings = vars(args).copy()
    settings.pop("job")
    settings.pop("preflight_only")
    settings.update(
        source=str(source),
        output=str(args.output.resolve()),
        source_hashes=source_inventory(),
        parent_hashes=population_inventory(source),
        software={
            name: duel.importlib.metadata.version(name)
            for name in ("torch", "numpy", "pandas", "pyarrow", "jsbsim", "gymnasium", "loky")
        },
        python=sys.version.split()[0],
    )
    check_seeds(source, settings)
    if plan.exists() and read(plan) != settings:
        raise ValueError(
            "Campaign settings/source/checkpoints/software changed; choose a new JOB directory"
        )
    if not plan.exists():
        write(plan, settings)
    if (args.output / "config.json").exists() and read(args.output / "config.json")[
        "source_experiment"
    ] != str(job):
        raise ValueError(
            "Output already belongs to another campaign; choose a new output directory"
        )
    torch.empty(1, device=args.device)  # Fail on an unavailable requested device before training.
    if args.inference_backend == "central_cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA inference requested but unavailable; choose auto or local_cpu")
    if args.preflight_only:
        from nn_duel_runtime import run_preflight

        run_preflight(job / "logs")
        roster = []
        for record in read(source / "pilots.json")["pilots"]:
            metadata = read(source / record["output_dir"] / "config.json")
            cfg = metadata["training"]
            roster.append(
                {
                    "pilot": record["pilot_id"],
                    **{
                        k: cfg[k]
                        for k in (
                            "hidden_size",
                            "transformer_heads",
                            "transformer_layers",
                            "value_coefficient",
                            "entropy_coefficient",
                        )
                    },
                    "reward_weights": metadata["reward"]["parameters"]["weights"],
                }
            )
        print(
            json.dumps(
                {
                    "source": str(source),
                    "job": str(job),
                    "output": str(args.output.resolve()),
                    "additional_epochs_per_pilot": {
                        "simple": args.simple_epochs,
                        "tactical": args.tactical_epochs,
                    },
                    "flights": args.flights,
                    "device": args.device,
                    "workers": args.workers,
                    "roster": roster,
                },
                indent=2,
            ),
            flush=True,
        )
        return
    # A local OS lock is released automatically after a process/kernel failure.
    with duel.campaign_lock(job):
        try:
            for stage in ("simple", "before_tactical", "tactical", "after_tactical", "corpus"):
                write(job / "status.json", {"stage": stage, "updated_utc": now()})
                if stage in ("simple", "tactical"):
                    train_population(job, stage, settings)
                elif stage == "before_tactical":
                    assess(job, job / "simple", stage, settings)
                elif stage == "after_tactical":
                    assess(job, job / "tactical", stage, settings)
                else:
                    collect_corpus(job, settings)
            write(job / "status.json", {"stage": "completed", "updated_utc": now()})
        except BaseException:
            write(job / "failure.json", {"updated_utc": now(), "traceback": traceback.format_exc()})
            raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--simple-epochs", type=int, default=200)
    p.add_argument("--tactical-epochs", type=int, default=300)
    p.add_argument("--assessment-scenarios", type=int, default=300)
    p.add_argument("--workers", type=int, default=max(1, min(32, int((os.cpu_count() or 1) * 0.8))))
    p.add_argument("--torch-threads", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--flights", type=int, default=10_000)
    p.add_argument("--flight-seconds", type=float, default=300)
    p.add_argument("--corpus-seed", type=int, default=800_000_000)
    p.add_argument("--shard-flights", type=int, default=100)
    p.add_argument("--benchmark-flights", type=int, default=32)
    p.add_argument(
        "--inference-backend",
        choices=("auto", "local_cpu", "central_cpu", "central_cuda"),
        default="auto",
    )
    p.add_argument(
        "--evaluation-interval",
        type=int,
        choices=(1,),
        default=1,
        help="Notebook 31 validates after every epoch",
    )
    for option in (
        "scenarios-per-epoch",
        "evaluation-scenarios-per-epoch",
        "update-epochs",
    ):
        p.add_argument("--" + option, type=int)
    p.add_argument("--episode-duration-s", type=float, default=300)
    return p


if __name__ == "__main__":
    arguments = parser().parse_args()
    if (
        min(
            arguments.workers,
            arguments.torch_threads,
            arguments.flights,
            arguments.shard_flights,
            arguments.simple_epochs,
            arguments.tactical_epochs,
            arguments.benchmark_flights,
        )
        < 1
        or arguments.assessment_scenarios < 3
    ):
        raise ValueError("Positive budgets and at least three assessment scenarios required")
    if not (
        0 < arguments.flight_seconds <= 300
        and np.isclose(arguments.flight_seconds / 0.1, round(arguments.flight_seconds / 0.1))
    ):
        raise ValueError("Flight duration must be a multiple of 0.1 seconds, at most 300")
    run(arguments)
