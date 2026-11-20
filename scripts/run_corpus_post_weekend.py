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
import signal
import sys
import time
import traceback
import zipfile
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path

from scripts.nn_duel_runtime import bootstrap_native_runtime

bootstrap_native_runtime()
# isort: off
import torch  # Torch must initialize before the other native libraries on Windows.
import numpy as np
# isort: on

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "bvr_sim_source")]

import scripts.generate_nn_duel_corpus as duel
from scripts.weekend_training_progress import EpochProgressReporter, metric_rows

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
# Only operational repairs from known notebook-31 revisions may upgrade
# an existing campaign. Hyperparameters, assets, parents and software stay pinned.
PERFORMANCE_FIX_BASE_COMMIT = '26b3446d839861ff68c3250a1dbcac34db0d2e34'
PERFORMANCE_FIX_ORIGINAL_COMMIT = "bc01bf41798d9858df739ec2e0bfefb4000fe3a9"
PERFORMANCE_FIX_BASE_HASHES = {
    # Known LF/CRLF versions from bc01bf4 and 26b3446; other source stays pinned.
    'scripts/run_varied_pilot_weekend.py': (
        'ebe16ca6eeb58b64a4753355ed2a13b76456e4d2ff340ea34104abb1e7ae927b',
        '078518d1ff5aa4506bd92383f67bc362673b2961d4ab82423dbe2094a5a5111d',
        '222025a69d59f1805cb152ba74f68433123cdf219752c19864b467979c391a4e',
        'b09e07393afaa14c09ad2c6c7b8bbd0bf31774f1f8d2ca0501af7d128036d3fa',
    ),
    'scripts/weekend_training_progress.py': (
        'd2e885ed5eaaf56fcc0fa2e22339ccd51ec0c830bf14ba5d1a270a2cb0e21e6c',
        '9ea08795c666c64289bc00d8a408defb0c26338934a943045ef78d4934497f81',
        '922d451cff3156a561d02ce2af7a596429c1e30804cfb5f7eb74a6f95abc77dc',
        'af11af629d0b838407fc249be0426d34053d9ab9251cb1936eaf1112c8a9a110',
    ),
    'bvr_behavior_prediction/rl/trainer.py': (
        '35ceffd912541a1a70d3b108977e50fd10d810b139de3951f8b9bdd51d5b78f2',
        '87e552c94dd91676df3e4086ccfb0006c7e0bdefc9106435941c0fdd95fb8c3b',
        'b46fe96f8f1881b78d84c91b3d34c9a2b0882dea0c87099d6fc90020275318a0',
        'dbad41f004088dd79222659b7ea6efb20168e13e70e601c70fd174398faa8e33',
    ),
    'bvr_behavior_prediction/rl/bvr_jsbsim_backend.py': (
        '0b69bcc1854ec0c468b10157e256a3fa0082652a6adca315130a387a2fc7c8af',
        'bb738123e87c442f32ccdef3eae9e6e7070c6b33f9be9dd32b4ea01549c251fe',
        'e2d1c46168ac54ba6af08120d1e2f8cd6889a9ef9084a31d9f42462812d60b33',
        '03d04271c97b632e2f555ca2505a5f07f9ea35ef3dedbe8eec95195b3b167a31',
    ),
    'bvr_behavior_prediction/rl/config.py': (
        '7c5161a82668c5190c7f7281f153750220192756cd91b400a679dbe4e371fdd3',
        '96b65acf5a38a936b6b28d52ddc64345dd31f49444826974906b667da9c4b1d1',
    ),
    'bvr_behavior_prediction/rl/pilot.py': (
        '6b89aea1b262befd27bcd01db69aa31805fd670dfa16983f2eea9b80720bba84',
        '95a45e59d279adb4160a0dd62fed8b59d4e382bd6692a36a09af574573909290',
    ),
    'bvr_behavior_prediction/rl/cuda_update.py': (
        'a4746f5a4c2e0dc024ba595f7e4ff17a24c5db94dc9638bf0cb6eb8ed48f05c3',
        'c2251a436ea03f274a9cb6038c2d608bf733680462fc29c21c68914dace9f26f',
    ),
    'bvr_behavior_prediction/rl/simulation_workers.py': (
        'f2d57b3276ec0a46a6a108613a218a60cb5a3b09a03592e42de69349a6e7f885',
        '23c8256cf96483c4a7b421c15ed64627c7ce533c8960f7d9bb77a1d2b3e37c2d',
    ),
    'bvr_sim_source/bvr_sim/bvr_env.py': (
        '9333c2ab18b033ad804736e095787afead1da2e73d0db607ac3837083f8e8a57',
        '2ce5e4d2b75f6e8b2a8b37f8d4e6c8789b05be66767f8b52115f2aef24c51661',
        '6cc8698f87e84eda9edda7fbdd8b0d29d05f2ee57ff701a91e45cdac83f6f78d',
        '3b89aa7598d3083d39baa69edc9347bc4cb8261f7e1b9a127fa876b6e330da6d',
    ),
    'bvr_sim_source/bvr_sim/src_py/simulator/aircraft/fdm/jsbsim_fdm.py': (
        'a8c0e76c02bee9257cf8a14d3a574d1e5231c159da06e1763c91ff0aa2b0a014',
        'ad248e52361a6dd39a6b00b32721ee490ee440771d48b783915052fae4a87bf5',
        '21bf0effc9ad7be4081ba8c5a09173a7353ba20249471daca392ab9e736db9a6',
        'b409797286559e96f31782c188e2ee9bde7798705d301d957b5bbff7cae39deb',
    ),
}


# The progress-reader repair also supports paired jobs from this branch tip.
# Keep these hashes separate from the historical serial acceleration repairs.
PROGRESS_FIX_BASE_COMMIT = "7b3916fcdcba935db97292267117f093c9b9833e"
PROGRESS_FIX_BASE_HASHES = {
    "scripts/run_varied_pilot_weekend.py": (
        "8ea42bfb12093323b8876f858a00698e15b5258bc9c4de51fecaca89c90d4b02",
        "eaca0324069a080f2946bdfc5b3f4fd47457e4592661d38d1b7b66cc720c6365",
    ),
    "scripts/weekend_training_progress.py": (
        "520ad10ecab6ab33d77ba2dae984a878d4094fabaf7bc4c43b8c0340b0353786",
        "0721cf752f22b284790a07e4475eb2d9bf0c21e0707427d177ca97ca5c220e2e",
    ),
}


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
            "paired_pilot_training.py",
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


def prepare_campaign(job, settings, accept_performance_fix=False):
    """Pin a new plan or record a narrowly scoped repair of an interrupted run."""
    plan = job / "campaign.json"
    # Preflight may upgrade provenance, so it must also exclude an active runner.
    with duel.campaign_lock(job):
        if not plan.exists():
            write(plan, settings)
            return
        previous = read(plan)
        if previous == settings:
            return
        error = (
            "Campaign settings/source/checkpoints/software changed; choose a new JOB directory. "
            "For known operational acceleration repairs, stop the old process and use "
            "--accept-performance-fix with otherwise identical settings."
        )
        if not accept_performance_fix:
            raise ValueError(error)
        if {k: v for k, v in previous.items() if k != "source_hashes"} != {
            k: v for k, v in settings.items() if k != "source_hashes"
        }:
            raise ValueError(error)
        before, after = previous["source_hashes"], settings["source_hashes"]
        changes = {name for name in before.keys() | after.keys()
                   if before.get(name) != after.get(name)}
        if not changes <= PERFORMANCE_FIX_BASE_HASHES.keys() or any(
            before.get(name) not in (
                *PERFORMANCE_FIX_BASE_HASHES[name], *PROGRESS_FIX_BASE_HASHES.get(name, ())
            ) or name not in after
            for name in changes
        ):
            raise ValueError("Performance repair cannot accept unrelated or unknown source changes")
        runner = "scripts/run_varied_pilot_weekend.py"
        if before.get(runner) in PROGRESS_FIX_BASE_HASHES[runner]:
            base_commit = PROGRESS_FIX_BASE_COMMIT
        else:
            base_commit = (PERFORMANCE_FIX_ORIGINAL_COMMIT
                           if before.get(runner) in PERFORMANCE_FIX_BASE_HASHES[runner][:2]
                           else PERFORMANCE_FIX_BASE_COMMIT)
        receipt = job / "performance_fix.json"
        # Commit provenance first. A crash between receipt and plan replacement
        # can be retried, and a later repair appends to the original history.
        history = read(receipt) if receipt.exists() else None
        if history is not None and history["updated_source_hashes"] == after:
            latest_previous = (history["upgrades"][-1]["previous_campaign"]
                               if history.get("upgrades") else history["original_campaign"])
            if latest_previous != previous:
                raise ValueError("An existing performance repair has different original provenance")
            write(plan, settings)
            return
        if history is not None and history["updated_source_hashes"] != before:
            raise ValueError("An existing performance repair has different original provenance")
        upgrades = list(history.get("upgrades", [])) if history else []
        if history and not upgrades:
            upgrades.append({
                "updated_utc": history["updated_utc"],
                "base_commit": history["base_commit"],
                "previous_campaign": history["original_campaign"],
                "updated_source_hashes": history["updated_source_hashes"],
                "changed_files": history["changed_files"],
            })
        upgrades.append({
            "updated_utc": now(), "base_commit": base_commit,
            "previous_campaign": previous, "updated_source_hashes": after,
            "changed_files": sorted(changes),
        })
        write(receipt, {
            "updated_utc": upgrades[-1]["updated_utc"],
            "base_commit": history["base_commit"] if history else base_commit,
            "original_campaign": history["original_campaign"] if history else previous,
            "updated_source_hashes": after,
            "changed_files": sorted(changes | set(history["changed_files"] if history else [])),
            "upgrades": upgrades,
            "note": "Operational acceleration and diagnostics repair; saved "
                    "model/optimizer/RNG/epochs and training budgets are retained.",
        })
        write(plan, settings)


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
        "minibatch_size",
        "simulator_transport",
        "target_kl",
    ):
        if settings.get(name) is not None:
            config[name] = settings[name]
    # Measure policy movement even when the inherited recipe has no KL limit.
    config["log_ppo_diagnostics"] = True
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


def restore_last_epoch_timing(metrics, recorded):
    """Recover post-checkpoint timings without accepting uncommitted metrics.

    A checkpoint cannot contain the duration of its own atomic write. The next
    checkpoint includes the previous epoch's finalized timings; only the last
    row needs recovery from JSONL. If a crash preceded that append, its compute
    timing remains available and checkpoint_timing_recorded stays zero.
    """
    if not metrics:
        return
    pending = metrics[-1]
    timing = {"epoch_seconds", "checkpoint_seconds", "checkpoint_timing_recorded"}
    for row in reversed(recorded):
        if row.get("epoch") == pending["epoch"]:
            if row.get("checkpoint_timing_recorded") == 1 and {
                k: v for k, v in row.items() if k not in timing
            } == {k: v for k, v in pending.items() if k not in timing}:
                metrics[-1] = row
            return


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
        restore_last_epoch_timing(metrics, metric_rows(trainer.diagnostics_path))
        trainer.diagnostics_path.write_text(
            "".join(json.dumps(m) + "\n" for m in metrics), encoding="utf8"
        )
        last_scores = trainer.weekend_evaluation_scores
        for epoch in range(start + 1, config.epochs + 1):
            started = time.perf_counter()
            if trainer.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(trainer.device)
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
            evaluation_profile = dict(trainer._last_collection_profile)
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
                "compute_seconds": 0.0,
                "checkpoint_seconds": 0.0,
                "checkpoint_timing_recorded": 0,
                **trainer._last_update_profile,
                **{f"training_{k}": v for k, v in collection_profile.items()},
                **{f"evaluation_{k}": v for k, v in evaluation_profile.items()},
            }
            row["update_seconds_per_minibatch"] = (
                update_seconds / row["update_minibatches"]
            )
            if trainer.device.type == "cuda":
                row.update(
                    cuda_allocated_mb=torch.cuda.memory_allocated(trainer.device) / 2**20,
                    cuda_reserved_mb=torch.cuda.memory_reserved(trainer.device) / 2**20,
                    cuda_peak_allocated_mb=torch.cuda.max_memory_allocated(trainer.device) / 2**20,
                )
            if not all(np.isfinite(v) for v in row.values()):
                raise ValueError("Nonfinite training metrics")
            metrics.append(row)
            report("saving checkpoint", epoch - 1, metrics[:-1])
            row["compute_seconds"] = time.perf_counter() - started
            row["epoch_seconds"] = row["compute_seconds"]
            checkpoint_started = time.perf_counter()
            save_state(trainer, epoch, metrics)
            if improved:
                save_best_weights(trainer)
            row.update(
                checkpoint_seconds=time.perf_counter() - checkpoint_started,
                epoch_seconds=time.perf_counter() - started,
                checkpoint_timing_recorded=1,
            )
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


def train_pair_member(job, stage, pilot, token=None):
    """A fresh interpreter owns this pilot's RNGs, CUDA context and simulators."""
    from paired_pilot_training import worker_budgets

    job = Path(job).resolve()
    settings = read(job / "campaign.json")
    settings["workers"] = worker_budgets(settings["workers"])[PILOTS.index(pilot) % 2]
    torch.set_num_threads(max(1, settings["torch_threads"] // 2))
    source = Path(settings["source"]) if stage == "simple" else job / "simple"
    records = {p["pilot_id"]: p for p in read(source / "pilots.json")["pilots"]}
    parent = source / records[pilot]["output_dir"]
    directory = job / stage / pilot

    def local_progress(path, value):
        write(directory / "progress.json", value)

    progress = EpochProgressReporter(job, settings, stage, pilot, local_progress)
    if token is not None:
        write(directory / "ready.json", {"token": token, "pid": os.getpid()})
        while not (job / "pair_release.json").exists() or read(
            job / "pair_release.json"
        ).get("token") != token:
            time.sleep(.05)
    execution = {"pid": os.getpid(), "started_at": time.time(),
                 "workers": settings["workers"], "stage": stage}
    write(directory / "execution.json", execution)
    try:
        result = train_one(parent, directory, stage_config(parent, directory, settings, stage),
                           stage, settings["device"], progress)
        progress("pilot complete", settings[f"{stage}_epochs"],
                 metric_rows(directory / "training_metrics.jsonl"))
        return result
    except BaseException:
        write(directory / "failure.json", {"updated_utc": now(),
                                            "traceback": traceback.format_exc()})
        raise
    finally:
        write(directory / "execution.json", {**execution, "finished_at": time.time()})


def train_population(job, stage, settings):
    if settings.get("pilot_schedule", "serial") == "serial":
        return _train_population_serial(job, stage, settings)
    from paired_pilot_training import run_pair

    output = job / stage
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for offset in range(0, len(PILOTS), 2):
        pair = PILOTS[offset:offset + 2]
        run_pair(job, stage, settings, pair, PILOTS, write)
        for pilot in pair:
            result = read(output / pilot / "manifest.json")
            if result["status"] != "completed":
                raise ValueError(f"Pair member did not finish: {stage}/{pilot}")
            load_trained_pilot(output / pilot)
            records.append(result)
        write(output / "pilots.json", {"pilots": records,
              "population_metadata": read(Path(settings["source"]) / "pilots.json")[
                  "population_metadata"]})
    return output


def _train_population_serial(job, stage, settings):
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
        if (job / "performance_fix.json").exists():
            z.write(job / "performance_fix.json", "provenance/performance_fix.json")
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
    if args.pilot_schedule == "paired":
        from paired_pilot_training import worker_budgets

        worker_budgets(args.workers)  # Fail before writing a plan or starting training.
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
    settings.pop("accept_performance_fix")
    settings.pop("pilot_worker")
    settings.pop("worker_stage")
    settings.pop("pair_token")
    for option in ("target_kl", "minibatch_size", "simulator_transport"):
        if settings[option] is None:
            settings.pop(option)  # Keep otherwise identical old campaign plans compatible.
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
    # prepare_campaign(job, settings, args.accept_performance_fix)
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
            # for stage in ("simple", "before_tactical", "tactical", "after_tactical", "corpus"):
            #     write(job / "status.json", {"stage": stage, "updated_utc": now()})
            #     if stage in ("simple", "tactical"):
            #         train_population(job, stage, settings)
            #     elif stage == "before_tactical":
            #         assess(job, job / "simple", stage, settings)
            #     elif stage == "after_tactical":
            #         assess(job, job / "tactical", stage, settings)
            #     else:
            collect_corpus(job, settings)
            write(job / "status.json", {"stage": "completed", "updated_utc": now()})
        except BaseException:
            write(job / "failure.json", {"updated_utc": now(), "traceback": traceback.format_exc()})
            raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--pilot-schedule", choices=("paired", "serial"), default="paired",
                   help="Train 1&2, then 3&4, etc.; workers is a shared total budget")
    p.add_argument("--pilot-worker", choices=PILOTS, help=argparse.SUPPRESS)
    p.add_argument("--worker-stage", choices=("simple", "tactical"), help=argparse.SUPPRESS)
    p.add_argument("--pair-token", help=argparse.SUPPRESS)
    p.add_argument(
        "--accept-performance-fix", action="store_true",
        help="After stopping the old process, accept only known operational acceleration repairs "
        "for the existing JOB and preserve its original campaign provenance",
    )
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
        "minibatch-size",
    ):
        p.add_argument("--" + option, type=int)
    p.add_argument(
        "--simulator-transport", choices=("auto", "shared_memory", "pipe"),
        help="Process observations use shared memory when available by default; "
        "pipe selects the reference transport",
    )
    p.add_argument(
        "--target-kl", type=float,
        help="Stop PPO passes once approximate KL exceeds 1.5 times this target; "
        "omit to inherit the saved pilot's setting (a changed recipe needs a new JOB)",
    )
    p.add_argument("--episode-duration-s", type=float, default=300)
    return p


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.pilot_worker:
        if arguments.worker_stage is None:
            raise ValueError("A pilot worker requires --worker-stage")

        def interrupted(signum, frame):
            raise KeyboardInterrupt("Pilot worker interrupted")

        signal.signal(signal.SIGTERM, interrupted)
        train_pair_member(arguments.job, arguments.worker_stage, arguments.pilot_worker,
                          arguments.pair_token)
        raise SystemExit(0)
    if arguments.target_kl is not None and (
        not np.isfinite(arguments.target_kl) or arguments.target_kl <= 0
    ):
        raise ValueError("target-kl must be finite and positive")
    if arguments.minibatch_size is not None and arguments.minibatch_size < 1:
        raise ValueError("minibatch-size must be positive")
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
