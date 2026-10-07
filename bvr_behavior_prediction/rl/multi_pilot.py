"""Train independent PPO pilots with explicit objectives and portable run records."""

import gc
import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path, PureWindowsPath

from .config import PilotTrainingConfig
from .reward import RewardDefinition, combat_reward_definition


@dataclass(frozen=True)
class PilotSpec:
    pilot_id: str
    seed: int
    reward: RewardDefinition = field(default_factory=combat_reward_definition)

    def __post_init__(self):
        if (
            not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", self.pilot_id)
            or PureWindowsPath(self.pilot_id).is_reserved()
        ):
            raise ValueError("pilot_id must be a portable name using letters, digits, _ or -")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("pilot seed must be an integer in [0, 2**32)")
        if not isinstance(self.reward, RewardDefinition):
            raise TypeError("reward must be a RewardDefinition")


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf8")
    temporary.replace(path)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_pilots(
    pilots,
    config: PilotTrainingConfig,
    env_factory_builder,
    observation_size: int,
    device=None,
    benchmark_reward=None,
):
    """Train fresh models sequentially; return completed pilot records.

    ``env_factory_builder(run_config)`` returns the usual
    ``env_factory(scenario, recording_path)``. Each pilot gets independent RNG
    initialization, optimizer, reward instances, directories, and MLflow run.
    The output root must be new. Interrupted batches retain completed runs and
    a status manifest; this function deliberately does not resume optimizers.

    Checkpoint selection uses each pilot's own reward. The selected policies
    are then scored with one common reward on identical validation scenarios.
    This validation set is used for selection; it is not an untouched test set.
    """
    import mlflow
    import numpy as np
    import torch

    from .trainer import PPOTrainer

    pilots = list(pilots)
    if not pilots or not all(isinstance(pilot, PilotSpec) for pilot in pilots):
        raise ValueError("provide at least one PilotSpec")
    if len({pilot.pilot_id.casefold() for pilot in pilots}) != len(pilots):
        raise ValueError("pilot IDs must be unique, including on Windows")
    benchmark_reward = benchmark_reward or combat_reward_definition("common_combat")
    evaluation_seed = config.evaluation_seed
    if evaluation_seed is None:
        evaluation_seed = max(pilot.seed for pilot in pilots) + config.scenarios_per_epoch + 100_000
    evaluation_end = evaluation_seed + config.evaluation_scenarios_per_epoch
    configs = {}
    root = Path(config.output_dir).resolve()
    for pilot in pilots:
        training_end = pilot.seed + config.scenarios_per_epoch
        if max(training_end, evaluation_end) > 2**32:
            raise ValueError("scenario seed ranges must fit in [0, 2**32)")
        if pilot.seed < evaluation_end and evaluation_seed < training_end:
            raise ValueError("training and evaluation seed ranges must not overlap")
        configs[pilot.pilot_id] = replace(
            config,
            seed=pilot.seed,
            evaluation_seed=evaluation_seed,
            output_dir=root / pilot.pilot_id,
        )

    # Reserve the whole batch before creating any simulator or checkpoint.
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "format_version": 1,
        "evaluation_seed": evaluation_seed,
        "evaluation_scenarios": config.evaluation_scenarios_per_epoch,
        "benchmark_reward": benchmark_reward.as_dict(),
        "pilots": [
            {
                "pilot_id": pilot.pilot_id,
                "seed": pilot.seed,
                "reward": pilot.reward.as_dict(),
                "status": "pending",
            }
            for pilot in pilots
        ],
    }
    manifest_path = root / "pilots.json"
    _write_json(manifest_path, manifest)
    results = []
    for pilot, record in zip(pilots, manifest["pilots"]):
        trainer = None
        record["status"] = "training"
        _write_json(manifest_path, manifest)
        try:
            run_config = configs[pilot.pilot_id]
            trainer = PPOTrainer(
                env_factory_builder(run_config),
                observation_size,
                run_config,
                device=device,
                reward_definition=pilot.reward,
            )
            print(f"Training {pilot.pilot_id}: reward={pilot.reward.name}, seed={pilot.seed}")
            trainer.train(
                run_name=pilot.pilot_id, tags={"pilot_id": pilot.pilot_id, "pilot_batch": root.name}
            )
            episodes = trainer.evaluate_episodes(reward_definition=benchmark_reward)
            scores = np.asarray([episode[1] for episode in episodes], dtype=np.float64)

            def outcome_rate(key, episodes=episodes):
                infos = [episode[2] for episode in episodes]
                if not all(key in info for info in infos):
                    return None
                return float(np.mean([bool(info[key]) for info in infos]))

            def info_mean(key, episodes=episodes):
                if not all(key in episode[2] for episode in episodes):
                    return None
                return float(np.mean([episode[2][key] for episode in episodes]))

            component_names = sorted({
                key for episode in episodes for key in episode[2]["reward_component_totals"]
            })

            result = {
                "pilot_id": pilot.pilot_id,
                "seed": pilot.seed,
                "reward_name": pilot.reward.name,
                "reward": pilot.reward.as_dict(),
                "output_dir": pilot.pilot_id,
                "checkpoint": f"{pilot.pilot_id}/best_model.pt",
                "checkpoint_sha256": _sha256(trainer.output / "best_model.pt"),
                "selection_return": trainer.best_score,
                "benchmark_return": float(scores.mean()),
                "benchmark_return_std": float(scores.std()),
                "survival_rate": outcome_rate("blue_alive"),
                "opponent_destroyed_rate": outcome_rate("opponent_destroyed"),
                "elimination_rate": outcome_rate("opponent_eliminated"),
                "mean_missiles_avoided": info_mean("missiles_avoided_total"),
                "mean_threatened_seconds": info_mean("threatened_time_s"),
                "benchmark_component_means": {
                    key: float(np.mean([
                        episode[2]["reward_component_totals"].get(key, 0.0) for episode in episodes
                    ])) for key in component_names
                },
                "evaluation_seed": evaluation_seed,
                "benchmark_reward": benchmark_reward.as_dict(),
                "mlflow_run_id": trainer.mlflow_run_id,
            }
            _write_json(trainer.output / "manifest.json", result)
            # Add comparable scores to the same pilot's MLflow run.
            with mlflow.start_run(run_id=trainer.mlflow_run_id):
                for key in (
                    "benchmark_return",
                    "benchmark_return_std",
                    "survival_rate",
                    "opponent_destroyed_rate",
                    "elimination_rate",
                    "mean_missiles_avoided",
                    "mean_threatened_seconds",
                ):
                    if result[key] is not None:
                        mlflow.log_metric(key, result[key])
                mlflow.log_artifact(str(trainer.output / "manifest.json"))
            results.append(result)
            record.update(result, status="completed")
        except BaseException as error:
            record.update(
                status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                error=f"{type(error).__name__}: {error}",
            )
            raise
        finally:
            _write_json(manifest_path, manifest)
            del trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return results


def load_trained_pilot(run_dir, device="cpu"):
    """Load one completed pilot for inference using its saved architecture.

    Checksums and skill/parameter ordering are checked before returning the model.
    Reward code is unnecessary for inference; use its recorded definition when
    evaluating a custom objective. This loads weights, not optimizer state.
    """
    import torch

    from .pilot import HybridSkillPilot

    run_dir = Path(run_dir)
    metadata = json.loads((run_dir / "config.json").read_text(encoding="utf8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf8"))
    checkpoint = run_dir / "best_model.pt"
    if _sha256(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("checkpoint SHA-256 does not match the pilot manifest")
    config = metadata["training"]
    pilot = HybridSkillPilot(
        metadata["observation_size"],
        config["hidden_size"],
        history_steps=round(config["history_duration_s"] / config["sample_interval_s"]),
        transformer_heads=config["transformer_heads"],
        transformer_layers=config["transformer_layers"],
    ).to(device)
    if (
        list(pilot.skill_names) != metadata["skill_names"]
        or list(pilot.parameter_names) != metadata["parameter_names"]
    ):
        raise ValueError("checkpoint skill catalogue differs from the installed catalogue")
    pilot.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return pilot.eval()
