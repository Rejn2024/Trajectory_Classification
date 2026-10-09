"""Train independent PPO pilots with explicit objectives and portable run records."""

import gc
import hashlib
import json
import re
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path, PureWindowsPath

from .config import PilotTrainingConfig
from .evaluation import OUTCOME_SCORES, summarize_test_episodes
from .reward import RewardDefinition, combat_reward_definition
from .scenarios import ScenarioSampler


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
    population_metadata=None,
    reuse_completed_from=None,
):
    """Train fresh models sequentially; return completed pilot records.

    ``env_factory_builder(run_config)`` returns the usual
    ``env_factory(scenario, recording_path)``. Each pilot gets independent RNG
    initialization, optimizer, reward instances, directories, and MLflow run.
    The output root must be new. Interrupted batches retain completed runs and
    a status manifest; this function deliberately does not resume optimizers.
    ``reuse_completed_from`` copies verified completed pilots from an older batch
    into the new root. Failed/pending pilots train afresh; the older batch stays intact.

    Checkpoint selection uses each pilot's own reward. The selected policies
    are then scored with one common reward on identical validation scenarios.
    This validation set is used for selection; it is not an untouched test set.
    If configured, a separate shared test suite runs only after checkpoint selection.
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
    population_metadata = json.loads(json.dumps(
        {} if population_metadata is None else population_metadata, allow_nan=False
    ))
    if not isinstance(population_metadata, dict):
        raise ValueError("population_metadata must be a JSON object")
    evaluation_seed = config.evaluation_seed
    if evaluation_seed is None:
        evaluation_seed = max(pilot.seed for pilot in pilots) + config.training_seed_count + 100_000
    evaluation_end = evaluation_seed + config.evaluation_scenarios_per_epoch
    test_seed = config.test_seed
    if config.test_scenarios_per_pilot:
        if test_seed is None:
            test_seed = max(
                evaluation_end, max(p.seed for p in pilots) + config.training_seed_count
            ) + 100_000
        test_end = test_seed + config.test_scenarios_per_pilot
        if max(test_seed, evaluation_seed) < min(test_end, evaluation_end):
            raise ValueError("test and validation seed ranges must not overlap")
    else:
        test_end = 0
    configs = {}
    root = Path(config.output_dir).resolve()
    for pilot in pilots:
        training_end = pilot.seed + config.training_seed_count
        if max(training_end, evaluation_end, test_end) > 2**32:
            raise ValueError("scenario seed ranges must fit in [0, 2**32)")
        if pilot.seed < evaluation_end and evaluation_seed < training_end:
            raise ValueError("training and evaluation seed ranges must not overlap")
        if config.test_scenarios_per_pilot and max(test_seed, pilot.seed) < min(test_end, training_end):
            raise ValueError("test and training seed ranges must not overlap")
        configs[pilot.pilot_id] = replace(
            config,
            seed=pilot.seed,
            evaluation_seed=evaluation_seed,
            test_seed=test_seed,
            output_dir=root / pilot.pilot_id,
        )

    test_scenarios = (
        ScenarioSampler(test_seed).sample_batch(config.test_scenarios_per_pilot)
        if config.test_scenarios_per_pilot else []
    )
    test_seeds = tuple(range(test_seed, test_end)) if test_scenarios else ()
    reusable = {}
    if reuse_completed_from is not None:
        previous_root = Path(reuse_completed_from).resolve()
        previous = json.loads((previous_root / "pilots.json").read_text(encoding="utf8"))
        if (previous["benchmark_reward"] != benchmark_reward.as_dict()
                or previous["evaluation_seed"] != evaluation_seed
                or previous["evaluation_scenarios"] != config.evaluation_scenarios_per_epoch
                or previous["test"]["seed"] != test_seed
                or previous["test"]["scenarios"] != config.test_scenarios_per_pilot):
            raise ValueError("reused population must have identical evaluation suites and reward")
        old_records = {p["pilot_id"]: p for p in previous["pilots"]}
        if set(old_records) != set(configs):
            raise ValueError("reused population must have the same pilot roster")
        for pilot in pilots:
            record = old_records[pilot.pilot_id]
            if record["seed"] != pilot.seed or record["reward"] != pilot.reward.as_dict():
                raise ValueError(f"reused pilot seed/reward differs: {pilot.pilot_id}")
            if record["status"] != "completed":
                continue
            directory = previous_root / pilot.pilot_id
            metadata = json.loads((directory / "config.json").read_text(encoding="utf8"))
            ignored = {"output_dir", "mlflow_tracking_uri"}
            expected = configs[pilot.pilot_id].as_dict()
            if ({k: v for k, v in metadata["training"].items() if k not in ignored}
                    != {k: v for k, v in expected.items() if k not in ignored}
                    or metadata["reward"] != pilot.reward.as_dict()
                    or metadata["observation_size"] != observation_size):
                raise ValueError(f"reused pilot training settings differ: {pilot.pilot_id}")
            result = json.loads((directory / "manifest.json").read_text(encoding="utf8"))
            checkpoint_hash = _sha256(directory / "best_model.pt")
            if (result["checkpoint_sha256"] != checkpoint_hash
                    or record["checkpoint_sha256"] != checkpoint_hash):
                raise ValueError(f"reused checkpoint SHA-256 mismatch: {pilot.pilot_id}")
            epochs = [json.loads(line)["epoch"] for line in
                      (directory / "training_metrics.jsonl").read_text().splitlines()]
            if epochs != list(range(1, config.epochs + 1)):
                raise ValueError(f"reused pilot did not finish all epochs: {pilot.pilot_id}")
            if test_scenarios:
                test = json.loads((directory / "test_episodes.json").read_text(encoding="utf8"))
                if (test["checkpoint_sha256"] != checkpoint_hash
                        or [r["seed"] for r in test["episodes"]] != list(test_seeds)):
                    raise ValueError(f"reused pilot test is incomplete: {pilot.pilot_id}")
            reusable[pilot.pilot_id] = (directory, result)
    # Reserve the whole batch before creating any simulator or checkpoint.
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "format_version": 1,
        "evaluation_seed": evaluation_seed,
        "evaluation_scenarios": config.evaluation_scenarios_per_epoch,
        "benchmark_reward": benchmark_reward.as_dict(),
        "population_metadata": population_metadata,
        "training_scenarios_resampled_each_epoch": config.resample_training_scenarios,
        "test": {
            "seed": test_seed, "scenarios": config.test_scenarios_per_pilot,
            "outcome_scores": dict(OUTCOME_SCORES),
            "role": "post-selection test; not used to select checkpoints",
        },
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
        if pilot.pilot_id in reusable:
            directory, result = reusable[pilot.pilot_id]
            destination = root / pilot.pilot_id
            shutil.copytree(directory, destination)
            for path in directory.rglob("*"):
                if path.is_file() and _sha256(path) != _sha256(destination / path.relative_to(directory)):
                    raise RuntimeError(f"copied pilot file checksum mismatch: {path}")
            record.update(result, status="completed", reused_from=str(directory))
            results.append(result)
            _write_json(manifest_path, manifest)
            print(f"Reused completed {pilot.pilot_id} unchanged from {directory}", flush=True)
            continue
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
            if test_scenarios:
                print(f"Testing {pilot.pilot_id} on {len(test_scenarios)} fresh scenarios")
                test_episodes = trainer.evaluate_scenarios(
                    test_scenarios, test_seeds, reward_definition=benchmark_reward
                )
                test_summary, test_records = summarize_test_episodes(
                    test_episodes, test_scenarios, test_seeds, trainer.pilot.skill_names
                )
                test_summary["benchmark_reward"] = benchmark_reward.as_dict()
                test_summary["episodes_file"] = "test_episodes.json"
                result["test"] = test_summary
                _write_json(trainer.output / "test_episodes.json", {
                    "checkpoint_sha256": result["checkpoint_sha256"],
                    "benchmark_reward": benchmark_reward.as_dict(),
                    "outcome_scores": dict(OUTCOME_SCORES), "episodes": test_records,
                })
            result["population_metadata"] = population_metadata
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
                if "test" in result:
                    mlflow.log_metrics({
                        f"test.{key}": value for key, value in result["test"].items()
                        if isinstance(value, (int, float))
                    })
                    mlflow.log_artifact(str(trainer.output / "test_episodes.json"))
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
        selected_skill_parameters=config.get("selected_skill_parameters", False),
    ).to(device)
    if (
        list(pilot.skill_names) != metadata["skill_names"]
        or list(pilot.parameter_names) != metadata["parameter_names"]
    ):
        raise ValueError("checkpoint skill catalogue differs from the installed catalogue")
    pilot.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return pilot.eval()
