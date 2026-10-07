"""Batched PPO training, diagnostics, MLflow tracking, and replay checkpoints."""

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import mlflow
import numpy as np
import torch
from tqdm.auto import tqdm

from .cuda_update import CUDABatchBackward
from .pilot import HybridSkillPilot
from .reward import combat_reward_definition
from .scenarios import ScenarioSampler
from .simulation_workers import ProcessSimulatorPool, ThreadSimulatorPool


@dataclass
class Rollout:
    observations: list
    skills: list
    parameters: list
    log_probs: list
    values: list
    rewards: list
    dones: list


class PPOTrainer:
    """Train once per scenario batch to minimize simulator/optimizer overhead."""

    def __init__(self, env_factory, observation_size, config, device=None, reward_definition=None):
        self.env_factory, self.config = env_factory, config
        self.observation_size = observation_size
        self.reward_definition = reward_definition or combat_reward_definition(weights=config.reward)
        self.mlflow_run_id = None
        self._simulator_pool = None
        self._training_active = False
        self._last_update_profile = {}
        # Reinitialize every pilot independently of the previous run's RNG state.
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.pilot = HybridSkillPilot(
            observation_size,
            config.hidden_size,
            history_steps=config.history_steps,
            transformer_heads=config.transformer_heads,
            transformer_layers=config.transformer_layers,
        ).to(self.device)
        adam_options = {"fused": True} if self.device.type == "cuda" else {}
        self.optimizer = torch.optim.Adam(
            self.pilot.parameters(), lr=config.learning_rate, **adam_options
        )
        self.use_mixed_precision = self.device.type == "cuda" and config.mixed_precision
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.use_mixed_precision)
        if self.device.type == "cuda":
            # Use TensorFloat-32 matrix multiplies where supported. This changes
            # neither the network nor the number of PPO updates.
            torch.set_float32_matmul_precision("high")
        self.sampler = ScenarioSampler(config.seed)
        # Establish the randomized training geometries and simulator seeds once.
        # Reusing both on every epoch prevents changing initial conditions from
        # obscuring the effect of successive policy updates.
        self.training_scenarios = self.sampler.sample_batch(config.scenarios_per_epoch)
        self.training_episode_seeds = tuple(
            config.seed + index for index in range(config.scenarios_per_epoch)
        )
        # Evaluation uses an independent sampler so training-batch sampling cannot
        # alter the benchmark.  Materialising the set once guarantees that every
        # epoch sees exactly the same initial geometries.
        self.evaluation_scenarios = ScenarioSampler(config.resolved_evaluation_seed).sample_batch(
            config.evaluation_scenarios_per_epoch
        )
        self.output = Path(config.output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.diagnostics_path = self.output / "training_metrics.jsonl"
        self.best_score = -float("inf")

    def _episode(self, scenario, seed, recording_path=None, deterministic=False):
        env = self.env_factory(scenario, recording_path)
        try:
            reward_fn = self.reward_definition.factory()
            rollout = Rollout([], [], [], [], [], [], [])
            component_totals = {}
            obs, total, final_info = env.reset(seed), 0.0, {}
            for _ in range(self.config.decisions_per_episode):
                name, params, logp, value, raw = self.pilot.act(obs, deterministic=deterministic)
                next_obs, terminated, truncated, info = env.step(name, params)
                reward, components = reward_fn(info)
                for key, component_value in components.items():
                    component_totals[key] = component_totals.get(key, 0.0) + float(component_value)
                rollout.observations.append(obs)
                rollout.skills.append(self.pilot._skill_indices[name])
                rollout.parameters.append(raw)
                rollout.log_probs.append(logp)
                rollout.values.append(value)
                rollout.rewards.append(reward)
                rollout.dones.append(terminated or truncated)
                total, obs, final_info = (
                    total + reward,
                    next_obs,
                    {**info, "reward_components": components,
                     "reward_component_totals": dict(component_totals)},
                )
                if terminated or truncated:
                    break
        finally:
            env.close()
        return rollout, total, final_info

    def _evaluate(self):
        """Evaluate deterministically on the fixed scenario and episode seeds."""
        return np.asarray([item[1] for item in self.evaluate_episodes()], dtype=np.float64)

    def evaluate_episodes(self, reward_definition=None):
        """Return deterministic rollouts, optionally scored by a common benchmark."""
        was_training = self.pilot.training
        self.pilot.eval()
        try:
            options = {"deterministic": True}
            if reward_definition is not None:
                options["reward_definition"] = reward_definition
            return self._episodes(
                self.evaluation_scenarios,
                tuple(
                    self.config.resolved_evaluation_seed + index
                    for index in range(len(self.evaluation_scenarios))
                ),
                **options,
            )
        finally:
            self.pilot.train(was_training)

    def _training_episodes(self):
        """Collect one epoch from the fixed training geometries and reset seeds."""
        return self._episodes(self.training_scenarios, self.training_episode_seeds)

    def _episodes(self, scenarios, seeds, deterministic=False, reward_definition=None):
        """Run independent simulators concurrently and batch policy work on the GPU."""
        collection_started = perf_counter()
        policy_seconds = 0.0
        simulator_wall_seconds = 0.0
        simulator_work_seconds = 0.0
        simulator_cpu_seconds = 0.0
        if len(scenarios) != len(seeds) or not scenarios:
            raise ValueError("provide matching nonempty scenarios and seeds")
        pool = self._simulator_pool
        if pool is None:
            pool_type = (
                ProcessSimulatorPool if self.config.simulator_executor == "process"
                else ThreadSimulatorPool
            )
            pool = pool_type(
                self.env_factory, min(self.config.resolved_simulator_workers, len(scenarios))
            )
            if self._training_active:
                self._simulator_pool = pool
        failed = True

        try:
            observations = pool.reset(scenarios, seeds)
            rollouts = [Rollout([], [], [], [], [], [], []) for _ in scenarios]
            totals = [0.0] * len(scenarios)
            component_totals = [{} for _ in scenarios]
            final_info = [{} for _ in scenarios]
            active = list(range(len(scenarios)))
            definition = reward_definition or self.reward_definition
            rewards = [definition.factory() for _ in scenarios]
            for _ in range(self.config.decisions_per_episode):
                if not active:
                    break
                policy_started = perf_counter()
                actions = self.pilot.act_batch(
                    np.stack([observations[index] for index in active]),
                    deterministic=deterministic,
                )
                policy_seconds += perf_counter() - policy_started
                simulator_started = perf_counter()
                timed_steps = pool.step([
                    (index, action[0], action[1]) for index, action in zip(active, actions)
                ])
                simulator_wall_seconds += perf_counter() - simulator_started
                simulator_work_seconds += sum(item[1] for item in timed_steps)
                simulator_cpu_seconds += sum(item[2] for item in timed_steps)
                next_active = []
                for index, action, timed_step_result in zip(active, actions, timed_steps):
                    step, _, _ = timed_step_result
                    name, _, logp, value, raw = action
                    next_obs, terminated, truncated, info = step
                    reward, components = rewards[index](info)
                    for key, component_value in components.items():
                        component_totals[index][key] = (
                            component_totals[index].get(key, 0.0) + float(component_value)
                        )
                    rollout = rollouts[index]
                    rollout.observations.append(observations[index])
                    rollout.skills.append(self.pilot._skill_indices[name])
                    rollout.parameters.append(raw)
                    rollout.log_probs.append(logp)
                    rollout.values.append(value)
                    rollout.rewards.append(reward)
                    rollout.dones.append(terminated or truncated)
                    totals[index] += reward
                    observations[index] = next_obs
                    final_info[index] = {
                        **info, "reward_components": components,
                        "reward_component_totals": dict(component_totals[index]),
                    }
                    if not (terminated or truncated):
                        next_active.append(index)
                active = next_active
            failed = False
        finally:
            if failed or not self._training_active:
                pool.close(force=failed)
                if self._simulator_pool is pool:
                    self._simulator_pool = None
            else:
                pool.release()
        self._last_collection_profile = {
            "collection_seconds": perf_counter() - collection_started,
            "policy_seconds": policy_seconds,
            "simulator_wall_seconds": simulator_wall_seconds,
            "simulator_work_seconds": simulator_work_seconds,
            "simulator_cpu_seconds": simulator_cpu_seconds,
            "simulator_effective_cores": (
                simulator_cpu_seconds / simulator_wall_seconds if simulator_wall_seconds else 0.0
            ),
            "simulator_cpu_utilization_percent": (
                100 * simulator_cpu_seconds / simulator_wall_seconds / (os.cpu_count() or 1)
                if simulator_wall_seconds else 0.0
            ),
        }
        return list(zip(rollouts, totals, final_info))

    def _advantages(self, rollout):
        advantages, gae, next_value = [], 0.0, 0.0
        for reward, value, done in reversed(
            list(zip(rollout.rewards, rollout.values, rollout.dones))
        ):
            delta = reward + self.config.discount * next_value * (not done) - value
            gae = delta + self.config.discount * self.config.gae_lambda * (not done) * gae
            advantages.append(gae)
            next_value = value
        advantages.reverse()
        return np.asarray(advantages, np.float32), np.asarray(advantages) + rollout.values

    def _backward_minibatch(self, batch, *, capture=False):
        observations, skills, raw, old, advantages, returns = batch
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.use_mixed_precision,
        ):
            # Distribution argument checks read GPU booleans on the CPU and cannot
            # run inside a CUDA graph. Collection/eager evaluation keep validation;
            # graph inputs are checked once per update and final loss must be finite.
            logp, entropy, values = self.pilot.evaluate(
                observations, skills, raw, validate_args=not capture,
            )
            ratio = (logp - old).exp()
            policy = -torch.minimum(
                ratio * advantages,
                ratio.clamp(1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantages,
            ).mean()
            loss = (
                policy + self.config.value_coefficient * (values - returns).pow(2).mean()
                - self.config.entropy_coefficient * entropy.mean()
            )
        self.grad_scaler.scale(loss).backward()
        return loss

    def _update(self, rollouts):
        advantages, returns = zip(*(self._advantages(r) for r in rollouts))
        obs = torch.as_tensor(
            np.concatenate([r.observations for r in rollouts]),
            dtype=torch.float32,
            device=self.device,
        )
        skills = torch.as_tensor(
            np.concatenate([np.asarray(r.skills) for r in rollouts]), device=self.device
        )
        raw = torch.as_tensor(
            np.concatenate([np.asarray(r.parameters) for r in rollouts]),
            dtype=torch.float32,
            device=self.device,
        )
        old = torch.as_tensor(
            np.concatenate([np.asarray(r.log_probs) for r in rollouts]),
            dtype=torch.float32,
            device=self.device,
        )
        adv = torch.as_tensor(np.concatenate(advantages), device=self.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ret = torch.as_tensor(np.concatenate(returns), dtype=torch.float32, device=self.device)
        n = len(obs)
        batch_tensors = (obs, skills, raw, old, adv, ret)
        graph_backward = None
        capture_seconds = 0.0
        if (self.config.cuda_graph_updates and self.device.type == "cuda"
                and n >= self.config.minibatch_size):
            valid = torch.stack([torch.isfinite(tensor).all() for tensor in batch_tensors]).all()
            valid = valid & ((skills >= 0) & (skills < len(self.pilot.skill_names))).all()
            if not valid.item():
                raise ValueError("PPO rollout contains non-finite data or invalid skill indices")
            capture_started = perf_counter()
            graph_backward = CUDABatchBackward(
                lambda batch: self._backward_minibatch(batch, capture=True),
                self.pilot.parameters(),
                tuple(tensor[:self.config.minibatch_size] for tensor in batch_tensors),
            )
            capture_seconds = perf_counter() - capture_started
        # Shuffle each tensor once per update epoch. Contiguous minibatch slices are
        # views; the previous advanced indexing performed five GPU gathers for every
        # minibatch. Keep losses on-device too, avoiding a device synchronization on
        # every loss.item() call.
        losses = []
        for _ in range(self.config.update_epochs):
            order = torch.randperm(n, device=self.device)
            shuffled = tuple(
                tensor.index_select(0, order) for tensor in batch_tensors
            )
            for left in range(0, n, self.config.minibatch_size):
                right = left + self.config.minibatch_size
                batch = tuple(tensor[left:right] for tensor in shuffled)
                if graph_backward is not None and len(batch[0]) == self.config.minibatch_size:
                    loss = graph_backward(batch)
                else:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss = self._backward_minibatch(batch)
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.pilot.parameters(), self.config.gradient_clip)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                losses.append(loss.detach())
        mean_loss = torch.stack(losses).mean().item()
        if not math.isfinite(mean_loss):
            raise FloatingPointError("PPO update produced a non-finite loss")
        self._last_update_profile = {
            "rollout_transitions": n,
            "update_minibatches": len(losses),
            "cuda_graph_updates": int(graph_backward is not None),
            "cuda_graph_capture_seconds": capture_seconds,
        }
        return mean_loss

    def _should_evaluate(self, epoch):
        """Evaluate at useful boundaries without doubling every training epoch."""
        return (
            epoch == 1
            or epoch == self.config.epochs
            or epoch % self.config.evaluation_interval == 0
        )

    def _estimated_epoch_speedup(self, rollout_seconds, update_seconds, evaluation_seconds):
        """Estimate steady-state speedup versus evaluating after every epoch.

        This phase-time model deliberately excludes checkpoint time and makes no
        claim about mixed-precision gains, which are hardware dependent.
        """
        baseline = rollout_seconds + update_seconds + evaluation_seconds
        amortized = (
            rollout_seconds
            + update_seconds
            + evaluation_seconds / self.config.evaluation_interval
        )
        return baseline / amortized if amortized else 1.0

    def train(self, run_name=None, tags=None):
        self._training_active = True
        try:
            return self._train(run_name, tags)
        finally:
            self._training_active = False
            self.close()

    def close(self):
        pool, self._simulator_pool = self._simulator_pool, None
        if pool is not None:
            pool.close()

    def _train(self, run_name=None, tags=None):
        metadata = {
            "training": self.config.as_dict(),
            "observation_size": self.observation_size,
            "reward": self.reward_definition.as_dict(),
            "skill_names": list(self.pilot.skill_names),
            "parameter_names": list(self.pilot.parameter_names),
            "checkpoint_format": "policy_state_dict",
            "torch_version": torch.__version__,
        }
        config_path = self.output / "config.json"
        config_path.write_text(json.dumps(metadata, indent=2), encoding="utf8")
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.mlflow_experiment)
        with mlflow.start_run(run_name=run_name, tags=tags) as active_run:
            self.mlflow_run_id = active_run.info.run_id
            run_parameters = self.config.as_dict()
            run_parameters.pop("reward")
            run_parameters.update(
                {
                    "reward.name": self.reward_definition.name,
                    "reward.version": self.reward_definition.version,
                    "reward.parameters": json.dumps(self.reward_definition.parameters),
                }
            )
            mlflow.log_params(run_parameters)
            mlflow.log_artifact(str(config_path))
            progress = tqdm(
                total=self.config.epochs,
                desc="Training",
                unit="epoch",
                bar_format=(
                    "{l_bar}{bar}| {n_fmt}/{total_fmt} epochs "
                    "[elapsed {elapsed} < ETA {remaining}, {rate_fmt}{postfix}]"
                ),
            )
            evaluation_scores = None
            evaluation_seconds_sample = 0.0
            for epoch in range(1, self.config.epochs + 1):
                epoch_started = perf_counter()
                progress.set_postfix(phase="collecting flights")
                episodes = self._training_episodes()
                rollout_seconds = perf_counter() - epoch_started
                training_collection_profile = self._last_collection_profile.copy()
                update_started = perf_counter()
                progress.set_postfix(phase="PPO update")
                loss, scores = (
                    self._update([item[0] for item in episodes]),
                    [item[1] for item in episodes],
                )
                update_seconds = perf_counter() - update_started
                evaluation_performed = self._should_evaluate(epoch)
                evaluation_seconds = 0.0
                if evaluation_performed:
                    progress.set_postfix(phase="validation flights")
                    evaluation_started = perf_counter()
                    evaluation_scores = self._evaluate()
                    evaluation_seconds = perf_counter() - evaluation_started
                    evaluation_seconds_sample = evaluation_seconds
                evaluation_mean = float(np.mean(evaluation_scores))
                metrics = {
                    "epoch": epoch,
                    "mean_return": evaluation_mean,
                    "evaluation_return_std": float(np.std(evaluation_scores)),
                    "training_mean_return": float(np.mean(scores)),
                    "loss": loss,
                    "episodes": len(scores),
                    "evaluation_episodes": len(evaluation_scores),
                    "best_return": max(self.best_score, evaluation_mean),
                    "rollout_seconds": rollout_seconds,
                    "update_seconds": update_seconds,
                    "evaluation_seconds": evaluation_seconds,
                    "evaluation_performed": int(evaluation_performed),
                    "estimated_epoch_speedup": self._estimated_epoch_speedup(
                        rollout_seconds, update_seconds, evaluation_seconds_sample
                    ),
                }
                metrics.update(training_collection_profile)
                metrics.update(self._last_update_profile)
                metrics["update_transitions_per_second"] = (
                    sum(len(item[0].rewards) for item in episodes) * self.config.update_epochs
                    / update_seconds if update_seconds else 0.0
                )
                checkpoint_started = perf_counter()
                if evaluation_performed and evaluation_mean > self.best_score:
                    self.best_score = evaluation_mean
                    torch.save(self.pilot.state_dict(), self.output / "best_model.pt")
                if epoch % self.config.checkpoint_interval == 0:
                    progress.set_postfix(phase="saving demonstration")
                    demo_dir = self.output / "demonstrations" / f"epoch_{epoch:04d}"
                    demo_dir.mkdir(parents=True, exist_ok=True)
                    # Fixed seed and first canonical geometry make progress comparable.
                    demo = ScenarioSampler(self.config.seed).sample_batch(3)[0]
                    rollout, score, info = self._episode(
                        demo, self.config.seed, demo_dir / "blue_vs_red.acmi"
                    )
                    np.savez_compressed(
                        demo_dir / "trajectory.npz",
                        observations=np.asarray(rollout.observations, dtype=np.float32),
                        skill_indices=np.asarray(rollout.skills, dtype=np.int64),
                        normalized_parameters=np.asarray(rollout.parameters, dtype=np.float32),
                        rewards=np.asarray(rollout.rewards, dtype=np.float32),
                        skill_names=np.asarray(self.pilot.skill_names),
                    )
                    (demo_dir / "summary.json").write_text(
                        json.dumps(
                            {"score": score, "scenario": demo.as_dict(), "final_info": info},
                            default=str,
                            indent=2,
                        )
                    )
                    mlflow.log_artifacts(
                        str(demo_dir), artifact_path=f"demonstrations/epoch_{epoch:04d}"
                    )
                metrics["checkpoint_seconds"] = perf_counter() - checkpoint_started
                metrics["epoch_seconds"] = perf_counter() - epoch_started
                with self.diagnostics_path.open("a", encoding="utf8") as stream:
                    stream.write(json.dumps(metrics) + "\n")
                mlflow.log_metrics({k: v for k, v in metrics.items() if k != "epoch"}, step=epoch)
                progress.set_postfix(
                    epoch=f"{metrics['epoch_seconds']:.1f}s",
                    loss=f"{loss:.4f}",
                    return_=f"{evaluation_mean:.2f}",
                    refresh=False,
                )
                progress.update()
            progress.close()
            self.pilot.load_state_dict(
                torch.load(
                    self.output / "best_model.pt", map_location=self.device, weights_only=True
                )
            )
            mlflow.log_artifact(str(self.output / "best_model.pt"))
            mlflow.log_artifact(str(self.diagnostics_path))
        return self.pilot
