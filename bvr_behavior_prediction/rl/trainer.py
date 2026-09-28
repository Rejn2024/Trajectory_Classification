"""Batched PPO training, diagnostics, MLflow tracking, and replay checkpoints."""

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from time import perf_counter

import mlflow
import numpy as np
import torch
from tqdm.auto import tqdm

from .pilot import HybridSkillPilot
from .reward import CombatReward
from .scenarios import ScenarioSampler


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

    def __init__(self, env_factory, observation_size, config, device=None):
        self.env_factory, self.config = env_factory, config
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
        self.evaluation_scenarios = ScenarioSampler(config.seed).sample_batch(
            config.evaluation_scenarios_per_epoch
        )
        self.output = Path(config.output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.diagnostics_path = self.output / "training_metrics.jsonl"
        self.best_score = -float("inf")

    def _episode(self, scenario, seed, recording_path=None, deterministic=False):
        env = self.env_factory(scenario, recording_path)
        reward_fn, rollout = CombatReward(self.config.reward), Rollout([], [], [], [], [], [], [])
        obs, total, final_info = env.reset(seed), 0.0, {}
        try:
            for _ in range(self.config.decisions_per_episode):
                name, params, logp, value, raw = self.pilot.act(obs, deterministic=deterministic)
                next_obs, terminated, truncated, info = env.step(name, params)
                reward, components = reward_fn(info)
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
                    {**info, "reward_components": components},
                )
                if terminated or truncated:
                    break
        finally:
            env.close()
        return rollout, total, final_info

    def _evaluate(self):
        """Evaluate deterministically on the fixed scenario and episode seeds."""
        was_training = self.pilot.training
        self.pilot.eval()
        try:
            scores = [
                item[1]
                for item in self._episodes(
                    self.evaluation_scenarios,
                    tuple(
                        self.config.seed + index for index in range(len(self.evaluation_scenarios))
                    ),
                    deterministic=True,
                )
            ]
        finally:
            self.pilot.train(was_training)
        return np.asarray(scores, dtype=np.float64)

    def _training_episodes(self):
        """Collect one epoch from the fixed training geometries and reset seeds."""
        return self._episodes(self.training_scenarios, self.training_episode_seeds)

    def _episodes(self, scenarios, seeds, deterministic=False):
        """Run independent simulators concurrently and batch policy work on the GPU."""
        collection_started = perf_counter()
        policy_seconds = 0.0
        simulator_wall_seconds = 0.0
        simulator_work_seconds = 0.0
        environments = [self.env_factory(scenario, None) for scenario in scenarios]
        rollouts = [Rollout([], [], [], [], [], [], []) for _ in environments]
        totals = [0.0] * len(environments)
        final_info = [{} for _ in environments]
        active = list(range(len(environments)))

        try:
            with ThreadPoolExecutor(
                max_workers=min(self.config.resolved_simulator_workers, len(environments))
            ) as executor:
                observations = list(
                    executor.map(lambda item: item[0].reset(item[1]), zip(environments, seeds))
                )
                rewards = [CombatReward(self.config.reward) for _ in environments]
                for _ in range(self.config.decisions_per_episode):
                    if not active:
                        break
                    policy_started = perf_counter()
                    actions = self.pilot.act_batch(
                        np.stack([observations[index] for index in active]),
                        deterministic=deterministic,
                    )
                    policy_seconds += perf_counter() - policy_started

                    def timed_step(item):
                        started = perf_counter()
                        result = item[0].step(item[1][0], item[1][1])
                        return result, perf_counter() - started

                    simulator_started = perf_counter()
                    timed_steps = list(
                        executor.map(
                            timed_step,
                            (
                                (environments[index], action)
                                for index, action in zip(active, actions)
                            ),
                        )
                    )
                    simulator_wall_seconds += perf_counter() - simulator_started
                    simulator_work_seconds += sum(item[1] for item in timed_steps)
                    next_active = []
                    for index, action, timed_step_result in zip(active, actions, timed_steps):
                        step, _ = timed_step_result
                        name, _, logp, value, raw = action
                        next_obs, terminated, truncated, info = step
                        reward, components = rewards[index](info)
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
                        final_info[index] = {**info, "reward_components": components}
                        if not (terminated or truncated):
                            next_active.append(index)
                    active = next_active
        finally:
            for environment in environments:
                environment.close()
        self._last_collection_profile = {
            "collection_seconds": perf_counter() - collection_started,
            "policy_seconds": policy_seconds,
            "simulator_wall_seconds": simulator_wall_seconds,
            "simulator_work_seconds": simulator_work_seconds,
            "simulator_parallel_speedup": (
                simulator_work_seconds / simulator_wall_seconds
                if simulator_wall_seconds
                else 1.0
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
        # Shuffle each tensor once per update epoch. Contiguous minibatch slices are
        # views; the previous advanced indexing performed five GPU gathers for every
        # minibatch. Keep losses on-device too, avoiding a device synchronization on
        # every loss.item() call.
        losses = []
        for _ in range(self.config.update_epochs):
            order = torch.randperm(n, device=self.device)
            shuffled = tuple(
                tensor.index_select(0, order) for tensor in (obs, skills, raw, old, adv, ret)
            )
            (
                shuffled_obs,
                shuffled_skills,
                shuffled_raw,
                shuffled_old,
                shuffled_adv,
                shuffled_ret,
            ) = shuffled
            for left in range(0, n, self.config.minibatch_size):
                right = left + self.config.minibatch_size
                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.use_mixed_precision,
                ):
                    logp, entropy, values = self.pilot.evaluate(
                        shuffled_obs[left:right],
                        shuffled_skills[left:right],
                        shuffled_raw[left:right],
                    )
                    ratio = (logp - shuffled_old[left:right]).exp()
                    policy = -torch.minimum(
                        ratio * shuffled_adv[left:right],
                        ratio.clamp(1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
                        * shuffled_adv[left:right],
                    ).mean()
                    loss = (
                        policy
                        + self.config.value_coefficient
                        * (values - shuffled_ret[left:right]).pow(2).mean()
                        - self.config.entropy_coefficient * entropy.mean()
                    )
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.pilot.parameters(), self.config.gradient_clip)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                losses.append(loss.detach())
        return torch.stack(losses).mean().item()

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

    def _estimated_total_speedup(
        self, rollout_seconds, update_seconds, evaluation_seconds, collection_profile=None
    ):
        """Estimate combined concurrency and evaluation-cadence acceleration.

        The counterfactual keeps measured policy/reset/processing time unchanged and
        replaces concurrent simulator wall time with the sum of worker step times.
        It is therefore deliberately conservative and does not guess at AMP gains.
        """
        profile = collection_profile or getattr(self, "_last_collection_profile", {})
        simulator_wall = profile.get("simulator_wall_seconds", 0.0)
        simulator_work = profile.get("simulator_work_seconds", simulator_wall)
        sequential_rollout = rollout_seconds + max(0.0, simulator_work - simulator_wall)
        baseline = sequential_rollout + update_seconds + evaluation_seconds
        optimized = (
            rollout_seconds
            + update_seconds
            + evaluation_seconds / self.config.evaluation_interval
        )
        return baseline / optimized if optimized else 1.0

    def train(self):
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.mlflow_experiment)
        with mlflow.start_run():
            run_parameters = self.config.as_dict()
            reward_parameters = run_parameters.pop("reward")
            run_parameters.update(
                {f"reward.{key}": value for key, value in reward_parameters.items()}
            )
            mlflow.log_params(run_parameters)
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
                episodes = self._training_episodes()
                rollout_seconds = perf_counter() - epoch_started
                training_collection_profile = self._last_collection_profile.copy()
                update_started = perf_counter()
                loss, scores = (
                    self._update([item[0] for item in episodes]),
                    [item[1] for item in episodes],
                )
                update_seconds = perf_counter() - update_started
                evaluation_performed = self._should_evaluate(epoch)
                evaluation_seconds = 0.0
                if evaluation_performed:
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
                    "estimated_total_speedup": self._estimated_total_speedup(
                        rollout_seconds,
                        update_seconds,
                        evaluation_seconds_sample,
                        training_collection_profile,
                    ),
                }
                metrics.update(training_collection_profile)
                if evaluation_performed and evaluation_mean > self.best_score:
                    self.best_score = evaluation_mean
                    torch.save(self.pilot.state_dict(), self.output / "best_model.pt")
                if epoch % self.config.checkpoint_interval == 0:
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
