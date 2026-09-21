"""Batched PPO training, diagnostics, MLflow tracking, and replay checkpoints."""

from dataclasses import dataclass
import json
from pathlib import Path

import mlflow
import numpy as np
import torch

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
        self.pilot = HybridSkillPilot(observation_size, config.hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.pilot.parameters(), lr=config.learning_rate)
        self.sampler = ScenarioSampler(config.seed)
        self.output = Path(config.output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.diagnostics_path = self.output / "training_metrics.jsonl"
        self.best_score = -float("inf")

    def _episode(self, scenario, seed, recording_path=None):
        env = self.env_factory(scenario, recording_path)
        reward_fn, rollout = CombatReward(self.config.reward), Rollout([], [], [], [], [], [], [])
        obs, total, final_info = env.reset(seed), 0.0, {}
        try:
            for _ in range(self.config.decisions_per_episode):
                name, params, logp, value, raw = self.pilot.act(obs)
                next_obs, terminated, truncated, info = env.step(name, params)
                reward, components = reward_fn(info)
                rollout.observations.append(obs)
                rollout.skills.append(self.pilot.skill_names.index(name))
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
        losses = []
        for _ in range(self.config.update_epochs):
            for indices in torch.randperm(n, device=self.device).split(self.config.minibatch_size):
                logp, entropy, values = self.pilot.evaluate(
                    obs[indices], skills[indices], raw[indices]
                )
                ratio = (logp - old[indices]).exp()
                policy = -torch.minimum(
                    ratio * adv[indices],
                    ratio.clamp(1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
                    * adv[indices],
                ).mean()
                loss = (
                    policy
                    + self.config.value_coefficient * (values - ret[indices]).pow(2).mean()
                    - self.config.entropy_coefficient * entropy.mean()
                )
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.pilot.parameters(), self.config.gradient_clip)
                self.optimizer.step()
                losses.append(float(loss.item()))
        return float(np.mean(losses))

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
            for epoch in range(1, self.config.epochs + 1):
                scenarios = self.sampler.sample_batch(self.config.scenarios_per_epoch)
                episodes = [
                    self._episode(s, self.config.seed + epoch * 1000 + i)
                    for i, s in enumerate(scenarios)
                ]
                loss, scores = (
                    self._update([item[0] for item in episodes]),
                    [item[1] for item in episodes],
                )
                metrics = {
                    "epoch": epoch,
                    "mean_return": float(np.mean(scores)),
                    "loss": loss,
                    "episodes": len(scores),
                    "best_return": max(self.best_score, float(np.mean(scores))),
                }
                with self.diagnostics_path.open("a", encoding="utf8") as stream:
                    stream.write(json.dumps(metrics) + "\n")
                mlflow.log_metrics({k: v for k, v in metrics.items() if k != "epoch"}, step=epoch)
                if epoch % self.config.diagnostic_interval == 0:
                    print(
                        f"epoch={epoch:04d} return={metrics['mean_return']:.2f} loss={loss:.4f}",
                        flush=True,
                    )
                if metrics["mean_return"] > self.best_score:
                    self.best_score = metrics["mean_return"]
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
            self.pilot.load_state_dict(
                torch.load(
                    self.output / "best_model.pt", map_location=self.device, weights_only=True
                )
            )
            mlflow.log_artifact(str(self.output / "best_model.pt"))
            mlflow.log_artifact(str(self.diagnostics_path))
        return self.pilot
