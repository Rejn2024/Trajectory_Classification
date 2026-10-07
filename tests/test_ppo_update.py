"""PPO correctness across rollout bookkeeping and optimized policy evaluation."""

from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mlflow")

from bvr_behavior_prediction.rl.config import PilotTrainingConfig
from bvr_behavior_prediction.rl.reward import RewardDefinition
from bvr_behavior_prediction.rl.trainer import PPOTrainer, Rollout


@pytest.mark.parametrize("batched", [False, True])
def test_reward_component_logging_preserves_critic_values(tmp_path, batched):
    class Environment:
        def reset(self, seed):
            return np.zeros((1, 1), dtype=np.float32)

        def step(self, name, parameters):
            return np.zeros((1, 1), dtype=np.float32), True, False, {}

        def close(self):
            pass

    config = PilotTrainingConfig(
        episode_duration_s=1, hidden_size=8, transformer_heads=2,
        transformer_layers=1, history_duration_s=1, sample_interval_s=1,
        simulator_workers=1, output_dir=tmp_path,
    )
    trainer = PPOTrainer(
        lambda *_: Environment(), 1, config, device="cpu",
        reward_definition=RewardDefinition(
            "fixture", factory=lambda: lambda info: (5.0, {"pursue": 3.0, "evade": 2.0}),
        ),
    )
    raw = np.zeros(len(trainer.pilot.parameter_names), dtype=np.float32)
    action = ("maintain_heading", {}, -1.0, 7.5, raw)
    trainer.pilot.act = lambda *args, **kwargs: action
    trainer.pilot.act_batch = lambda observations, **kwargs: (action,) * len(observations)
    if batched:
        episodes = trainer._episodes([object()] * 3, (1, 2, 3))
    else:
        episodes = [trainer._episode(object(), 1)]
    for rollout, total, info in episodes:
        assert rollout.values == [7.5]
        assert total == 5.0
        assert info["reward_component_totals"] == {"pursue": 3.0, "evade": 2.0}
        advantages, returns = trainer._advantages(rollout)
        np.testing.assert_allclose(advantages, [-2.5])
        np.testing.assert_allclose(returns, [5.0])


def _rollout(trainer, count):
    rng = np.random.default_rng(count)
    observations = rng.normal(size=(count, 2, 3)).astype(np.float32)
    actions = trainer.pilot.act_batch(observations, deterministic=True)
    return Rollout(
        list(observations), [trainer.pilot._skill_indices[a[0]] for a in actions],
        [a[4] for a in actions], [a[2] for a in actions], [a[3] for a in actions],
        list(rng.normal(size=count)), [False] * (count - 1) + [True],
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("mixed_precision", [False, True])
def test_graphed_ppo_preserves_updates_scaler_and_remainder_batches(
    tmp_path, device, mixed_precision,
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required to test graph replay")
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = PilotTrainingConfig(
            hidden_size=16, transformer_heads=2, transformer_layers=1,
            history_duration_s=2, sample_interval_s=1, minibatch_size=8,
            update_epochs=3, mixed_precision=mixed_precision, output_dir=tmp_path / "eager",
        )
        eager = PPOTrainer(lambda *_: None, 3, config, device=device)
        graphed = PPOTrainer(
            lambda *_: None, 3,
            replace(config, cuda_graph_updates=True, output_dir=tmp_path / "graphed"),
            device=device,
        )
        initial = {key: value.clone() for key, value in eager.pilot.state_dict().items()}
        # Exercise partial batches, exact multiples, below-threshold batches, then
        # return to capture, all with persistent Adam/scaler state between updates.
        for count in (19, 16, 5, 21):
            rollout = _rollout(eager, count)
            torch.manual_seed(313)
            expected_loss = eager._update([rollout])
            torch.manual_seed(313)
            actual_loss = graphed._update([rollout])
            assert actual_loss == pytest.approx(expected_loss, rel=1e-6, abs=1e-6)
            assert eager.grad_scaler.state_dict() == graphed.grad_scaler.state_dict()
            for key, value in eager.pilot.state_dict().items():
                torch.testing.assert_close(value, graphed.pilot.state_dict()[key], rtol=0, atol=0)
            for left, right in zip(eager.optimizer.state.values(), graphed.optimizer.state.values()):
                for key, value in left.items():
                    torch.testing.assert_close(value, right[key], rtol=0, atol=0)
            profile = graphed._last_update_profile
            assert profile["cuda_graph_updates"] == int(device == "cuda" and count >= 8)
            assert profile["update_minibatches"] == 3 * ((count + 7) // 8)
        assert any(not torch.equal(initial[key], value)
                   for key, value in eager.pilot.state_dict().items())
    finally:
        torch.set_num_threads(old_threads)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph validation")
def test_graph_rejects_invalid_rollout_before_replay(tmp_path):
    config = PilotTrainingConfig(
        hidden_size=16, transformer_heads=2, transformer_layers=1,
        history_duration_s=2, sample_interval_s=1, minibatch_size=8,
        update_epochs=1, cuda_graph_updates=True, output_dir=tmp_path,
    )
    trainer = PPOTrainer(lambda *_: None, 3, config, device="cuda")
    rollout = _rollout(trainer, 8)
    rollout.parameters[0][0] = float("nan")
    with pytest.raises(ValueError, match="non-finite data"):
        trainer._update([rollout])
