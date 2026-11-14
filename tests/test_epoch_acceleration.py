"""Action validation and graph caching preserve valid policy/RNG semantics."""

from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mlflow")

from bvr_behavior_prediction.rl.config import PilotTrainingConfig
from bvr_behavior_prediction.rl.pilot import HybridSkillPilot
from bvr_behavior_prediction.rl.trainer import PPOTrainer


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("conditioned", [False, True])
@pytest.mark.parametrize("deterministic", [False, True])
def test_deferred_validation_preserves_actions_and_rng(device, conditioned, deterministic):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pilot = HybridSkillPilot(3, 8, history_steps=2, transformer_heads=2,
                             transformer_layers=1, skill_conditioned_parameters=conditioned).to(device)
    observations = torch.randn(7, 2, 3, device=device)
    with torch.inference_mode():
        torch.manual_seed(313)
        expected = pilot._rollout_tensors(observations, deterministic, defer_validation=False)
        expected_rng = torch.get_rng_state()
        expected_cuda_rng = torch.cuda.get_rng_state() if device == "cuda" else None
        torch.manual_seed(313)
        actual = pilot._rollout_tensors(observations, deterministic, defer_validation=True)
        for left, right in zip(expected[:4], actual[:4]):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        assert torch.equal(expected_rng, torch.get_rng_state())
        if device == "cuda":
            assert torch.equal(expected_cuda_rng, torch.cuda.get_rng_state())
        received = pilot._rollout_outputs_to_cpu(*actual[:4], valid=actual[4])
        np.testing.assert_array_equal(received[1], expected[1].cpu().numpy())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("bad_head", ["skill", "mean", "scale", "value", "observation"])
def test_invalid_distributions_fail_before_actions_without_poisoning_device(device, bad_head):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pilot = HybridSkillPilot(3, 8, history_steps=2, transformer_heads=2,
                             transformer_layers=1).to(device)
    observations = torch.zeros(3, 2, 3, device=device)
    with torch.no_grad():
        if bad_head == "skill":
            pilot.skill_head.bias.fill_(float("nan"))
        elif bad_head == "mean":
            pilot.parameter_mean.bias.fill_(float("inf"))
        elif bad_head == "scale":
            pilot.parameter_log_std.fill_(float("nan"))
        elif bad_head == "value":
            pilot.value_head.bias.fill_(float("inf"))
        else:
            observations.fill_(float("nan"))
    with torch.inference_mode():
        outputs = pilot._rollout_tensors(observations, False, defer_validation=True)
        with pytest.raises(ValueError, match="Nonfinite"):
            pilot._rollout_outputs_to_cpu(*outputs[:4], valid=outputs[4])
    # A device-side assertion would surface here and break all later CUDA work.
    assert torch.ones(3, device=device).sum().item() == 3


def test_score_only_rejects_nonfinite_logits_and_unused_parameter_heads():
    pilot = HybridSkillPilot(3, 8, history_steps=2, transformer_heads=2,
                             transformer_layers=1, skill_conditioned_parameters=True)
    observations = np.zeros((3, 2, 3), np.float32)
    with torch.no_grad():
        pilot.skill_head.bias.fill_(float("nan"))
    with pytest.raises(ValueError, match="Nonfinite"):
        pilot.act_deterministic_batch(observations)
    with torch.no_grad():
        pilot.skill_head.bias.fill_(0)
        pilot.parameter_mean.bias[-1] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        pilot.act_deterministic_batch(observations)


def test_deferred_distribution_checks_do_not_read_tensor_booleans():
    pilot = HybridSkillPilot(3, 8, history_steps=2, transformer_heads=2, transformer_layers=1)
    observations = torch.zeros(3, 2, 3)
    with torch.inference_mode(), torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
    ) as profile:
        pilot._rollout_tensors(observations, False, defer_validation=True)
    assert not any(event.key == "aten::is_nonzero" for event in profile.key_averages())


def test_graph_cache_reuses_and_invalidates_captured_storage(tmp_path, monkeypatch):
    import bvr_behavior_prediction.rl.trainer as training

    created = []

    class Graph:
        def __init__(self, *args):
            self.closed = False
            created.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(training, "CUDABatchBackward", Graph)
    config = PilotTrainingConfig(hidden_size=8, transformer_heads=2, transformer_layers=1,
                                 output_dir=tmp_path)
    trainer = PPOTrainer(lambda *_: None, 3, config, "cpu")
    batch = (torch.zeros(8, 2, 3), torch.zeros(8, dtype=torch.long),
             torch.zeros(8, 3), torch.zeros(8), torch.zeros(8), torch.zeros(8))
    first, _, hit = trainer._get_cuda_backward(batch)
    assert not hit
    again, seconds, hit = trainer._get_cuda_backward(tuple(t.clone() for t in batch))
    assert again is first and seconds == 0 and hit
    # In-place policy updates/load_state_dict keep capture-time storage valid.
    with torch.no_grad():
        next(trainer.pilot.parameters()).add_(1)
    assert trainer._get_cuda_backward(batch)[0] is first
    trainer.config = replace(config, entropy_coefficient=.02)
    second = trainer._get_cuda_backward(batch)[0]
    assert second is not first and first.closed
    trainer.pilot.eval()
    third = trainer._get_cuda_backward(batch)[0]
    assert third is not second and second.closed
    parameter = next(trainer.pilot.parameters())
    parameter.data = parameter.data.clone()
    fourth = trainer._get_cuda_backward(batch)[0]
    assert fourth is not third and third.closed
    fifth = trainer._get_cuda_backward(tuple(t[:4] for t in batch))[0]
    assert fifth is not fourth and fourth.closed
    smaller = tuple(t[:4] for t in batch)
    trainer.grad_scaler._scale = torch.tensor(128.)
    sixth = trainer._get_cuda_backward(smaller)[0]
    assert sixth is not fifth and fifth.closed
    trainer.grad_scaler._scale.fill_(256.)
    assert trainer._get_cuda_backward(smaller)[0] is sixth
    trainer.grad_scaler._scale = trainer.grad_scaler._scale.clone()
    seventh = trainer._get_cuda_backward(smaller)[0]
    assert seventh is not sixth and sixth.closed
    trainer.close()
    assert seventh.closed and trainer._cuda_backward is None
    assert len(created) == 7
