"""Conditional hybrid policy math, initialization and checkpoint compatibility."""

import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bvr_behavior_prediction.rl.config import PilotTrainingConfig
from bvr_behavior_prediction.rl.pilot import HybridSkillPilot


def model(conditioned=True, device="cpu"):
    return HybridSkillPilot(3, 8, history_steps=2, transformer_heads=2,
                           transformer_layers=1, selected_skill_parameters=True,
                           skill_conditioned_parameters=conditioned).to(device)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_initial_policy_and_random_draws_match_shared_head(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(71)
    baseline = model(False, device)
    expected_next = torch.rand(3)
    torch.manual_seed(71)
    pilot = model(True, device)
    torch.testing.assert_close(torch.rand(3), expected_next, rtol=0, atol=0)
    for key, value in baseline.state_dict().items():
        new = pilot.state_dict()[key]
        if key.startswith("parameter_"):
            new = new.reshape(len(pilot.skill_names), *value.shape)
            for head in new:
                torch.testing.assert_close(head, value, rtol=0, atol=0)
        else:
            torch.testing.assert_close(new, value, rtol=0, atol=0)
    observations = np.random.default_rng(4).normal(size=(40, 2, 3)).astype(np.float32)
    torch.manual_seed(9)
    before = baseline.act_batch(observations)
    torch.manual_seed(9)
    after = pilot.act_batch(observations)
    for left, right in zip(before, after):
        assert left[0] == right[0]
        assert left[1].keys() == right[1].keys()
        for name, value in left[1].items():
            if isinstance(value, (int, float)):
                assert right[1][name] == pytest.approx(value, rel=2e-5, abs=2e-5)
            else:
                assert right[1][name] == value
        np.testing.assert_allclose(left[2:4], right[2:4], rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(left[4], right[4], rtol=2e-5, atol=2e-5)
    obs = torch.tensor(observations, device=device)
    skills = torch.tensor([pilot._skill_indices[a[0]] for a in after], device=device)
    raw = torch.tensor(np.stack([a[4] for a in after]), device=device)
    logp, entropy, values = pilot.evaluate(obs, skills, raw)
    torch.testing.assert_close(logp.detach(), torch.tensor([a[2] for a in after], device=device))
    torch.testing.assert_close((logp.detach() - torch.tensor([a[2] for a in after], device=device)).exp(),
                               torch.ones(len(after), device=device))
    _, expected_entropy, expected_values = baseline.evaluate(obs, skills, raw)
    torch.testing.assert_close(entropy, expected_entropy)
    torch.testing.assert_close(values, expected_values)


def test_selected_head_probability_and_gradients_exclude_other_heads_and_coordinates():
    pilot = model()
    obs = torch.randn(1, 2, 3)
    index = pilot._skill_indices["maintain_heading"]
    skills = torch.tensor([index])
    with torch.no_grad():
        # Heads must be able to disagree even when two skills share a parameter.
        pilot.parameter_mean.bias.reshape(len(pilot.skill_names), -1).add_(
            torch.arange(len(pilot.skill_names))[:, None] / 10)
    categorical, gaussian, _ = pilot.distributions(obs)
    assert gaussian.mean.shape == (1, len(pilot.skill_names), len(pilot.parameter_names))
    assert not torch.equal(gaussian.mean[:, 0], gaussian.mean[:, 1])
    raw = gaussian.mean[:, index].detach() + 0.75
    mask = pilot._skill_parameter_mask[index].bool()
    logp, _, _ = pilot.evaluate(obs, skills, raw)
    expected = categorical.log_prob(skills) + torch.distributions.Normal(
        gaussian.loc[:, index, mask], gaussian.scale[:, index, mask]
    ).log_prob(raw[:, mask]).sum(-1)
    torch.testing.assert_close(logp, expected)
    changed = raw.clone()
    changed[:, ~mask] += 50
    actual, _, _ = pilot.evaluate(obs, skills, changed)
    torch.testing.assert_close(actual, logp, rtol=0, atol=0)
    (-logp.sum()).backward()
    for gradient in (pilot.parameter_mean.bias.grad.reshape(len(pilot.skill_names), -1),
                     pilot.parameter_log_std.grad):
        assert torch.count_nonzero(gradient[index, mask]) == int(mask.sum())
        assert torch.count_nonzero(gradient[index, ~mask]) == 0
        assert torch.count_nonzero(gradient[torch.arange(len(pilot.skill_names)) != index]) == 0
    with torch.no_grad():
        pilot.skill_head.weight.zero_()
        pilot.skill_head.bias.fill_(-100)
        pilot.skill_head.bias[index] = 100
    action = pilot.act_batch(obs.numpy(), deterministic=True)[0]
    assert action[0] == "maintain_heading"
    np.testing.assert_allclose(action[4], gaussian.mean[:, index].detach().numpy()[0])
    assert action[1] == pilot.decode_parameters(index, action[4])


def test_entropy_integrates_conditional_heads_and_retains_categorical_gradient():
    pilot = model()
    with torch.no_grad():
        pilot.parameter_log_std.copy_(torch.linspace(-1, 0.5, pilot.parameter_log_std.numel()).reshape_as(
            pilot.parameter_log_std))
    obs = torch.randn(2, 2, 3)
    categorical, gaussian, _ = pilot.distributions(obs)
    expected = categorical.entropy()
    for index in range(len(pilot.skill_names)):
        mask = pilot._skill_parameter_mask[index].bool()
        expected = expected + categorical.probs[:, index] * gaussian.entropy()[:, index, mask].sum(-1)
    raw = gaussian.mean[:, 0].detach()
    _, entropy, _ = pilot.evaluate(obs, torch.tensor([0, 1]), raw)
    _, other, _ = pilot.evaluate(obs, torch.tensor([2, 3]), raw)
    torch.testing.assert_close(entropy, expected)
    torch.testing.assert_close(entropy, other, rtol=0, atol=0)
    actual_grad = torch.autograd.grad(entropy.sum(), (pilot.skill_head.weight, pilot.parameter_log_std))
    expected_grad = torch.autograd.grad(expected.sum(), (pilot.skill_head.weight, pilot.parameter_log_std))
    for actual, target in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, target)
    assert torch.count_nonzero(actual_grad[0]) > 0
    assert torch.count_nonzero(actual_grad[1][pilot._skill_parameter_mask == 0]) == 0


@pytest.mark.parametrize("flag", [None, False, True])
def test_loads_conditioned_and_historical_checkpoints(tmp_path, flag):
    from bvr_behavior_prediction.rl.multi_pilot import load_trained_pilot

    pilot = model(bool(flag))
    config = PilotTrainingConfig(hidden_size=8, transformer_heads=2, transformer_layers=1,
                                 history_duration_s=2, sample_interval_s=1,
                                 selected_skill_parameters=True, skill_conditioned_parameters=bool(flag))
    training = config.as_dict()
    if flag is None:
        del training["skill_conditioned_parameters"]
    torch.save(pilot.state_dict(), tmp_path / "best_model.pt")
    (tmp_path / "config.json").write_text(json.dumps({
        "training": training, "observation_size": 3,
        "skill_names": pilot.skill_names, "parameter_names": pilot.parameter_names,
    }))
    (tmp_path / "manifest.json").write_text(json.dumps({
        "checkpoint_sha256": hashlib.sha256((tmp_path / "best_model.pt").read_bytes()).hexdigest(),
    }))
    loaded = load_trained_pilot(tmp_path)
    assert loaded.skill_conditioned_parameters is bool(flag)
    for key, value in pilot.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[key], rtol=0, atol=0)


def test_config_requires_boolean_and_selected_skill_objective():
    assert PilotTrainingConfig().skill_conditioned_parameters is False
    with pytest.raises(ValueError, match="boolean"):
        PilotTrainingConfig(skill_conditioned_parameters="false")
    with pytest.raises(ValueError, match="require selected_skill_parameters"):
        PilotTrainingConfig(skill_conditioned_parameters=True)
    with pytest.raises(ValueError, match="require selected_skill_parameters"):
        HybridSkillPilot(3, skill_conditioned_parameters=True)
