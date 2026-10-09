"""The hybrid action objective must ignore unused values without changing actions."""

import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bvr_behavior_prediction.rl.config import PilotTrainingConfig
from bvr_behavior_prediction.rl.pilot import HybridSkillPilot


@pytest.fixture
def pilot():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(123)
    yield HybridSkillPilot(3, 8, history_steps=2, transformer_heads=2,
                           transformer_layers=1, selected_skill_parameters=True)
    torch.set_num_threads(previous)


def active_indices(pilot, skill_index):
    properties = pilot.manager.get_contract(pilot.skill_names[skill_index])["parameter_schema"]["properties"]
    return [pilot.parameter_names.index(name) for name, spec in properties.items()
            if spec["type"] == "number"]


def test_actions_architecture_and_sampling_match_legacy(pilot):
    torch.manual_seed(123)
    legacy = HybridSkillPilot(3, 8, history_steps=2, transformer_heads=2, transformer_layers=1)
    assert pilot.state_dict().keys() == legacy.state_dict().keys()
    for name, value in legacy.state_dict().items():
        torch.testing.assert_close(value, pilot.state_dict()[name], rtol=0, atol=0)
    observations = np.random.default_rng(77).normal(size=(40, 2, 3)).astype(np.float32)
    torch.manual_seed(9)
    before = legacy.act_batch(observations)
    torch.manual_seed(9)
    after = pilot.act_batch(observations)
    for old, new in zip(before, after):
        assert old[:2] == new[:2]  # Skill and simulator-bound parameter dictionary.
        assert old[3] == new[3]  # Critic prediction.
        np.testing.assert_array_equal(old[4], new[4])
    skills = torch.tensor([pilot._skill_indices[a[0]] for a in after])
    raw = torch.tensor(np.stack([a[4] for a in after]))
    logp, _, _ = pilot.evaluate(torch.tensor(observations), skills, raw)
    torch.testing.assert_close(logp.detach(), torch.tensor([a[2] for a in after]))
    torch.testing.assert_close((logp.detach() - torch.tensor([a[2] for a in after])).exp(),
                               torch.ones(len(after)))


def test_log_probability_and_gradient_ignore_only_unused_parameters(pilot):
    obs = torch.randn(1, 2, 3)
    skill_index = pilot._skill_indices["maintain_heading"]
    skills = torch.tensor([skill_index])
    active = active_indices(pilot, skill_index)
    inactive = [j for j in range(len(pilot.parameter_names)) if j not in active]
    categorical, gaussian, _ = pilot.distributions(obs)
    raw = gaussian.mean.detach() + 1.0
    logp, _, _ = pilot.evaluate(obs, skills, raw)
    expected = categorical.log_prob(skills) + gaussian.log_prob(raw)[:, active].sum(-1)
    torch.testing.assert_close(logp, expected)
    changed = raw.clone()
    changed[:, inactive] += 100.0
    unchanged, _, _ = pilot.evaluate(obs, skills, changed)
    torch.testing.assert_close(logp, unchanged, rtol=0, atol=0)
    changed[:, active[0]] += 1.0
    different, _, _ = pilot.evaluate(obs, skills, changed)
    assert not torch.allclose(logp, different)
    (-logp.sum()).backward()
    assert torch.count_nonzero(pilot.parameter_mean.bias.grad[inactive]) == 0
    assert torch.count_nonzero(pilot.parameter_log_std.grad[inactive]) == 0
    assert torch.count_nonzero(pilot.parameter_mean.bias.grad[active]) == len(active)


def test_entropy_averages_all_skills_and_keeps_skill_probability_gradients(pilot):
    obs = torch.randn(2, 2, 3)
    categorical, gaussian, _ = pilot.distributions(obs)
    expected = categorical.entropy()
    for index in range(len(pilot.skill_names)):
        expected = expected + categorical.probs[:, index] * gaussian.entropy()[
            :, active_indices(pilot, index)
        ].sum(-1)
    raw = gaussian.mean.detach()
    _, entropy, _ = pilot.evaluate(obs, torch.tensor([0, 1]), raw)
    _, other_entropy, _ = pilot.evaluate(obs, torch.tensor([2, 3]), raw)
    torch.testing.assert_close(entropy, expected)
    torch.testing.assert_close(entropy, other_entropy, rtol=0, atol=0)
    actual_gradient = torch.autograd.grad(entropy.sum(), pilot.skill_head.weight)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), pilot.skill_head.weight)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert torch.count_nonzero(actual_gradient) > 0


@pytest.mark.parametrize("saved_flag", [None, False, True])
def test_checkpoint_loader_restores_objective_and_accepts_legacy_metadata(pilot, tmp_path, saved_flag):
    from bvr_behavior_prediction.rl.multi_pilot import load_trained_pilot

    config = PilotTrainingConfig(hidden_size=8, transformer_heads=2, transformer_layers=1,
                                 history_duration_s=2, sample_interval_s=1,
                                 selected_skill_parameters=bool(saved_flag))
    training = config.as_dict()
    if saved_flag is None:
        del training["selected_skill_parameters"]
    torch.save(pilot.state_dict(), tmp_path / "best_model.pt")
    (tmp_path / "config.json").write_text(json.dumps({
        "training": training, "observation_size": 3,
        "skill_names": pilot.skill_names, "parameter_names": pilot.parameter_names,
    }))
    (tmp_path / "manifest.json").write_text(json.dumps({
        "checkpoint_sha256": hashlib.sha256((tmp_path / "best_model.pt").read_bytes()).hexdigest(),
    }))
    loaded = load_trained_pilot(tmp_path)
    assert loaded.selected_skill_parameters is bool(saved_flag)
    for name, value in pilot.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[name], rtol=0, atol=0)


def test_mode_requires_a_boolean_and_defaults_to_legacy():
    assert PilotTrainingConfig().selected_skill_parameters is False
    with pytest.raises(ValueError, match="boolean"):
        PilotTrainingConfig(selected_skill_parameters="false")
