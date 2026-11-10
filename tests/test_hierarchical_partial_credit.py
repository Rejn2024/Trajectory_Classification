"""Family recognition earns base credit; correct handedness earns only a joint bonus."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bvr_behavior_prediction import paired_skill_forecasting as paired
from tests.test_hierarchical_skill_forecasting import network


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def confidence_loss(family_probability, side_probability, family_reward=1., handedness_reward=1.):
    model = network(["launch", "notch_left", "notch_right"])
    family = torch.log(torch.tensor([[[1 - family_probability, family_probability]] * 3]))
    direction = torch.log(torch.tensor([[[[side_probability, 1 - side_probability]]] * 3]))
    labels = torch.ones(1, 3, dtype=torch.long)  # notch_left
    return paired.masked_hierarchical_skill_loss(family, direction, labels,
        paired.hierarchy_mapping_tensors(model.skill_hierarchy, "cpu"), hierarchical_loss="partial_credit",
        family_reward=family_reward, handedness_reward=handedness_reward)


def test_correct_family_with_wrong_side_gets_partial_credit_and_has_bounded_side_penalty():
    correct = confidence_loss(.99, .99)
    family_only = confidence_loss(.99, .01)
    wrong_family = confidence_loss(.01, .99)
    assert correct < family_only < wrong_family
    for p_family, p_side in ((.99, .99), (.99, .01), (.01, .99)):
        expected_credit = p_family * (1 + p_side) / 2
        # FP32 logits/loss are compared with a float64 analytical reference.
        assert confidence_loss(p_family, p_side).item() == pytest.approx(-np.log(expected_credit), rel=1e-6, abs=2e-7)
    extreme_side = confidence_loss(.99, 1e-30)
    assert extreme_side.item() - (-np.log(.99)) == pytest.approx(np.log(2), abs=1e-6)
    assert extreme_side < confidence_loss(.01, .999)


@pytest.mark.parametrize("a,b", [(2., 1.), (1., 3.), (.5, .1), (1., 0.)])
def test_reward_weights_and_disabled_side_bonus_follow_normalized_credit(a, b):
    loss = confidence_loss(.8, .25, a, b)
    assert loss.item() == pytest.approx(-np.log(.8 * (a + b * .25) / (a + b)), rel=2e-6)
    if b == 0:
        torch.testing.assert_close(confidence_loss(.8, .01, a, b), confidence_loss(.8, .99, a, b))


def test_leaf_weights_missing_labels_and_singletons_preserve_masks_and_normalization():
    model = network(["launch", "notch_left", "notch_right"])
    family = torch.zeros(2, 3, 2, requires_grad=True)
    direction = torch.zeros(2, 3, 1, 2, requires_grad=True)
    labels = torch.tensor([[0, 0, -1], [1, -1, -1]])
    weights = torch.tensor([[2., 4., 4.]] * 3)
    loss = paired.masked_hierarchical_skill_loss(family, direction, labels,
        paired.hierarchy_mapping_tensors(model.skill_hierarchy, "cpu"), weights, hierarchical_loss="partial_credit")
    singleton = np.log(2)
    pair = -np.log(.5 * (1 + .5) / 2)
    expected = ((2 * singleton + 4 * pair) / 2 + 2 * singleton) / 2
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert direction.grad[1, 0].abs().sum() > 0
    assert not direction.grad[0].any()  # Singleton has no side loss or artificial bonus.
    assert not direction.grad[:, 1:].any()
    assert not family.grad[:, 2].any()
    assert torch.isfinite(direction.grad).all() and torch.isfinite(family.grad).all()


def test_all_missing_and_extreme_wrong_direction_have_finite_loss_and_gradients():
    model = network(["launch", "notch_left", "notch_right"])
    family = torch.zeros(1, 3, 2, requires_grad=True)
    direction = torch.tensor([[[[-10000., 10000.]]] * 3], requires_grad=True)
    mapping = paired.hierarchy_mapping_tensors(model.skill_hierarchy, "cpu")
    loss = paired.masked_hierarchical_skill_loss(family, direction, torch.ones(1, 3, dtype=torch.long),
                                                mapping, hierarchical_loss="partial_credit")
    assert loss.item() == pytest.approx(np.log(4))
    loss.backward()
    assert torch.isfinite(family.grad).all() and torch.isfinite(direction.grad).all()
    family.grad = direction.grad = None
    missing = paired.masked_hierarchical_skill_loss(family, direction, torch.full((1, 3), -1),
                                                   mapping, hierarchical_loss="partial_credit")
    missing.backward()
    assert missing.item() == 0 and not family.grad.any() and not direction.grad.any()


def test_evaluation_reports_realized_credit_without_awarding_side_bonus_to_wrong_family(monkeypatch):
    model = network(["launch", "notch_left", "notch_right"]).eval()
    family_prob = torch.tensor([[[.01, .99]] * 3, [[.01, .99]] * 3, [[.99, .01]] * 3])
    direction_prob = torch.tensor([[[[.99, .01]]] * 3, [[[.01, .99]]] * 3, [[[.99, .01]]] * 3])
    family, direction = family_prob.log(), direction_prob.log()
    outputs = {"family_logits": family, "direction_logits": direction,
               "skill_log_probs": model.compose_skill_log_probabilities(family, direction),
               "intent_logits": torch.zeros(3, 3, 3)}
    monkeypatch.setattr(model, "forward", lambda *args, **kwargs: outputs)
    targets = np.ones((3, 3), dtype=np.int64)
    targets[2, 2] = -1
    data = SimpleNamespace(indices={"test": np.arange(3)}, targets=targets,
        classes=["launch", "notch_left", "notch_right"], intent_targets=targets.copy(),
        intent_classes=["offensive", "defensive", "context_dependent"],
        family_targets=targets.copy(), family_classes=["launch", "notch"],
        direction_targets=np.where(targets >= 0, 0, -1), direction_classes=["left", "right"])
    data.batch = lambda ids, device: (torch.zeros(len(ids), 51, 85), torch.as_tensor(targets[ids]))
    result = paired.evaluate(model, data, "test", "cpu", hierarchical_loss="partial_credit", intent_loss_weight=.25)
    reward = result["hierarchical_reward"]
    np.testing.assert_allclose(reward.hierarchical_credit, [.5, .5, .75])
    np.testing.assert_allclose(reward.family_accuracy, [2 / 3, 2 / 3, 1.])
    assert result["skill_data_loss"] < result["joint_skill_data_loss"]
    assert result["data_loss"] == pytest.approx(result["skill_data_loss"] + .25 * np.log(3))
    # Reported skill NLL remains the joint 33-class-style metric, not reward loss.
    assert result["summary"].nll.mean() == pytest.approx(result["joint_skill_data_loss"])


@pytest.mark.parametrize("settings", [dict(hierarchical_loss="unknown"), dict(family_reward=0),
    dict(family_reward=True), dict(family_reward=float("nan")), dict(handedness_reward=-1),
    dict(handedness_reward=float("inf"))])
def test_invalid_partial_credit_configuration_fails_before_writing(tmp_path, settings):
    with pytest.raises(ValueError):
        paired.train_model(None, tmp_path / "unused", {"architecture": "hgt_hierarchical_intent", **settings})
    assert not (tmp_path / "unused").exists()
