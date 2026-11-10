"""Conditional L/R balancing, independent family balance, and direction-only gradients."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bvr_behavior_prediction import paired_skill_forecasting as paired
from bvr_behavior_prediction import skill_intent_forecasting as intent


CLASSES = ["beam_missile_left", "beam_missile_right", "launch", "notch_left", "notch_right"]


def fixture_data(train_only=False):
    labels = np.array([
        [3] * 9 + [4] + [0] + [1] * 9 + [2] * 8,
        [3] + [4] * 9 + [2] * 18,
        [4] * 10 + [2] * 17 + [-1],
    ]).T
    train_ids = np.arange(len(labels))

    class TrainingTargetsOnly:
        def __getitem__(self, ids):
            np.testing.assert_array_equal(ids, train_ids)
            return labels

    data = SimpleNamespace(classes=CLASSES,
        targets=TrainingTargetsOnly() if train_only else labels,
        manifest={"skill_hierarchy": intent.hierarchy_policy(CLASSES)},
        indices={"train": train_ids, "test": np.array([100]), "validation": np.array([200])})
    return data


def test_handedness_weights_are_conditional_per_pair_horizon_and_training_only():
    data = fixture_data(train_only=True)
    state = paired.fit_handedness_weighting(data, {"handedness_weighting": "sqrt_inverse"})
    counts, weights = np.asarray(state["counts"]), np.asarray(state["weights"])
    np.testing.assert_array_equal(counts, [[[1, 9], [9, 1]], [[0, 0], [1, 9]], [[0, 0], [0, 10]]])
    assert weights[0, 0, 0] / weights[0, 0, 1] == pytest.approx(3)
    assert weights[0, 1, 1] / weights[0, 1, 0] == pytest.approx(3)
    assert weights[1, 1, 0] / weights[1, 1, 1] == pytest.approx(3)  # Imbalance flips at +2s.
    for h in range(3):
        for p in range(2):
            if counts[h, p].sum():
                assert np.dot(counts[h, p], weights[h, p]) / counts[h, p].sum() == pytest.approx(1)
            else:
                assert not weights[h, p].any()
    np.testing.assert_array_equal(weights[2, 1], [0, 1])  # A single supported side has unit weight.
    assert not weights[counts == 0].any()
    table = paired.handedness_weight_table(state)
    assert table.shape == (12, 6)
    assert table.training_support.sum() == counts.sum()
    assert paired.handedness_weight_tensor(state, "cpu").shape == (3, 2, 2)


@pytest.mark.parametrize("mode,cap,expected_ratio", [("sqrt_inverse", 2., 2.), ("inverse", 5., 5.),
                                                    ("sqrt_inverse", 1., 1.)])
def test_conditional_weight_ratios_are_capped_before_normalization(mode, cap, expected_ratio):
    state = paired.fit_handedness_weighting(fixture_data(),
        {"handedness_weighting": mode, "max_handedness_weight_ratio": cap})
    weights = np.asarray(state["weights"])
    assert weights[0, 1, 1] / weights[0, 1, 0] == pytest.approx(expected_ratio)
    assert np.isfinite(weights).all() and (weights >= 0).all() and (weights <= cap + 1e-12).all()


def test_family_weights_pool_mirror_leaves_and_do_not_reapply_side_imbalance():
    data = fixture_data(train_only=True)
    state = paired.fit_class_weighting(data, {**paired.DEFAULT_TRAINING, "class_weighting": "family_sqrt_inverse"})
    weights = np.asarray(state["weights"])
    counts = np.asarray(state["counts"])
    np.testing.assert_array_equal(weights[:, 0], weights[:, 1])
    np.testing.assert_array_equal(weights[:, 3], weights[:, 4])
    np.testing.assert_allclose((weights * counts).sum(1) / counts.sum(1), 1)
    assert state["weighting_level"] == "skill_family"
    table = paired.class_weight_table(state)
    assert "family" in table and set(table.family) == {"beam_missile", "launch", "notch"}
    assert weights[2, 3] == weights[2, 4] > 0  # Known family weight survives an unseen left side.


@pytest.mark.parametrize("objective", ["partial_credit", "joint_nll"])
def test_rare_side_gradient_is_upweighted_without_changing_family_gradients(objective):
    data = fixture_data()
    state = paired.fit_handedness_weighting(data, {"handedness_weighting": "sqrt_inverse"})
    hand_weights = paired.handedness_weight_tensor(state, "cpu")
    mapping = paired.hierarchy_mapping_tensors(data.manifest["skill_hierarchy"], "cpu")
    family = torch.zeros(2, 3, 3, requires_grad=True)
    direction = torch.zeros(2, 3, 2, 2, requires_grad=True)
    targets = torch.tensor([[3, -1, -1], [4, -1, -1]])
    balanced = paired.masked_hierarchical_skill_loss(family, direction, targets, mapping,
        hierarchical_loss=objective, handedness_weights=hand_weights)
    family_grad, side_grad = torch.autograd.grad(balanced, (family, direction), retain_graph=True)
    unbalanced = paired.masked_hierarchical_skill_loss(family, direction, targets, mapping, hierarchical_loss=objective)
    original_family_grad = torch.autograd.grad(unbalanced, family)[0]
    torch.testing.assert_close(family_grad, original_family_grad, atol=0, rtol=0)
    assert side_grad[1, 0, 1].abs().sum() / side_grad[0, 0, 1].abs().sum() == pytest.approx(3)
    assert not side_grad[:, 1:].any() and not side_grad[:, :, 0].any()
    assert torch.isfinite(side_grad).all()


def test_unseen_side_singleton_and_missing_labels_do_not_gain_direction_loss():
    data = fixture_data()
    state = paired.fit_handedness_weighting(data, {"handedness_weighting": "sqrt_inverse"})
    mapping = paired.hierarchy_mapping_tensors(data.manifest["skill_hierarchy"], "cpu")
    family = torch.zeros(1, 3, 3, requires_grad=True)
    direction = torch.zeros(1, 3, 2, 2, requires_grad=True)
    # +5s left notch was never observed in training; family remains supervised.
    targets = torch.tensor([[2, -1, 3]])
    loss = paired.masked_hierarchical_skill_loss(family, direction, targets, mapping,
        hierarchical_loss="partial_credit", handedness_weights=paired.handedness_weight_tensor(state, "cpu"))
    loss.backward()
    assert loss.item() == pytest.approx(np.log(3))
    assert not direction.grad.any()
    assert family.grad[0, 2].abs().sum() > 0 and not family.grad[0, 1].any()


def test_no_mirrored_families_and_disabled_weighting_are_supported():
    assert paired.fit_handedness_weighting(None, {}) is None
    policy = intent.hierarchy_policy(["launch", "maintain_heading"])
    data = SimpleNamespace(classes=["launch", "maintain_heading"], targets=np.zeros((2, 3), dtype=np.int64),
                           indices={"train": np.arange(2)}, manifest={"skill_hierarchy": policy})
    state = paired.fit_handedness_weighting(data, {"handedness_weighting": "sqrt_inverse"})
    assert paired.handedness_weight_tensor(state, "cpu").shape == (3, 0, 2)
    assert paired.handedness_weight_table(state).empty
    targets = torch.zeros(2, 3, dtype=torch.long)
    selected = paired.target_handedness_weights(targets, paired.hierarchy_mapping_tensors(policy, "cpu"),
                                                paired.handedness_weight_tensor(state, "cpu"))
    torch.testing.assert_close(selected, torch.ones(2, 3))


@pytest.mark.parametrize("settings", [{"handedness_weighting": "unknown"},
    {"handedness_weighting": "sqrt_inverse", "max_handedness_weight_ratio": 0},
    {"handedness_weighting": "sqrt_inverse", "max_handedness_weight_ratio": True},
    {"handedness_weighting": "sqrt_inverse", "max_handedness_weight_ratio": float("nan")}])
def test_invalid_handedness_configuration_fails_before_writing(tmp_path, settings):
    with pytest.raises(ValueError):
        paired.train_model(None, tmp_path / "unused", {"architecture": "hgt_hierarchical_intent", **settings})
    assert not (tmp_path / "unused").exists()
