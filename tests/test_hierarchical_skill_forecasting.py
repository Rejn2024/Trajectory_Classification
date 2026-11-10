"""Family/direction partition, normalized leaf probabilities and exact hierarchical NLL."""

import copy

import numpy as np
import pytest
import torch

from bvr_behavior_prediction import paired_skill_forecasting as paired
from bvr_behavior_prediction import skill_intent_forecasting as intent
from bvr_behavior_prediction.enemy_missile_models import HGTHierarchicalIntentTransformer
from tests.test_hgt_missile_models import target_example
from tests.test_skill_intent_forecasting import prepare_base, ROOT


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def network(classes=None, **kwargs):
    classes = paired.catalogue(ROOT) if classes is None else classes
    policy = intent.hierarchy_policy(classes)
    return HGTHierarchicalIntentTransformer(num_classes=len(classes), input_features=82 + len(classes),
        skill_hierarchy=policy, width=12, heads=6, layers=1, graph_width=8, graph_heads=2,
        graph_layers=1, dropout=0, observer_acceleration=True, target_relative_missiles=True,
        target_relative_missile_velocities=True, **kwargs)


def test_full_catalogue_has_29_families_four_complete_pairs_and_correct_masks():
    classes = paired.catalogue(ROOT)
    policy = intent.hierarchy_policy(classes)
    assert len(policy["families"]) == 29
    assert policy["paired_families"] == ["beam_missile", "beam_target", "crank_target", "notch"]
    assert policy["directions"] == ["left", "right"]
    for family, names in intent.MIRROR_PAIRS.items():
        left, right = [classes.index(name) for name in names]
        assert policy["skill_to_family"][left] == policy["skill_to_family"][right] == policy["families"].index(family)
        assert [policy["skill_to_direction"][left], policy["skill_to_direction"][right]] == [0, 1]
    labels = np.array([[classes.index("notch_left"), classes.index("launch"), -1]])
    family_y = intent.map_intent_targets(labels, policy["skill_to_family"])
    direction_y = intent.map_intent_targets(labels, policy["skill_to_direction"])
    np.testing.assert_array_equal(family_y, [[policy["families"].index("notch"), policy["families"].index("launch"), -1]])
    np.testing.assert_array_equal(direction_y, [[0, -1, -1]])
    with pytest.raises(ValueError, match="Incomplete mirror pair"):
        intent.hierarchy_policy(["launch", "notch_left"])


def test_joint_probabilities_are_normalized_and_singletons_have_no_direction_mass():
    model = network()
    torch.manual_seed(71)
    families = torch.randn(2, 3, 29, requires_grad=True)
    directions = torch.randn(2, 3, 4, 2, requires_grad=True)
    p = model.compose_skill_log_probabilities(families, directions).exp()
    torch.testing.assert_close(p.sum(-1), torch.ones(2, 3))
    for i, name in enumerate(model.skill_hierarchy["classes"]):
        f = model.skill_hierarchy["skill_to_family"][i]
        d = model.skill_hierarchy["skill_to_direction"][i]
        expected = families.softmax(-1)[:, :, f]
        if d >= 0:
            pair = model.skill_hierarchy["skill_to_pair"][i]
            expected = expected * directions.softmax(-1)[:, :, pair, d]
        torch.testing.assert_close(p[:, :, i], expected)


@pytest.mark.parametrize("weighted,missing", [(False, False), (True, False), (True, True)])
def test_hierarchical_loss_and_gradients_equal_weighted_joint_leaf_nll(weighted, missing):
    model = network()
    torch.manual_seed(72)
    family_logits = torch.randn(2, 3, 29, requires_grad=True)
    direction_logits = torch.randn(2, 3, 4, 2, requires_grad=True)
    classes = model.skill_hierarchy["classes"]
    targets = torch.tensor([[classes.index("launch"), classes.index("notch_right"), -1],
                            [classes.index("crank_target_left"), -1, classes.index("maintain_heading")]])
    if missing:
        targets[:] = -1
    weights = torch.rand(3, 33) + .1 if weighted else None
    mapping = paired.hierarchy_mapping_tensors(model.skill_hierarchy, "cpu")
    actual = paired.masked_hierarchical_skill_loss(family_logits, direction_logits, targets, mapping, weights)
    leaf = model.compose_skill_log_probabilities(family_logits, direction_logits)
    expected = paired.masked_loss(leaf, targets, weights)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    actual_grads = torch.autograd.grad(actual, (family_logits, direction_logits), retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (family_logits, direction_logits))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-6, rtol=1e-6)
        assert torch.isfinite(actual_grad).all()
    if missing:
        assert actual.item() == 0 and not any(gradient.any() for gradient in actual_grads)


def test_only_true_pair_direction_is_supervised_even_when_family_prediction_is_wrong():
    model = network()
    classes = model.skill_hierarchy["classes"]
    family = torch.zeros(1, 3, 29)
    notch = model.skill_hierarchy["families"].index("notch")
    family[:, :, notch] = -100  # The family prediction is deliberately very wrong.
    family.requires_grad_()
    direction = torch.zeros(1, 3, 4, 2, requires_grad=True)
    targets = torch.tensor([[classes.index("notch_left"), classes.index("launch"), -1]])
    loss = paired.masked_hierarchical_skill_loss(family, direction, targets,
                                                paired.hierarchy_mapping_tensors(model.skill_hierarchy, "cpu"))
    loss.backward()
    pair = model.skill_hierarchy["paired_families"].index("notch")
    assert direction.grad[0, 0, pair].abs().sum() > 0
    other = direction.grad.clone()
    other[0, 0, pair] = 0
    assert not other.any()
    assert not family.grad[0, 2].any()


def test_direction_uses_all_valid_skill_labels_as_denominator_and_greedy_can_differ_from_joint():
    model = network(["launch", "notch_left", "notch_right"])
    family_logits = torch.zeros(2, 3, 2, requires_grad=True)
    direction_logits = torch.zeros(2, 3, 1, 2, requires_grad=True)
    labels = torch.tensor([[0, 0, 0], [1, 1, 1]])
    loss = paired.masked_hierarchical_skill_loss(family_logits, direction_logits, labels,
                                                paired.hierarchy_mapping_tensors(model.skill_hierarchy, "cpu"))
    assert loss.item() == pytest.approx(1.5 * np.log(2))
    family_logits = torch.log(torch.tensor([[[.45, .55]] * 3]))
    direction_logits = torch.zeros(1, 3, 1, 2)
    joint = model.compose_skill_log_probabilities(family_logits, direction_logits)
    assert (joint.argmax(-1) == 0).all()  # Launch has .45 versus .275 for each notch direction.
    greedy = model.decode_hierarchy(family_logits, direction_logits)
    assert (greedy["skill_indices"] == 1).all()  # Family-first chooses notch, then left on the tie.


def test_no_directional_skills_and_empty_direction_horizons_are_valid():
    model = network(["launch", "maintain_heading"])
    result = model(torch.zeros(1, 51, 84), return_hierarchy=True)
    assert result["direction_logits"].shape == (1, 3, 0, 2)
    torch.testing.assert_close(result["skill_log_probs"].exp().sum(-1), torch.ones(1, 3))
    _, _, matrices = paired.metric_tables(np.full((1, 3), -1), np.full((1, 3, 2), .5),
                                          ["left", "right"], "test", allow_empty=True)
    assert not matrices.sum()


@pytest.mark.parametrize("autocast", [False, True])
def test_forward_joint_loss_shared_gradients_and_factory_reload(autocast):
    raw, mean, std = target_example()
    model = network(mean=mean, std=std).eval()
    x = (raw - mean) / std
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        outputs = model(x, return_hierarchy=True)
    assert outputs["family_logits"].shape == (2, 3, 29)
    assert outputs["direction_logits"].shape == (2, 3, 4, 2)
    policy = model.skill_hierarchy
    labels = torch.full((2, 3), policy["classes"].index("notch_right"), dtype=torch.long)
    loss = paired.masked_hierarchical_skill_loss(outputs["family_logits"], outputs["direction_logits"], labels,
                                                paired.hierarchy_mapping_tensors(policy, "cpu"))
    loss = loss + .25 * paired.masked_loss(outputs["intent_logits"], torch.ones_like(labels))
    loss.backward()
    assert model.family_head[-1].weight.grad.abs().sum() > 0
    assert model.direction_head[-1].weight.grad.abs().sum() > 0
    assert model.intent_head[-1].weight.grad.abs().sum() > 0
    assert model.messages[0].edge_attention.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    config = dict(architecture="hgt_hierarchical_intent", width=12, heads=6, layers=1,
                  dropout=0, graph_width=8, graph_heads=2, graph_layers=1,
                  observer_acceleration=True, target_relative_missiles=True, target_relative_missile_velocities=True)
    settings = paired.model_configuration(config, 115, policy)
    restored = paired.make_model(33, settings).eval()
    restored.load_state_dict(model.state_dict())
    with torch.inference_mode():
        torch.testing.assert_close(restored(x), model(x), atol=0, rtol=0)


def test_dataset_rejects_tampered_direction_labels_even_after_rehash(tmp_path, monkeypatch):
    base = prepare_base(tmp_path, monkeypatch)
    output = tmp_path / "hierarchy"
    intent.prepare_intent_dataset(base, output)
    data = intent.IntentNNDuelData(output)
    labels = np.load(output / "direction_targets.npy")
    row, horizon = np.argwhere(labels >= 0)[0]
    labels[row, horizon] = 1 - labels[row, horizon]
    np.save(output / "direction_targets.npy", labels)
    manifest = intent.read_json(output / "manifest.json")
    manifest["files"]["direction_targets.npy"] = intent.sha256(output / "direction_targets.npy")
    intent.write_json(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Direction labels"):
        intent.IntentNNDuelData(output)
    manifest["data_contract"] = "nn-duel-target-relative-missile-skill-intent-horizon-v1"
    intent.write_json(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="rerun notebook 26"):
        intent.IntentNNDuelData(output)


@pytest.mark.skipif(not hasattr(torch.export, "Dim"), reason="Dynamic export requires newer PyTorch")
def test_hierarchical_model_dynamic_export_preserves_original_skill_interface():
    model = network().eval()
    windows = torch.zeros(2, 51, 115)
    program = torch.export.export(model, (windows,), dynamic_shapes=(
        {0: torch.export.Dim("batch", min=1, max=4)},))
    with torch.inference_mode():
        torch.testing.assert_close(program.module()(windows[:1]), model(windows[:1]))
