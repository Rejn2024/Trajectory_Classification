"""HGT attention, target-distance weighting, masking and portable configuration."""

import pytest
import torch

from bvr_behavior_prediction.enemy_missile_models import HGTTransformer
from bvr_behavior_prediction.paired_skill_forecasting import make_model, model_configuration
from tests.test_enemy_missile_models import example


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def model(mean=None, std=None, **kwargs):
    return HGTTransformer(width=12, heads=6, layers=1, graph_width=8, graph_heads=2,
                          graph_layers=2, dropout=0, mean=mean, std=std, **kwargs)


def target_example():
    raw, mean, std = example()
    order = [*range(3, 54), 0, 1, 2, *range(54, 115)]
    return raw[:, :, order], mean[order], std[order]


def attention_call(network, raw, slots, nodes, present):
    geometry = network._edge_geometry(raw, slots)
    prior = network._edge_log_prior(geometry, present)
    result, attention = network.messages[0](
        nodes, present, geometry, network.sources, network.destinations,
        network.relations, network.roles, prior, return_attention=True)
    return result, attention, prior


def test_distance_prior_uses_target_not_owner_and_both_directions():
    network = model(observer_acceleration=True, target_relative_missiles=True,
                    target_relative_missile_velocities=True)
    raw = torch.zeros(1, 84)
    raw[:, 6] = 1  # Opponent at 50 km from observer.
    slots = torch.zeros(1, 8, 7)
    slots[0, :4, 0] = torch.tensor([.02, .18, 0, .02])  # Friendly target distances: 1, 9, absent, 1 km.
    slots[0, 4:, 0] = torch.tensor([.4, .04, 0, 0])     # Hostile target distances: 20, 2 km.
    present = torch.tensor([[True, True, True, True, False, True, True, True, False, False]])
    geometry = network._edge_geometry(raw, slots)
    prior = network._edge_log_prior(geometry, present).exp()
    target_forward = network.target_edge_indices
    expected = torch.tensor([[1., .2, 1., 1., 3 / 21, 1., 1., 1.]])
    torch.testing.assert_close(prior[:, target_forward], expected)
    for edge in torch.where(network.relations == 3)[0]:
        reverse = (network.sources == network.destinations[edge]) & (network.destinations == network.sources[edge])
        torch.testing.assert_close(prior[:, edge], prior[:, reverse].squeeze(1))
    assert torch.all(prior[:, network.relations < 3] == 1)
    # Increasing target distance reduces the corresponding prior.
    slots[0, 1, 0] = .38
    changed = network._edge_log_prior(network._edge_geometry(raw, slots), present).exp()
    assert changed[0, target_forward[1]] < prior[0, target_forward[1]]


def test_attention_softmax_masks_and_prior_ratios_with_equal_learned_scores():
    raw, mean, std = target_example()
    network = model(mean, std, observer_acceleration=True, target_relative_missiles=True,
                    target_relative_missile_velocities=True)
    block = network.messages[0]
    # Zero learned logits to isolate the distance prior's effect on attention.
    with torch.no_grad():
        for projection in [*block.query, *block.key, block.edge_attention]:
            for parameter in projection.parameters():
                parameter.zero_()
    decoded, slots, nodes, present = network._decode((raw - mean) / std)
    result, attention, prior = attention_call(network, decoded, slots, nodes, present)
    active = present[:, network.sources] & present[:, network.destinations]
    assert not attention[~active].any()
    assert not result[~present].any()
    for destination in range(10):
        incoming = network.destinations == destination
        expected_sum = (active[:, incoming].any(1)).float()[:, None].expand(-1, 2)
        torch.testing.assert_close(attention[:, incoming].sum(1), expected_sum)
    # The ratio between two missile->same-target attention values equals their prior ratio.
    a, b = network.target_edge_indices[[0, 2]]
    torch.testing.assert_close(attention[:, a] / attention[:, b],
                               (prior[:, a] - prior[:, b]).exp()[:, None].expand(-1, 2))


@pytest.mark.parametrize("occupied", [False, True])
@pytest.mark.parametrize("autocast", [False, True])
def test_empty_slots_and_bfloat16_training_have_finite_gradients(occupied, autocast):
    raw, mean, std = target_example()
    if not occupied:
        raw[..., 16:44] = 0
        raw[..., 54:82] = 0
    network = model(mean, std, observer_acceleration=True, target_relative_missiles=True,
                    target_relative_missile_velocities=True)
    normalized = ((raw - mean) / std).requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        output = network(normalized)
    output.float().square().mean().backward()
    assert torch.isfinite(output).all() and torch.isfinite(normalized.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in network.parameters() if p.grad is not None)
    if occupied:
        assert normalized.grad[..., 16:23].abs().sum() > 0
        assert network.messages[0].relation_attention.grad.abs().sum() > 0
        assert network.messages[0].edge_attention.weight.grad.abs().sum() > 0
    else:
        assert not normalized.grad[..., 16:44].any()
        assert not normalized.grad[..., 54:82].any()


def test_slot_permutations_are_invariant_and_frames_do_not_mix():
    raw, mean, std = target_example()
    network = model(mean, std, observer_acceleration=True, target_relative_missiles=True,
                    target_relative_missile_velocities=True).eval()
    changed = raw.clone()
    for frame in range(51):
        for start in (16, 54):
            slots = raw[:, frame, start:start + 28].reshape(2, 4, 7)
            changed[:, frame, start:start + 28] = slots[:, torch.randperm(4)].reshape(2, 28)
    with torch.no_grad():
        torch.testing.assert_close(network((raw - mean) / std), network((changed - mean) / std),
                                   atol=2e-6, rtol=2e-6)
        baseline = network.encode_frames((raw - mean) / std)
        changed = raw.clone()
        changed[0, 50, 16:23] += .1
        after = network.encode_frames((changed - mean) / std)
        torch.testing.assert_close(baseline[0, :50], after[0, :50], atol=0, rtol=0)
        torch.testing.assert_close(baseline[1], after[1], atol=0, rtol=0)


def test_factory_restores_heads_weighting_and_predictions():
    raw, mean, std = target_example()
    config = dict(architecture="hgt", width=12, heads=6, layers=1, dropout=0,
                  graph_width=8, graph_heads=2, graph_layers=2, observer_acceleration=True,
                  target_relative_missiles=True, target_relative_missile_velocities=True,
                  missile_distance_floor_m=500., missile_distance_power=2.)
    settings = model_configuration(config, 115)
    original = make_model(33, settings, mean, std).eval()
    restored = make_model(33, settings).eval()
    restored.load_state_dict(original.state_dict())
    assert isinstance(restored, HGTTransformer)
    assert restored.messages[0].heads == 2
    assert restored.missile_distance_floor_m == 500.
    assert restored.missile_distance_power == 2.
    assert restored.messages[0].query[0].weight is not restored.messages[0].query[2].weight
    with torch.no_grad():
        torch.testing.assert_close(original((raw - mean) / std), restored((raw - mean) / std),
                                   atol=0, rtol=0)


@pytest.mark.parametrize("settings", [dict(graph_heads=3), dict(missile_distance_floor_m=0),
                                     dict(missile_distance_power=-1), dict(missile_distance_power=float("nan"))])
def test_invalid_attention_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        HGTTransformer(**settings)
