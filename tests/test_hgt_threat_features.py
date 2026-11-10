"""Independent physics, separated relation families and trainable threat priorities."""

import numpy as np
import pytest
import torch

from bvr_behavior_prediction.enemy_missile_models import HGT_THREAT_EDGE_FEATURES
from tests.test_hgt_missile_models import model


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def network():
    return model(hgt_version=2, observer_acceleration=True, target_relative_missiles=True,
                 target_relative_missile_velocities=True)


def test_closest_approach_alignment_speed_and_specific_energy_match_world_state():
    graph = network()
    observer = np.array([100., 200., 5000.])
    opponent = np.array([10100., 200., 6000.])
    observer_velocity = np.array([200., 0, 0])
    opponent_velocity = np.array([-100., 0, 0])
    missile = np.array([5100., 200., 6500.])
    missile_velocity = np.array([600., 0, -10.])
    raw = torch.zeros(1, 115)
    raw[0, :3] = torch.tensor(observer_velocity / 600)
    raw[0, 3] = observer[2] / 10000
    raw[0, 6:9] = torch.tensor((opponent - observer) / [50000, 50000, 10000])
    raw[0, 9:12] = torch.tensor((opponent_velocity - observer_velocity) / 600)
    slots = torch.zeros(1, 8, 7)
    slots[0, 0, :3] = torch.tensor((missile - opponent) / [50000, 50000, 10000])
    slots[0, 0, 3:6] = torch.tensor((missile_velocity - opponent_velocity) / 1200)
    features = graph._edge_geometry(raw, slots)
    assert features.shape == (1, 34, len(HGT_THREAT_EDGE_FEATURES))
    edge = graph.target_edge_indices[0]
    r, v = missile - opponent, missile_velocity - opponent_velocity
    cpa_time = np.clip(-np.dot(r, v) / np.dot(v, v), 0, 30)
    cpa_distance = np.linalg.norm(r + cpa_time * v)
    alignment = -np.dot(r, missile_velocity) / (np.linalg.norm(r) * np.linalg.norm(missile_velocity))
    np.testing.assert_allclose(features[0, edge, 8:12].detach().numpy(),
                               [cpa_time / 30, cpa_distance / 50000, alignment, np.linalg.norm(v) / 1200],
                               rtol=2e-6, atol=1e-7)
    source_kinetic = .5 * np.dot(missile_velocity, missile_velocity)
    source_potential = 9.80665 * missile[2]
    target_kinetic = .5 * np.dot(opponent_velocity, opponent_velocity)
    target_potential = 9.80665 * opponent[2]
    np.testing.assert_allclose(features[0, edge, 12:].detach().numpy(), np.array([
        source_kinetic, source_potential, source_kinetic + source_potential,
        target_kinetic, target_potential, target_kinetic + target_potential]) / 1e6, rtol=2e-6)
    # Changing the current altitude datum affects potential energy, not approach geometry or kinetic energy.
    raw[0, 3] += .1
    higher = graph._edge_geometry(raw, slots)
    torch.testing.assert_close(higher[:, :, :12], features[:, :, :12])
    torch.testing.assert_close(higher[:, :, [12, 15]], features[:, :, [12, 15]])
    torch.testing.assert_close(higher[:, :, [13, 14, 16, 17]] - features[:, :, [13, 14, 16, 17]],
                               torch.full((1, 34, 4), 9.80665 * 1000 / 1e6), atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("relative_velocity,expected_time", [(0., 0.), (100., 0.), (-10., 30.)])
def test_stationary_receding_and_distant_approaches_are_bounded(relative_velocity, expected_time):
    graph = network()
    raw, slots = torch.zeros(1, 115), torch.zeros(1, 8, 7)
    slots[0, 0, 0] = .1  # 5 km from target.
    slots[0, 0, 3] = relative_velocity / 1200
    features = graph._edge_geometry(raw, slots)
    assert torch.isfinite(features).all()
    assert features[0, graph.target_edge_indices[0], 8].item() * 30 == pytest.approx(expected_time)


def test_relation_families_have_independent_attention_and_masked_learned_gates():
    graph = network()
    block = graph.messages[0]
    with torch.no_grad():
        for projection in [*block.query, *block.key, block.edge_attention, block.family_gate[-1]]:
            for parameter in projection.parameters():
                parameter.zero_()
    nodes = torch.randn(1, 10, 8)
    present = torch.ones(1, 10, dtype=torch.bool)
    features = torch.zeros(1, 34, 18)
    prior = torch.zeros(1, 34, 2)
    output, attention, gates = block(nodes, present, features, graph.sources, graph.destinations,
                                     graph.relations, graph.roles, prior, return_family_gates=True)
    torch.testing.assert_close(gates[0, 0], torch.full((3, 2), 1 / 3))
    # Four missile edges must not crowd out the aircraft edge: each family gets one third.
    families = torch.where(graph.relations == 0, 0, torch.where(graph.relations <= 2, 1, 2))
    for family in range(3):
        edges = (graph.destinations == 0) & (families == family)
        torch.testing.assert_close(attention[:, edges].sum(1), torch.full((1, 2), 1 / 3))
    output.square().mean().backward()
    assert block.family_gate[-1].weight.grad.abs().sum() > 0
    present[:, 6:] = False  # No hostile missiles => no incoming target family at observer.
    output, attention, gates = block(nodes, present, features, graph.sources, graph.destinations,
                                     graph.relations, graph.roles, prior, return_family_gates=True)
    assert not gates[:, 0, 2].any()
    torch.testing.assert_close(gates[:, 0, :2], torch.full((1, 2, 2), .5))
    assert not attention[:, ~present[0, graph.sources] | ~present[0, graph.destinations]].any()
    assert not output[:, 6:].any()


def test_learned_prior_can_favor_intercepting_missile_over_closer_receding_missile():
    graph = network()
    raw, slots = torch.zeros(1, 115), torch.zeros(1, 8, 7)
    raw[0, 6] = 1
    raw[0, 3] = .5
    slots[0, :2, 0] = torch.tensor([.04, -.2])  # 2 km receding versus 10 km on collision course.
    slots[0, :2, 3] = torch.tensor([500 / 1200, 1000 / 1200])
    present = torch.tensor([[True, True, True, True, False, False, False, False, False, False]])
    features = graph._edge_geometry(raw, slots)
    prior = graph._edge_log_prior(features, present)
    near, far = graph.target_edge_indices[:2]
    assert torch.all(prior[0, far] > prior[0, near])
    assert not prior[:, graph.relations < 3].any()
    for edge in torch.where(graph.relations == 3)[0]:
        reverse = (graph.sources == graph.destinations[edge]) & (graph.destinations == graph.sources[edge])
        torch.testing.assert_close(prior[:, edge], prior[:, reverse].squeeze(1))
    prior[:, graph.target_edge_indices[:2]].sum().backward()
    assert graph.threat_prior_raw_strength.grad.abs().sum() > 0
    assert graph.threat_prior[-1].weight.grad.abs().sum() > 0
    torch.testing.assert_close(torch.nn.functional.softplus(graph.threat_prior_raw_strength), torch.ones(2))


@pytest.mark.parametrize("settings", [dict(hgt_version=3), dict(threat_cpa_horizon_s=0),
                                     dict(threat_time_scale_s=float("inf")), dict(threat_prior_strength=-1)])
def test_invalid_physical_prior_settings_are_rejected(settings):
    from bvr_behavior_prediction.enemy_missile_models import HGTTransformer

    with pytest.raises(ValueError):
        HGTTransformer(**settings)


@pytest.mark.skipif(not hasattr(torch.export, "Dim"), reason="Dynamic export requires newer PyTorch")
def test_hgt_v2_dynamic_batch_export_matches_eager_inference():
    graph = network().eval()
    windows = torch.zeros(2, 51, 115)
    windows[:, :, 82] = 1
    windows[:, :, 16:23] = torch.tensor([.1, .02, .03, -.5, .1, .02, .11])
    program = torch.export.export(graph, (windows,), dynamic_shapes=(
        {0: torch.export.Dim("batch", min=1, max=4)},))
    with torch.inference_mode():
        torch.testing.assert_close(program.module()(windows[:1]), graph(windows[:1]))
