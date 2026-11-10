"""Enemy slots must match same-time physical missile tracks in either view."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bvr_behavior_prediction import nn_duel_skill_forecasting as duel


@pytest.mark.parametrize('observer_id', ['A01', 'B01'])
@pytest.mark.parametrize('missile_count', [0, 1, 4, 5])
def test_enemy_slots_match_real_compact_extraction_and_world_tracks(observer_id, missile_count):
    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location('compact_test', root / 'bvr_sim_source/bvr_sim/src_py/observation_space.py')
    compact = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compact)
    spec = importlib.util.spec_from_file_location('schema_test', root / 'scripts/tactical_corpus_schema.py')
    schema = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(schema)

    def aircraft(uid, color, position, velocity, heading):
        return SimpleNamespace(uid=uid, color=color, is_alive=True, partners=[],
                               position=np.asarray(position, dtype=float),
                               velocity=np.asarray(velocity, dtype=float),
                               get_speed=lambda: float(np.linalg.norm(velocity)),
                               get_altitude=lambda: position[2], get_heading=lambda: heading)

    a = aircraft('A01', 'Red', [100, 300, 5000], [200, 100, 20], .4)
    b = aircraft('B01', 'Blue', [8000, -6000, 4000], [-300, -20, -10], 2.)
    a.enemies, b.enemies = [b], [a]
    missiles = {}
    for owner, target in ((a, b), (b, a)):
        for i in range(missile_count):
            missiles[f'{owner.uid}_{i}'] = SimpleNamespace(
                color=owner.color, is_alive=True, target=target,
                position=owner.position + [1000 * (i + 1), -300 * (i + 1), 200 * (i + 1)],
                velocity=owner.velocity + [500 + i, -100 - i, 30 + i])
        # Dead missiles must not use a slot, even when fewer than four are active.
        missiles[f'{owner.uid}_dead'] = SimpleNamespace(
            color=owner.color, is_alive=False, target=target,
            position=np.ones(3) * 1e8, velocity=np.ones(3) * 1e8)
    sim = SimpleNamespace(agents={'A01': a, 'B01': b}, missiles=missiles,
                          obs_manager=compact.CompactObsSpace())
    target_id = 'B01' if observer_id == 'A01' else 'A01'
    observer = sim.agents[observer_id]
    raw = schema.observer_frame(sim, observer_id, target_id)[None]
    other = schema.observer_frame(sim, target_id, observer_id)[None]
    original = other.copy()
    result = duel.reconstruct_enemy_missiles(raw, other).reshape(4, 7)
    expected = np.zeros((4, 7))
    enemy = [m for m in missiles.values() if m.is_alive and m.color != observer.color][:4]
    for i, missile in enumerate(enemy):
        expected[i, :3] = (missile.position - observer.position) / [50000, 50000, 10000]
        expected[i, 3:6] = (missile.velocity - observer.velocity) / 1200
        expected[i, 6] = np.linalg.norm(missile.position - observer.position) / 50000
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(result[len(enemy):], 0)
    np.testing.assert_array_equal(other, original)
    assert result.dtype == np.float32
    # Notebook 24 expresses friendly position and velocity relative to its target.
    friendly_result = duel.target_relative_friendly_missiles(raw).reshape(4, 7)
    friendly_expected = np.zeros((4, 7))
    friendly = [m for m in missiles.values() if m.is_alive and m.color == observer.color][:4]
    target = sim.agents[target_id]
    for i, missile in enumerate(friendly):
        friendly_expected[i, :3] = (missile.position - target.position) / [50000, 50000, 10000]
        friendly_expected[i, 3:6] = (missile.velocity - target.velocity) / 1200
        friendly_expected[i, 6] = np.linalg.norm(missile.position - target.position) / 50000
    np.testing.assert_allclose(friendly_result, friendly_expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(friendly_result[len(friendly):], 0)
    # Opponent headings, energy, own state and observer-friendly slots are not
    # additional sources of enemy information in this transform.
    altered = other.copy()
    altered[:, :19] = 123
    altered[:, 47:] = 456
    np.testing.assert_array_equal(duel.reconstruct_enemy_missiles(raw, altered).reshape(4, 7), result)


@pytest.mark.parametrize('failure', ['shape', 'nan', 'rows'])
def test_invalid_opposite_arrays_fail(failure):
    raw = np.zeros((10, 51))
    other = raw.copy()
    if failure == 'shape':
        other = other[:, :-1]
    elif failure == 'nan':
        other[0, 19] = np.nan
    else:
        other = other[:-1]
    with pytest.raises(ValueError, match='aligned finite'):
        duel.reconstruct_enemy_missiles(raw, other)


def test_no_future_missile_frames_enter_past_features():
    raw = np.zeros((100, 51), dtype=np.float32)
    raw[:, 9:15] = .2
    other = np.zeros_like(raw)
    other[:61, 19:26] = .1
    before = duel.reconstruct_enemy_missiles(raw, other)
    other[61:, 19:47] = 500
    after = duel.reconstruct_enemy_missiles(raw, other)
    np.testing.assert_array_equal(before[:61], after[:61])
    assert not np.array_equal(before[61:], after[61:])


@pytest.mark.parametrize('flag,nonoverlap', [('yes', True), (True, False)])
def test_enemy_variant_requires_boolean_and_nonoverlap(tmp_path, flag, nonoverlap):
    with pytest.raises(ValueError, match='enemy_missiles'):
        duel.prepare_dataset(tmp_path, tmp_path / 'unused', enemy_missiles=flag,
                             nonoverlapping_train=nonoverlap)
    assert not (tmp_path / 'unused').exists()
