"""Reproducible sampling and actual per-pilot model/loss configuration."""

import json
from dataclasses import replace

import numpy as np
import pytest

from bvr_behavior_prediction.rl.config import PilotTrainingConfig
from bvr_behavior_prediction.rl.multi_pilot import PilotSpec, load_trained_pilot, train_pilots
from bvr_behavior_prediction.rl.population_variations import sample_training_variations
from tests.test_multi_pilot import TinyEnvironment, definition


@pytest.fixture
def config(tmp_path):
    import torch

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield PilotTrainingConfig(
        epochs=1, episode_duration_s=2, scenarios_per_epoch=3,
        evaluation_scenarios_per_epoch=3, simulator_workers=1, evaluation_seed=100000,
        history_duration_s=2, sample_interval_s=1, hidden_size=8, transformer_heads=2,
        transformer_layers=1, update_epochs=1, minibatch_size=6, checkpoint_interval=1,
        output_dir=tmp_path / 'batch', mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
    )
    torch.set_num_threads(previous)


def test_sampling_is_reproducible_prefix_stable_and_leaves_global_rng_alone():
    before = np.random.get_state()
    samples = sample_training_variations(10, seed=42)
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    assert samples == sample_training_variations(10, seed=42)
    assert samples != sample_training_variations(10, seed=43)
    assert samples == sample_training_variations(11, seed=42)[:10]
    assert json.loads(json.dumps(samples, allow_nan=False)) == samples
    assert {row['hidden_size'] for row in samples} == {56, 64, 72}
    assert {row['transformer_layers'] for row in samples} == {1, 2, 3}
    for row in samples:
        assert row['hidden_size'] % 4 == 0
        assert .45 <= row['value_coefficient'] <= .55
        assert .009 <= row['entropy_coefficient'] <= .011


@pytest.mark.parametrize('settings', [
    {'count': 0}, {'count': True}, {'seed': -1}, {'widths': (63, 64)},
    {'widths': ()}, {'layers': (0,)}, {'layers': (1, 1)},
    {'value_coefficient_range': (.6, .4)}, {'entropy_coefficient_range': (-.1, .1)},
    {'entropy_coefficient_range': (float('nan'), .1)},
])
def test_invalid_sampling_options_are_rejected(settings):
    with pytest.raises(ValueError):
        sample_training_variations(**{'count': 10, **settings})


@pytest.mark.parametrize('overrides', [
    {'seed': 123}, {'output_dir': 'elsewhere'}, {'hidden_size': 0},
    {'transformer_layers': True}, {'value_coefficient': -1}, {'entropy_coefficient': float('nan')},
])
def test_per_pilot_overrides_cannot_change_scenarios_or_use_invalid_values(overrides):
    with pytest.raises(ValueError):
        PilotSpec('pilot_a', 7, training_overrides=overrides)


def test_varied_models_and_loss_weights_train_load_and_reuse(config):
    first = {'hidden_size': 8, 'transformer_layers': 1, 'value_coefficient': .45, 'entropy_coefficient': .009}
    second = {'hidden_size': 12, 'transformer_layers': 2, 'value_coefficient': .55, 'entropy_coefficient': .011}
    pilots = [PilotSpec('first', 7, definition(1), first), PilotSpec('second', 7, definition(2), second)]
    factory = lambda _: lambda scenario, path=None: TinyEnvironment(path)
    results = train_pilots(pilots, config, factory, 1, 'cpu')
    batch = json.loads((config.output_dir / 'pilots.json').read_text())
    assert len({r['network_parameters'] for r in results}) == 2
    for pilot, result, record in zip(pilots, results, batch['pilots']):
        run = config.output_dir / pilot.pilot_id
        metadata = json.loads((run / 'config.json').read_text())
        assert result['training_overrides'] == record['training_overrides'] == pilot.training_overrides
        for key, value in pilot.training_overrides.items():
            assert metadata['training'][key] == value
        model = load_trained_pilot(run)
        assert model.input_projection.out_features == pilot.training_overrides['hidden_size']
        assert len(model.encoder.layers) == pilot.training_overrides['transformer_layers']
        assert model.skill_names == ('change_trajectory', 'launch')
    recovery = replace(config, output_dir=config.output_dir.parent / 'reused')
    reused = train_pilots(pilots, recovery, factory, 1, 'cpu', reuse_completed_from=config.output_dir)
    assert [r['checkpoint_sha256'] for r in results] == [r['checkpoint_sha256'] for r in reused]
    incompatible = [replace(pilots[0], training_overrides=second), pilots[1]]
    rejected = replace(config, output_dir=config.output_dir.parent / 'rejected')
    with pytest.raises(ValueError, match='training settings differ'):
        train_pilots(incompatible, rejected, factory, 1, 'cpu', reuse_completed_from=config.output_dir)
    assert not rejected.output_dir.exists()


def test_incompatible_width_is_rejected_before_any_output_or_simulation(config):
    pilot = PilotSpec('first', 7, definition(1), {'hidden_size': 9})
    with pytest.raises(ValueError, match='divisible'):
        train_pilots([pilot], config, lambda _: pytest.fail('Simulator should not be built'), 1, 'cpu')
    assert not config.output_dir.exists()


def test_planned_variations_are_saved_before_training_failure(config):
    overrides = sample_training_variations(2)
    pilots = [PilotSpec(f'pilot_{i:03}', 7, definition(i + 1), row)
              for i, row in enumerate(overrides)]

    def fail_before_simulation(run_config):
        recorded = json.loads((config.output_dir / 'pilots.json').read_text())
        assert [p['training_overrides'] for p in recorded['pilots']] == overrides
        assert run_config.hidden_size == overrides[0]['hidden_size']
        raise RuntimeError('fixture simulator startup failure')

    with pytest.raises(RuntimeError, match='startup failure'):
        train_pilots(pilots, config, fail_before_simulation, 1, 'cpu',
                     population_metadata={'variation_seed': 42})
    recorded = json.loads((config.output_dir / 'pilots.json').read_text())
    assert recorded['population_metadata']['variation_seed'] == 42
    assert [p['status'] for p in recorded['pilots']] == ['failed', 'pending']
    assert [p['training_overrides'] for p in recorded['pilots']] == overrides
