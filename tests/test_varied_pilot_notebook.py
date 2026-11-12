"""Run the ten-pilot notebook through real PPO with a small flight fixture."""

import ast
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from bvr_behavior_prediction.rl.bvr_jsbsim_backend import BVRJSBSimBackend
from bvr_behavior_prediction.rl.multi_pilot import load_trained_pilot

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/30_train_varied_transformer_pilots.ipynb'


def test_varied_notebook_trains_records_and_replays_ten_distinct_configurations(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    monkeypatch.chdir(ROOT)
    for key in list(os.environ):
        if key.startswith('BVR_PILOT_'):
            monkeypatch.delenv(key)
    for key, value in {
        'BVR_PILOT_EPOCHS': '1', 'BVR_PILOT_EPISODE_SECONDS': '1',
        'BVR_PILOT_SCENARIOS': '3', 'BVR_PILOT_EVALUATION_SCENARIOS': '3',
        'BVR_PILOT_UPDATE_EPOCHS': '1', 'BVR_PILOT_WORKERS': '1',
        'BVR_PILOT_TORCH_THREADS': '1', 'BVR_PILOT_DEVICE': 'cpu',
        'BVR_PILOT_EXECUTOR': 'thread', 'BVR_PILOT_VIEW': 'pilot_010',
        'BVR_PILOT_OUTPUT': str(tmp_path / 'batch'),
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(plt, 'show', lambda: None)
    instances = []

    class Backend:
        HOSTILE_MISSILE_OBSERVATION_SIZE = BVRJSBSimBackend.HOSTILE_MISSILE_OBSERVATION_SIZE
        HOSTILE_MISSILE_SLOTS = BVRJSBSimBackend.HOSTILE_MISSILE_SLOTS
        HOSTILE_MISSILE_FEATURE_NAMES = BVRJSBSimBackend.HOSTILE_MISSILE_FEATURE_NAMES

        def __init__(self, scenario_dict, recording_path=None, log_dir=None,
                     include_hostile_missiles=False):
            assert include_hostile_missiles is True
            self.path, self.closed = recording_path, False
            self.info = {
                'blue_alive': True, 'blue_altitude_m': 6000, 'red_altitude_m': 6000,
                'blue_speed_mps': 250, 'red_speed_mps': 250,
                'target_range_m': 30000, 'target_alignment': 1,
                'initial_target_range_m': 30000, 'initial_target_alignment': 1,
                'missiles_avoided_total': 0, 'missiles_launched_total': 0,
                'opponent_eliminated': False, 'opponent_destroyed': False,
                'incoming_missiles': 1, 'target_locked': True,
            }
            instances.append(self)

        def reset(self, seed):
            self.steps = 0
            if self.path:
                self.path.write_text('fixture replay')
            return self.observation(), self.info

        def observation(self):
            # A live incoming missile exercises the new policy input during PPO.
            frame = np.zeros(self.HOSTILE_MISSILE_OBSERVATION_SIZE, np.float32)
            frame[18] = 1
            frame[47:57] = [1, 0.1, 0, 0, -0.5, 0, 0, 0.1, 0.5, 1]
            return frame

        def step(self, blue_action, red_action):
            self.steps += 1
            return self.observation(), 0., self.steps >= 10, self.info

        def close(self):
            self.closed = True

    notebook = json.loads(NOTEBOOK.read_text())
    namespace = {'display': lambda *_: None}
    threads = torch.get_num_threads()
    try:
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] != 'code':
                continue
            source = ''.join(cell['source'])
            ast.parse(source)
            assert cell['execution_count'] is None and not cell['outputs']
            exec(compile(source, f'{NOTEBOOK}:cell{i}', 'exec'), namespace)  # noqa: S102
            if i == 2:
                namespace['BVRJSBSimBackend'] = Backend
            if i == 7:
                namespace['config'] = replace(namespace['config'],
                    mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    finally:
        torch.set_num_threads(threads)
        plt.close('all')
    assert namespace['PILOT_COUNT'] == len(namespace['results']) == 10
    assert namespace['OBSERVATION_SIZE'] == 91
    batch = json.loads((tmp_path / 'batch/pilots.json').read_text())
    assert batch['population_metadata']['variation_seed'] == 42
    assert batch['population_metadata']['roster'] == namespace['VARIATION_ROSTER']
    assert len({row['network_parameters'] for row in namespace['results']}) > 1
    assert (tmp_path / 'batch/variant_roster.csv').is_file()
    assert all(instance.closed for instance in instances)
    for spec, result, record in zip(namespace['PILOTS'], namespace['results'], batch['pilots']):
        metadata = json.loads((tmp_path / 'batch' / spec.pilot_id / 'config.json').read_text())
        assert metadata['observation_size'] == 91
        assert spec.training_overrides == result['training_overrides'] == record['training_overrides']
        assert metadata['reward']['parameters']['weights'] == spec.reward.parameters['weights']
        for key, value in spec.training_overrides.items():
            assert metadata['training'][key] == value
        assert result['benchmark_reward']['name'] == 'common_hybrid_dogfight'
    selected = load_trained_pilot(tmp_path / 'batch/pilot_010')
    assert selected.input_projection.in_features == 91
    with pytest.raises(ValueError, match='expected 91 observation features, got 51'):
        selected.act(np.zeros((20, 51), np.float32))
    assert len(selected.encoder.layers) == namespace['selected_config'].transformer_layers
    assert selected.input_projection.out_features == namespace['selected_config'].hidden_size
    assert len(namespace['final_evaluation_manifest']) == 3

    csv_path = tmp_path / 'batch/variant_roster.csv'
    preserved = csv_path.read_bytes()
    with pytest.raises(ValueError, match='fresh'):
        exec(compile(''.join(notebook['cells'][9]['source']), 'rerun-training', 'exec'), namespace)  # noqa: S102
    assert csv_path.read_bytes() == preserved
