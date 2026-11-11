"""Continuous forecast targets, circular headings, grouped preparation and NN reload."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from bvr_behavior_prediction.forecasting import flight_state as state
from bvr_behavior_prediction.forecasting.data import (
    OBS_COLUMNS,
    grouped_split,
    observed_kinematics,
    prepare_dataset,
    read_json,
    sha256,
    write_json,
)
from bvr_behavior_prediction.forecasting.models import MotionForecaster, masked_loss
from bvr_behavior_prediction.forecasting.train import train_model

ROOT = Path(__file__).resolve().parents[1]


def episode(n=121):
    t = np.arange(n) * .1
    heading = np.deg2rad(359 + t)
    speed = 200 + 2 * t
    velocity = np.column_stack((speed * np.cos(heading), -speed * np.sin(heading), np.full(n, 2.)))
    position = np.cumsum(velocity * .1, axis=0) + [1000, 2000, 4000]
    obs = np.zeros((n, 51), dtype=np.float32)
    obs[:, 9:12] = position / [50000, 50000, 10000]
    obs[:, 12:15] = velocity / 600
    obs[:, 15] = np.linalg.norm(position, axis=1) / 50000
    tracks = pd.DataFrame(np.column_stack((position, velocity)),
                          columns=[f'target_{c}' for c in ('x', 'y', 'z', 'vx', 'vy', 'vz')])
    tracks['observer_alive'] = tracks['target_alive'] = True
    return obs, tracks


def test_future_states_exact_horizons_and_no_future_inputs():
    obs, tracks = episode()
    features, ends, target, _valid, _ = state.episode_examples(obs, tracks)
    origin = state.state_from_kinematics(*observed_kinematics(obs)[:2])
    decoded = state.decode(target[0], origin[ends[0]])
    future = ends[0] + np.array([10, 20, 50])
    expected = state.state_from_kinematics(tracks[['target_x', 'target_y', 'target_z']].to_numpy(),
                                          tracks[['target_vx', 'target_vy', 'target_vz']].to_numpy())[future]
    np.testing.assert_allclose(decoded[:, [0, 2]], expected[:, [0, 2]], atol=1e-4)
    np.testing.assert_allclose(np.degrees(decoded[:, 1]), (359 + future * .1) % 360, atol=1e-4)
    changed_obs = obs.copy()
    changed_obs[ends[0] + 1:] += 999
    new_features, *_ = state.episode_examples(changed_obs, tracks)
    np.testing.assert_array_equal(features[:ends[0] + 1], new_features[:ends[0] + 1])
    changed_tracks = tracks.copy()
    changed_tracks.loc[future, 'target_z'] += 100
    same_features, _, changed_targets, *_ = state.episode_examples(obs, changed_tracks)
    np.testing.assert_array_equal(features, same_features)
    assert (changed_targets[0, :, 3] > target[0, :, 3]).all()
    tracks.loc[105:, 'target_alive'] = False
    assert state.episode_examples(obs, tracks)[3][0].tolist() == [True, True, False]


def test_circular_heading_error_across_north():
    origin = np.array([200., np.deg2rad(359), 5000.])
    truth = state.encode(np.array([210., np.deg2rad(1), 5010.]), origin)
    prediction = state.encode(np.array([212., np.deg2rad(359), 5030.]), origin)
    samples = pd.DataFrame([dict(zip(state.ORIGIN_COLUMNS, origin)) | {'episode_id': 'e', 'pilot_id': 'p'}])
    metrics = state.metrics(np.broadcast_to(prediction, (1, 3, 4)), np.broadcast_to(truth, (1, 3, 4)),
                            np.ones((1, 3), bool), samples, 'test')
    population = metrics.query("pilot_id == 'population'")
    assert np.allclose(population.query("quantity == 'heading'").mean_error, 2)
    assert np.allclose(population.query("quantity == 'speed'").mean_error, 2)
    assert np.allclose(population.query("quantity == 'altitude'").mean_error, 20)


def corpus_fixture(tmp_path):
    corpus = tmp_path / 'corpus'
    cases = [{'scenario_group': f'g{i:02}', 'scenario': {'range_m': 18000}} for i in range(10)]
    groups = grouped_split(cases)
    write_json(corpus / 'config.json', {'observer_id': 'B01', 'target_id': 'A01', 'sample_interval_s': .1, 'cases': cases})
    write_json(corpus / 'schema.json', {})
    writers, episodes = {}, []
    for group, assignment in sorted(groups.items()):
        obs, track = episode()
        if assignment['split'] == 'test':
            obs[:] = np.nan  # Preparation must not read reserved test episodes.
        if assignment['split'] == 'validation':
            obs[:, 0] = 100
        eid = f'p_{group}'
        observation = pd.DataFrame(obs, columns=OBS_COLUMNS)
        for frame in (observation, track):
            frame['episode_id'] = eid
            frame['step'] = np.arange(len(frame))
            frame['time_s'] = frame.step * .1
        for name, frame in [('observations', observation), ('tracks', track)]:
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if name not in writers:
                (corpus / name).mkdir()
                writers[name] = pq.ParquetWriter(corpus / name / 'p.parquet', table.schema)
            writers[name].write_table(table)
        episodes.append({'episode_id': eid, 'pilot_id': 'p', 'scenario_group': group, 'seed': 1, 'frames': len(track)})
    for writer in writers.values():
        writer.close()
    pd.DataFrame(episodes).to_parquet(corpus / 'episodes.parquet', index=False)
    write_json(corpus / 'manifest.json', {'complete': True, 'episodes': 10, 'files': {
        p.relative_to(corpus).as_posix(): {'sha256': sha256(p)} for p in corpus.rglob('*') if p.is_file()}})
    return corpus


@pytest.mark.parametrize('architecture', ['gru', 'transformer'])
def test_prepare_train_reload_and_protect_contract(tmp_path, architecture):
    torch.set_num_threads(1)
    corpus = corpus_fixture(tmp_path)
    data = tmp_path / 'data'
    manifest = prepare_dataset(corpus, data, target='flight_state')
    assert manifest['target_contract'] == state.CONTRACT
    assert manifest['counts']['test']['samples'] == 0
    assert read_json(data / 'scaling.json')['feature_mean'][0] == 0
    assert np.load(data / 'residuals.npy').shape[-2:] == (3, 4)
    config = {'device': 'cpu', 'cpu_threads': 1, 'seed': 7, 'epochs': 1, 'patience': 3,
              'min_delta': .001, 'batch_size': 32, 'evaluation_batch_size': 32,
              'learning_rate': .001, 'weight_decay': .01, 'gradient_clip': 1,
              'model': {'width': 8, 'layers': 1, 'heads': 2, 'dropout': 0}}
    output = tmp_path / architecture
    summary = train_model(ROOT, data, output, architecture, config)
    assert summary['epochs_completed'] == 1 and not summary['test_evaluated']
    checkpoint = torch.load(output / 'best.pt', weights_only=False)
    assert checkpoint['model_config']['output_size'] == 4
    raw, _ = episode()
    predictor = state.FlightStatePredictor(output / 'best.pt')
    prediction = predictor.predict(raw[:70])
    assert set(prediction) == {'horizons_s', 'speed_mps', 'heading_deg', 'altitude_m'}
    assert all(np.isfinite(values).all() for values in prediction.values())
    assert ((prediction['heading_deg'] >= 0) & (prediction['heading_deg'] < 360)).all()
    from bvr_behavior_prediction.forecasting.evaluation import build_report
    report = build_report(tmp_path)
    assert set(report.columns.get_level_values(0)) == {"speed", "heading", "altitude"}
    html = (tmp_path / "report.html").read_text()
    assert "speed, circular heading and altitude" in html
    resumed = train_model(ROOT, data, output, architecture, config, resume=True)
    assert resumed['epochs_completed'] == 1
    repeated = state.FlightStatePredictor(output / 'best.pt').predict(raw[:70])
    for key in prediction:
        np.testing.assert_array_equal(prediction[key], repeated[key])
    checkpoint['target_contract'] = 'old-skill-classification'
    torch.save(checkpoint, tmp_path / 'wrong.pt')
    with pytest.raises(ValueError, match='contract'):
        state.FlightStatePredictor(tmp_path / 'wrong.pt')
    model = MotionForecaster(architecture=architecture, history_steps=7, width=8, heads=2, layers=1, output_size=4)
    result = model(torch.zeros(2, 7, 54))
    assert result.shape == (2, 3, 4)
    valid = torch.tensor([[True, False, False]] * 2)
    masked_loss(result, torch.ones_like(result), valid).backward()
    assert model.head[-1].weight.grad[:4].abs().sum() > 0
    assert model.head[-1].weight.grad[4:].abs().sum() == 0


def test_standard_cli_prepares_continuous_targets_and_notebook_compiles(tmp_path):
    import ast
    import json
    import subprocess
    import sys

    corpus = corpus_fixture(tmp_path)
    output = tmp_path / 'cli_data'
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/train_skill_forecaster.py'),
                             'prepare', '--corpus', str(corpus), '--data', str(output)],
                            capture_output=True, text=True, check=True)
    assert 'train' in result.stdout
    assert read_json(output / 'data_manifest.json')['target_contract'] == state.CONTRACT
    notebook = json.loads((ROOT / 'notebooks/29_trajectory_state_forecasting.ipynb').read_text())
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
            assert cell['execution_count'] is None and not cell['outputs']
