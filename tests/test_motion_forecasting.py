"""Forecasting boundaries, one-sided inputs, physical targets, and masked losses."""

import numpy as np
import pandas as pd
import pytest
import torch

from bvr_behavior_prediction.forecasting.data import (
    episode_examples,
    fit_scaling,
    grouped_split,
    observed_kinematics,
)
from bvr_behavior_prediction.forecasting.evaluation import motion_metrics, selection_score
from bvr_behavior_prediction.forecasting.inference import MotionPredictor, history_features
from bvr_behavior_prediction.forecasting.models import MotionForecaster, masked_loss


def synthetic_episode(n=121, acceleration=2.0):
    time = np.arange(n) * 0.1
    position = np.column_stack((1000 + 100 * time + 0.5 * acceleration * time ** 2,
                                np.full(n, 2000), np.full(n, 4000)))
    velocity = np.column_stack((100 + acceleration * time, np.zeros(n), np.zeros(n)))
    obs = np.zeros((n, 51), dtype=np.float32)
    obs[:, 9:12] = position / [50000, 50000, 10000]
    obs[:, 12:15] = velocity / 600
    obs[:, 15] = np.linalg.norm(position, axis=1) / 50000
    tracks = pd.DataFrame(np.concatenate((position, velocity), axis=1),
                          columns=[f"target_{c}" for c in ("x", "y", "z", "vx", "vy", "vz")])
    tracks["observer_alive"] = tracks["target_alive"] = True
    return obs, tracks


def test_split_is_deterministic_grouped_and_range_stratified():
    cases = [{"scenario_group": f"case_{r}_{i:03d}", "scenario": {"range_m": r}}
             for r in (18000, 30000, 45000) for i in range(100)]
    result = grouped_split(cases)
    assert result == grouped_split(list(reversed(cases)))
    assert result != grouped_split(cases, seed=123)
    for distance in (18000, 30000, 45000):
        counts = pd.Series([v["split"] for v in result.values() if v["requested_range_m"] == distance]).value_counts()
        assert counts.to_dict() == {"train": 70, "validation": 15, "test": 15}


def test_future_targets_have_exact_horizons_and_do_not_enter_features():
    obs, tracks = synthetic_episode()
    features, ends, residual, mask, ca = episode_examples(obs, tracks)
    assert ends[0] == 60  # complete 5 s history, excluding reset frame
    assert mask[0].all()
    np.testing.assert_allclose(residual[0, :, 0], [1, 4, 25], atol=0.001)
    np.testing.assert_allclose(residual[0, :, 3], [2, 4, 10], atol=0.001)
    np.testing.assert_allclose(ca[0, :, 0], [1, 4, 25], atol=0.001)
    changed = tracks.copy()
    changed.loc[100:, "target_x"] += 9999
    features2, ends2, labels2, *_ = episode_examples(obs, changed)
    np.testing.assert_array_equal(features, features2)
    np.testing.assert_array_equal(ends, ends2)
    assert labels2[0, 2, 0] != residual[0, 2, 0]
    changed_obs = obs.copy()
    changed_obs[81:] = 10
    for before, after in zip(observed_kinematics(obs), observed_kinematics(changed_obs), strict=True):
        np.testing.assert_array_equal(before[:81], after[:81])


def test_terminal_and_partial_horizons_are_masked_without_crossing_episode():
    obs, tracks = synthetic_episode()
    tracks.loc[105:, "target_alive"] = False
    _, ends, _, mask, _ = episode_examples(obs, tracks)
    assert np.all(ends < 105)
    assert mask[0].tolist() == [True, True, False]  # 11 s target is dead
    assert mask[-1].tolist() == [True, False, False]
    assert np.all(ends - 50 > 0)
    short_obs, short_tracks = synthetic_episode(n=50)
    assert len(episode_examples(short_obs, short_tracks)[1]) == 0


def test_zero_residual_is_constant_velocity_and_scaling_uses_only_supplied_training():
    obs, tracks = synthetic_episode(acceleration=0)
    _, _, residual, mask, _ = episode_examples(obs, tracks)
    np.testing.assert_allclose(residual[mask], 0, atol=0.002)
    training = np.arange(20 * 54).reshape(20, 54).astype(float) / 100
    scale = fit_scaling(training.sum(0), (training ** 2).sum(0), len(training), residual, mask)
    np.testing.assert_allclose(scale["feature_mean"], training.mean(0))
    assert scale["fitted_on"] == "train only"
    assert np.asarray(scale["residual_scale"]).min() >= 1


@pytest.mark.parametrize("architecture", ["transformer", "gru"])
def test_forecaster_output_contract_and_masked_gradient(architecture):
    torch.set_num_threads(1)
    model = MotionForecaster(architecture=architecture, history_steps=7, width=16, heads=4, layers=2)
    inputs = torch.randn(4, 7, 54)
    prediction = model(inputs)
    assert prediction.shape == (4, 3, 6)
    assert torch.count_nonzero(prediction) == 0
    target = torch.ones_like(prediction)
    valid = torch.tensor([[True, False, False]] * 4)
    loss = masked_loss(prediction, target, valid)
    loss.backward()
    assert model.head[-1].weight.grad[:6].abs().sum() > 0
    assert model.head[-1].weight.grad[6:].abs().sum() == 0
    changed_target = target.clone()
    changed_target[:, 1:] = 999
    assert masked_loss(prediction.detach(), changed_target, valid) == loss.detach()


def test_metrics_weight_episodes_equally_and_report_physical_vector_error():
    samples = pd.DataFrame({"episode_id": ["short", "long", "long", "long"], "pilot_id": ["pilot_001"] * 4})
    truth = np.zeros((4, 3, 6))
    prediction = np.zeros_like(truth)
    prediction[0, :, 0], prediction[1:, :, 0] = 10, 2
    prediction[:, :, 3] = 1
    metrics = motion_metrics(prediction, truth, np.ones((4, 3), dtype=bool), samples, "example")
    first = metrics.query("pilot_id == 'population' and quantity == 'position' and horizon_s == 1").iloc[0]
    assert first.episode_mean_error == 6
    assert first.mean_error == 4
    assert selection_score(metrics, metrics) == 1


def test_streaming_features_match_offline_features_and_zero_model_is_cv(tmp_path):
    obs, tracks = synthetic_episode()
    features, ends, *_ = episode_examples(obs, tracks)
    anchor = ends[0]
    streaming, position, velocity = history_features(obs[anchor - 55:anchor + 1])
    np.testing.assert_array_equal(streaming, features[anchor - 50:anchor + 1])
    config = {"architecture": "transformer", "width": 16, "layers": 1, "history_steps": 51}
    model = MotionForecaster(**config)
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_config": config, "model_state": model.state_dict(),
                "scaling": {"feature_mean": [0] * 54, "feature_std": [1] * 54,
                            "residual_scale": np.ones((3, 6)).tolist()}}, checkpoint)
    predictor = MotionPredictor(checkpoint)
    predicted = predictor.predict(obs[:anchor + 1])
    np.testing.assert_allclose(predicted["position_nwu_m"], position + np.array([1, 2, 5])[:, None] * velocity)
    np.testing.assert_allclose(predicted["velocity_nwu_mps"], np.broadcast_to(velocity, (3, 3)))
    with pytest.raises(ValueError, match="at least 56"):
        predictor.predict(obs[:51])
