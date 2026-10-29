"""Leakage boundaries for nested selection, geometry, long history and ensembles."""

import numpy as np
import pandas as pd
import pytest
import torch

from bvr_behavior_prediction.forecasting.data import grouped_split, observed_kinematics, write_json
from bvr_behavior_prediction.skill_forecasting.improved_features import (
    geometry_features,
    history_offsets,
    raw_history_features,
    window_indices,
)
from bvr_behavior_prediction.skill_forecasting.improved_models import (
    ImprovedForecaster,
    supervised_loss,
)
from bvr_behavior_prediction.skill_forecasting.improvement import (
    DevelopmentData,
    ImprovedPredictor,
    inner_partition,
    paired_group_intervals,
)


def raw_observations(n=401):
    time = np.arange(n) * 0.1
    obs = np.zeros((n, 51), dtype=np.float32)
    obs[:, 3] = 0.3
    obs[:, 9] = 0.3 - time * 0.002
    obs[:, 10] = 0.1
    obs[:, 11] = 0.2
    obs[:, 12] = -0.1 + 0.02 * np.sin(time)
    obs[:, 13] = 0.02 * np.cos(time)
    obs[:, 15] = 0.3
    return obs


def test_inner_selection_keeps_scenarios_and_outer_partitions_separate():
    cases = [{"scenario_group": f"r{r}_g{i:03}", "scenario": {"range_m": r}}
             for r in (18000, 30000, 45000) for i in range(100)]
    outer = grouped_split(cases)
    inner = inner_partition(outer, 271828)
    assert inner == inner_partition(dict(reversed(list(outer.items()))), 271828)
    assert inner != inner_partition(outer, 314159)
    assert pd.Series(inner).value_counts().to_dict() == {"fit": 147, "select": 42, "stop": 21}
    assert all(outer[group]["split"] == "train" for group in inner)
    assert set(inner) == {g for g, value in outer.items() if value["split"] == "train"}


@pytest.mark.parametrize("history_seconds,recent_stride", [(5, 1), (20, 1), (20, 5), (60, 5)])
def test_long_history_never_crosses_episode_or_future_and_matches_streaming(history_seconds, recent_stride):
    obs = raw_observations()
    _, _, acc = observed_kinematics(obs)
    frames = np.column_stack((obs, acc / 100)).astype(np.float32)
    features, _ = geometry_features(frames)
    offsets = history_offsets(history_seconds, recent_stride)
    origins = np.array([60, 120, 210, 400])
    indices, valid = window_indices(origins + 1000, origins, offsets)
    assert np.all(indices <= origins[:, None] + 1000)
    assert indices.min() >= 1006
    for row, origin in enumerate(origins):
        streaming, mask = raw_history_features(obs[1:origin + 1], offsets, "geometry")
        np.testing.assert_array_equal(mask, valid[row])
        np.testing.assert_allclose(streaming[mask], features[indices[row, mask] - 1000], atol=1e-6)
    with pytest.raises(ValueError, match="future"):
        window_indices(origins, origins, np.array([-1, 1, 0]))


def test_geometry_is_causal_and_rotation_invariants_are_consistent():
    obs = raw_observations()
    _, _, acc = observed_kinematics(obs)
    frames = np.column_stack((obs, acc / 100)).astype(np.float32)
    result, names = geometry_features(frames)
    changed = frames.copy()
    changed[100:] += 10
    np.testing.assert_array_equal(result[:100], geometry_features(changed)[0][:100])
    rotated = frames.copy()
    for start in (0, 3, 9, 12, 51):
        rotated[:, start] = -frames[:, start + 1]
        rotated[:, start + 1] = frames[:, start]
    transformed, _ = geometry_features(rotated)
    invariant_names = ["target_speed", "observer_speed", "target_aspect_cos", "target_aspect_sin",
                       "relative_course_cos", "target_course_rate", "cpa_time", "cpa_distance"]
    columns = [names.index(name) for name in invariant_names]
    np.testing.assert_allclose(result[:, columns], transformed[:, columns], atol=1e-6)


@pytest.mark.parametrize("architecture", ["mlp", "transformer"])
def test_padding_values_and_privileged_auxiliary_targets_do_not_enter_predictions(architecture):
    torch.set_num_threads(1)
    model = ImprovedForecaster(architecture, 10, 12, num_classes=4, width=16, layers=1, dropout=0,
                               auxiliary=True).eval()
    features = torch.randn(3, 12, 10)
    valid = torch.ones(3, 12, dtype=torch.bool)
    valid[:, :4] = False
    logits = model(features, valid)
    altered = features.clone()
    altered[:, :4] = 123456
    torch.testing.assert_close(logits, model(altered, valid), atol=0, rtol=0)
    targets = torch.tensor([[1, -1, 2]] * 3)
    current = torch.tensor([0, 1, 2])
    other = torch.tensor([3, 3, 3])
    assert supervised_loss(logits, targets, current) == supervised_loss(logits, targets, other)
    logits.retain_grad()
    supervised_loss(logits, targets, current, auxiliary_weight=0.25).backward()
    assert logits.grad[:, 1].abs().sum() == 0
    assert logits.grad[:, 3].abs().sum() > 0


def test_inner_scaling_excludes_stop_and_outer_validation_groups(tmp_path):
    features = np.zeros((300, 54), dtype=np.float32)
    features[:100, 0], features[100:200, 0], features[200:, 0] = 1, 111, 999
    samples = pd.DataFrame({"scenario_group": ["fit", "stop", "outer"], "step": [60] * 3,
                            "split": ["train", "train", "validation"]})
    samples.to_parquet(tmp_path / "samples.parquet", index=False)
    for name, values in [("features", features), ("anchors", np.array([60, 160, 260])),
                         ("targets", np.zeros((3, 3), dtype=np.int64)), ("current", np.zeros(3, dtype=np.int64))]:
        np.save(tmp_path / f"{name}.npy", values)
    write_json(tmp_path / "data_manifest.json", {"files": {}, "classes": ["a", "b"]})
    candidate = {"history_seconds": 20, "recent_stride": 1, "features": "raw"}
    data = DevelopmentData(tmp_path, candidate, ["fit"], device="cpu")
    assert data.scaling["mean"][0] == 1
    assert data.scaling["fit_groups"] == ["fit"]
    with pytest.raises(ValueError, match="training"):
        DevelopmentData(tmp_path, candidate, ["outer"], device="cpu")


def test_uniform_ensemble_reload_matches_separate_observation_only_members(tmp_path):
    torch.set_num_threads(1)
    obs = raw_observations(121)[1:]
    offsets = history_offsets(20, 5)
    frame, _ = raw_history_features(obs, offsets, "geometry")
    config = {"architecture": "mlp", "input_size": frame.shape[1], "tokens": len(offsets),
              "num_classes": 4, "width": 16, "layers": 1, "dropout": 0, "auxiliary": False}
    paths = []
    for seed in (42, 43):
        torch.manual_seed(seed)
        model = ImprovedForecaster(**config)
        path = tmp_path / f"seed{seed}.pt"
        torch.save({"model_config": config, "model_state": model.state_dict(),
                    "candidate": {"features": "geometry"}, "offsets": offsets.tolist(),
                    "scaling": {"mean": [0] * frame.shape[1], "std": [1] * frame.shape[1]},
                    "classes": list("abcd")}, path)
        paths.append(path)
    expected = np.mean([ImprovedPredictor([path]).predict(obs)["probabilities"] for path in paths], axis=0)
    actual = ImprovedPredictor(paths).predict(obs)
    np.testing.assert_array_equal(actual["probabilities"], expected)
    np.testing.assert_allclose(actual["probabilities"].sum(-1), 1, atol=1e-6)
    assert actual["top3_skills"].shape == (3, 3)


def test_uncertainty_resamples_scenario_groups_with_partial_horizon_masks():
    samples = pd.DataFrame({"scenario_group": ["a", "a", "b", "b"]})
    targets = np.array([[0, 0, -1], [0, 0, 0], [0, 0, 0], [0, -1, -1]])
    new = np.broadcast_to([1.0, 0.0], (4, 3, 2))
    old = np.broadcast_to([0.0, 1.0], (4, 3, 2))
    result = paired_group_intervals(new, old, targets, samples, draws=100)
    assert result.accuracy_gain.tolist() == [1, 1, 1]
    assert result.lower_95.tolist() == result.upper_95.tolist() == [1, 1, 1]
    assert result.scenario_groups.tolist() == [2, 2, 2]


def test_temperature_reduces_overconfidence_without_changing_class_order():
    from bvr_behavior_prediction.skill_forecasting.temperature import fit_temperature
    from bvr_behavior_prediction.skill_forecasting.train import validation_nll

    probabilities = np.broadcast_to([0.999, 0.001], (100, 3, 2)).copy()
    targets = np.zeros((100, 3), dtype=np.int64)
    targets[:20] = 1
    temperature = fit_temperature(probabilities, targets)
    assert 1 < temperature <= 4
    adjusted = np.exp(np.log(probabilities) / temperature)
    adjusted /= adjusted.sum(-1, keepdims=True)
    assert validation_nll(adjusted, targets) < validation_nll(probabilities, targets)
    np.testing.assert_array_equal(probabilities.argmax(-1), adjusted.argmax(-1))


def test_mixture_requires_identical_forecast_origins(tmp_path):
    from bvr_behavior_prediction.skill_forecasting.ensemble_search import combine_predictions

    paths = [tmp_path / "a", tmp_path / "b"]
    for i, path in enumerate(paths):
        path.mkdir()
        np.save(path / "selection_indices.npy", [1, 3])
        np.save(path / "selection_probabilities.npy", np.full((2, 3, 4), 0.25))
    probability, indices = combine_predictions(paths)
    assert probability.shape == (2, 3, 4)
    assert indices.tolist() == [1, 3]
    np.save(paths[1] / "selection_indices.npy", [3, 1])
    with pytest.raises(ValueError, match="different"):
        combine_predictions(paths)
