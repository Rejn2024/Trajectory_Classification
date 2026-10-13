"""Forecast data must be causal, correctly oriented, and terminal-aware."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "tactical_corpus_schema", Path(__file__).parents[1] / "scripts/tactical_corpus_schema.py"
)
schema = importlib.util.module_from_spec(spec)
spec.loader.exec_module(schema)


def trajectory(n=131, rotation=0.02):
    time = np.arange(n) * 0.1
    heading = rotation * time
    return pd.DataFrame({
        "episode_id": "example", "step": np.arange(n), "time_s": time,
        "observer_x": 250 * time, "observer_y": 0., "observer_z": 5000.,
        "observer_vx": 250., "observer_vy": 0., "observer_vz": 0.,
        "target_x": 10000 + 250 * time, "target_y": 1000., "target_z": 6000 + 3 * time,
        "target_vx": (250 + time) * np.cos(heading),
        "target_vy": (250 + time) * np.sin(heading), "target_vz": 3.,
        "observer_alive": True, "target_alive": True,
    })


def test_features_do_not_change_when_future_is_removed_or_changed():
    raw = trajectory()
    whole, _ = schema.derive_tables(raw)
    prefix, _ = schema.derive_tables(raw.iloc[:51])
    pd.testing.assert_frame_equal(whole.iloc[:51], prefix)
    raw.loc[51:, ["target_x", "target_vy", "target_z"]] += 500
    changed, _ = schema.derive_tables(raw)
    pd.testing.assert_frame_equal(whole.iloc[:51], changed.iloc[:51])


@pytest.mark.parametrize("rotation,label", [(0.02, 1), (-0.02, 2), (0., 0)])
def test_nwu_motion_signs_and_future_labels(rotation, label):
    observations, labels = schema.derive_tables(trajectory(rotation=rotation))
    assert observations.relative_left_m.iloc[10] == 1000  # north-facing, west on left
    assert labels.current_lateral.iloc[10] == label
    assert labels.current_vertical.iloc[10] == 1
    assert labels.current_energy.iloc[10] == 1
    assert labels.future_lateral_8s.iloc[10] == label


def test_warmup_and_unobserved_future_are_null_not_negative_examples():
    _, labels = schema.derive_tables(trajectory())
    assert not labels.current_valid.iloc[:10].any()
    assert labels.current_lateral.iloc[:10].isna().all()
    assert labels.valid_8s.iloc[50]
    assert not labels.valid_8s.iloc[51:].any()
    assert labels.future_lateral_8s.iloc[51:].isna().all()


def test_death_censors_forecasts_even_when_simulator_continues():
    raw = trajectory()
    raw.loc[70:, "target_alive"] = False
    features, labels = schema.derive_tables(raw)
    assert not features.input_valid.iloc[70:].any()
    assert labels.valid_2s.iloc[49]
    assert not labels.valid_2s.iloc[50:].any()
    assert labels.future_vertical_2s.iloc[50:].isna().all()


def test_heading_wrap_and_episode_boundary():
    raw = trajectory(rotation=0.02)
    raw.target_vx *= -1
    raw.target_vy *= -1
    _, labels = schema.derive_tables(raw)
    assert labels.current_lateral.dropna().eq(1).all()
    raw.loc[60:, "episode_id"] = "other"
    with pytest.raises(ValueError, match="exactly one episode"):
        schema.derive_tables(raw)


def test_rejects_nonfinite_and_broken_timestamps():
    raw = trajectory()
    raw.loc[4, "target_x"] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        schema.derive_tables(raw)
    raw = trajectory()
    raw.loc[4, "time_s"] = 0.1
    with pytest.raises(ValueError, match="cadence"):
        schema.derive_tables(raw)
