from types import SimpleNamespace

import numpy as np
import pytest

from bvr_behavior_prediction.rl.observation_features import (
    HOSTILE_MISSILE_OBSERVATION_SIZE,
    POLICY_OBSERVATION_SIZE,
    RADAR_FEATURE_SIZE,
    RADAR_POLICY_OBSERVATION_SIZE,
    append_hostile_missile_features,
    append_radar_features,
    hostile_missile_features,
    observation_feature_schema,
    radar_features,
)


def test_hostile_missiles_are_sorted_and_zero_padded():
    aircraft = SimpleNamespace(
        is_alive=True, color="red", position=np.zeros(3),
        velocity=np.array([300.0, 0.0, 0.0]), under_missiles=[],
    )
    aircraft.under_missiles.extend([
        SimpleNamespace(uid="far", color="blue", is_alive=True, target=aircraft,
                        position=np.array([2000.0, 0.0, 0.0]),
                        velocity=np.array([-300.0, 0.0, 0.0]), radar_on=False),
        SimpleNamespace(uid="near", color="blue", is_alive=True, target=aircraft,
                        position=np.array([1000.0, 0.0, 0.0]),
                        velocity=np.array([-300.0, 0.0, 0.0]), radar_on=True),
    ])
    encoded = hostile_missile_features(aircraft)
    assert encoded.shape == (40,)
    assert encoded[0] == 1.0
    assert encoded[8] > 0.0
    assert encoded[9] == 1.0
    assert encoded[10] == 1.0
    assert np.all(encoded[20:] == 0.0)


def test_full_policy_schema_matches_runtime_frame():
    compact = np.zeros(47, dtype=np.float32)
    aircraft = SimpleNamespace(
        is_alive=True, color="red", position=np.zeros(3), velocity=np.zeros(3),
        under_missiles=[],
    )
    assert append_hostile_missile_features(compact, aircraft).shape == (87,)
    assert len(observation_feature_schema(POLICY_OBSERVATION_SIZE)) == POLICY_OBSERVATION_SIZE


def test_radar_features_are_fixed_numeric_and_body_relative():
    radar = SimpleNamespace(
        radar_mode="stt", track_targets={"B01": object()},
        RadarAzimuth=np.pi / 2, RadarElevation=0.0,
        RadarDirectionVec=np.array([0.0, 1.0, 0.0]),
        scan_zone=SimpleNamespace(
            now_scan_zone_rpy=np.array([0.0, 0.1, -0.2]),
            now_beam_angle_hori=0.2, now_beam_angle_vert=-0.1,
            beam_step_right=True,
        ),
    )
    aircraft = SimpleNamespace(
        radar=radar, get_roll=lambda: 0.0, get_pitch=lambda: 0.0,
        get_heading=lambda: 0.0,
    )
    encoded = radar_features(aircraft)
    assert encoded.shape == (RADAR_FEATURE_SIZE,)
    assert encoded[1] == 1.0  # STT
    assert encoded[3] == 1.0  # track present
    assert encoded[8:11].tolist() == pytest.approx([0.0, 1.0, 0.0])
    assert encoded[21] == 1.0
    assert encoded[22] == 1.0
    assert np.isfinite(encoded).all()
    assert append_radar_features(
        np.zeros(HOSTILE_MISSILE_OBSERVATION_SIZE, dtype=np.float32), aircraft
    ).shape == (HOSTILE_MISSILE_OBSERVATION_SIZE + RADAR_FEATURE_SIZE,)
    assert len(observation_feature_schema(RADAR_POLICY_OBSERVATION_SIZE)) == RADAR_POLICY_OBSERVATION_SIZE
