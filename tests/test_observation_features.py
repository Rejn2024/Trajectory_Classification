from types import SimpleNamespace

import numpy as np

from bvr_behavior_prediction.rl.observation_features import (
    POLICY_OBSERVATION_SIZE,
    append_hostile_missile_features,
    hostile_missile_features,
    observation_feature_schema,
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
