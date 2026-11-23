"""Shared policy-frame and forecasting-feature encoding.

The compact simulator observation is deliberately kept separate from the
policy-frame contract.  A legacy policy consumes 47 compact values plus four
energy values (51 total); notebook 31 policies consume the same values plus
four hostile-missile slots (91 total).  Radar-enabled policies append a fixed
23-value numeric radar state before the four energy values (114 total).
Keeping the encoder here makes runtime inference and corpus recording use
exactly the same ordering and scaling.
"""

from __future__ import annotations

import json

import numpy as np


COMPACT_OBSERVATION_SIZE = 47
HOSTILE_MISSILE_SLOTS = 4
HOSTILE_MISSILE_FEATURE_NAMES = (
    "present",
    "relative_north",
    "relative_west",
    "relative_up",
    "relative_velocity_north",
    "relative_velocity_west",
    "relative_velocity_up",
    "range",
    "closing_speed",
    "seeker_active",
)
HOSTILE_MISSILE_FEATURES_PER_SLOT = len(HOSTILE_MISSILE_FEATURE_NAMES)
HOSTILE_MISSILE_FEATURE_SIZE = HOSTILE_MISSILE_SLOTS * HOSTILE_MISSILE_FEATURES_PER_SLOT
HOSTILE_MISSILE_OBSERVATION_SIZE = COMPACT_OBSERVATION_SIZE + HOSTILE_MISSILE_FEATURE_SIZE
ENERGY_FEATURE_NAMES = (
    "own_kinetic_energy",
    "own_potential_energy",
    "target_kinetic_energy",
    "target_potential_energy",
)
POLICY_OBSERVATION_SIZE = HOSTILE_MISSILE_OBSERVATION_SIZE + len(ENERGY_FEATURE_NAMES)
RADAR_FEATURE_NAMES = (
    "radar_mode_scan", "radar_mode_stt", "radar_track_count",
    "radar_track_present", "radar_azimuth_sin", "radar_azimuth_cos",
    "radar_elevation_sin", "radar_elevation_cos", "radar_direction_body_x",
    "radar_direction_body_y", "radar_direction_body_z", "scan_zone_roll_sin",
    "scan_zone_roll_cos", "scan_zone_pitch_sin", "scan_zone_pitch_cos",
    "scan_zone_yaw_sin", "scan_zone_yaw_cos", "scan_beam_azimuth_sin",
    "scan_beam_azimuth_cos", "scan_beam_elevation_sin",
    "scan_beam_elevation_cos", "scan_beam_step_right", "radar_direction_valid",
)
RADAR_FEATURE_SIZE = len(RADAR_FEATURE_NAMES)
COMPACT_RADAR_OBSERVATION_SIZE = COMPACT_OBSERVATION_SIZE + RADAR_FEATURE_SIZE
RADAR_OBSERVATION_SIZE = COMPACT_RADAR_OBSERVATION_SIZE
HOSTILE_MISSILE_RADAR_OBSERVATION_SIZE = HOSTILE_MISSILE_OBSERVATION_SIZE + RADAR_FEATURE_SIZE
RADAR_POLICY_OBSERVATION_SIZE = HOSTILE_MISSILE_RADAR_OBSERVATION_SIZE + len(ENERGY_FEATURE_NAMES)

_MISSILE_SCALES = (1, 50_000, 50_000, 10_000, 1_200, 1_200,
                   1_200, 50_000, 1_200, 1)


def hostile_missile_features(controlled) -> np.ndarray:
    """Encode the nearest live hostile missiles targeting ``controlled``.

    This is simulator truth, matching the compact observation's truth-based
    hostile-missile flag.  Empty slots are zero padded and sorted by range,
    then UID for deterministic replay and forecasting labels.
    """
    slots = np.zeros((HOSTILE_MISSILE_SLOTS, HOSTILE_MISSILE_FEATURES_PER_SLOT), dtype=np.float32)
    if not getattr(controlled, "is_alive", True):
        return slots.reshape(-1)
    own_position = np.asarray(controlled.position, dtype=np.float64)
    own_velocity = np.asarray(controlled.velocity, dtype=np.float64)
    threats = [
        missile for missile in getattr(controlled, "under_missiles", ())
        if getattr(missile, "is_alive", False)
        and getattr(missile, "target", None) is controlled
        and getattr(missile, "color", getattr(controlled, "color", None)) != getattr(controlled, "color", None)
    ]
    threats.sort(key=lambda missile: (
        float(np.linalg.norm(np.asarray(missile.position) - own_position)),
        str(getattr(missile, "uid", "")),
    ))
    for row, missile in zip(slots, threats[:HOSTILE_MISSILE_SLOTS]):
        relative_position = np.asarray(missile.position, dtype=np.float64) - own_position
        relative_velocity = np.asarray(missile.velocity, dtype=np.float64) - own_velocity
        distance = float(np.linalg.norm(relative_position))
        closing_speed = (
            -float(np.dot(relative_position, relative_velocity)) / distance
            if distance > 0.0 else 0.0
        )
        row[:] = np.asarray((
            1.0,
            relative_position[0] / 50_000.0,
            relative_position[1] / 50_000.0,
            relative_position[2] / 10_000.0,
            *(relative_velocity / 1_200.0),
            distance / 50_000.0,
            closing_speed / 1_200.0,
            float(getattr(missile, "radar_on", False)),
        ), dtype=np.float32)
    return slots.reshape(-1)


def append_hostile_missile_features(compact, controlled) -> np.ndarray:
    """Return a 87-value compact-plus-threat vector."""
    compact = np.asarray(compact, dtype=np.float32).reshape(-1)
    if compact.size != COMPACT_OBSERVATION_SIZE:
        raise ValueError(f"expected {COMPACT_OBSERVATION_SIZE} compact features, got {compact.size}")
    return np.concatenate((compact, hostile_missile_features(controlled))).astype(np.float32, copy=False)


def _angle_pair(angle):
    """Encode an angle without a discontinuity at +/-pi."""
    angle = float(angle)
    if not np.isfinite(angle):
        angle = 0.0
    return np.sin(angle), np.cos(angle)


def _vector_values(value):
    """Read a simulator vector or Vector3-like object as a unit vector."""
    if value is None:
        return None
    raw = getattr(value, "vec", value)
    vector = np.asarray(raw, dtype=np.float64).reshape(-1)
    if vector.size != 3 or not np.isfinite(vector).all():
        return None
    magnitude = float(np.linalg.norm(vector))
    return vector / magnitude if magnitude > 1e-9 else None


def _body_direction(world_direction, aircraft):
    """Transform a world/NWU direction into the aircraft body frame.

    This mirrors Radar._sync_radar_antenna_ego's inverse Z-Y-X rotation so the
    feature is invariant to aircraft heading, pitch, and roll.
    """
    roll = float(getattr(aircraft, "get_roll", lambda: 0.0)())
    pitch = float(getattr(aircraft, "get_pitch", lambda: 0.0)())
    heading = float(getattr(aircraft, "get_heading", lambda: 0.0)())
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(heading), np.sin(heading)
    inverse_roll = np.array(((1.0, 0.0, 0.0), (0.0, cr, sr), (0.0, -sr, cr)))
    inverse_pitch = np.array(((cp, 0.0, -sp), (0.0, 1.0, 0.0), (sp, 0.0, cp)))
    inverse_heading = np.array(((cy, sy, 0.0), (-sy, cy, 0.0), (0.0, 0.0, 1.0)))
    return inverse_roll @ inverse_pitch @ inverse_heading @ world_direction


def radar_features(controlled) -> np.ndarray:
    """Encode the controlled aircraft's radar state as 23 finite values.

    Strings and object references are reduced to categorical/count features.
    Angles use sine/cosine pairs, the radar direction is body-relative, and
    scan-zone values describe the current scan centre and beam position.
    Missing radar state is represented by zeros and ``radar_direction_valid``
    remains zero.
    """
    result = np.zeros(RADAR_FEATURE_SIZE, dtype=np.float32)
    radar = getattr(controlled, "radar", None)
    if radar is None:
        return result
    mode = getattr(radar, "radar_mode", None)
    result[0:2] = (float(mode == "scan"), float(mode == "stt"))
    tracks = getattr(radar, "track_targets", {})
    try:
        track_count = len(tracks)
    except TypeError:
        track_count = 0
    result[2] = np.clip(float(track_count) / 4.0, 0.0, 1.0)
    result[3] = float(track_count > 0)

    azimuth = float(getattr(radar, "RadarAzimuth", 0.0))
    elevation = float(getattr(radar, "RadarElevation", 0.0))
    result[4:6] = _angle_pair(azimuth)
    result[6:8] = _angle_pair(elevation)

    direction = _vector_values(getattr(radar, "RadarDirectionVec", None))
    if direction is not None:
        result[8:11] = _body_direction(direction, controlled).astype(np.float32)
        result[22] = 1.0

    zone = getattr(radar, "scan_zone", None)
    rpy = np.asarray(getattr(zone, "now_scan_zone_rpy", np.zeros(3)), dtype=np.float64).reshape(-1)
    if rpy.size != 3 or not np.isfinite(rpy).all():
        rpy = np.zeros(3)
    result[11:13] = _angle_pair(rpy[0])
    result[13:15] = _angle_pair(rpy[1])
    result[15:17] = _angle_pair(rpy[2])
    result[17:19] = _angle_pair(getattr(zone, "now_beam_angle_hori", 0.0))
    result[19:21] = _angle_pair(getattr(zone, "now_beam_angle_vert", 0.0))
    result[21] = float(bool(getattr(zone, "beam_step_right", False)))
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def append_radar_features(observation, controlled) -> np.ndarray:
    """Append the fixed radar block to a compact/hostile-missile vector."""
    observation = np.asarray(observation, dtype=np.float32).reshape(-1)
    if observation.size not in (COMPACT_OBSERVATION_SIZE, HOSTILE_MISSILE_OBSERVATION_SIZE):
        raise ValueError(
            "expected a 47- or 87-feature vector before radar features, "
            f"got {observation.size}"
        )
    return np.concatenate((observation, radar_features(controlled))).astype(np.float32, copy=False)


def observation_feature_schema(observation_size: int = POLICY_OBSERVATION_SIZE):
    """Return machine-readable names and scales for stored policy columns."""
    fields = [(f"own_{axis}", scale) for axis, scale in zip(
        ("north", "west", "up", "velocity_north", "velocity_west", "velocity_up"),
        (50_000, 50_000, 10_000, 600, 600, 600), strict=True)]
    fields += [("own_altitude", 10_000), ("own_heading_sin", 1), ("own_heading_cos", 1)]
    fields += [(f"target_relative_{axis}", scale) for axis, scale in zip(
        ("north", "west", "up", "velocity_north", "velocity_west", "velocity_up", "range"),
        (50_000, 50_000, 10_000, 600, 600, 600, 50_000), strict=True)]
    fields += [("target_bearing_sin", 1), ("target_bearing_cos", 1),
               ("hostile_missile_incoming", 1)]
    for slot in range(4):
        fields += [(f"friendly_{slot}_relative_{axis}", scale) for axis, scale in zip(
            ("north", "west", "up", "velocity_north", "velocity_west", "velocity_up", "range"),
            (50_000, 50_000, 10_000, 1_200, 1_200, 1_200, 50_000), strict=True)]
    if observation_size in (HOSTILE_MISSILE_OBSERVATION_SIZE, POLICY_OBSERVATION_SIZE,
                            HOSTILE_MISSILE_RADAR_OBSERVATION_SIZE,
                            RADAR_POLICY_OBSERVATION_SIZE):
        for slot in range(4):
            fields += [(f"hostile_{slot}_{name}", scale) for name, scale in zip(
                HOSTILE_MISSILE_FEATURE_NAMES, _MISSILE_SCALES, strict=True)]
    elif observation_size not in (COMPACT_OBSERVATION_SIZE + len(ENERGY_FEATURE_NAMES),
                                  COMPACT_RADAR_OBSERVATION_SIZE):
        raise ValueError(f"unsupported policy observation size: {observation_size}")
    if observation_size in (RADAR_OBSERVATION_SIZE, HOSTILE_MISSILE_RADAR_OBSERVATION_SIZE,
                            RADAR_POLICY_OBSERVATION_SIZE):
        fields += [(name, "sin/cos or bounded radar-state encoding")
                   for name in RADAR_FEATURE_NAMES]
    if observation_size in (51, POLICY_OBSERVATION_SIZE, RADAR_POLICY_OBSERVATION_SIZE):
        fields += [(f"{role}_{kind}", formula) for role in ("own", "target")
                   for kind, formula in (("kinetic_energy", "(speed_mps / 400)^2"),
                                         ("potential_energy", "altitude_m / 12000"))]
    return [{"column": f"obs_{index:02d}", "name": name,
             **({"divisor": value} if isinstance(value, int) else {"formula": value})}
            for index, (name, value) in enumerate(fields)]


def encode_skill_decision(skill_name, parameters, skill_names):
    """Encode a selected skill for forecasting without losing its parameters."""
    names = tuple(skill_names)
    if skill_name not in names:
        raise ValueError(f"unknown skill {skill_name!r}")
    return {
        "skill": skill_name,
        "skill_index": int(names.index(skill_name)),
        "skill_one_hot": [int(name == skill_name) for name in names],
        "parameters_json": json.dumps(parameters, sort_keys=True),
    }
