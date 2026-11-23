"""Shared policy-frame and forecasting-feature encoding.

The compact simulator observation is deliberately kept separate from the
policy-frame contract.  A legacy policy consumes 47 compact values plus four
energy values (51 total); notebook 31 policies consume the same values plus
four hostile-missile slots (91 total).  Keeping the encoder here makes runtime
inference and corpus recording use exactly the same ordering and scaling.
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
    if observation_size in (HOSTILE_MISSILE_OBSERVATION_SIZE, POLICY_OBSERVATION_SIZE):
        for slot in range(4):
            fields += [(f"hostile_{slot}_{name}", scale) for name, scale in zip(
                HOSTILE_MISSILE_FEATURE_NAMES, _MISSILE_SCALES, strict=True)]
    elif observation_size != COMPACT_OBSERVATION_SIZE + len(ENERGY_FEATURE_NAMES):
        raise ValueError(f"unsupported policy observation size: {observation_size}")
    if observation_size in (51, POLICY_OBSERVATION_SIZE):
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
