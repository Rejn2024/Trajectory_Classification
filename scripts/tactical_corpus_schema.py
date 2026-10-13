"""Causal features and censored forecasting labels for the tactical corpus.

This schema is independent of the RL observation vector and legacy skill labels.
Coordinates are North/West/Up; the observer course frame is forward/left/up.
"""

import numpy as np
import pandas as pd

DT = 0.1
HORIZONS = (2, 4, 8)
RATE_WINDOW = 10  # one second, trailing only
FEATURES = (
    "observer_altitude_m", "observer_speed_mps", "observer_climb_mps",
    "relative_forward_m", "relative_left_m", "relative_up_m",
    "relative_vforward_mps", "relative_vleft_mps", "relative_vup_mps",
    "range_m", "range_rate_mps", "bearing_sin", "bearing_cos", "elevation_rad",
    "relative_course_sin", "relative_course_cos", "target_speed_mps",
    "target_turn_rate_radps", "target_climb_mps", "target_acceleration_mps2",
)
KEYS = ("episode_id", "step", "time_s")
LABEL_MAP = {
    "lateral": {"0": "straight", "1": "left", "2": "right"},
    "vertical": {"0": "level", "1": "climb", "2": "descend"},
    "energy": {"0": "steady", "1": "accelerate", "2": "decelerate"},
    "thresholds": {"turn_radps": 0.01, "climb_mps": 2.0, "acceleration_mps2": 0.5},
    "rate_window_s": 1.0,
    "horizons_s": list(HORIZONS),
    "definition": "Actual motion over the trailing second at t+h; not controller intent.",
}


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def derive_tables(raw):
    """Process exactly one whole episode; future values never enter features."""
    if raw.episode_id.nunique() != 1:
        raise ValueError("derive_tables requires exactly one episode")
    if not np.array_equal(raw.step.to_numpy(), np.arange(len(raw))):
        raise ValueError("Steps must start at zero and increase by one")
    if not np.allclose(raw.time_s, raw.step * DT, atol=1e-8):
        raise ValueError("Unexpected sample cadence")
    o = raw[[f"observer_{k}" for k in ("x", "y", "z", "vx", "vy", "vz")]].to_numpy()
    t = raw[[f"target_{k}" for k in ("x", "y", "z", "vx", "vy", "vz")]].to_numpy()
    if not np.isfinite(o).all() or not np.isfinite(t).all():
        raise ValueError("Non-finite aircraft state")
    oc, tc = np.arctan2(o[:, 4], o[:, 3]), np.arctan2(t[:, 4], t[:, 3])
    ospeed, tspeed = np.linalg.norm(o[:, 3:], axis=1), np.linalg.norm(t[:, 3:], axis=1)
    dp, dv = t[:, :3] - o[:, :3], t[:, 3:] - o[:, 3:]
    c, s = np.cos(oc), np.sin(oc)
    forward, left = c * dp[:, 0] + s * dp[:, 1], -s * dp[:, 0] + c * dp[:, 1]
    vf, vl = c * dv[:, 0] + s * dv[:, 1], -s * dv[:, 0] + c * dv[:, 1]
    distance = np.linalg.norm(dp, axis=1)
    bearing = np.arctan2(left, forward)
    turn = np.zeros(len(raw))
    climb = np.zeros(len(raw))
    acceleration = np.zeros(len(raw))
    w = RATE_WINDOW
    turn[w:] = wrap(tc[w:] - tc[:-w]) / (w * DT)
    climb[w:] = (t[w:, 2] - t[:-w, 2]) / (w * DT)
    acceleration[w:] = (tspeed[w:] - tspeed[:-w]) / (w * DT)
    alive = raw.observer_alive.to_numpy() & raw.target_alive.to_numpy()
    valid = pd.Series(alive).rolling(w + 1).sum().eq(w + 1).to_numpy(copy=True)
    valid &= (np.hypot(o[:, 3], o[:, 4]) > 1) & (np.hypot(t[:, 3], t[:, 4]) > 1)
    features = raw[list(KEYS)].copy()
    values = (
        o[:, 2], ospeed, o[:, 5], forward, left, dp[:, 2], vf, vl, dv[:, 2],
        distance, (dp * dv).sum(axis=1) / np.maximum(distance, 1e-6),
        np.sin(bearing), np.cos(bearing), np.arctan2(dp[:, 2], np.hypot(forward, left)),
        np.sin(tc - oc), np.cos(tc - oc), tspeed, turn, climb, acceleration,
    )
    for name, value in zip(FEATURES, values, strict=True):
        features[name] = value.astype(np.float32)
    features["input_valid"] = valid
    labels = raw[list(KEYS)].copy()
    # Positive course rotation in NWU is a left turn, not a right turn.
    classes = {
        "lateral": np.where(turn > 0.01, 1, np.where(turn < -0.01, 2, 0)),
        "vertical": np.where(climb > 2.0, 1, np.where(climb < -2.0, 2, 0)),
        "energy": np.where(acceleration > 0.5, 1, np.where(acceleration < -0.5, 2, 0)),
    }
    labels["current_valid"] = valid
    for name, values in classes.items():
        labels[f"current_{name}"] = pd.array(np.where(valid, values, None), dtype="Int8")
    for horizon in HORIZONS:
        shift = round(horizon / DT)
        future_valid = np.zeros(len(raw), dtype=bool)
        if shift < len(raw):
            # Alive states are terminal: no resurrection within an episode.
            future_valid[:-shift] = valid[:-shift] & valid[shift:]
        labels[f"valid_{horizon}s"] = future_valid
        for name, values in classes.items():
            future = np.empty(len(raw), dtype=object)
            future[:] = None
            if shift < len(raw):
                future[:-shift] = values[shift:]
            future[~future_valid] = None
            labels[f"future_{name}_{horizon}s"] = pd.array(future, dtype="Int8")
    return features, labels
