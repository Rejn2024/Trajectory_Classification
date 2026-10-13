"""Storage contract for raw pilot observations; no forecasting preprocessing."""

import numpy as np

DT = 0.1
KEYS = ("episode_id", "step", "time_s")
OBSERVATION_COLUMNS = tuple(f"obs_{i:02d}" for i in range(51))
STATE_COLUMNS = tuple(f"{role}_{name}" for role in ("observer", "target")
                      for name in ("x", "y", "z", "vx", "vy", "vz"))


def reconstruct_policy_history(frames, step, history_steps=20):
    """Reconstruct the controller input for verification, without saving windows."""
    indices = np.maximum(0, np.arange(step - history_steps + 1, step + 1))
    return np.asarray(frames, dtype=np.float32)[indices]


def validate_episode(observations, tracks):
    """Reject missing, misaligned or nonfinite samples without transforming them."""
    if observations.episode_id.nunique() != 1 or tracks.episode_id.nunique() != 1:
        raise ValueError("Expected exactly one episode")
    if len(observations) != len(tracks):
        raise ValueError("Observation and track lengths differ")
    if not observations[list(KEYS)].equals(tracks[list(KEYS)]):
        raise ValueError("Observation and track timestamps differ")
    if not np.array_equal(tracks.step.to_numpy(), np.arange(len(tracks))):
        raise ValueError("Missing or repeated simulator step")
    if not np.allclose(tracks.time_s, tracks.step * DT, atol=1e-8):
        raise ValueError("Unexpected sample cadence")
    if not np.isfinite(observations[list(OBSERVATION_COLUMNS)].to_numpy()).all():
        raise ValueError("Non-finite pilot observation")
    if not np.isfinite(tracks[list(STATE_COLUMNS)].to_numpy()).all():
        raise ValueError("Non-finite aircraft state")
    for column in ("observer_alive", "target_alive"):
        if np.any(np.diff(tracks[column].to_numpy(dtype=np.int8)) > 0):
            raise ValueError("Aircraft resurrected within an episode")
