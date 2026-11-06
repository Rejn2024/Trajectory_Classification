"""Storage contract for raw pilot observations; no forecasting preprocessing."""

import numpy as np

DT = 0.1
KEYS = ("episode_id", "step", "time_s")
OBSERVATION_COLUMNS = tuple(f"obs_{i:02d}" for i in range(51))
STATE_COLUMNS = tuple(f"{role}_{name}" for role in ("observer", "target")
                      for name in ("x", "y", "z", "vx", "vy", "vz"))


def observer_frame(simulator, observer_id, target_id):
    """Read the compact interface for either aircraft, with observer-first energies.

    The scripted tactical controller uses aircraft objects directly. This is a
    side-specific recording interface, not a claim that it consumes this vector.
    Read-only extraction does not alter either controller or their actions.
    """
    observer = simulator.agents[observer_id]
    target = simulator.agents[target_id]
    compact = simulator.obs_manager.extract_obs(observer, simulator.agents, simulator.missiles)
    energy = [(float(aircraft.get_speed()) / 400.0) ** 2
              if kind == "speed" else float(aircraft.get_altitude()) / 12000.0
              for aircraft in (observer, target) for kind in ("speed", "altitude")]
    frame = np.concatenate((np.asarray(compact, dtype=np.float32),
                            np.asarray(energy, dtype=np.float32)))
    if frame.shape != (51,):
        raise ValueError("Expected a 51-feature observer frame")
    return frame


def validate_observer_geometry(observations, tracks):
    """Independently detect side swaps and sign errors against physical tracks."""
    observer_alive = tracks.observer_alive.to_numpy(dtype=bool)
    target_alive = tracks.target_alive.to_numpy(dtype=bool)
    observed = observations[list(OBSERVATION_COLUMNS)].to_numpy()
    own = tracks[[f"observer_{k}" for k in ("x", "y", "z", "vx", "vy", "vz")]].to_numpy()
    other = tracks[[f"target_{k}" for k in ("x", "y", "z", "vx", "vy", "vz")]].to_numpy()
    scale = np.array([50000, 50000, 10000, 600, 600, 600])
    expected_own = np.where(observer_alive[:, None], own / scale, 0)
    relative = other - own
    visible = observer_alive & target_alive
    expected_target = np.where(visible[:, None], relative / scale, 0)
    expected_range = np.where(visible, np.linalg.norm(relative[:, :3], axis=1) / 50000, 0)
    for actual, expected in ((observed[:, :6], expected_own),
                             (observed[:, 9:15], expected_target),
                             (observed[:, 15], expected_range)):
        if not np.allclose(actual, expected, rtol=1e-6, atol=1e-7):
            raise ValueError("Observer geometry does not match the declared aircraft roles")


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
