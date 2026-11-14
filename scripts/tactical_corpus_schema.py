"""Storage contract for raw pilot observations; no forecasting preprocessing."""

import numpy as np

DT = 0.1
KEYS = ("episode_id", "step", "time_s")
OBSERVATION_COLUMNS = tuple(f"obs_{i:02d}" for i in range(51))
STATE_COLUMNS = tuple(f"{role}_{name}" for role in ("observer", "target")
                      for name in ("x", "y", "z", "vx", "vy", "vz"))


def observation_feature_schema(observation_size=51):
    """Names/scales in the exact CompactObsSpace + threat + energy order."""
    fields = [(f"own_{axis}", scale) for axis, scale in zip(
        ("north", "west", "up", "velocity_north", "velocity_west", "velocity_up"),
        (50000, 50000, 10000, 600, 600, 600), strict=True)]
    fields += [("own_altitude", 10000), ("own_heading_sin", 1), ("own_heading_cos", 1)]
    fields += [(f"target_relative_{axis}", scale) for axis, scale in zip(
        ("north", "west", "up", "velocity_north", "velocity_west", "velocity_up", "range"),
        (50000, 50000, 10000, 600, 600, 600, 50000), strict=True)]
    fields += [("target_bearing_sin", 1), ("target_bearing_cos", 1), ("hostile_missile_incoming", 1)]
    for slot in range(4):
        fields += [(f"friendly_{slot}_relative_{axis}", scale) for axis, scale in zip(
            ("north", "west", "up", "velocity_north", "velocity_west", "velocity_up", "range"),
            (50000, 50000, 10000, 1200, 1200, 1200, 50000), strict=True)]
    if observation_size == 91:
        for slot in range(4):
            fields += [(f"hostile_{slot}_{name}", scale) for name, scale in zip(
                ("present", "relative_north", "relative_west", "relative_up", "relative_velocity_north",
                 "relative_velocity_west", "relative_velocity_up", "range", "closing_speed", "seeker_active"),
                (1, 50000, 50000, 10000, 1200, 1200, 1200, 50000, 1200, 1), strict=True)]
    elif observation_size != 51:
        raise ValueError("Unknown observation contract")
    fields += [(f"{role}_{kind}", formula) for role in ("own", "target")
               for kind, formula in (("kinetic_energy", "(speed_mps / 400)^2"),
                                     ("potential_energy", "altitude_m / 12000"))]
    return [{"column": f"obs_{i:02d}", "name": name,
             **({"divisor": scale} if isinstance(scale, int) else {"formula": scale})}
            for i, (name, scale) in enumerate(fields)]


def observer_frame(simulator, observer_id, target_id, observation_size=51):
    """Read the compact interface for either aircraft, with observer-first energies.

    The scripted tactical controller uses aircraft objects directly. This is a
    side-specific recording interface, not a claim that it consumes this vector.
    Read-only extraction does not alter either controller or their actions.
    """
    observer = simulator.agents[observer_id]
    target = simulator.agents[target_id]
    compact = simulator.obs_manager.extract_obs(observer, simulator.agents, simulator.missiles)
    if observation_size == 91:
        from bvr_behavior_prediction.rl.bvr_jsbsim_backend import BVRJSBSimBackend

        compact = np.concatenate((compact, BVRJSBSimBackend.hostile_missile_features(observer)))
    elif observation_size != 51:
        raise ValueError("Expected a 51- or 91-feature observation contract")
    energy = [(float(aircraft.get_speed()) / 400.0) ** 2
              if kind == "speed" else float(aircraft.get_altitude()) / 12000.0
              for aircraft in (observer, target) for kind in ("speed", "altitude")]
    frame = np.concatenate((np.asarray(compact, dtype=np.float32),
                            np.asarray(energy, dtype=np.float32)))
    if frame.shape != (observation_size,):
        raise ValueError(f"Expected a {observation_size}-feature observer frame")
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


def validate_episode(observations, tracks, observation_columns=OBSERVATION_COLUMNS):
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
    if not np.isfinite(observations[list(observation_columns)].to_numpy()).all():
        raise ValueError("Non-finite pilot observation")
    if not np.isfinite(tracks[list(STATE_COLUMNS)].to_numpy()).all():
        raise ValueError("Non-finite aircraft state")
    for column in ("observer_alive", "target_alive"):
        if np.any(np.diff(tracks[column].to_numpy(dtype=np.int8)) > 0):
            raise ValueError("Aircraft resurrected within an episode")
