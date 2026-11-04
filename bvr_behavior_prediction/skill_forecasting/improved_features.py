"""Observable encounter geometry and strictly past-only, multiscale histories."""

import numpy as np

from bvr_behavior_prediction.forecasting.data import FEATURE_NAMES, observed_kinematics


def geometry_features(frames, mode="geometry"):
    """Each row uses that observation and its already-causal acceleration only."""
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 2 or frames.shape[1] != 54 or not np.isfinite(frames).all():
        raise ValueError("Expected finite 54-feature causal frames")
    if mode == "raw":
        return frames.copy(), list(FEATURE_NAMES)
    if mode != "geometry":
        raise ValueError("Unknown feature mode")
    pieces, names = [frames], list(FEATURE_NAMES)

    def add(values, labels):
        values = np.asarray(values, dtype=np.float32)
        pieces.append(values.reshape(len(frames), -1))
        names.extend(labels)

    obs = frames[:, :51]
    position = (obs[:, :3] + obs[:, 9:12])
    target_velocity = (obs[:, 3:6] + obs[:, 12:15]) * 600
    observer_velocity = obs[:, 3:6] * 600
    relative_position = obs[:, 9:12] * np.array([50000, 50000, 10000], dtype=np.float32)
    relative_velocity = obs[:, 12:15] * 600
    target_speed = np.linalg.norm(target_velocity, axis=1)
    observer_speed = np.linalg.norm(observer_velocity, axis=1)
    target_horizontal = np.maximum(np.linalg.norm(target_velocity[:, :2], axis=1), 1)
    observer_horizontal = np.maximum(np.linalg.norm(observer_velocity[:, :2], axis=1), 1)
    target_unit = target_velocity[:, :2] / target_horizontal[:, None]
    observer_unit = observer_velocity[:, :2] / observer_horizontal[:, None]
    distance = np.linalg.norm(relative_position, axis=1)
    los = relative_position / np.maximum(distance[:, None], 1)
    horizontal_los = relative_position[:, :2] / np.maximum(
        np.linalg.norm(relative_position[:, :2], axis=1, keepdims=True), 1)
    acceleration = frames[:, 51:54] * 100
    add(position, [f"reconstructed_target_position_{axis}" for axis in "nwu"])
    add(target_velocity / 600, [f"reconstructed_target_velocity_{axis}" for axis in "nwu"])
    add(np.column_stack((target_speed, observer_speed)) / 600, ["target_speed", "observer_speed"])
    add(target_unit, ["target_course_n", "target_course_w"])
    add(observer_unit, ["observer_course_n", "observer_course_w"])
    add(los, [f"line_of_sight_{axis}" for axis in "nwu"])
    for velocity, name in [(relative_velocity, "relative"), (target_velocity, "target"),
                           (observer_velocity, "observer")]:
        add(np.sum(velocity * los, axis=1) / 600, [f"{name}_radial_speed"])
    for a, b, name in [(target_unit, -horizontal_los, "target_aspect"),
                       (observer_unit, horizontal_los, "observer_aspect"),
                       (target_unit, observer_unit, "relative_course")]:
        add(np.column_stack((np.sum(a * b, axis=1), a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])),
            [f"{name}_cos", f"{name}_sin"])
    forward_acc = np.sum(acceleration[:, :2] * target_unit, axis=1)
    lateral_acc = target_unit[:, 0] * acceleration[:, 1] - target_unit[:, 1] * acceleration[:, 0]
    add(np.column_stack((forward_acc / 100, lateral_acc / 100, lateral_acc / target_horizontal)),
        ["target_forward_acceleration", "target_lateral_acceleration", "target_course_rate"])
    add(np.log1p(distance / 1000), ["log_range_km"])
    # A constant-velocity geometric projection from current observations, not a future track.
    cpa_time = np.clip(-np.sum(relative_position * relative_velocity, axis=1) /
                       np.maximum(np.sum(relative_velocity ** 2, axis=1), 1), 0, 120)
    cpa_distance = np.linalg.norm(relative_position + cpa_time[:, None] * relative_velocity, axis=1)
    add(np.column_stack((cpa_time / 120, cpa_distance / 50000)), ["cpa_time", "cpa_distance"])
    add(obs[:, 49:51] - obs[:, 47:49], ["kinetic_energy_difference", "potential_energy_difference"])
    missile_present = np.column_stack([obs[:, 25 + 7 * slot] > 0 for slot in range(4)]).astype(np.float32)
    add(missile_present, [f"observer_missile_slot_{slot}_present" for slot in range(4)])
    add(missile_present.sum(1) / 4, ["observer_missile_count"])
    result = np.concatenate(pieces, axis=1)
    if not np.isfinite(result).all() or result.shape[1] != len(names):
        raise ValueError("Invalid engineered features")
    return result, names


def history_offsets(history_seconds=5, recent_stride=1):
    if history_seconds not in (5, 20, 60) or recent_stride not in (1, 5):
        raise ValueError("Supported histories: 5/20/60 seconds; recent stride: 1/5 ticks")
    recent = np.arange(-50, 1, recent_stride, dtype=np.int64)
    older = np.arange(-history_seconds * 10, -50, 10, dtype=np.int64)
    return np.r_[older, recent]


def window_indices(anchors, origin_steps, offsets):
    """Padding never crosses reset/episode boundaries or reads after the origin."""
    anchors, origin_steps, offsets = map(np.asarray, (anchors, origin_steps, offsets))
    if np.any(offsets > 0) or offsets[-1] != 0:
        raise ValueError("Histories must end at the origin and contain no future offsets")
    valid = origin_steps[:, None] + offsets[None] >= 6
    starts = anchors - origin_steps
    indices = np.maximum(anchors[:, None] + offsets[None], starts[:, None] + 6)
    return indices, valid


def raw_history_features(observations, offsets, mode):
    """Streaming parity: retain 0.5 seconds before the oldest requested token."""
    observations = np.asarray(observations, dtype=np.float32)
    if (observations.ndim != 2 or observations.shape[1] != 51 or len(observations) < 56
            or not np.isfinite(observations).all()):
        raise ValueError("Supply at least 56 finite, consecutive 10 Hz B01 observation frames")
    required = int(-np.min(offsets)) + 6
    observations = observations[-required:]
    if np.any(observations[:, 15] <= 0):
        raise ValueError("A live target observation is required throughout supplied history")
    _, _, acceleration = observed_kinematics(observations)
    base = np.column_stack((observations, acceleration / 100)).astype(np.float32)
    frames, _ = geometry_features(base, mode)
    indices = len(frames) - 1 + np.asarray(offsets)
    # Local frame 0 can be a valid non-reset frame; five earlier frames suffice here.
    valid = indices >= 5
    return frames[np.maximum(indices, 5)], valid
