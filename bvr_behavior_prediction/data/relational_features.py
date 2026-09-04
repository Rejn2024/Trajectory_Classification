import math

from .transforms import to_observer_body, wrap_angle


def relational_features(observer: dict, target: dict) -> dict[str, float]:
    """Create target-centric features with +x forward, +y right and +z up."""
    dp = tuple(target[k] - observer[k] for k in ("x", "y", "z"))
    dv = tuple(target[k] - observer[k] for k in ("vx", "vy", "vz"))
    rp = to_observer_body(dp, observer["heading"])
    rv = to_observer_body(dv, observer["heading"])
    distance = math.sqrt(sum(x * x for x in dp))
    horizontal = math.hypot(rp[0], rp[1])
    range_rate = sum(p * v for p, v in zip(dp, dv)) / distance if distance else 0.0
    target_speed = target.get("speed", math.sqrt(sum(target[k] ** 2 for k in ("vx", "vy", "vz"))))
    observer_speed = observer.get("speed", math.sqrt(sum(observer[k] ** 2 for k in ("vx", "vy", "vz"))))
    bearing = math.atan2(rp[1], rp[0])
    heading_rel = wrap_angle(target["heading"] - observer["heading"])
    target_to_observer = math.atan2(-dp[1], -dp[0])
    aspect = wrap_angle(target["heading"] - target_to_observer)
    return {
        "rel_x_body": rp[0], "rel_y_body": rp[1], "rel_z_body": rp[2],
        "rel_vx_body": rv[0], "rel_vy_body": rv[1], "rel_vz_body": rv[2],
        "range": distance, "range_rate": range_rate, "relative_bearing": bearing,
        "relative_heading": heading_rel, "relative_altitude": dp[2],
        "relative_speed": target_speed - observer_speed, "target_aspect": aspect,
        "angle_off": abs(wrap_angle(heading_rel - bearing)), "los_azimuth": bearing,
        "los_elevation": math.atan2(rp[2], horizontal),
    }


def add_temporal_dynamics(rows: list[dict]) -> list[dict]:
    """Add causal finite-difference motion features in-place."""
    previous = None
    previous_los = None
    for row in rows:
        if previous is None:
            turn = climb = acceleration = los_rate = 0.0
        else:
            dt = row["time_s"] - previous["time_s"]
            if dt <= 0:
                raise ValueError("Rows must have strictly increasing time_s")
            turn = wrap_angle(row["target_heading"] - previous["target_heading"]) / dt
            climb = (row["target_z"] - previous["target_z"]) / dt
            acceleration = (row["target_speed"] - previous["target_speed"]) / dt
            los_rate = wrap_angle(row["los_azimuth"] - previous_los) / dt
        row.update(target_turn_rate=turn, target_climb_rate=climb,
                   target_acceleration=acceleration, los_rate=los_rate)
        previous, previous_los = row, row["los_azimuth"]
    return rows

