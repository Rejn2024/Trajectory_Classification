from dataclasses import dataclass

IDENTITY_COLUMNS = ("episode_id", "perspective_id", "step", "time_s", "observer_id", "target_id")
STATE_COLUMNS = tuple(
    f"{role}_{field}" for role in ("observer", "target")
    for field in ("x", "y", "z", "vx", "vy", "vz", "heading", "pitch", "roll", "speed")
)
RELATIVE_COLUMNS = (
    "rel_x_body", "rel_y_body", "rel_z_body", "rel_vx_body", "rel_vy_body", "rel_vz_body",
    "range", "range_rate", "relative_bearing", "relative_heading", "relative_altitude",
    "relative_speed", "target_aspect", "angle_off", "los_azimuth", "los_elevation",
    "los_rate", "target_turn_rate", "target_climb_rate", "target_acceleration",
)
SENSOR_COLUMNS = ("track_valid", "track_age", "sensor_mode")
AIRCRAFT_COLUMNS = ("observer_aircraft_type", "target_aircraft_type")
TRAJECTORY_COLUMNS = IDENTITY_COLUMNS + AIRCRAFT_COLUMNS + STATE_COLUMNS + RELATIVE_COLUMNS + SENSOR_COLUMNS

EPISODE_COLUMNS = (
    "episode_id", "seed", "bvr_sim_commit", "bvr_sim_backend", "bvr_sim_config_hash",
    "jsbsim_version", "observer_aircraft_type", "target_aircraft_type", "observer_model_version",
    "target_model_version", "initial_range_nm", "initial_altitude_difference_m",
    "initial_heading_difference_deg", "initial_speed_difference_ms", "initial_aspect_bin",
    "scenario_bin", "observer_policy", "target_policy", "weapons_enabled", "sensor_mode",
    "episode_duration_s", "termination_reason",
)


@dataclass(frozen=True)
class SchemaVersions:
    feature: str = "1.0.0"
    label: str = "1.0.0"


def validate_columns(columns, required=TRAJECTORY_COLUMNS) -> None:
    missing = set(required) - set(columns)
    if missing:
        raise ValueError(f"Missing schema columns: {sorted(missing)}")

