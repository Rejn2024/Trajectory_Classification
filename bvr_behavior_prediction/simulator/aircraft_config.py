from dataclasses import dataclass


@dataclass(frozen=True)
class AircraftConfig:
    aircraft_type: str
    bvr_unit_spec: str
    fdm_type: str = "jsbsim"
    controller_id: str | None = "default"
    model_version: str | None = "repo-pinned"
    reference_speed_ms: float | None = None
    reference_altitude_m: float | None = None
    max_g: float | None = None

