from .aircraft_config import AircraftConfig

AIRCRAFT_REGISTRY: dict[str, AircraftConfig] = {
    "F16": AircraftConfig("F16", "F16", reference_speed_ms=300.0,
                          reference_altitude_m=10_000.0, max_g=9.0),
}


def get_aircraft(name: str) -> AircraftConfig:
    try:
        return AIRCRAFT_REGISTRY[name.upper()]
    except KeyError as exc:
        raise KeyError(f"Unknown aircraft {name!r}; available: {sorted(AIRCRAFT_REGISTRY)}") from exc


def register_aircraft(config: AircraftConfig, *, replace: bool = False) -> None:
    key = config.aircraft_type.upper()
    if key in AIRCRAFT_REGISTRY and not replace:
        raise ValueError(f"Aircraft {key} is already registered")
    AIRCRAFT_REGISTRY[key] = config

