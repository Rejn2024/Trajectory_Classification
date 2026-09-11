from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True)
class ScenarioConfig:
    observer_aircraft: str = "F16"
    target_aircraft: str = "F16"
    backend: str = "cpp"
    dt: float = 0.4
    max_steps: int = 400
    weapons_enabled: bool = False
    observation_type: str = "entity"
    sensor_mode: str = "truth"

    def __post_init__(self) -> None:
        if self.backend not in {"python", "cpp"}:
            raise ValueError("backend must be 'python' or 'cpp'")
        if self.dt <= 0 or self.max_steps <= 0:
            raise ValueError("dt and max_steps must be positive")

    @property
    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()
