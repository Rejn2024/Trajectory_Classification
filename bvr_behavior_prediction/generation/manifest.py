from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import platform


@dataclass
class DatasetManifest:
    dataset_id: str
    simulator: dict
    aircraft: dict
    generation: dict
    fdm: dict = field(default_factory=lambda: {"type": "jsbsim"})
    observation: dict = field(default_factory=lambda: {"type": "entity"})
    weapons: dict = field(default_factory=lambda: {"enabled": False})
    feature_schema_version: str = "1.0.0"
    label_schema_version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    python_version: str = field(default_factory=platform.python_version)
    os: str = field(default_factory=platform.platform)
    def as_dict(self): return asdict(self)

