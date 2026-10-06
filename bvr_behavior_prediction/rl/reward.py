"""Event-aware reward shaping for a BVR engagement."""

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from functools import partial

from .config import RewardWeights


class CombatReward:
    """Compute rewards from simulator ``info`` without repeatedly paying events.

    Boolean event keys may remain true in later frames; edge detection makes target
    lock, launch, missile evasion and kill rewards one-shot. Ground clearance is a
    small survival reward and is paid only while safely airborne.
    """

    EVENT_FIELDS = (
        "target_locked",
        "fired_with_lock",
        "missile_avoided",
        "opponent_destroyed",
        "crashed",
        "shot_down",
    )

    def __init__(self, weights=None, safe_altitude_m=500.0):
        self.weights = weights or RewardWeights()
        self.safe_altitude_m = safe_altitude_m
        self.previous = {}

    def reset(self):
        self.previous = {}

    def __call__(self, info: dict) -> tuple[float, dict[str, float]]:
        components = {}
        altitude = float(info.get("blue_altitude_m", 0.0))
        airborne = bool(info.get("blue_alive", True)) and altitude >= self.safe_altitude_m
        components["ground_clearance"] = self.weights.ground_clearance if airborne else 0.0
        mapping = {
            "target_locked": self.weights.target_lock_acquired,
            "fired_with_lock": self.weights.fired_with_lock,
            "missile_avoided": self.weights.incoming_missile_avoided,
            "opponent_destroyed": self.weights.opponent_destroyed,
            "crashed": self.weights.crashed,
            "shot_down": self.weights.shot_down,
        }
        for event, value in mapping.items():
            components[event] = (
                value if info.get(event, False) and not self.previous.get(event, False) else 0.0
            )
        components["fired_without_lock"] = (
            self.weights.fired_without_lock
            if info.get("fired", False) and not info.get("target_locked", False)
            else 0.0
        )
        self.previous = {field: bool(info.get(field, False)) for field in self.EVENT_FIELDS}
        return sum(components.values()), components


@dataclass(frozen=True)
class RewardDefinition:
    """A named, versioned recipe creating a fresh reward callable per episode.

    ``factory()`` must return a callable taking simulator info and returning
    ``(scalar_reward, component_dict)``. Store all tunable settings in parameters
    and change version when the implementation changes. Never return a shared
    stateful reward object from the factory.
    """

    name: str
    factory: Callable = field(repr=False)
    parameters: dict = field(default_factory=dict)
    version: str = "1"

    def __post_init__(self):
        if not self.name or not self.version or not callable(self.factory):
            raise ValueError("reward name, version, and a callable factory are required")
        # Copy and validate metadata before any expensive training starts.
        parameters = json.loads(json.dumps(self.parameters, allow_nan=False))
        if not isinstance(parameters, dict):
            raise TypeError("reward parameters must be a JSON object")
        object.__setattr__(self, "parameters", parameters)

    def as_dict(self):
        return {"name": self.name, "version": self.version, "parameters": self.parameters}


def combat_reward_definition(name="combat", weights=None, safe_altitude_m=500.0):
    """Wrap any event-weight configuration; names do not select fixed pilot roles."""
    weights = weights or RewardWeights()
    return RewardDefinition(
        name=name,
        factory=partial(CombatReward, weights=weights, safe_altitude_m=safe_altitude_m),
        parameters={"weights": asdict(weights), "safe_altitude_m": safe_altitude_m},
    )
