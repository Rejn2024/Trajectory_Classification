"""Event-aware reward shaping for a BVR engagement."""

import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from functools import partial

from .config import RewardWeights


class CombatReward:
    """Compute rewards from simulator ``info`` without repeatedly paying events.

    Substep event counts preserve short-lived and consecutive events. Boolean-only
    backends use edge detection for one-shot events. Ground clearance is a
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
        counts = info.get("reward_event_counts", {})
        for event, value in mapping.items():
            count = counts.get(event, info.get(event, False) and not self.previous.get(event, False))
            components[event] = value * count
        components["fired_without_lock"] = self.weights.fired_without_lock * counts.get(
            "fired_without_lock", info.get("fired", False) and not info.get("target_locked", False)
        )
        self.previous = {field: bool(info.get(field, False)) for field in self.EVENT_FIELDS}
        return sum(components.values()), components


class CleanWinCombatReward(CombatReward):
    """Combat v2 plus one terminal bonus for a confirmed elimination and survival.

    The collector supplies episode_ended from its termination/truncation flags.
    A kill during a continuing flight is insufficient: a later death must exclude
    the bonus. The episode's existing event rewards remain unchanged.
    """

    def __init__(self, weights=None, safe_altitude_m=500.0, clean_win_bonus=250.0):
        if (isinstance(clean_win_bonus, bool) or not math.isfinite(clean_win_bonus)
                or clean_win_bonus <= 0):
            raise ValueError("clean_win_bonus must be finite and positive")
        super().__init__(weights, safe_altitude_m)
        self.clean_win_bonus = float(clean_win_bonus)
        self._terminal_scored = False

    def reset(self):
        super().reset()
        self._terminal_scored = False

    def __call__(self, info):
        if type(info.get("episode_ended")) is not bool:
            raise ValueError("clean-win reward requires an explicit boolean episode_ended")
        bonus = 0.0
        if info["episode_ended"] and not self._terminal_scored:
            for key in ("blue_alive", "opponent_eliminated", "opponent_destroyed"):
                if key not in info:
                    raise ValueError(f"clean-win reward requires terminal {key}")
            if info["opponent_eliminated"] and not info["opponent_destroyed"]:
                raise ValueError("a confirmed elimination requires opponent destruction")
            if info["blue_alive"] and info["opponent_eliminated"]:
                bonus = self.clean_win_bonus
            self._terminal_scored = True
        reward, components = super().__call__(info)
        components["clean_win_bonus"] = bonus
        return reward + bonus, components


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
        version="2",
    )


def clean_win_reward_definition(name="combat_clean_win", weights=None, safe_altitude_m=500.0,
                                clean_win_bonus=250.0):
    """Versioned opt-in recipe; common evaluation continues using combat v2."""
    weights = weights or RewardWeights()
    # Validate before preparing simulators or writing a training batch.
    CleanWinCombatReward(weights, safe_altitude_m, clean_win_bonus)
    return RewardDefinition(
        name=name,
        factory=partial(CleanWinCombatReward, weights=weights, safe_altitude_m=safe_altitude_m,
                        clean_win_bonus=clean_win_bonus),
        parameters={"weights": asdict(weights), "safe_altitude_m": safe_altitude_m,
                    "clean_win_bonus": float(clean_win_bonus)},
        version="3",
    )
