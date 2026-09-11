"""Seeded stochastic tactical-skill selection for BVR Sim rollouts."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import random
from typing import Any

from .base import NativeAction, TacticalPolicy

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SkillSpec:
    """Bounds used to sample one invocation of a skill."""

    duration_s: tuple[float, float]
    parameters: dict[str, tuple[float, float]]


DEFAULT_SKILLS = {
    "maintain_position": SkillSpec((4.0, 10.0), {"altitude_target": (6500, 10500), "speed_target": (220, 310)}),
    "pursue_target": SkillSpec((5.0, 14.0), {"lead_angle": (0.0, 0.25), "speed_target": (260, 360)}),
    "launch": SkillSpec((0.5, 1.5), {"max_launch_range": (18000, 40000)}),
    "crank_maneuver": SkillSpec((6.0, 16.0), {"offset_angle": (25, 55), "speed_target": (260, 350)}),
    "support_missile": SkillSpec((5.0, 14.0), {"offset_angle": (10, 35), "speed_target": (250, 340)}),
    "turn_cold": SkillSpec((6.0, 14.0), {"offset_angle": (145, 180), "speed_target": (300, 390)}),
    "recommit": SkillSpec((5.0, 12.0), {"lead_angle": (0.0, 0.2), "speed_target": (280, 370)}),
    "missile_evasion": SkillSpec((4.0, 10.0), {"break_duration": (4, 10), "max_g": (6, 9)}),
    "disengage": SkillSpec((15.0, 30.0), {"climb_rate": (-25, 60), "speed_target": (310, 400)}),
    "search": SkillSpec((5.0, 15.0), {"sweep_angle": (20, 60), "speed_target": (220, 310)}),
}

SEQUENCE = (
    "maintain_position", "pursue_target", "launch", "crank_maneuver",
    "support_missile", "turn_cold", "recommit",
)


class StochasticSkillPolicy(TacticalPolicy):
    """Execute persistent, randomized ``SkillManager`` skills.

    Emergency predicates are evaluated every tick, while normal edges are only
    considered after the sampled duration has elapsed. All randomness comes from
    the RNG supplied to :meth:`reset`, making dataset episodes replayable.
    """

    label = "maintain_position"

    def __init__(self, skill_manager: Any, specs: dict[str, SkillSpec] | None = None):
        self.manager = skill_manager
        self.specs = specs or DEFAULT_SKILLS
        self.rng = random.Random()
        self._skill = None
        self._started_at = 0.0
        self._ends_at = 0.0
        self._params: dict[str, Any] = {}
        self._reason = "reset"
        self.transition_log: list[dict[str, Any]] = []

    def reset(self, rng: random.Random) -> None:
        self.rng = rng
        self.transition_log.clear()
        self._activate("maintain_position", 0.0, "episode_start")

    @property
    def selector_state(self) -> dict[str, Any]:
        return {
            "skill": self.label,
            "parameters": dict(self._params),
            "started_at_s": self._started_at,
            "ends_at_s": self._ends_at,
            "transition_reason": self._reason,
        }

    def _activate(self, name: str, time_s: float, reason: str) -> None:
        spec = self.specs[name]
        params = {key: self.rng.uniform(*bounds) for key, bounds in spec.parameters.items()}
        if name in {"crank_maneuver", "missile_evasion"}:
            key = "direction" if name == "crank_maneuver" else "break_direction"
            params[key] = self.rng.choice(("left", "right"))
        duration = self.rng.uniform(*spec.duration_s)
        skill = self.manager.create_skill(name, params)
        if skill is None or getattr(skill, "skill_name", name) != name:
            raise ValueError(f"SkillManager does not provide required skill {name!r}")
        previous = self.label
        self._skill = skill
        self.label, self._params = name, params
        self._started_at, self._ends_at, self._reason = time_s, time_s + duration, reason
        event = {"time_s": time_s, "from_skill": previous, "to_skill": name, **self.selector_state}
        self.transition_log.append(event)
        LOGGER.info("skill_transition", extra={"selector_state": event})

    def _interrupt(self, own: dict, opponent: dict) -> tuple[str, str] | None:
        if own.get("incoming_missile_active") or own.get("missile_warning") == "active":
            return "missile_evasion", "incoming_active_missile"
        if own.get("fuel_fraction", 1.0) <= 0.12:
            return "disengage", "low_fuel"
        if own.get("weapons_remaining", 1) <= 0:
            return "disengage", "no_weapons"
        if opponent.get("destroyed") or opponent.get("alive") is False:
            return "search", "target_destroyed"
        if self.label == "pursue_target" and _range(own, opponent) <= 18_000:
            return "launch", "target_entered_launch_envelope"
        return None

    def _normal_successor(self, own: dict, opponent: dict) -> tuple[str, str]:
        index = SEQUENCE.index(self.label) if self.label in SEQUENCE else -1
        candidate = SEQUENCE[(index + 1) % len(SEQUENCE)]
        distance = _range(own, opponent)
        if self.label in {"maintain_position", "recommit"} and distance > 55_000:
            candidate = "pursue_target"
        elif self.label == "pursue_target" and distance <= 35_000 and self.rng.random() < 0.8:
            candidate = "launch"
        return candidate, "duration_elapsed_geometry"

    def act(self, own_state: dict, opponent_state: dict, time_s: float) -> NativeAction:
        interrupt = self._interrupt(own_state, opponent_state)
        if interrupt:
            if interrupt[0] != self.label or self._reason != interrupt[1]:
                self._activate(interrupt[0], time_s, interrupt[1])
        elif time_s >= self._ends_at:
            name, reason = self._normal_successor(own_state, opponent_state)
            self._activate(name, time_s, reason)

        obs = {"time": time_s, "self_status": _skill_status(own_state), "target": opponent_state}
        command, completed = self._skill.execute(obs)
        minimum = self.specs[self.label].duration_s[0]
        if completed and time_s >= self._started_at + minimum:
            name, reason = self._normal_successor(own_state, opponent_state)
            self._activate(name, time_s, "skill_completed_" + reason)
            command, _ = self._skill.execute(obs)
        return _native_action(command)


def _range(own: dict, opponent: dict) -> float:
    if "range" in own:
        return float(own["range"])
    return math.sqrt(sum((float(opponent.get(k, 0)) - float(own.get(k, 0))) ** 2 for k in ("x", "y", "z")))


def _skill_status(state: dict) -> dict:
    return {
        "position": {"altitude_m": state.get("altitude_m", state.get("z", 8000))},
        "performance": {"speed_mps": state.get("speed_mps", state.get("speed", 250))},
    }


def _native_action(command: dict) -> NativeAction:
    """Quantize SkillManager's continuous command to BVR Sim's native action."""
    heading = max(0, min(14, 7 + round(float(command.get("delta_heading", 0)) * 7 / math.pi)))
    altitude = max(0, min(14, 7 + round(float(command.get("delta_altitude", 0)) / 50)))
    speed = max(0, min(8, 4 + round(float(command.get("delta_speed", 0)) / 25)))
    return NativeAction(heading, altitude, speed, int(bool(command.get("shoot", 0))))
