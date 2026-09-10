"""Validated, stateful tactical skills for high-level BVR controllers.

Skills emit *tactical deltas*: heading in radians, altitude in metres and speed
in metres/second.  Quantisation into the simulator's ``MultiDiscrete`` action
belongs at the environment adapter boundary, not inside individual skills.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple, Type

COMMAND_SEMANTICS = "tactical_deltas_v1"


class SkillError(ValueError):
    """Base error raised by the tactical skill layer."""


class UnknownSkillError(SkillError):
    """Raised when a selector asks for an unregistered skill."""


class InvalidSkillParameters(SkillError):
    """Raised when skill parameters do not match the declared schema."""


@dataclass(frozen=True)
class SkillCreationEvent:
    requested_skill: str
    executed_skill: str
    used_fallback: bool = False
    fallback_reason: Optional[str] = None


@dataclass
class SkillContext:
    """Small validated view over the dictionary observations used by BVR Sim."""

    time_s: float
    heading_rad: float = 0.0
    altitude_m: float = 8000.0
    speed_mps: float = 250.0
    target_bearing_rad: Optional[float] = None
    target_range_m: Optional[float] = None
    home_bearing_rad: Optional[float] = None
    threat_bearing_rad: Optional[float] = None
    missile_warning: bool = False

    @classmethod
    def from_observation(cls, observation: Mapping[str, Any]) -> "SkillContext":
        if not isinstance(observation, Mapping):
            raise TypeError("skill observations must be mappings, not serialized strings")
        status = observation.get("self_status", {}) or {}
        position = status.get("position", {}) or {}
        performance = status.get("performance", {}) or {}
        target = observation.get("target", {}) or {}
        threat = observation.get("threat", {}) or {}
        return cls(
            time_s=float(observation.get("time_s", observation.get("time", 0.0))),
            heading_rad=float(position.get("heading_rad", status.get("heading_rad", 0.0))),
            altitude_m=float(position.get("altitude_m", 8000.0)),
            speed_mps=float(performance.get("speed_mps", 250.0)),
            target_bearing_rad=_optional_float(
                observation.get("target_bearing_rad", target.get("bearing_rad"))
            ),
            target_range_m=_optional_float(
                observation.get("target_range_m", target.get("range_m"))
            ),
            home_bearing_rad=_optional_float(observation.get("home_bearing_rad")),
            threat_bearing_rad=_optional_float(
                observation.get("threat_bearing_rad", threat.get("bearing_rad"))
            ),
            missile_warning=bool(observation.get("missile_warning", threat.get("missile_warning", False))),
        )


def _optional_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class TacticalSkill:
    """Base class for persistent, interruptible tactical manoeuvres."""

    skill_name = "tactical_skill"
    default_params: Dict[str, Any] = {}
    parameter_schema: Dict[str, Dict[str, Any]] = {}

    def __init__(self, params: Optional[Mapping[str, Any]] = None):
        supplied = dict(params or {})
        self.params = {**self.default_params, **supplied}
        self._validate_parameters(supplied)
        self.started_at: Optional[float] = None
        self.interrupted = False
        self.interruption_reason: Optional[str] = None

    def execute(self, observation: Mapping[str, Any]) -> Tuple[Dict[str, Any], bool]:
        context = SkillContext.from_observation(observation)
        if self.started_at is None:
            self.started_at = context.time_s
        command, completed = self._execute(context)
        command.update(
            skill_name=self.skill_name,
            command_semantics=COMMAND_SEMANTICS,
            completed=bool(completed),
        )
        return command, bool(completed)

    def _execute(self, context: SkillContext) -> Tuple[Dict[str, Any], bool]:
        raise NotImplementedError

    def interrupt(self, reason: str) -> None:
        self.interrupted = True
        self.interruption_reason = reason

    @classmethod
    def get_schema(cls) -> Dict[str, Any]:
        return {"parameters": cls.parameter_schema, "command_semantics": COMMAND_SEMANTICS}

    def elapsed(self, context: SkillContext) -> float:
        return max(0.0, context.time_s - (self.started_at if self.started_at is not None else context.time_s))

    def _validate_parameters(self, supplied: Mapping[str, Any]) -> None:
        unknown = set(supplied) - set(self.default_params)
        if unknown:
            raise InvalidSkillParameters(
                "%s received unknown parameters: %s" % (self.skill_name, sorted(unknown))
            )
        for name, rule in self.parameter_schema.items():
            value = self.params.get(name)
            if "enum" in rule and value not in rule["enum"]:
                raise InvalidSkillParameters("%s must be one of %s" % (name, rule["enum"]))
            expected = rule.get("type")
            if expected == "number" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
                raise InvalidSkillParameters("%s must be numeric" % name)
            if expected == "boolean" and not isinstance(value, bool):
                raise InvalidSkillParameters("%s must be boolean" % name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if "min" in rule and value < rule["min"]:
                    raise InvalidSkillParameters("%s must be >= %s" % (name, rule["min"]))
                if "max" in rule and value > rule["max"]:
                    raise InvalidSkillParameters("%s must be <= %s" % (name, rule["max"]))

    @staticmethod
    def command(delta_heading: float = 0.0, delta_altitude: float = 0.0,
                delta_speed: float = 0.0, shoot: bool = False) -> Dict[str, Any]:
        return {
            "delta_heading": float(delta_heading),
            "delta_altitude": float(delta_altitude),
            "delta_speed": float(delta_speed),
            "shoot": 1.0 if shoot else 0.0,
        }


class TimedSkill(TacticalSkill):
    default_params = {"duration_s": 10.0}
    parameter_schema = {"duration_s": {"type": "number", "min": 0.1, "max": 600.0}}

    def is_complete(self, context: SkillContext) -> bool:
        return self.elapsed(context) >= float(self.params["duration_s"])


class MaintainPositionSkill(TimedSkill):
    """Maintain the present heading while converging on altitude and speed targets."""

    skill_name = "maintain_position"
    default_params = {"altitude_target": 8000.0, "speed_target": 250.0, "duration_s": 10.0}
    parameter_schema = {
        "altitude_target": {"type": "number", "min": 1000.0, "max": 15000.0},
        "speed_target": {"type": "number", "min": 100.0, "max": 500.0},
        "duration_s": TimedSkill.parameter_schema["duration_s"],
    }

    def _execute(self, context: SkillContext) -> Tuple[Dict[str, Any], bool]:
        return self.command(
            delta_altitude=(self.params["altitude_target"] - context.altitude_m) / 10.0,
            delta_speed=(self.params["speed_target"] - context.speed_mps) / 10.0,
        ), self.is_complete(context)


class TurnToHeadingSkill(TimedSkill):
    skill_name = "turn_to_heading"
    default_params = {"heading_rad": 0.0, "gain": 1.0, "tolerance_rad": 0.03, "duration_s": 30.0}
    parameter_schema = {
        "heading_rad": {"type": "number", "min": -math.pi, "max": math.pi},
        "gain": {"type": "number", "min": 0.01, "max": 10.0},
        "tolerance_rad": {"type": "number", "min": 0.001, "max": 1.0},
        "duration_s": TimedSkill.parameter_schema["duration_s"],
    }

    def _execute(self, context: SkillContext) -> Tuple[Dict[str, Any], bool]:
        error = _wrap_angle(self.params["heading_rad"] - context.heading_rad)
        done = abs(error) <= self.params["tolerance_rad"] or self.is_complete(context)
        return self.command(delta_heading=error * self.params["gain"]), done


class FlightPathSkill(TimedSkill):
    """Reusable primitive for climb/descend/accelerate/decelerate generation."""

    skill_name = "flight_path"
    default_params = {
        "altitude_rate": 0.0, "speed_rate": 0.0, "duration_s": 8.0,
    }
    parameter_schema = {
        "altitude_rate": {"type": "number", "min": -500.0, "max": 500.0},
        "speed_rate": {"type": "number", "min": -200.0, "max": 200.0},
        "duration_s": TimedSkill.parameter_schema["duration_s"],
    }

    def _execute(self, context: SkillContext) -> Tuple[Dict[str, Any], bool]:
        return self.command(
            delta_altitude=self.params["altitude_rate"], delta_speed=self.params["speed_rate"]
        ), self.is_complete(context)


class ClimbSkill(FlightPathSkill):
    skill_name = "climb"
    default_params = {**FlightPathSkill.default_params, "altitude_rate": 100.0}


class DescendSkill(FlightPathSkill):
    skill_name = "descend"
    default_params = {**FlightPathSkill.default_params, "altitude_rate": -100.0}


class AccelerateSkill(FlightPathSkill):
    skill_name = "accelerate"
    default_params = {**FlightPathSkill.default_params, "speed_rate": 25.0}


class DecelerateSkill(FlightPathSkill):
    skill_name = "decelerate"
    default_params = {**FlightPathSkill.default_params, "speed_rate": -25.0}


class TargetRelativeSkill(TimedSkill):
    default_params = {"offset_angle": 0.0, "heading_gain": 1.0, "duration_s": 15.0}
    parameter_schema = {
        "offset_angle": {"type": "number", "min": -180.0, "max": 180.0},
        "heading_gain": {"type": "number", "min": 0.01, "max": 10.0},
        "duration_s": TimedSkill.parameter_schema["duration_s"],
    }

    def desired_heading(self, context: SkillContext) -> float:
        if context.target_bearing_rad is None:
            raise SkillError("%s requires target_bearing_rad" % self.skill_name)
        return _wrap_angle(context.target_bearing_rad + math.radians(self.params["offset_angle"]))

    def _execute(self, context: SkillContext) -> Tuple[Dict[str, Any], bool]:
        error = _wrap_angle(self.desired_heading(context) - context.heading_rad)
        return self.command(delta_heading=error * self.params["heading_gain"]), self.is_complete(context)


class PursuitSkill(TargetRelativeSkill):
    skill_name = "pursuit"


class BeamSkill(TargetRelativeSkill):
    skill_name = "beam"
    default_params = {**TargetRelativeSkill.default_params, "offset_angle": 90.0}


class ExtendSkill(TargetRelativeSkill):
    skill_name = "extend"
    default_params = {**TargetRelativeSkill.default_params, "offset_angle": 180.0}


class RecommitSkill(TargetRelativeSkill):
    skill_name = "recommit"
    default_params = {**TargetRelativeSkill.default_params, "offset_angle": 0.0}


class CrankManeuverSkill(TargetRelativeSkill):
    skill_name = "crank_maneuver"
    default_params = {
        "direction": "left", "offset_angle": 30.0, "switch_frequency": 15.0,
        "altitude_change": -50.0, "speed_target": 300.0, "heading_gain": 1.0,
        "duration_s": 45.0,
    }
    parameter_schema = {
        "direction": {"enum": ("left", "right")},
        "offset_angle": {"type": "number", "min": 0.0, "max": 90.0},
        "switch_frequency": {"type": "number", "min": 0.1, "max": 300.0},
        "altitude_change": {"type": "number", "min": -500.0, "max": 500.0},
        "speed_target": {"type": "number", "min": 100.0, "max": 500.0},
        "heading_gain": TargetRelativeSkill.parameter_schema["heading_gain"],
        "duration_s": TimedSkill.parameter_schema["duration_s"],
    }

    def __init__(self, params: Optional[Mapping[str, Any]] = None):
        super().__init__(params)
        self.direction = self.params["direction"]
        self.last_switch_time: Optional[float] = None

    def _execute(self, context: SkillContext) -> Tuple[Dict[str, Any], bool]:
        if context.target_bearing_rad is None:
            raise SkillError("crank_maneuver requires target_bearing_rad")
        if self.last_switch_time is None:
            self.last_switch_time = context.time_s
        if context.time_s - self.last_switch_time >= self.params["switch_frequency"]:
            self.direction = "right" if self.direction == "left" else "left"
            self.last_switch_time = context.time_s
        sign = -1.0 if self.direction == "left" else 1.0
        desired = _wrap_angle(
            context.target_bearing_rad + sign * math.radians(self.params["offset_angle"])
        )
        heading_error = _wrap_angle(desired - context.heading_rad)
        return self.command(
            delta_heading=heading_error * self.params["heading_gain"],
            delta_altitude=self.params["altitude_change"],
            delta_speed=self.params["speed_target"] - context.speed_mps,
        ), self.is_complete(context)


class MissileEvasionSkill(TimedSkill):
    skill_name = "missile_evasion"
    default_params = {
        "break_direction": "right", "break_duration": 8.0, "max_g": 9.0,
        "dive_rate": -100.0, "speed_increase": 100.0,
    }
    parameter_schema = {
        "break_direction": {"enum": ("left", "right")},
        "break_duration": {"type": "number", "min": 0.1, "max": 60.0},
        "max_g": {"type": "number", "min": 1.0, "max": 12.0},
        "dive_rate": {"type": "number", "min": -500.0, "max": 500.0},
        "speed_increase": {"type": "number", "min": -200.0, "max": 200.0},
    }

    def _execute(self, context: SkillContext) -> Tuple[Dict[str, Any], bool]:
        done = self.elapsed(context) >= self.params["break_duration"]
        if done:
            return self.command(), True
        sign = 1.0 if self.params["break_direction"] == "right" else -1.0
        # Scale the normalized break command so max_g is no longer dead metadata.
        break_command = sign * min(1.0, self.params["max_g"] / 9.0)
        return self.command(
            delta_heading=break_command,
            delta_altitude=self.params["dive_rate"],
            delta_speed=self.params["speed_increase"],
        ), False


class DisengageSkill(TimedSkill):
    skill_name = "disengage"
    default_params = {
        "heading_home": 0.0, "climb_rate": 50.0, "speed_target": 350.0,
        "heading_gain": 1.0, "duration_s": 60.0,
    }
    parameter_schema = {
        "heading_home": {"type": "number", "min": -180.0, "max": 360.0},
        "climb_rate": {"type": "number", "min": -500.0, "max": 500.0},
        "speed_target": {"type": "number", "min": 100.0, "max": 500.0},
        "heading_gain": {"type": "number", "min": 0.01, "max": 10.0},
        "duration_s": TimedSkill.parameter_schema["duration_s"],
    }

    def _execute(self, context: SkillContext) -> Tuple[Dict[str, Any], bool]:
        desired = context.home_bearing_rad
        if desired is None:
            desired = math.radians(self.params["heading_home"])
        return self.command(
            delta_heading=_wrap_angle(desired - context.heading_rad) * self.params["heading_gain"],
            delta_altitude=self.params["climb_rate"],
            delta_speed=self.params["speed_target"] - context.speed_mps,
        ), self.is_complete(context)


DEFAULT_SKILLS: Tuple[Type[TacticalSkill], ...] = (
    CrankManeuverSkill, MissileEvasionSkill, DisengageSkill, MaintainPositionSkill,
    TurnToHeadingSkill, ClimbSkill, DescendSkill, AccelerateSkill, DecelerateSkill,
    PursuitSkill, BeamSkill, ExtendSkill, RecommitSkill,
)


class SkillManager:
    """Registry/factory with explicit fallback and parameter-validation events."""

    def __init__(self, strict: bool = True, fallback_skill: str = "maintain_position"):
        self.skills: Dict[str, Type[TacticalSkill]] = {}
        self.strict = strict
        self.fallback_skill = fallback_skill
        self.last_creation_event: Optional[SkillCreationEvent] = None
        for skill_class in DEFAULT_SKILLS:
            self.register(skill_class)

    def register(self, skill_class: Type[TacticalSkill]) -> None:
        name = skill_class.skill_name
        if not name or name == TacticalSkill.skill_name:
            raise SkillError("registered skills must define a unique skill_name")
        if name in self.skills:
            raise SkillError("skill already registered: %s" % name)
        self.skills[name] = skill_class

    def create_skill(self, skill_name: str,
                     params: Optional[Mapping[str, Any]] = None) -> TacticalSkill:
        skill_class = self.skills.get(skill_name)
        if skill_class is None:
            if self.strict:
                raise UnknownSkillError("unknown skill: %s" % skill_name)
            reason = "unknown skill: %s" % skill_name
            skill_class = self.skills[self.fallback_skill]
            skill = skill_class({})
            self.last_creation_event = SkillCreationEvent(
                skill_name, skill.skill_name, True, reason
            )
            return skill
        skill = skill_class(params)
        self.last_creation_event = SkillCreationEvent(skill_name, skill.skill_name)
        return skill

    def list_skills(self) -> Tuple[str, ...]:
        return tuple(self.skills)

    def schemas(self) -> Dict[str, Dict[str, Any]]:
        return {name: skill.get_schema() for name, skill in self.skills.items()}
