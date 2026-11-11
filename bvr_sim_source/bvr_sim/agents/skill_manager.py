"""Versioned catalogue of physically distinct simulator guidance commands.

Only commands that change heading, altitude, speed, or missile launch requests are
exposed to agents. Names and parameters describe the controls actually executed by
the simulator rather than unimplemented tactical intentions.
"""

import math
from copy import deepcopy
from dataclasses import dataclass
from numbers import Real

CATALOGUE_VERSION = "2.0.0"

ACTION_SCHEMA = {
    "type": "object",
    "required": ["skill_name", "delta_heading", "delta_altitude", "delta_speed", "shoot", "completed"],
    "properties": {
        "skill_name": {"type": "string"},
        "delta_heading": {"type": "number", "description": "Requested heading change in radians."},
        "delta_altitude": {"type": "number", "description": "Requested altitude change in metres."},
        "delta_speed": {"type": "number", "description": "Requested true-airspeed change in m/s."},
        "shoot": {"type": "integer", "enum": [0, 1]},
        "completed": {"type": "boolean"},
    },
    "additionalProperties": False,
}


def _number(description, default, minimum, maximum):
    return {
        "type": "number", "description": description, "default": default,
        "minimum": minimum, "maximum": maximum,
    }


@dataclass(frozen=True)
class SkillSpec:
    """Immutable, serialisable lifecycle contract for one simulator command."""

    name: str
    category: str
    start_condition: str
    termination_condition: str
    interruption_conditions: tuple
    parameters: dict
    version: str = CATALOGUE_VERSION

    def contract(self):
        return {
            "name": self.name,
            "category": self.category,
            "version": self.version,
            "start_condition": self.start_condition,
            "output_contract": deepcopy(ACTION_SCHEMA),
            "termination_condition": self.termination_condition,
            "interruption_conditions": list(self.interruption_conditions),
            "parameter_schema": {
                "type": "object",
                "properties": deepcopy(self.parameters),
                "additionalProperties": False,
            },
        }


COMMON_INTERRUPTS = (
    "A higher-priority safety or collision-avoidance command is issued.",
    "The planner explicitly cancels or replaces the skill.",
)


def _catalogue():
    controllable = "Aircraft is controllable."
    specs = [
        SkillSpec(
            "turn_to_heading", "kinematic", controllable,
            "The commanded heading is reached, or the command is replaced.", COMMON_INTERRUPTS,
            {"heading_deg": _number("Commanded true heading in degrees.", 0, 0, 360)},
        ),
        SkillSpec(
            "set_altitude", "kinematic", controllable,
            "The commanded altitude is reached, or the command is replaced.", COMMON_INTERRUPTS,
            {"altitude_m": _number("Commanded altitude above mean sea level.", 8000, 0, 20000)},
        ),
        SkillSpec(
            "set_speed", "kinematic", controllable,
            "The commanded speed is reached, or the command is replaced.", COMMON_INTERRUPTS,
            {"speed_mps": _number("Commanded true airspeed.", 250, 30, 800)},
        ),
        SkillSpec(
            "launch", "weapon_employment", "The simulator can accept a launch request.",
            "One launch request is emitted; ammunition and launch timing are enforced by the simulator.",
            COMMON_INTERRUPTS, {},
        ),
    ]
    return {spec.name: spec for spec in specs}


class TacticalSkill:
    """Executable instance of a :class:`SkillSpec`."""

    def __init__(self, spec, params=None):
        self.spec = spec
        self.skill_name = spec.name
        self.params = {name: definition["default"] for name, definition in spec.parameters.items()}
        self.params.update(params or {})
        self._validate_params()

    @property
    def version(self):
        return self.spec.version

    def _validate_params(self):
        unknown = set(self.params) - set(self.spec.parameters)
        if unknown:
            raise ValueError(f"Unknown parameters for {self.skill_name}: {sorted(unknown)}")
        for name, value in self.params.items():
            schema = self.spec.parameters[name]
            if not isinstance(value, Real) or isinstance(value, bool):
                raise TypeError(f"{name} must be a number")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < schema["minimum"]:
                raise ValueError(f"{name} must be >= {schema['minimum']}")
            if value > schema["maximum"]:
                raise ValueError(f"{name} must be <= {schema['maximum']}")

    def get_schema(self):
        """Return the complete lifecycle contract (keeps the historical method name)."""
        return self.spec.contract()

    def execute(self, obs):
        """Emit guidance containing only the selected skill's physical control."""
        position = obs.get("self_status", {}).get("position", {})
        performance = obs.get("self_status", {}).get("performance", {})
        heading = position.get("heading_deg", performance.get("heading_deg", 0))
        altitude = position.get("altitude_m", 0)
        speed = performance.get("speed_mps", 0)
        delta_heading = 0.0
        delta_altitude = 0.0
        delta_speed = 0.0
        completed = False

        if self.skill_name == "turn_to_heading":
            error = (self.params["heading_deg"] - heading + 180) % 360 - 180
            delta_heading = math.radians(error)
            completed = error == 0
        elif self.skill_name == "set_altitude":
            delta_altitude = self.params["altitude_m"] - altitude
            completed = delta_altitude == 0
        elif self.skill_name == "set_speed":
            delta_speed = self.params["speed_mps"] - speed
            completed = delta_speed == 0
        elif self.skill_name == "launch":
            completed = True
        action = {
            "skill_name": self.skill_name, "delta_heading": delta_heading,
            "delta_altitude": delta_altitude, "delta_speed": delta_speed,
            "shoot": int(self.skill_name == "launch"), "completed": bool(completed),
        }
        return action, bool(completed)


class SkillManager:
    """Registry and factory for the versioned physical-control vocabulary."""

    CATALOGUE_VERSION = CATALOGUE_VERSION

    def __init__(self):
        self.skills = _catalogue()

    def create_skill(self, skill_name, params=None):
        if skill_name not in self.skills:
            raise KeyError(f"Unknown skill: {skill_name}")
        return TacticalSkill(self.skills[skill_name], params)

    def list_skills(self, category=None):
        """List stable skill names, optionally restricted to a category."""
        return [name for name, spec in self.skills.items() if category is None or spec.category == category]

    def get_contract(self, skill_name):
        """Return a defensive copy of a skill's complete public contract."""
        if skill_name not in self.skills:
            raise KeyError(f"Unknown skill: {skill_name}")
        return self.skills[skill_name].contract()

    def describe_skills(self, category=None):
        """Return contracts suitable for planner prompts or API discovery."""
        return [self.get_contract(name) for name in self.list_skills(category)]

    def get_catalogue(self):
        """Return version and ordered contracts for checkpoint compatibility checks."""
        return {"version": self.CATALOGUE_VERSION, "skills": self.describe_skills()}
