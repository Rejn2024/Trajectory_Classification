"""Versioned catalogue of physically distinct simulator guidance commands.

Only commands that change heading, altitude, speed, or missile launch requests are
exposed to agents. Names and parameters describe the controls actually executed by
the simulator rather than unimplemented tactical intentions.
"""

import math
from copy import deepcopy
from dataclasses import dataclass
from numbers import Real

CATALOGUE_VERSION = "4.0.0"

ACTION_SCHEMA = {
    "type": "object",
    "required": ["skill_name", "delta_heading", "delta_altitude", "delta_speed", "shoot", "completed"],
    "properties": {
        "skill_name": {"type": "string", "enum": ["change_trajectory", "launch"]},
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
    # Altitude/speed limits match CampusActionSpace without importing its optional
    # simulator dependencies into this lightweight catalogue.
    controllable = "Aircraft is controllable."
    specs = [
        SkillSpec(
            "change_trajectory", "kinematic", controllable,
            "All relative commands are zero, or the planner replaces them.", COMMON_INTERRUPTS,
            {
                "delta_heading_deg": _number("Signed relative heading change in degrees.", 0, -180, 180),
                "delta_altitude_m": _number("Signed relative altitude change in metres.", 0, -80, 80),
                "delta_speed_mps": _number("Signed relative airspeed change in m/s.", 0, -80, 80),
            },
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
        """Emit the selected relative command at each simulator substep.

        Parameters are already deltas; never subtract the current state or retain
        an absolute target between calls. The planner holds or replaces the delta
        on its existing decision cadence, and the simulator applies control limits.
        """
        del obs
        delta_heading = 0.0
        delta_altitude = 0.0
        delta_speed = 0.0
        completed = False

        if self.skill_name == "change_trajectory":
            delta_heading = math.radians(self.params["delta_heading_deg"])
            delta_altitude = self.params["delta_altitude_m"]
            delta_speed = self.params["delta_speed_mps"]
            completed = delta_heading == delta_altitude == delta_speed == 0
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
