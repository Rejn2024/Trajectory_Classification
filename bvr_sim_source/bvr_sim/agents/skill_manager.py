"""Versioned tactical-skill catalogue used by agents and planners.

The catalogue deliberately describes *what* a skill promises rather than tying the
planner to a particular flight-control implementation.  Every skill emits the same
small guidance command; an aircraft-specific controller can translate that command
to its native action space.
"""

from copy import deepcopy
from dataclasses import dataclass
import math


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


def _number(description, default, minimum=None, maximum=None):
    result = {"type": "number", "description": description, "default": default}
    if minimum is not None:
        result["minimum"] = minimum
    if maximum is not None:
        result["maximum"] = maximum
    return result


def _string(description, default, enum=None):
    result = {"type": "string", "description": description, "default": default}
    if enum:
        result["enum"] = enum
    return result


@dataclass(frozen=True)
class SkillSpec:
    """Immutable, serialisable lifecycle contract for one tactical skill."""

    name: str
    category: str
    start_condition: str
    termination_condition: str
    interruption_conditions: tuple
    parameters: dict
    version: str = "1.0.0"

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
    "The selected target becomes invalid or unavailable when the skill requires one.",
    "The planner explicitly cancels or replaces the skill.",
)

HEADING = {"heading_deg": _number("Commanded true heading in degrees.", 0, 0, 360),
           "tolerance_deg": _number("Completion tolerance in degrees.", 3, 0.1, 30)}
ALTITUDE = {"altitude_m": _number("Commanded altitude above mean sea level.", 8000, 0, 20000),
            "tolerance_m": _number("Completion tolerance in metres.", 50, 1, 1000)}
SPEED = {"speed_mps": _number("Commanded true airspeed.", 250, 30, 800),
         "tolerance_mps": _number("Completion tolerance in m/s.", 5, 0.1, 100)}
TARGET = {"target_id": _string("Track identifier of the target aircraft.", "primary")}
SIDE = {"side": _string("Side on which to place the threat.", "left", ["left", "right"])}
TIMEOUT = {"timeout_s": _number("Maximum execution time before termination.", 60, 1, 600)}


def _spec(name, category, start, stop, parameters=None, interruptions=COMMON_INTERRUPTS):
    return SkillSpec(name, category, start, stop, interruptions, parameters or {})


def _catalogue():
    specs = []
    add = specs.append
    # Basic kinematic primitives
    add(_spec("maintain_heading", "kinematic", "Aircraft is controllable and a heading is available.",
              "Cancelled, interrupted, or timeout_s expires.", {**HEADING, **TIMEOUT}))
    add(_spec("turn_to_heading", "kinematic", "Aircraft is controllable and differs from heading_deg by more than tolerance_deg.",
              "Heading error is at most tolerance_deg or timeout_s expires.", {**HEADING, **TIMEOUT}))
    add(_spec("climb_to_altitude", "kinematic", "Current altitude is below altitude_m minus tolerance_m.",
              "Altitude is within tolerance_m of altitude_m or timeout_s expires.", {**ALTITUDE, **TIMEOUT}))
    add(_spec("descend_to_altitude", "kinematic", "Current altitude is above altitude_m plus tolerance_m.",
              "Altitude is within tolerance_m of altitude_m or timeout_s expires.", {**ALTITUDE, **TIMEOUT}))
    add(_spec("accelerate_to_speed", "kinematic", "Current speed is below speed_mps minus tolerance_mps.",
              "Speed is within tolerance_mps of speed_mps or timeout_s expires.", {**SPEED, **TIMEOUT}))
    add(_spec("decelerate_to_speed", "kinematic", "Current speed is above speed_mps plus tolerance_mps.",
              "Speed is within tolerance_mps of speed_mps or timeout_s expires.", {**SPEED, **TIMEOUT}))

    pursuit = {**TARGET, "offset_deg": _number("Angular offset from pursuit geometry.", 0, -90, 90), **TIMEOUT}
    relational = {
        "pursue_target": ("A valid target track exists.", "Target is inside capture_angle_deg or timeout_s expires."),
        "lead_pursuit": ("A valid maneuvering target track with velocity exists.", "Lead geometry is established or timeout_s expires."),
        "lag_pursuit": ("A valid target track exists and closure should be reduced.", "Lag geometry is established or timeout_s expires."),
        "beam_target_left": ("A valid target track exists.", "Target aspect is 90 degrees off the left wing or timeout_s expires."),
        "beam_target_right": ("A valid target track exists.", "Target aspect is 90 degrees off the right wing or timeout_s expires."),
        "crank_target_left": ("A valid locked target exists and can remain supported.", "The configured left crank angle is established or timeout_s expires."),
        "crank_target_right": ("A valid locked target exists and can remain supported.", "The configured right crank angle is established or timeout_s expires."),
        "turn_cold": ("Disengagement is commanded and a threat axis is known.", "Heading is away from the threat within tolerance_deg or timeout_s expires."),
        "extend": ("Separation from a threat or target is required.", "range_m is reached or timeout_s expires."),
        "recommit": ("Extension is complete, fuel permits, and a valid target track exists.", "The target is inside commit_range_m or timeout_s expires."),
    }
    for name, (start, stop) in relational.items():
        params = dict(pursuit)
        params["capture_angle_deg"] = _number("Desired angular-geometry tolerance.", 5, 1, 45)
        params["range_m"] = _number("Desired separation range.", 30000, 1000, 200000)
        params["commit_range_m"] = _number("Range at which recommit completes.", 50000, 1000, 200000)
        add(_spec(name, "relational", start, stop, params))

    weapon = {
        "search": ("Sensors are operational and no suitable track is selected.", "A track meeting detection criteria is found or timeout_s expires."),
        "lock_target": ("A valid detectable target track exists.", "A fire-control lock is confirmed or the track is lost."),
        "commit": ("A valid target meets the configured commit criteria.", "The target enters launch_range_m, criteria fail, or timeout_s expires."),
        "launch": ("A valid locked target is inside launch constraints and a weapon is available.", "One launch command is emitted or launch authorization is withdrawn."),
        "support_missile": ("A supported friendly missile is in flight.", "The missile becomes autonomous, impacts, misses, or support_timeout_s expires."),
        "abort_support": ("Missile support is active and defensive risk exceeds the abort threshold.", "Support is dropped and defensive geometry begins."),
        "secondary_shot": ("A target remains valid after a first shot and another weapon is available.", "One follow-up launch command is emitted or shot criteria fail."),
        "short_range_attack": ("A valid target is within short_range_m and a short-range weapon is available.", "A shot is launched, the target leaves constraints, or timeout_s expires."),
    }
    weapon_params = {**TARGET, "launch_range_m": _number("Maximum authorized launch range.", 40000, 100, 200000),
                     "short_range_m": _number("Maximum short-range attack distance.", 5000, 100, 20000),
                     "support_timeout_s": _number("Maximum missile-support duration.", 45, 1, 180), **TIMEOUT}
    for name, (start, stop) in weapon.items():
        add(_spec(name, "weapon_employment", start, stop, weapon_params))

    defensive = {
        "preemptive_crank": "A credible threat can engage but no missile warning is active.",
        "notch_left": "A radar-guided threat is active and left-side notch geometry is feasible.",
        "notch_right": "A radar-guided threat is active and right-side notch geometry is feasible.",
        "beam_missile_left": "An inbound missile is tracked and left-side beam geometry is feasible.",
        "beam_missile_right": "An inbound missile is tracked and right-side beam geometry is feasible.",
        "drag_missile": "An inbound missile is tracked and separation can defeat it kinematically.",
        "dive_defense": "An inbound missile is tracked and safe altitude is available below.",
        "last_ditch_break": "Missile time-to-impact is below the configured critical threshold.",
        "defensive_reversal": "A defensive maneuver has forced an overshoot or reversal criteria are met.",
    }
    defense_params = {"threat_id": _string("Track identifier of the threat or missile.", "highest_priority"),
                      "max_g": _number("Maximum commanded load factor.", 9, 2, 12),
                      "min_altitude_m": _number("Hard altitude floor for the maneuver.", 500, 0, 10000), **TIMEOUT}
    for name, start in defensive.items():
        add(_spec(name, "defensive", start,
                  "Threat is defeated or no longer tracked, safety limits are reached, or timeout_s expires.", defense_params))
    return {spec.name: spec for spec in specs}


class TacticalSkill:
    """Executable instance of a :class:`SkillSpec`."""

    def __init__(self, spec, params=None):
        self.spec = spec
        self.skill_name = spec.name
        self.params = {name: definition.get("default") for name, definition in spec.parameters.items()}
        self.params.update(params or {})
        self._validate_params()
        self.start_time = None

    @property
    def version(self):
        return self.spec.version

    def _validate_params(self):
        unknown = set(self.params) - set(self.spec.parameters)
        if unknown:
            raise ValueError(f"Unknown parameters for {self.skill_name}: {sorted(unknown)}")
        for name, value in self.params.items():
            schema = self.spec.parameters[name]
            if schema["type"] == "number" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
                raise TypeError(f"{name} must be a number")
            if "enum" in schema and value not in schema["enum"]:
                raise ValueError(f"{name} must be one of {schema['enum']}")
            if "minimum" in schema and value < schema["minimum"]:
                raise ValueError(f"{name} must be >= {schema['minimum']}")
            if "maximum" in schema and value > schema["maximum"]:
                raise ValueError(f"{name} must be <= {schema['maximum']}")

    def get_schema(self):
        """Return the complete lifecycle contract (keeps the historical method name)."""
        return self.spec.contract()

    def execute(self, obs):
        """Emit a stable guidance action and completion flag."""
        now = obs.get("time", 0)
        if self.start_time is None:
            self.start_time = now
        position = obs.get("self_status", {}).get("position", {})
        performance = obs.get("self_status", {}).get("performance", {})
        heading = position.get("heading_deg", performance.get("heading_deg", 0))
        altitude = position.get("altitude_m", 0)
        speed = performance.get("speed_mps", 0)
        delta_heading = 0.0
        delta_altitude = 0.0
        delta_speed = 0.0
        completed = now - self.start_time >= self.params.get("timeout_s", math.inf)

        if "heading_deg" in self.params:
            error = (self.params["heading_deg"] - heading + 180) % 360 - 180
            delta_heading = math.radians(error)
            completed |= abs(error) <= self.params.get("tolerance_deg", 0)
        if "altitude_m" in self.params:
            error = self.params["altitude_m"] - altitude
            delta_altitude = error
            completed |= abs(error) <= self.params.get("tolerance_m", 0)
        if "speed_mps" in self.params:
            error = self.params["speed_mps"] - speed
            delta_speed = error
            completed |= abs(error) <= self.params.get("tolerance_mps", 0)

        left = self.skill_name.endswith("_left")
        right = self.skill_name.endswith("_right")
        if left or right:
            delta_heading = math.radians(-90 if left else 90)
        if self.skill_name in {"launch", "secondary_shot", "short_range_attack"}:
            completed = True
        action = {"skill_name": self.skill_name, "delta_heading": delta_heading,
                  "delta_altitude": delta_altitude, "delta_speed": delta_speed,
                  "shoot": int(self.skill_name in {"launch", "secondary_shot", "short_range_attack"}),
                  "completed": bool(completed)}
        return action, bool(completed)


class SkillManager:
    """Registry and factory for the complete, versioned tactical vocabulary."""

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
