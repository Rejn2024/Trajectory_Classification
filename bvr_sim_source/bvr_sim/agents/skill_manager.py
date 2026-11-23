"""Versioned tactical-skill catalogue used by agents and planners.

The catalogue deliberately describes *what* a skill promises rather than tying the
planner to a particular flight-control implementation.  Every skill emits the same
small guidance command; an aircraft-specific controller can translate that command
to its native action space.
"""

from copy import deepcopy
from dataclasses import dataclass
import math
from collections.abc import Mapping


ACTION_SCHEMA = {
    "type": "object",
    "required": ["skill_name", "delta_heading", "delta_altitude", "delta_speed", "shoot", "completed"],
    "properties": {
        "skill_name": {"type": "string"},
        "delta_heading": {"type": "number", "description": "Signed heading error from the current aircraft heading to the heading_deg setpoint, in radians."},
        "delta_altitude": {"type": "number", "description": "Altitude error from the current aircraft altitude to the altitude_m setpoint, in metres."},
        "delta_speed": {"type": "number", "description": "True-airspeed error from the current aircraft speed to the speed_mps setpoint, in m/s."},
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
TRAJECTORY = {**HEADING, **ALTITUDE, **SPEED, **TIMEOUT}

# The skill layer emits physical deltas.  Aircraft-specific adapters may
# normalise these values before handing them to an FDM.  These limits are used
# only to keep generated manoeuvre commands bounded and physically plausible.
GRAVITY_MPS2 = 9.80665
MAX_TURN_RATE_RAD_S = math.radians(9.0)
MAX_CLIMB_RATE_MPS = 250.0
MAX_ACCELERATION_MPS2 = 5.0


def _value(container, key, default=None):
    """Read a field from either a mapping or a simulator-style object."""
    if isinstance(container, Mapping):
        return container.get(key, default)
    return getattr(container, key, default)


def _first(container, *keys, default=None):
    for key in keys:
        result = _value(container, key)
        if result is not None:
            return result
    return default


def _vector(value):
    """Return a 3-vector from common dict, object, or sequence formats."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        candidates = (
            ("north", "west", "up"), ("x", "y", "z"),
            ("vx", "vy", "vz"), ("north_m", "west_m", "up_m"),
        )
        for names in candidates:
            if all(_value(value, name) is not None for name in names):
                return tuple(float(_value(value, name)) for name in names)
        return None
    try:
        if len(value) >= 3:
            return tuple(float(value[i]) for i in range(3))
    except (TypeError, ValueError, IndexError):
        pass
    return None


def _track_id(track):
    return _first(track, "uid", "id", "track_id", "name")


def _track_heading(track):
    """Return a tracked aircraft/missile heading in radians when available."""
    heading_deg = _first(track, "heading_deg", "course_deg", "track_heading_deg")
    if heading_deg is not None:
        return math.radians(float(heading_deg))

    heading_rad = _first(track, "heading_rad", "course_rad", "track_heading_rad")
    if heading_rad is not None:
        return float(heading_rad)

    # Guidance observations commonly expose velocity rather than orientation.
    # In the NWU frame, north is x and west is y, matching _bearing().
    velocity = _track_velocity(track)
    if velocity is not None and math.hypot(velocity[0], velocity[1]) > 1e-9:
        return math.atan2(velocity[1], velocity[0])
    return None


def _track_candidates(obs, keys):
    """Yield tracks from mapping/sequence observation fields."""
    for key in keys:
        collection = _value(obs, key)
        if collection is None:
            continue
        if isinstance(collection, Mapping):
            candidates = collection.values()
        else:
            try:
                candidates = iter(collection)
            except TypeError:
                continue
        yield from candidates


def _find_nearest_incoming_missile(obs, own, identifier="highest_priority"):
    """Select the nearest live missile track, rather than an aircraft threat."""
    candidates = []
    direct = _value(obs, "missile")
    if direct is not None:
        candidates.append(direct)
    candidates.extend(_track_candidates(
        obs, (
            "incoming_missiles", "incoming_missile_tracks", "missiles",
            "enemy_missiles", "missile_tracks", "active_missiles",
        )
    ))

    filtered = []
    for candidate in candidates:
        if _first(candidate, "is_alive", "alive", default=True) is False:
            continue
        if identifier not in (None, "primary", "highest_priority") and _track_id(candidate) != identifier:
            continue
        filtered.append(candidate)

    if not filtered:
        return None
    return min(
        enumerate(filtered),
        key=lambda item: (
            _range(item[1], own) is None,
            _range(item[1], own) if _range(item[1], own) is not None else math.inf,
            item[0],
        ),
    )[1]


def _find_track(obs, identifier, threat=False):
    """Find a target/threat without requiring one observation representation."""
    direct_keys = ("threat", "missile", "active_threat") if threat else (
        "target", "primary_target")
    collection_keys = (
        ("threats", "missiles", "enemy_missiles", "missile_tracks", "active_missiles")
        if threat else ("targets", "target_tracks", "enemies", "enemy_tracks", "tracks")
    )

    for key in direct_keys:
        candidate = _value(obs, key)
        if candidate is not None:
            if identifier in (None, "primary", "highest_priority"):
                return candidate
            if _track_id(candidate) == identifier:
                return candidate

    for key in collection_keys:
        collection = _value(obs, key)
        if collection is None:
            continue
        if isinstance(collection, Mapping):
            if identifier in collection:
                return collection[identifier]
            values = collection.values()
        else:
            try:
                values = iter(collection)
            except TypeError:
                continue
        first_candidate = None
        for candidate in values:
            if first_candidate is None:
                first_candidate = candidate
            if identifier in (None, "primary", "highest_priority") or _track_id(candidate) == identifier:
                return candidate
        if first_candidate is not None and identifier in (None, "primary", "highest_priority"):
            return first_candidate
    return None


def _find_active_support_missile(obs, identifier="primary"):
    """Return the selected live friendly missile from a guidance observation."""
    for key in ("support_missiles", "friendly_missiles", "launched_missiles"):
        collection = _value(obs, key)
        if collection is None:
            continue
        if isinstance(collection, Mapping):
            if identifier not in (None, "primary") and identifier in collection:
                candidates = (collection[identifier],)
            else:
                candidates = collection.values()
        else:
            try:
                candidates = iter(collection)
            except TypeError:
                continue
        first_live = None
        for candidate in candidates:
            if _first(candidate, "is_alive", "alive", default=True) is False:
                continue
            if first_live is None:
                first_live = candidate
            if identifier in (None, "primary") or _track_id(candidate) == identifier:
                return candidate
        if first_live is not None and identifier in (None, "primary"):
            return first_live
    return None


def _own_state(obs):
    status = _value(obs, "self_status", {}) or {}
    position = _value(status, "position", {}) or {}
    performance = _value(status, "performance", {}) or {}
    position_vector = _vector(_first(position, "position", "coordinates", default=position))
    velocity = _vector(_first(performance, "velocity", "linear_velocity"))
    if velocity is None:
        velocity = _vector(_first(status, "velocity", "linear_velocity"))
    heading_deg = _first(position, "heading_deg", default=None)
    if heading_deg is None:
        heading_deg = _first(performance, "heading_deg", default=None)
    if heading_deg is None and velocity is not None:
        heading_deg = math.degrees(math.atan2(velocity[1], velocity[0]))
    altitude = _first(position, "altitude_m", "height_m", default=None)
    if altitude is None and position_vector is not None:
        altitude = position_vector[2]
    speed = _first(performance, "speed_mps", "true_airspeed_mps", default=None)
    if speed is None and velocity is not None:
        speed = math.sqrt(sum(component * component for component in velocity))
    return {
        "position": position_vector or (0.0, 0.0, float(altitude or 0.0)),
        "velocity": velocity or (0.0, 0.0, 0.0),
        "heading": math.radians(float(heading_deg or 0.0)),
        "altitude": float(altitude or 0.0),
        "speed": float(speed or 0.0),
    }


def _track_position(track):
    return _vector(_first(track, "position", "coordinates", "location"))


def _track_velocity(track):
    return _vector(_first(track, "velocity", "linear_velocity"))


def _bearing(track, own):
    position = _track_position(track)
    if position is not None:
        return math.atan2(position[1] - own["position"][1], position[0] - own["position"][0])
    relative = _first(track, "relative_bearing_deg", default=None)
    if relative is not None:
        return own["heading"] + math.radians(float(relative))
    absolute = _first(track, "bearing_deg", default=None)
    if absolute is not None:
        return math.radians(float(absolute))
    return None


def _range(track, own):
    position = _track_position(track)
    if position is not None:
        return math.sqrt(sum((position[i] - own["position"][i]) ** 2 for i in range(3)))
    result = _first(track, "range_m", "distance_m", default=None)
    return None if result is None else float(result)


def _wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _clip(value, lower, upper):
    return max(lower, min(upper, value))


def _spec(name, category, start, stop, parameters=None, interruptions=COMMON_INTERRUPTS):
    return SkillSpec(name, category, start, stop, interruptions, parameters or {})


def _catalogue():
    specs = []
    add = specs.append
    # One kinematic command controls all three trajectory setpoints.  The
    # executor emits the error from the current aircraft state to each target.
    add(_spec("set_trajectory", "kinematic",
              "Aircraft is controllable and trajectory setpoints are available.",
              "Heading, altitude, and speed are within their tolerances or timeout_s expires.",
              TRAJECTORY))

    pursuit = {
        **TARGET,
        "offset_deg": _number("Angular offset from pursuit geometry.", 0, -90, 90),
        "lead_time_s": _number("Prediction time used by lead pursuit.", 2, 0, 10),
        "lag_angle_deg": _number("Trailing angle used by lag pursuit.", 20, 0, 90),
        "crank_angle_deg": _number("Target offset used by crank pursuit.", 45, 0, 90),
        **TIMEOUT,
    }
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
        "launch": ("A valid locked target is inside launch constraints and a weapon is available.", "One launch command is emitted or launch authorization is withdrawn."),
        "support_missile": ("A supported friendly missile is in flight.", "The missile becomes autonomous, impacts, misses, or support_timeout_s expires."),
    }
    weapon_params = {**TARGET, "launch_range_m": _number("Maximum authorized launch range.", 40000, 100, 200000), **TIMEOUT}
    search_params = {
        **TARGET,
        "sector_width_deg": _number("Total heading width of the radar search sector.", 60, 1, 180),
        "sweep_period_s": _number("Time for one complete left-to-right-to-left sweep.", 20, 2, 300),
        "pattern": _string("Heading sweep profile.", "sinusoidal", ["sinusoidal", "triangle"]),
        "direction": _string("Initial sweep direction.", "right", ["left", "right"]),
        **TIMEOUT,
    }
    support_params = {
        **TARGET,
        "missile_id": _string("Friendly missile to support.", "primary"),
        "support_side": _string("Side on which to place the target in the radar cone.", "left", ["left", "right"]),
        "radar_half_fov_deg": _number("Radar half-cone angle in degrees.", 60, 5, 89),
        "radar_margin_deg": _number("Safety margin inside the radar cone.", 5, 1, 30),
        "max_g": _number("Maximum commanded support turn load factor.", 9, 2, 12),
        "support_timeout_s": _number("Maximum missile-support duration.", 45, 1, 180),
        **TIMEOUT,
    }
    for name, (start, stop) in weapon.items():
        add(_spec(name, "weapon_employment", start, stop,
                  search_params if name == "search" else
                  support_params if name == "support_missile" else weapon_params))

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
    defense_params = {
        "threat_id": _string("Track identifier of the threat or missile.", "highest_priority"),
        **SIDE,
        "crank_angle_deg": _number("Threat offset used by a pre-emptive crank.", 45, 0, 90),
        "max_g": _number("Maximum commanded load factor.", 9, 2, 12),
        "min_altitude_m": _number("Hard altitude floor for the maneuver.", 500, 0, 10000),
        **TIMEOUT,
    }
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
        self._search_origin_heading = None

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

    def _bounded_heading_delta(self, desired, own, max_g=None):
        delta = _wrap(desired - own["heading"])
        if max_g is None:
            return delta
        max_delta = min(
            max_g * GRAVITY_MPS2 / max(own["speed"], 1.0),
            MAX_TURN_RATE_RAD_S,
            math.pi / 2,
        )
        return _clip(delta, -max_delta, max_delta)

    def _target_heading(self, obs, own, track, predicted=False):
        """Return a target bearing, optionally using constant-velocity lead."""
        if not predicted:
            return _bearing(track, own)
        position = _track_position(track)
        velocity = _track_velocity(track)
        if position is None or velocity is None:
            return _bearing(track, own)
        lead_time = self.params.get("lead_time_s", 0.0)
        predicted_track = {"position": tuple(
            position[i] + lead_time * velocity[i] for i in range(3)
        )}
        return _bearing(predicted_track, own)

    def _execute_relational(self, obs, own, action):
        name = self.skill_name
        track = _find_track(obs, self.params.get("target_id", "primary"))
        if track is None:
            # Preserve the old safe fallback for labels whose name specifies a
            # side, while all other relational skills hold their current path.
            if name.endswith("_left"):
                action["delta_heading"] = -math.pi / 2
            elif name.endswith("_right"):
                action["delta_heading"] = math.pi / 2
            return

        bearing = self._target_heading(obs, own, track, name == "lead_pursuit")
        if bearing is None:
            return

        offset = math.radians(self.params.get("offset_deg", 0.0))
        if name == "pursue_target":
            desired = bearing + offset
        elif name == "lead_pursuit":
            desired = bearing + offset
        elif name == "lag_pursuit":
            desired = bearing - math.radians(self.params.get("lag_angle_deg", 20.0)) + offset
        elif name == "beam_target_left":
            desired = bearing - math.pi / 2
        elif name == "beam_target_right":
            desired = bearing + math.pi / 2
        elif name == "crank_target_left":
            desired = bearing - math.radians(self.params.get("crank_angle_deg", 45.0)) + offset
        elif name == "crank_target_right":
            desired = bearing + math.radians(self.params.get("crank_angle_deg", 45.0)) + offset
        elif name in {"turn_cold", "extend"}:
            desired = bearing + math.pi
        elif name == "recommit":
            desired = bearing
        else:
            return

        action["delta_heading"] = self._bounded_heading_delta(desired, own)
        distance = _range(track, own)
        if name == "extend" and distance is not None and distance < self.params.get("range_m", math.inf):
            action["delta_speed"] = MAX_ACCELERATION_MPS2
        elif name == "recommit" and distance is not None and distance > self.params.get("commit_range_m", 0.0):
            action["delta_speed"] = MAX_ACCELERATION_MPS2

        tolerance = math.radians(self.params.get("capture_angle_deg", 5.0))
        aligned = abs(_wrap(desired - own["heading"])) <= tolerance
        if name in {"pursue_target", "lead_pursuit", "lag_pursuit", "beam_target_left",
                    "beam_target_right", "crank_target_left", "crank_target_right", "turn_cold"}:
            action["completed"] |= aligned
        elif name == "extend" and distance is not None:
            action["completed"] |= distance >= self.params.get("range_m", math.inf)
        elif name == "recommit" and distance is not None:
            action["completed"] |= distance <= self.params.get("commit_range_m", 0.0)

    def _execute_defensive(self, obs, own, action):
        name = self.skill_name
        identifier = self.params.get("threat_id", "highest_priority")
        if name in {"notch_left", "notch_right"}:
            # Notching is aircraft-versus-aircraft geometry: place our nose
            # across the hostile aircraft's heading, not across the current
            # line of sight to it.
            track = _find_track(obs, identifier, threat=False)
            reference_heading = None if track is None else _track_heading(track)
            bearing = None
        elif name in {"beam_missile_left", "beam_missile_right"}:
            # Missile-beam geometry is instead referenced to the nearest live
            # incoming missile track.
            track = _find_nearest_incoming_missile(obs, own, identifier)
            reference_heading = None
            bearing = None if track is None else _bearing(track, own)
        else:
            track = _find_track(obs, identifier, threat=True)
            reference_heading = None
            bearing = None if track is None else _bearing(track, own)
        side = -1 if self.params.get("side", "left") == "left" else 1
        max_g = self.params.get("max_g", 9.0)

        if name == "preemptive_crank" and bearing is not None:
            desired = bearing + side * math.radians(self.params.get("crank_angle_deg", 45.0))
            action["delta_heading"] = self._bounded_heading_delta(desired, own, max_g)
        elif name == "notch_left" and reference_heading is not None:
            action["delta_heading"] = self._bounded_heading_delta(
                reference_heading - math.pi / 2, own, max_g
            )
        elif name == "notch_right" and reference_heading is not None:
            action["delta_heading"] = self._bounded_heading_delta(
                reference_heading + math.pi / 2, own, max_g
            )
        elif name == "beam_missile_left" and bearing is not None:
            action["delta_heading"] = self._bounded_heading_delta(bearing - math.pi / 2, own, max_g)
        elif name == "beam_missile_right" and bearing is not None:
            action["delta_heading"] = self._bounded_heading_delta(bearing + math.pi / 2, own, max_g)
        elif name == "drag_missile" and bearing is not None:
            action["delta_heading"] = self._bounded_heading_delta(bearing + math.pi, own, max_g)
            action["delta_speed"] = MAX_ACCELERATION_MPS2
        elif name == "dive_defense":
            if bearing is not None:
                action["delta_heading"] = self._bounded_heading_delta(bearing + math.pi, own, max_g)
            action["delta_altitude"] = -MAX_CLIMB_RATE_MPS if own["altitude"] > self.params.get("min_altitude_m", 500.0) else 0.0
        elif name == "last_ditch_break":
            # A one-second maximum-lateral-turn command, bounded by max_g.
            action["delta_heading"] = side * _clip(
                min(max_g * GRAVITY_MPS2 / max(own["speed"], 1.0), MAX_TURN_RATE_RAD_S),
                0.0,
                math.pi / 2,
            )
        elif name == "defensive_reversal":
            desired = bearing + math.pi if bearing is not None else own["heading"] + side * math.pi / 2
            action["delta_heading"] = self._bounded_heading_delta(desired, own, max_g)

        if (bearing is not None or reference_heading is not None) and name in {
            "preemptive_crank", "notch_left", "notch_right",
            "beam_missile_left", "beam_missile_right"
        }:
            action["completed"] |= abs(action["delta_heading"]) <= math.radians(5.0)

    def _execute_search(self, obs, own, action):
        """Sweep the aircraft nose through a radar search sector."""
        target = _find_track(obs, self.params.get("target_id", "primary"))
        if target is not None:
            # A track is now available; stop sweeping so the next planner
            # decision can lock, pursue, or employ a weapon.
            action["completed"] = True
            return

        if self._search_origin_heading is None:
            self._search_origin_heading = own["heading"]

        elapsed = max(0.0, float(_value(obs, "time", self.start_time) - self.start_time))
        period = self.params["sweep_period_s"]
        phase = 2.0 * math.pi * elapsed / period
        direction = -1.0 if self.params["direction"] == "left" else 1.0
        amplitude = math.radians(self.params["sector_width_deg"]) / 2.0
        if self.params["pattern"] == "triangle":
            # asin(sin(.)) is a continuous triangular wave in [-pi/2, pi/2].
            offset = amplitude * (2.0 / math.pi) * math.asin(math.sin(phase))
        else:
            offset = amplitude * math.sin(phase)
        desired = self._search_origin_heading + direction * offset
        action["delta_heading"] = _wrap(desired - own["heading"])

    def _execute_support_missile(self, obs, own, action):
        """Crank to the radar-cone edge while retaining a live missile track."""
        missile = _find_active_support_missile(obs, self.params.get("missile_id", "primary"))
        target = _find_track(obs, self.params.get("target_id", "primary"))
        elapsed = max(0.0, float(_value(obs, "time", self.start_time) - self.start_time))
        if missile is None or target is None or elapsed >= self.params["support_timeout_s"]:
            # Never continue a support turn after the missile or radar track is
            # gone; the planner can transition to another skill immediately.
            action["completed"] = True
            return

        bearing = _bearing(target, own)
        if bearing is None:
            action["completed"] = True
            return

        cone_edge = math.radians(
            max(0.0, self.params["radar_half_fov_deg"] - self.params["radar_margin_deg"])
        )
        # In the NWU convention used by the existing beam skills, left places
        # the target at positive relative bearing and right at negative.
        sign = 1.0 if self.params["support_side"] == "left" else -1.0
        desired = bearing - sign * cone_edge
        action["delta_heading"] = self._bounded_heading_delta(
            desired, own, self.params["max_g"]
        )

    def execute(self, obs):
        """Emit a geometry-aware guidance action and completion flag.

        ``obs`` may be the structured agent observation or a light-weight
        mapping containing ``self_status`` plus target/threat tracks.  Missing
        tracks result in a bounded no-op (or the historical left/right
        fallback), which keeps catalogue selection safe during track loss.
        """
        now = _value(obs, "time", 0) or 0
        if self.start_time is None:
            self.start_time = now
        own = _own_state(obs)
        delta_heading = 0.0
        delta_altitude = 0.0
        delta_speed = 0.0
        completed = now - self.start_time >= self.params.get("timeout_s", math.inf)

        trajectory_complete = []
        if "heading_deg" in self.params:
            error = (self.params["heading_deg"] - math.degrees(own["heading"]) + 180) % 360 - 180
            delta_heading = math.radians(error)
            trajectory_complete.append(abs(error) <= self.params.get("tolerance_deg", 0))
        if "altitude_m" in self.params:
            error = self.params["altitude_m"] - own["altitude"]
            delta_altitude = error
            trajectory_complete.append(abs(error) <= self.params.get("tolerance_m", 0))
        if "speed_mps" in self.params:
            error = self.params["speed_mps"] - own["speed"]
            delta_speed = error
            trajectory_complete.append(abs(error) <= self.params.get("tolerance_mps", 0))
        if self.skill_name == "set_trajectory" and trajectory_complete:
            completed |= all(trajectory_complete)

        action = {"skill_name": self.skill_name, "delta_heading": delta_heading,
                  "delta_altitude": delta_altitude, "delta_speed": delta_speed,
                  "shoot": int(self.skill_name == "launch"),
                  "completed": bool(completed)}

        relational = {"pursue_target", "lead_pursuit", "lag_pursuit", "beam_target_left",
                      "beam_target_right", "crank_target_left", "crank_target_right",
                      "turn_cold", "extend", "recommit"}
        defensive = {"preemptive_crank", "notch_left", "notch_right", "beam_missile_left",
                     "beam_missile_right", "drag_missile", "dive_defense", "last_ditch_break",
                     "defensive_reversal"}
        if self.skill_name == "search":
            self._execute_search(obs, own, action)
        elif self.skill_name == "support_missile":
            self._execute_support_missile(obs, own, action)
        elif self.skill_name in relational:
            self._execute_relational(obs, own, action)
        elif self.skill_name in defensive:
            self._execute_defensive(obs, own, action)
        if self.skill_name == "launch":
            action["completed"] = True
        return action, bool(action["completed"])


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
