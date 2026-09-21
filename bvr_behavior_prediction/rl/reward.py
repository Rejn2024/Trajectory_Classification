"""Event-aware reward shaping for a BVR engagement."""

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
        for field, value in mapping.items():
            components[field] = (
                value if info.get(field, False) and not self.previous.get(field, False) else 0.0
            )
        components["fired_without_lock"] = (
            self.weights.fired_without_lock
            if info.get("fired", False) and not info.get("target_locked", False)
            else 0.0
        )
        self.previous = {field: bool(info.get(field, False)) for field in self.EVENT_FIELDS}
        return sum(components.values()), components
