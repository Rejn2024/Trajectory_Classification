from dataclasses import dataclass
from .base import NativeAction, TacticalPolicy


@dataclass
class CrankPolicy(TacticalPolicy):
    direction: str = "left"
    offset_degrees: float = 40.0
    label = "CRANK"
    def __post_init__(self): self.label = f"CRANK_{self.direction.upper()}"
    def act(self, own_state, opponent_state, time_s):
        magnitude = max(1, min(6, round(self.offset_degrees / 7.5)))
        return NativeAction(heading=7 - magnitude if self.direction == "left" else 7 + magnitude)

