from dataclasses import dataclass
from .base import NativeAction, TacticalPolicy


class MaintainPolicy(TacticalPolicy):
    label = "MAINTAIN"
    def act(self, own_state, opponent_state, time_s): return NativeAction()


@dataclass
class TurnPolicy(TacticalPolicy):
    direction: str = "left"
    label = "TURN"
    def act(self, own_state, opponent_state, time_s):
        return NativeAction(heading=3 if self.direction == "left" else 11)


@dataclass
class ClimbPolicy(TacticalPolicy):
    direction: str = "climb"
    label = "CLIMB"
    def act(self, own_state, opponent_state, time_s):
        return NativeAction(altitude=11 if self.direction == "climb" else 3)


class DescendPolicy(ClimbPolicy):
    direction = "descend"
    def __init__(self): super().__init__("descend")


@dataclass
class SpeedPolicy(TacticalPolicy):
    accelerate: bool = True
    label = "ACCELERATE"
    def act(self, own_state, opponent_state, time_s):
        return NativeAction(speed=7 if self.accelerate else 1)


class AcceleratePolicy(SpeedPolicy): pass
class DeceleratePolicy(SpeedPolicy):
    label = "DECELERATE"
    def __init__(self): super().__init__(False)

