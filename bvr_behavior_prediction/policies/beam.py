from dataclasses import dataclass
from .base import NativeAction, TacticalPolicy
from .pursuit import PursuitPolicy


@dataclass
class BeamPolicy(TacticalPolicy):
    direction: str = "left"
    label = "BEAM"
    def act(self, own_state, opponent_state, time_s):
        pursuit = PursuitPolicy().act(own_state, opponent_state, time_s)
        return NativeAction(heading=1 if self.direction == "left" else 13,
                            altitude=pursuit.altitude, speed=pursuit.speed)

