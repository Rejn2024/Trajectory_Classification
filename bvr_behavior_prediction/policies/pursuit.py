import math
from .base import NativeAction, TacticalPolicy
from ..data.transforms import wrap_angle


class PursuitPolicy(TacticalPolicy):
    label = "PURSUE"
    def act(self, own_state, opponent_state, time_s):
        desired = math.atan2(opponent_state["y"] - own_state["y"],
                             opponent_state["x"] - own_state["x"])
        error = wrap_angle(desired - own_state["heading"])
        heading = 7 if abs(error) < 0.03 else (11 if error > 0 else 3)
        return NativeAction(heading=heading)

