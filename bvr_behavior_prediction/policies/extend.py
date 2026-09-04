from .base import NativeAction, TacticalPolicy


class ExtendPolicy(TacticalPolicy):
    label = "EXTEND"
    def act(self, own_state, opponent_state, time_s): return NativeAction(heading=0, speed=8)

