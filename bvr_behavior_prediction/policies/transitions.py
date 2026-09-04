from dataclasses import dataclass


@dataclass
class ScheduledPolicy:
    schedule: list[tuple[float, object]]
    def reset(self, rng):
        for _, policy in self.schedule: policy.reset(rng)
    @property
    def label(self):
        return self._label
    @property
    def _label(self):
        # EpisodeRunner obtains the exact active label through this object after act().
        return getattr(self, "_active_label", self.schedule[0][1].label)
    def active(self, time_s):
        eligible = [policy for start, policy in self.schedule if start <= time_s]
        return eligible[-1] if eligible else self.schedule[0][1]
    def act(self, own_state, opponent_state, time_s):
        policy = self.active(time_s); self._active_label = policy.label
        return policy.act(own_state, opponent_state, time_s)
