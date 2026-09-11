import random

from ..data.labels import annotate_transitions
from ..data.relational_features import add_temporal_dynamics, relational_features


class EpisodeRunner:
    """Runs adapters exposing optional state in info['states']; intentionally backend-neutral."""
    def __init__(self, adapter, observer_policy, target_policy):
        self.adapter, self.observer_policy, self.target_policy = adapter, observer_policy, target_policy

    def run(self, seed: int, episode_id: str) -> list[dict]:
        rng = random.Random(seed); self.observer_policy.reset(rng); self.target_policy.reset(rng)
        _, info = self.adapter.reset(seed); rows = []
        for step in range(self.adapter.config.max_steps):
            own, target = info["states"]["observer"], info["states"]["target"]
            time_s = step * self.adapter.config.dt
            oa = self.observer_policy.act(own, target, time_s)
            ta = self.target_policy.act(target, own, time_s)
            _, _, terminated, truncated, info = self.adapter.step((oa.as_tuple(), ta.as_tuple()))
            row = {"episode_id": episode_id, "perspective_id": f"{episode_id}:observer",
                   "step": step, "time_s": time_s, "observer_id": "observer", "target_id": "target",
                   "observer_aircraft_type": self.adapter.config.observer_aircraft,
                   "target_aircraft_type": self.adapter.config.target_aircraft,
                   **{f"observer_{k}": v for k, v in own.items()},
                   **{f"target_{k}": v for k, v in target.items()},
                   **relational_features(own, target), "track_valid": True, "track_age": 0.0,
                   "sensor_mode": self.adapter.config.sensor_mode, "target_skill": self.target_policy.label,
                   "target_action_heading_bin": ta.heading, "target_action_altitude_bin": ta.altitude,
                   "target_action_speed_bin": ta.speed, "target_action_fire": ta.fire}
            selector = getattr(self.target_policy, "selector_state", None)
            if selector:
                row["selector_state"] = selector
                row["transition_reason"] = selector["transition_reason"]
            rows.append(row)
            if terminated or truncated: break
        return annotate_transitions(add_temporal_dynamics(rows))
