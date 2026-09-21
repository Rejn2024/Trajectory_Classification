"""Simulator-neutral boundary enforcing blue control and constant-course red flight."""

from collections.abc import Callable


class BluePilotEnvironment:
    """Adapt a JSBSim-backed duel to skill-level planning decisions.

    ``backend_factory`` receives the scenario dictionary and recording path. Its
    object must implement ``reset(seed)``, ``step(blue_action, red_action)`` and
    ``close()``. One step is one planning horizon; a backend may internally integrate
    several JSBSim frames for speed. Observations must be flat numeric vectors.
    """

    def __init__(
        self,
        backend_factory: Callable,
        scenario,
        planning_horizon_s=1.0,
        episode_duration_s=120.0,
        recording_path=None,
    ):
        if episode_duration_s > 120:
            raise ValueError("episodes may not exceed 120 in-game seconds")
        self.horizon = planning_horizon_s
        self.limit = episode_duration_s
        self.backend = backend_factory(scenario.as_dict(), recording_path)
        self.elapsed = 0.0

    def reset(self, seed):
        self.elapsed = 0.0
        result = self.backend.reset(seed)
        return result[0] if isinstance(result, tuple) else result

    def step(self, skill_name, parameters):
        blue_action = {"skill_name": skill_name, "parameters": parameters}
        red_action = {"skill_name": "maintain_heading", "parameters": {}}
        observation, simulator_reward, terminated, info = self.backend.step(blue_action, red_action)
        self.elapsed += self.horizon
        truncated = self.elapsed >= self.limit
        info = dict(info)
        info.update(
            {
                "controlled_team": "blue",
                "red_policy": "constant_course",
                "elapsed_game_s": self.elapsed,
                "simulator_reward": simulator_reward,
            }
        )
        return observation, bool(terminated), bool(truncated), info

    def close(self):
        self.backend.close()
