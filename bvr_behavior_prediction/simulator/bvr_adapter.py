"""A dependency-injected adapter; no BVR Sim internals leak into the ML pipeline."""
from collections.abc import Callable
from typing import Any

from .scenario_config import ScenarioConfig


class BVRSimAdapter:
    def __init__(self, config: ScenarioConfig, env_factory: Callable[[dict], Any]):
        self.config = config
        self._env = env_factory(self.to_bvr_config())

    def to_bvr_config(self) -> dict:
        return {
            "dt": self.config.dt,
            "max_steps": self.config.max_steps,
            "obs_type": self.config.observation_type,
            "backend": self.config.backend,
            "weapons_enabled": self.config.weapons_enabled,
            "red": {"unit_spec": self.config.observer_aircraft},
            "blue": {"unit_spec": self.config.target_aircraft},
        }

    def reset(self, seed: int):
        return self._env.reset(seed=seed)

    def step(self, actions):
        return self._env.step(actions)

    def close(self) -> None:
        self._env.close()

    @staticmethod
    def validate_action(action) -> tuple[int, int, int, int]:
        values = tuple(int(v) for v in action)
        if len(values) != 4 or any(v < 0 or v >= n for v, n in zip(values, (15, 15, 9, 2))):
            raise ValueError("action must belong to MultiDiscrete([15, 15, 9, 2])")
        return values

