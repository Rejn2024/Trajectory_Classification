"""Simulator-neutral boundary for temporally informed blue-pilot control."""

from collections import deque
from collections.abc import Callable, Mapping

import numpy as np


ENERGY_FEATURE_NAMES = (
    "blue_kinetic_energy",
    "blue_potential_energy",
    "red_kinetic_energy",
    "red_potential_energy",
)


class BluePilotEnvironment:
    """Adapt a duel backend to fixed-rate, skill-level planning decisions.

    The backend receives one call per ``sample_interval_s`` (rather than one call
    per policy decision), making the temporal resolution independent of the
    planning horizon.  Its ``reset`` may return either an observation or an
    ``(observation, info)`` pair, and ``step`` returns
    ``(observation, reward, terminated, info)``.  Energy inputs are calculated
    from ``blue/red_speed_mps`` and ``blue/red_altitude_m`` in ``info``.  Missing
    values are carried forward (and are zero before the first measurement).

    A policy observation has shape ``(history_steps, backend_features + 4)``.
    The four extra values are dimensionless approximations of each aircraft's
    specific kinetic and potential energy, kept separate so the policy can
    reason about an energy trade rather than a single total-energy scalar.
    """

    def __init__(
        self,
        backend_factory: Callable,
        scenario,
        planning_horizon_s=1.0,
        episode_duration_s=120.0,
        recording_path=None,
        history_duration_s=2.0,
        sample_interval_s=0.1,
        speed_scale_mps=400.0,
        altitude_scale_m=12_000.0,
    ):
        if episode_duration_s > 120:
            raise ValueError("episodes may not exceed 120 in-game seconds")
        if min(planning_horizon_s, history_duration_s, sample_interval_s) <= 0:
            raise ValueError(
                "planning horizon, history duration, and sample interval must be positive"
            )
        if min(speed_scale_mps, altitude_scale_m) <= 0:
            raise ValueError("energy reference scales must be positive")
        substeps = planning_horizon_s / sample_interval_s
        history_steps = history_duration_s / sample_interval_s
        if not np.isclose(substeps, round(substeps)):
            raise ValueError("planning_horizon_s must be divisible by sample_interval_s")
        if not np.isclose(history_steps, round(history_steps)):
            raise ValueError("history_duration_s must be divisible by sample_interval_s")
        self.horizon = planning_horizon_s
        self.limit = episode_duration_s
        self.sample_interval_s = sample_interval_s
        self.substeps = int(round(substeps))
        self.history_steps = int(round(history_steps))
        self.speed_scale_mps = speed_scale_mps
        self.altitude_scale_m = altitude_scale_m
        self.backend = backend_factory(scenario.as_dict(), recording_path)
        self.history = deque(maxlen=self.history_steps)
        self._energy_measurements = {name: 0.0 for name in ENERGY_FEATURE_NAMES}
        self.elapsed = 0.0

    def _energy_features(self, info: Mapping | None):
        info = info or {}
        for team in ("blue", "red"):
            speed = info.get(f"{team}_speed_mps")
            altitude = info.get(f"{team}_altitude_m")
            if speed is not None:
                # Specific KE / (0.5 * reference-speed**2); mass cancels.
                self._energy_measurements[f"{team}_kinetic_energy"] = (
                    float(speed) / self.speed_scale_mps
                ) ** 2
            if altitude is not None:
                # Specific PE / (g * reference-altitude); gravity cancels.
                self._energy_measurements[f"{team}_potential_energy"] = (
                    float(altitude) / self.altitude_scale_m
                )
        return np.asarray(
            [self._energy_measurements[name] for name in ENERGY_FEATURE_NAMES],
            dtype=np.float32,
        )

    def _frame(self, observation, info=None):
        observation = np.asarray(observation, dtype=np.float32)
        if observation.ndim != 1:
            raise ValueError("backend observations must be flat numeric vectors")
        return np.concatenate((observation, self._energy_features(info)))

    def _stacked_history(self):
        return np.stack(self.history).astype(np.float32, copy=False)

    def reset(self, seed):
        self.elapsed = 0.0
        self.history.clear()
        self._energy_measurements = {name: 0.0 for name in ENERGY_FEATURE_NAMES}
        result = self.backend.reset(seed)
        if isinstance(result, tuple):
            observation, info = result
        else:
            observation, info = result, {}
        frame = self._frame(observation, info)
        # Left padding represents the best estimate before episode start.
        self.history.extend(frame.copy() for _ in range(self.history_steps))
        return self._stacked_history()

    def step(self, skill_name, parameters):
        blue_action = {"skill_name": skill_name, "parameters": parameters}
        red_action = {"skill_name": "maintain_heading", "parameters": {}}
        simulator_reward, terminated, info = 0.0, False, {}
        for _ in range(self.substeps):
            observation, reward, terminated, step_info = self.backend.step(
                blue_action, red_action
            )
            simulator_reward += reward
            info.update(step_info)
            self.history.append(self._frame(observation, step_info))
            self.elapsed += self.sample_interval_s
            if terminated or self.elapsed >= self.limit:
                break
        truncated = self.elapsed >= self.limit
        info.update(
            {
                "controlled_team": "blue",
                "red_policy": "constant_course",
                "elapsed_game_s": self.elapsed,
                "simulator_reward": simulator_reward,
            }
        )
        return self._stacked_history(), bool(terminated), bool(truncated), info

    def close(self):
        self.backend.close()
