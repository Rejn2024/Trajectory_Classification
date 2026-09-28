"""Real BVR Sim/JSBSim backend for the hybrid-skill pilot."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bvr_sim.agents.skill_manager import SkillManager


class BVRJSBSimBackend:
    """Adapt the Python BVR environment to :class:`BluePilotEnvironment`.

    BVR Sim calls its learned side ``red`` internally.  This adapter deliberately
    exposes that aircraft as ``blue`` to preserve the public blue-pilot contract
    used by the training pipeline.  Both aircraft are F-16s using BVR Sim's
    JSBSim FDM, and the opposing (internally blue) aircraft uses the simulator's
    simple baseline policy.
    """

    CONTROLLED_ID = "A01"
    OPPONENT_ID = "B01"
    OBSERVATION_SIZE = 47  # compact 1-v-1 observation supplied by BVR Sim

    def __init__(self, scenario: dict, recording_path=None, env_factory=None, log_dir=None):
        self.scenario = scenario
        self.recording_path = Path(recording_path) if recording_path else None
        self.skill_manager = SkillManager()
        self.current_skill = None
        self._current_parameters = None
        self._previous_missiles_threatening = 0
        self._previous_friendly_missiles = 0

        if env_factory is None:
            from bvr_sim import BVR3DEnv

            env_factory = BVR3DEnv
        log_dir = Path(log_dir or "artifacts/rl_pilot_notebook/simulator_logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        self.env = env_factory(self._sim_config(), logdir=str(log_dir))
        if self.recording_path:
            self.recording_path.parent.mkdir(parents=True, exist_ok=True)
            self.env.enable_render(str(self.recording_path))

    def _sim_config(self) -> dict:
        duration = 120.0
        dt = 0.1
        altitudes = (self.scenario["blue_altitude_m"], self.scenario["red_altitude_m"])
        return {
            "dt": dt,
            "max_steps": int(duration / dt),
            "red_fighters": {
                self.CONTROLLED_ID: {"model": "F16", "record": False},
            },
            "blue_fighters": {
                self.OPPONENT_ID: {"model": "F16", "record": False},
            },
            "obs_type": "compact",
            "blue_opponent_type": "simple",
            "initial_separation_nm": self.scenario["range_m"] / 1852.0,
            "formation_max_spread_nm": 0.0,
            # SpawnManager samples the two aircraft independently in this interval.
            "min_altitude": float(min(altitudes)),
            "max_altitude": float(max(altitudes)),
            "ground_units": {},
        }

    @staticmethod
    def _flat_observation(observation) -> np.ndarray:
        result = np.asarray(observation, dtype=np.float32)
        if result.ndim == 2 and result.shape[0] == 1:
            result = result[0]
        if result.ndim != 1:
            raise ValueError(f"expected one flat BVR observation, got shape {result.shape}")
        return result

    def _aircraft_info(self, fired=False, missile_avoided=False) -> dict:
        controlled = self.env.agents[self.CONTROLLED_ID]
        opponent = self.env.agents[self.OPPONENT_ID]
        locked = bool(controlled.enemies_lock)
        return {
            "blue_speed_mps": float(controlled.get_speed()),
            "blue_altitude_m": float(controlled.get_altitude()),
            "red_speed_mps": float(opponent.get_speed()),
            "red_altitude_m": float(opponent.get_altitude()),
            "blue_alive": bool(controlled.is_alive),
            "target_locked": locked,
            "fired": bool(fired),
            "fired_with_lock": bool(fired and locked),
            "missile_avoided": bool(missile_avoided),
            "opponent_destroyed": not bool(opponent.is_alive),
            "crashed": not controlled.is_alive and controlled.get_altitude() <= 500.0,
            "shot_down": not controlled.is_alive and controlled.get_altitude() > 500.0,
        }

    def reset(self, seed):
        observation, _ = self.env.reset(seed=seed)
        self.current_skill = None
        self._current_parameters = None
        self._previous_missiles_threatening = 0
        self._previous_friendly_missiles = 0
        if self.recording_path:
            self.env.render()
        return self._flat_observation(observation), self._aircraft_info()

    def _guidance_observation(self) -> dict:
        aircraft = self.env.agents[self.CONTROLLED_ID]
        return {
            "time": self.env.current_step * self.env.dt,
            "self_status": {
                "position": {
                    "heading_deg": float(np.degrees(aircraft.get_heading()) % 360.0),
                    "altitude_m": float(aircraft.get_altitude()),
                },
                "performance": {"speed_mps": float(aircraft.get_speed())},
            },
        }

    def step(self, blue_action, red_action):
        # The opponent is owned by BVR Sim's simple baseline; the second action is
        # accepted only to satisfy BluePilotEnvironment's simulator-neutral API.
        del red_action
        name = blue_action["skill_name"]
        parameters = blue_action["parameters"]
        if (
            self.current_skill is None
            or self.current_skill.skill_name != name
            or self._current_parameters != parameters
        ):
            self.current_skill = self.skill_manager.create_skill(name, parameters)
            self._current_parameters = dict(parameters)
        guidance, _ = self.current_skill.execute(self._guidance_observation())
        normalized = self.env.act_manager.legacy_norm_campus_action(guidance, to_std1=True)

        previous_threats = self._previous_missiles_threatening
        previous_friendly = self._previous_friendly_missiles
        observation, reward, done, simulator_info = self.env.step(
            {self.CONTROLLED_ID: normalized}
        )
        if self.recording_path:
            self.env.render()

        controlled = self.env.agents[self.CONTROLLED_ID]
        self._previous_missiles_threatening = len(controlled.under_missiles)
        self._previous_friendly_missiles = len(controlled.launched_missiles)
        fired = self._previous_friendly_missiles > previous_friendly
        avoided = previous_threats > 0 and self._previous_missiles_threatening == 0
        info = {
            **simulator_info,
            **self._aircraft_info(fired=fired, missile_avoided=avoided),
        }
        simulator_reward = float(np.asarray(reward).reshape(-1)[0])
        terminated = bool(np.asarray(done).reshape(-1)[0]) or bool(
            simulator_info.get("episode_done", False)
        )
        return self._flat_observation(observation), simulator_reward, terminated, info

    def close(self):
        self.env.close()
