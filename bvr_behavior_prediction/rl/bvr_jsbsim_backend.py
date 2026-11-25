"""Real BVR Sim/JSBSim backend for the hybrid-skill pilot."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from bvr_sim.agents.skill_manager import SkillManager

from .observation_features import (
    COMPACT_OBSERVATION_SIZE,
    HOSTILE_MISSILE_FEATURE_NAMES,
    HOSTILE_MISSILE_OBSERVATION_SIZE,
    HOSTILE_MISSILE_RADAR_OBSERVATION_SIZE,
    HOSTILE_MISSILE_SLOTS,
    RADAR_FEATURE_SIZE,
    RADAR_OBSERVATION_SIZE,
    append_radar_features,
    append_hostile_missile_features,
    hostile_missile_features,
)


class BVRJSBSimBackend:
    """Adapt the Python BVR environment to :class:`BluePilotEnvironment`.

    BVR Sim calls its learned side ``red`` internally.  This adapter deliberately
    exposes that aircraft as ``blue`` to preserve the public blue-pilot contract
    used by the training pipeline.  Both aircraft are F-16s using BVR Sim's
    JSBSim FDM, and the opposing (internally blue) aircraft uses the simulator's
    selected scripted baseline policy (simple by default).  Optional radar
    features are appended after compact/hostile-missile features and before the
    environment's four energy features.
    """

    CONTROLLED_ID = "A01"
    OPPONENT_ID = "B01"
    OBSERVATION_SIZE = COMPACT_OBSERVATION_SIZE  # compact 1-v-1 observation supplied by BVR Sim
    HOSTILE_MISSILE_SLOTS = HOSTILE_MISSILE_SLOTS
    HOSTILE_MISSILE_FEATURE_NAMES = HOSTILE_MISSILE_FEATURE_NAMES
    HOSTILE_MISSILE_OBSERVATION_SIZE = HOSTILE_MISSILE_OBSERVATION_SIZE
    RADAR_FEATURE_SIZE = RADAR_FEATURE_SIZE
    RADAR_OBSERVATION_SIZE = RADAR_OBSERVATION_SIZE
    HOSTILE_MISSILE_RADAR_OBSERVATION_SIZE = HOSTILE_MISSILE_RADAR_OBSERVATION_SIZE

    def __init__(self, scenario: dict, recording_path=None, env_factory=None, log_dir=None,
                 opponent_type="simple", include_hostile_missiles=False,
                 episode_duration_s=120.0, include_radar_features=False,
                 terminate_on_aircraft_elimination=False):
        if opponent_type not in ("simple", "tactical"):
            raise ValueError("opponent_type must be simple or tactical")
        if type(include_hostile_missiles) is not bool:
            raise ValueError("include_hostile_missiles must be a boolean")
        if type(include_radar_features) is not bool:
            raise ValueError("include_radar_features must be a boolean")
        if type(terminate_on_aircraft_elimination) is not bool:
            raise ValueError("terminate_on_aircraft_elimination must be a boolean")
        if not np.isfinite(episode_duration_s) or not 0 < episode_duration_s <= 300:
            raise ValueError("episode_duration_s must be in (0, 300]")
        self.episode_duration_s = float(episode_duration_s)
        self.include_hostile_missiles = include_hostile_missiles
        self.include_radar_features = include_radar_features
        self.terminate_on_aircraft_elimination = terminate_on_aircraft_elimination
        self.observation_size = self.OBSERVATION_SIZE
        if include_hostile_missiles:
            self.observation_size = self.HOSTILE_MISSILE_OBSERVATION_SIZE
        if include_radar_features:
            self.observation_size += self.RADAR_FEATURE_SIZE
        self.opponent_type = opponent_type
        self.scenario = scenario
        self.recording_path = Path(recording_path) if recording_path else None
        self.skill_manager = SkillManager()
        self.current_skill = None
        self._current_parameters = None
        self._previous_friendly_missiles = 0
        self._tracked_threats = {}
        self._resolved_threats = set()
        self._missiles_avoided_total = 0
        self._initial_geometry = {}

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
        duration = self.episode_duration_s
        dt = 0.1
        altitudes = (self.scenario["blue_altitude_m"], self.scenario["red_altitude_m"])
        return {
            "dt": dt,
            "max_steps": round(duration / dt),
            "red_fighters": {
                self.CONTROLLED_ID: {"model": "F16", "record": False},
            },
            "blue_fighters": {
                self.OPPONENT_ID: {"model": "F16", "record": False},
            },
            "obs_type": "compact",
            "blue_opponent_type": self.opponent_type,
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

    def _policy_observation(self, observation) -> np.ndarray:
        """Return the configured compact, missile, and radar policy vector."""
        compact = self._flat_observation(observation)
        controlled = self.env.agents[self.CONTROLLED_ID]
        if self.include_hostile_missiles:
            compact = append_hostile_missile_features(compact, controlled)
        if self.include_radar_features:
            compact = append_radar_features(compact, controlled)
        if compact.size != self.observation_size:
            raise ValueError(
                f"expected {self.observation_size} backend features, got {compact.size}"
            )
        return compact

    @classmethod
    def hostile_missile_features(cls, controlled):
        """Expose the shared threat encoder for corpus recording."""
        return hostile_missile_features(controlled)

    def _target_geometry(self):
        controlled = self.env.agents[self.CONTROLLED_ID]
        opponent = self.env.agents[self.OPPONENT_ID]
        relative = np.asarray(opponent.position) - np.asarray(controlled.position)
        distance = float(np.linalg.norm(relative))
        velocity = np.asarray(controlled.velocity)
        speed = float(np.linalg.norm(velocity))
        alignment = float(np.dot(velocity, relative) / (speed * distance)) if speed * distance else 0.0
        return {"target_range_m": distance, "target_alignment": float(np.clip(alignment, -1, 1))}

    def _aircraft_info(self, skill_name=None, fired=0, missiles_avoided=0) -> dict:
        controlled = self.env.agents[self.CONTROLLED_ID]
        opponent = self.env.agents[self.OPPONENT_ID]
        locked = bool(controlled.enemies_lock)
        # A launch command is retained as a reward event even when the
        # simulator rejects the shot. This is notebook 31's un-locked launch
        # accounting and drives the dumb-launch penalty.
        launch_ordered = skill_name == "launch"
        fired_for_reward = bool(launch_ordered or fired)
        threats = [
            missile for missile in controlled.under_missiles
            if missile.is_alive and missile.target is controlled
        ]
        eliminated = not opponent.is_alive and any(
            missile.is_success and missile.target is opponent
            for missile in controlled.launched_missiles
        )
        supporting = controlled.is_alive and opponent.is_alive and opponent in controlled.enemies_lock and any(
            missile.is_alive and missile.target is opponent
            and not getattr(missile, "radar_on", True)
            for missile in controlled.launched_missiles
        )
        return {
            **self._target_geometry(),
            **self._initial_geometry,
            "red_policy": self.opponent_type,
            "blue_speed_mps": float(controlled.get_speed()),
            "blue_altitude_m": float(controlled.get_altitude()),
            "red_speed_mps": float(opponent.get_speed()),
            "red_altitude_m": float(opponent.get_altitude()),
            "blue_alive": bool(controlled.is_alive),
            "target_locked": locked,
            "supporting_missile": bool(supporting),
            "fired": bool(fired),
            "fired_with_lock": bool(fired and locked),
            "fired_without_lock": bool(fired and not locked),
            "missile_avoided": bool(missiles_avoided),
            "incoming_missiles": len(threats),
            "incoming_missiles_seen": len(self._tracked_threats),
            "missiles_avoided_total": self._missiles_avoided_total,
            "missiles_launched_total": len(controlled.launched_missiles),
            "opponent_eliminated": bool(eliminated),
            "opponent_destroyed": not bool(opponent.is_alive),
            "crashed": not controlled.is_alive and controlled.get_altitude() <= 500.0,
            "shot_down": not controlled.is_alive and controlled.get_altitude() > 500.0,
            "reward_event_counts": {
                "fired_with_lock": int(fired_for_reward if locked else 0),
                "fired_without_lock": int(fired_for_reward if not locked else 0),
                "missile_avoided": missiles_avoided,
            },
        }

    def reset(self, seed):
        observation, _ = self.env.reset(seed=seed)
        self.current_skill = None
        self._current_parameters = None
        controlled = self.env.agents[self.CONTROLLED_ID]
        self._previous_friendly_missiles = len(controlled.launched_missiles)
        self._tracked_threats = {
            missile.uid: missile for missile in controlled.under_missiles
            if missile.is_alive and missile.target is controlled
        }
        self._resolved_threats = {
            missile.uid for missile in controlled.under_missiles if not missile.is_alive
        }
        self._missiles_avoided_total = 0
        self._initial_geometry = {
            f"initial_{key}": value for key, value in self._target_geometry().items()
        }
        if self.recording_path:
            self.env.render()
        return self._policy_observation(observation), self._aircraft_info()

    def _guidance_observation(self) -> dict:
        aircraft = self.env.agents[self.CONTROLLED_ID]
        targets = {}
        incoming_missiles = []
        support_missiles = []
        if aircraft.enemies_lock:
            target = min(
                aircraft.enemies_lock,
                key=lambda candidate: np.linalg.norm(candidate.position - aircraft.position),
            )
            targets["primary"] = {
                "id": target.uid,
                "position": np.asarray(target.position, dtype=float).tolist(),
                "velocity": np.asarray(target.velocity, dtype=float).tolist(),
            }
        for missile in aircraft.under_missiles:
            if not missile.is_alive or missile.target is not aircraft:
                continue
            incoming_missiles.append({
                "id": missile.uid,
                "target_id": getattr(getattr(missile, "target", None), "uid", None),
                "position": np.asarray(missile.position, dtype=float).tolist(),
                "velocity": np.asarray(missile.velocity, dtype=float).tolist(),
                "is_alive": True,
            })
        for missile in aircraft.launched_missiles:
            if not missile.is_alive:
                continue
            support_missiles.append({
                "id": missile.uid,
                "target_id": getattr(getattr(missile, "target", None), "uid", None),
                "position": np.asarray(missile.position, dtype=float).tolist(),
                "velocity": np.asarray(missile.velocity, dtype=float).tolist(),
                "is_alive": True,
            })
        return {
            "time": self.env.current_step * self.env.dt,
            "self_status": {
                "position": {
                    "heading_deg": float(np.degrees(aircraft.get_heading()) % 360.0),
                    "altitude_m": float(aircraft.get_altitude()),
                },
                "performance": {"speed_mps": float(aircraft.get_speed())},
            },
            "targets": targets,
            "incoming_missiles": incoming_missiles,
            "support_missiles": support_missiles,
        }

    def step(self, blue_action, red_action):
        # The opponent is owned by BVR Sim's scripted baseline; the second action is
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

        previous_friendly = self._previous_friendly_missiles
        observation, reward, done, simulator_info = self.env.step(
            {self.CONTROLLED_ID: normalized}
        )
        if self.recording_path:
            self.env.render()

        controlled = self.env.agents[self.CONTROLLED_ID]
        self._previous_friendly_missiles = len(controlled.launched_missiles)
        fired = max(0, self._previous_friendly_missiles - previous_friendly)
        for missile in controlled.under_missiles:
            if missile.target is controlled and missile.uid not in self._resolved_threats:
                self._tracked_threats[missile.uid] = missile
        avoided = 0
        for uid, missile in self._tracked_threats.items():
            if uid in self._resolved_threats or missile.is_alive:
                continue
            self._resolved_threats.add(uid)
            # Historical missile entries remain in under_missiles. Credit each
            # failed threat once, only if its intended victim actually survived.
            if controlled.is_alive and missile.target is controlled and not missile.is_success:
                avoided += 1
        self._missiles_avoided_total += avoided
        info = {
            **simulator_info,
            **self._aircraft_info(skill_name=name, fired=fired, missiles_avoided=avoided),
        }
        simulator_reward = float(np.asarray(reward).reshape(-1)[0])
        terminated = bool(np.asarray(done).reshape(-1)[0]) or bool(
            simulator_info.get("episode_done", False)
        )
        if self.terminate_on_aircraft_elimination:
            terminated = terminated or not controlled.is_alive or not self.env.agents[
                self.OPPONENT_ID
            ].is_alive
        return self._policy_observation(observation), simulator_reward, terminated, info

    def close(self):
        self.env.close()
