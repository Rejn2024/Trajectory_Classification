"""Collect a forecasting-ready corpus from a newly trained pilot population.

The collector deliberately records the exact frame submitted to each runtime
controller.  It does not reconstruct a reduced observation later, which keeps
hostile-missile geometry, energy features, and future skill labels aligned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT), str(ROOT / "bvr_sim_source")]

from bvr_behavior_prediction.rl.multi_pilot import load_trained_pilot
from bvr_behavior_prediction.rl.observation_features import POLICY_OBSERVATION_SIZE, observation_feature_schema
from bvr_behavior_prediction.rl.scenarios import ScenarioSampler
from tactical_corpus_schema import DT, KEYS, observer_frame, observation_columns
from bvr_sim.agents.skill_manager import SkillManager

SIDES = ("A01", "B01")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")


def write_parquet(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")


def skill_vector(model, name, parameters, raw):
    names = tuple(model.skill_names)
    if name not in names:
        raise ValueError(f"Unknown selected skill: {name}")
    return {
        "skill": name,
        "skill_index": names.index(name),
        "skill_one_hot": [int(item == name) for item in names],
        "parameters_json": json.dumps(parameters, sort_keys=True),
        "parameter_vector_json": json.dumps(
            np.asarray(raw, dtype=np.float32).tolist(), separators=(",", ":")
        ),
    }


def guidance_observation(env, side, other):
    aircraft = env.agents[side]
    target = env.agents[other]
    targets = {
        "primary": {
            "id": target.uid,
            "position": np.asarray(target.position, dtype=float).tolist(),
            "velocity": np.asarray(target.velocity, dtype=float).tolist(),
        }
    } if target.is_alive else {}
    incoming = []
    support = []
    for missile in aircraft.under_missiles:
        if missile.is_alive and missile.target is aircraft:
            incoming.append({
                "id": missile.uid, "target_id": aircraft.uid,
                "position": np.asarray(missile.position, dtype=float).tolist(),
                "velocity": np.asarray(missile.velocity, dtype=float).tolist(),
                "is_alive": True,
            })
    for missile in aircraft.launched_missiles:
        if missile.is_alive:
            support.append({
                "id": missile.uid,
                "target_id": getattr(getattr(missile, "target", None), "uid", None),
                "position": np.asarray(missile.position, dtype=float).tolist(),
                "velocity": np.asarray(missile.velocity, dtype=float).tolist(),
                "is_alive": True,
            })
    return {
        "time": env.current_step * env.dt,
        "self_status": {
            "position": {
                "heading_deg": float(np.degrees(aircraft.get_heading()) % 360),
                "altitude_m": float(aircraft.get_altitude()),
            },
            "performance": {"speed_mps": float(aircraft.get_speed())},
        },
        "targets": targets,
        "incoming_missiles": incoming,
        "support_missiles": support,
    }


def scenario_case(scenario, index, pilot_a, pilot_b, seed):
    return {
        "episode_id": f"nn_{seed}_{index:05d}",
        "scenario_group": f"nn_{seed}_{index:05d}",
        "seed": seed + index,
        "pilots": {"A01": pilot_a, "B01": pilot_b},
        "scenario": scenario.as_dict(),
    }


def collect_case(case, models, output, history_steps=20, episode_seconds=120.0):
    from bvr_sim import BVR3DEnv

    episode_id = case["episode_id"]
    scenario = case["scenario"]
    config = {
        "dt": DT,
        "max_steps": round(episode_seconds / DT),
        "red_fighters": {"A01": {"model": "F16", "record": False}},
        "blue_fighters": {"B01": {"model": "F16", "record": False}},
        "blue_opponent_type": None,
        "obs_type": "compact",
        "initial_separation_nm": float(scenario["range_m"]) / 1852.0,
        "formation_max_spread_nm": 0.0,
        "min_altitude": min(float(scenario["blue_altitude_m"]), float(scenario["red_altitude_m"])),
        "max_altitude": max(float(scenario["blue_altitude_m"]), float(scenario["red_altitude_m"])),
        "ground_units": {},
    }
    env = BVR3DEnv(config, logdir=str(output / "logs" / episode_id))
    frames = {side: [] for side in SIDES}
    histories = {side: deque(maxlen=history_steps) for side in SIDES}
    tracks = {side: [] for side in SIDES}
    decisions = {side: [] for side in SIDES}
    managers = {side: SkillManager() for side in SIDES}
    active = {}
    skills = {}

    def snapshot():
        for side, other in (("A01", "B01"), ("B01", "A01")):
            frame = observer_frame(env, side, other, POLICY_OBSERVATION_SIZE)
            frames[side].append(frame)
            histories[side].append(frame)
            if len(histories[side]) < history_steps:
                padded = np.stack([frame] * (history_steps - len(histories[side])))
                histories[side] = deque(
                    [*padded, *histories[side]], maxlen=history_steps
                )
            row = {"episode_id": episode_id, "step": env.current_step,
                   "time_s": round(env.current_step * DT, 8)}
            for role, uid in (("observer", side), ("target", other)):
                aircraft = env.agents[uid]
                for name, value in zip(
                    ("x", "y", "z", "vx", "vy", "vz"),
                    [*aircraft.position, *aircraft.velocity], strict=True,
                ):
                    row[f"{role}_{name}"] = float(value)
                row[f"{role}_alive"] = bool(aircraft.is_alive)
                row[f"{role}_launches"] = len(aircraft.launched_missiles)
                row[f"{role}_missiles_remaining"] = int(aircraft.num_left_missiles)
                row[f"{role}_incoming_missiles"] = sum(
                    missile.is_alive for missile in aircraft.under_missiles
                )
            tracks[side].append(row)

    try:
        env.reset(seed=case["seed"])
        snapshot()
        max_steps = config["max_steps"]
        for step in range(max_steps):
            if step % 10 == 0:
                for side, other in (("A01", "B01"), ("B01", "A01")):
                    pilot_id = case["pilots"][side]
                    name, parameters, _, _, raw = models[pilot_id].act(
                        np.stack(histories[side]), deterministic=True
                    )
                    choice = (name, json.dumps(parameters, sort_keys=True))
                    if active.get(side) != choice:
                        skills[side] = managers[side].create_skill(name, parameters)
                        active[side] = choice
                    decisions[side].append({
                        "episode_id": episode_id, "aircraft_id": side,
                        "step": step, "time_s": round(step * DT, 8),
                        **skill_vector(models[pilot_id], name, parameters, raw),
                        "history_start_step": max(0, step - history_steps + 1),
                        "history_end_step": step,
                        "policy_history_sha256": hashlib.sha256(
                            np.asarray(histories[side], dtype="<f4").tobytes()
                        ).hexdigest(),
                    })
            actions = {}
            for side, other in (("A01", "B01"), ("B01", "A01")):
                guidance, _ = skills[side].execute(guidance_observation(env, side, other))
                actions[side] = env.act_manager.legacy_norm_campus_action(guidance, to_std1=True)
            env.step(actions)
            snapshot()
            if any(not env.agents[side].is_alive for side in SIDES):
                break
    finally:
        env.close()

    for side in SIDES:
        observation = pd.DataFrame(frames[side], columns=observation_columns(POLICY_OBSERVATION_SIZE))
        key_frame = pd.DataFrame(tracks[side])[list(KEYS)]
        observation = pd.concat([key_frame.reset_index(drop=True), observation], axis=1)
        track = pd.DataFrame(tracks[side])
        decision = pd.DataFrame(decisions[side])
        directory = output / side
        directory.mkdir(parents=True, exist_ok=True)
        # One file per side keeps future grouped splits simple and preserves the
        # exact raw feature channels independently for both viewpoints.
        write_parquet(directory / "observations" / f"{episode_id}.parquet", observation)
        write_parquet(directory / "tracks" / f"{episode_id}.parquet", track)
        write_parquet(directory / "target_decisions" / f"{episode_id}.parquet", decision)
    return {
        "episode_id": episode_id,
        "scenario_group": case["scenario_group"],
        "seed": case["seed"],
        "pilots": case["pilots"],
        "frames": len(tracks["A01"]),
        "duration_s": float(tracks["A01"][-1]["time_s"]),
        "scenario": scenario,
    }


def generate(population, output, flights=100, seed=26110600, episode_seconds=120.0):
    population = Path(population).resolve()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Corpus output must be empty; choose a new directory")
    records = json.loads((population / "pilots.json").read_text(encoding="utf8"))["pilots"]
    records = [record for record in records if record.get("status") == "completed"]
    if len(records) != 10:
        raise ValueError("Expected exactly ten completed pilots")
    models = {}
    for record in records:
        run = population / record["output_dir"]
        models[record["pilot_id"]] = load_trained_pilot(run, device="cpu")
    sampler = ScenarioSampler(seed)
    scenarios = sampler.sample_batch(flights)
    cases = []
    pilot_ids = [record["pilot_id"] for record in records]
    for index, scenario in enumerate(scenarios):
        cases.append(scenario_case(
            scenario, index, pilot_ids[index % 10], pilot_ids[(index // 10) % 10], seed
        ))
    output.mkdir(parents=True, exist_ok=True)
    episode_records = []
    for case in cases:
        print(f"Collecting {case['episode_id']}", flush=True)
        episode_records.append(collect_case(case, models, output, episode_seconds=episode_seconds))
    write_parquet(output / "episodes.parquet", pd.DataFrame([
        {
            "episode_id": record["episode_id"],
            "scenario_group": record["scenario_group"],
            "seed": record["seed"],
            "pilot_A01": record["pilots"]["A01"],
            "pilot_B01": record["pilots"]["B01"],
            "frames": record["frames"],
            "duration_s": record["duration_s"],
        }
        for record in episode_records
    ]))
    classes = list(models[pilot_ids[0]].skill_names)
    parameter_names = list(models[pilot_ids[0]].parameter_names)
    config = {
        "schema_version": "nn-duels-current-v1",
        "observation_size": POLICY_OBSERVATION_SIZE,
        "sample_interval_s": DT,
        "decision_interval_s": 1.0,
        "history_steps": 20,
        "episode_limit_s": episode_seconds,
        "flights": flights,
        "pilots": pilot_ids,
        "skill_catalogue": classes,
        "parameter_names": parameter_names,
        "source_population": str(population),
        "cases": cases,
        "episodes": episode_records,
    }
    write_json(output / "config.json", config)
    write_json(output / "skill_catalogue.json", {
        "version": SkillManager().skills[next(iter(classes))].version,
        "skills": SkillManager().describe_skills(),
        "classes": classes,
        "parameter_names": parameter_names,
    })
    write_json(output / "schema.json", {
        "schema_version": "nn-duels-current-v1",
        "observation_columns": list(observation_columns(POLICY_OBSERVATION_SIZE)),
        "features": observation_feature_schema(POLICY_OBSERVATION_SIZE),
        "history": "Chronological 20 frames at 10 Hz; reset frame is repeated for left padding.",
        "decision_labels": {
            "skill_index": "catalogue index",
            "skill_one_hot": "one-hot vector in skill_catalogue.classes order",
            "parameters_json": "decoded public SkillManager parameters",
            "parameter_vector_json": "raw NN Gaussian output in parameter_names order",
        },
        "tracks": "Simulator-truth physical state retained for trajectory/maneuver targets.",
    })
    files = {p.relative_to(output).as_posix(): sha256(p)
             for p in sorted(output.rglob("*")) if p.is_file()}
    write_json(output / "manifest.json", {
        "complete": True, "schema_version": "nn-duels-current-v1",
        "observation_size": POLICY_OBSERVATION_SIZE, "episodes": len(episode_records),
        "physical_flights": len(episode_records), "perspective_episodes": len(episode_records) * 2,
        "files": files,
    })
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flights", type=int, default=100)
    parser.add_argument("--seed", type=int, default=26110600)
    parser.add_argument("--episode-seconds", type=float, default=120.0)
    args = parser.parse_args()
    generate(args.population, args.output, args.flights, args.seed, args.episode_seconds)


if __name__ == "__main__":
    main()
