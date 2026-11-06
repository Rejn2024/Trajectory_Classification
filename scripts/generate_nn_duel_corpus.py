"""Frozen notebook-07 NN versus NN flights, with restartable two-sided recording.

Run this file in a subprocess (including from Jupyter). Worker imports deliberately
use the verified tactical reproduction snapshot, not the caller's installed pilot.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import zipfile
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_context
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tactical_corpus_schema import (
    DT,
    KEYS,
    OBSERVATION_COLUMNS,
    observer_frame,
    validate_episode,
    validate_observer_geometry,
)

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "shared/corpora/trained-pilot-episodes-v1/reproduction.zip"
PARENT = ARCHIVE.parent
DEFAULT_WORK = ROOT / "artifacts/nn_duels_v1"
DEFAULT_OUTPUT = ROOT / "shared/corpora/nn-duels-v1"
SIDES = ("A01", "B01")
PARTS = ("observations", "tracks", "target_decisions")
FAMILIES = (
    "mixed_aspect",
    "crossing_left",
    "crossing_right",
    "tail_chase",
    "cold_recommit",
    "close_merge",
    "low_energy_climb",
    "high_low",
    "offset_support",
    "defensive_beam",
)
STATE = {}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def durable_replace(temp, path):
    with Path(temp).open("rb+") as f:
        os.fsync(f.fileno())
    for attempt in range(20):
        try:
            os.replace(temp, path)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)
    if os.name != "nt":
        fd = os.open(Path(path).parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
    durable_replace(temp, path)


def save_table(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    frame.to_parquet(temp, index=False, compression="zstd", compression_level=3)
    durable_replace(temp, path)


def scenarios(count=10_000, seed=26_110_600):
    """Stratified continuous geometries; every engagement has a new physical state.

    Coordinates and mathematical headings are North/West/Up. They are distinct
    from the simulator's guidance headings (clockwise from north).
    """
    if count < 1 or not 0 <= seed < 2**32 - count:
        raise ValueError("Positive count and an unoverflowed uint32 seed range required")
    rng = np.random.default_rng(seed)
    lhs = np.column_stack([(rng.permutation(count) + rng.random(count)) / count for _ in range(9)])
    # Balanced ordered pilot pairs, including ten self-pairs. Permute each cycle.
    pairs = [(a, b) for a in range(1, 11) for b in range(1, 11)]
    pair_schedule = []
    while len(pair_schedule) < count:
        pair_schedule.extend(pairs[i] for i in rng.permutation(len(pairs)))
    result = []
    for i, u in enumerate(lhs):
        family = FAMILIES[i % len(FAMILIES)]
        axis = 360 * u[0]
        distance = 8_000 + 57_000 * u[1]
        h_a, h_b = axis + 360 * u[2], axis + 360 * u[3]
        alt_a, alt_b = 1_500 + 10_000 * u[4], 1_500 + 10_000 * u[5]
        speed_a, speed_b = 200 + 180 * u[6], 200 + 180 * u[7]
        if family in ("crossing_left", "crossing_right"):
            sign = 1 if family == "crossing_left" else -1
            h_a, h_b = axis + 20 * (u[2] - 0.5), axis + sign * (65 + 50 * u[3])
        elif family == "tail_chase":
            h_a, h_b = axis + 40 * (u[2] - 0.5), axis + 40 * (u[3] - 0.5)
        elif family == "cold_recommit":
            h_a, h_b = axis + 150 + 60 * u[2], axis + 40 * (u[3] - 0.5)
        elif family == "close_merge":
            distance = 6_000 + 8_000 * u[1]
        elif family == "low_energy_climb":
            alt_a, alt_b = 2_000 + 2_000 * u[4], 6_000 + 4_000 * u[5]
            speed_a = 200 + 45 * u[6]
        elif family == "high_low":
            alt_a, alt_b = 8_000 + 3_500 * u[4], 1_500 + 3_000 * u[5]
        elif family == "offset_support":
            distance = 18_000 + 22_000 * u[1]
            h_a, h_b = axis + 25 + 35 * u[2], axis + 145 + 30 * u[3]
        elif family == "defensive_beam":
            distance = 10_000 + 15_000 * u[1]
            h_a, h_b = axis + 75 + 30 * u[2], axis + 160 + 35 * u[3]
        # Exclude the old corpus's exactly antiparallel initial velocities.
        if abs(((h_b - h_a) % 360) - 180) < 5:
            h_b += 11
        delta = distance * np.array([math.cos(math.radians(axis)), math.sin(math.radians(axis))])
        center = 4_000 * (u[8] - 0.5) * np.array([1.0, -0.7])
        a, b = pair_schedule[i]
        result.append(
            {
                "episode_id": f"nn_{seed}_{i:05d}",
                "scenario_group": f"nn_{seed}_{i:05d}",
                "seed": seed + i,
                "family": family,
                "pilots": {"A01": f"pilot_{a:03d}", "B01": f"pilot_{b:03d}"},
                "initial": {
                    "A01": {
                        "position": [*(center - delta / 2), alt_a],
                        "heading_nwu_deg": h_a % 360,
                        "speed_mps": speed_a,
                    },
                    "B01": {
                        "position": [*(center + delta / 2), alt_b],
                        "heading_nwu_deg": h_b % 360,
                        "speed_mps": speed_b,
                    },
                },
            }
        )
    signatures = [json.dumps(c["initial"], sort_keys=True) for c in result]
    if len(set(signatures)) != count:
        raise ValueError("Repeated geometry")
    return result


def verify_snapshot(job):
    plan = read(job / "experiment.json")
    if "20261010T200223_142625Z_tactical" not in plan["source_experiment"]:
        raise ValueError("Expected the post-tactical population used by notebook 07")
    for relative, expected in plan["source_hashes"].items():
        if sha256(job / "source" / relative) != expected:
            raise ValueError(f"Frozen source changed: {relative}")
    pilots = {}
    for number in range(1, 11):
        name = f"pilot_{number:03d}"
        directory = job / "training" / name
        manifest = read(directory / "manifest.json")
        checksum = sha256(directory / "best_model.pt")
        if checksum != manifest["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint mismatch: {name}")
        config = read(directory / "config.json")
        if (
            config["observation_size"] != 51
            or config["training"]["sample_interval_s"] != DT
            or config["training"]["planning_horizon_s"] != 1
        ):
            raise ValueError("Expected notebook-07's 51-feature, 1 Hz pilots")
        pilots[name] = {
            "checkpoint_sha256": checksum,
            "config_sha256": sha256(directory / "config.json"),
            "history_steps": round(config["training"]["history_duration_s"] / DT),
        }
    return pilots


def prepare(work, flights=10_000, seed=26_110_600, seconds=300, deterministic=True):
    work = Path(work).resolve()
    if not math.isfinite(seconds) or not 0 < seconds <= 300 or not float(seconds / DT).is_integer():
        raise ValueError("Duration must be a multiple of 0.1 seconds, at most 300")
    expected = read(PARENT / "manifest.json")["files"]["reproduction.zip"]["sha256"]
    if sha256(ARCHIVE) != expected:
        raise ValueError("Fetch the verified tactical reproduction.zip with Git LFS first")
    job = work / "tactical_snapshot"
    if not (job / "experiment.json").exists():
        job.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ARCHIVE) as z:
            for info in z.infolist():
                destination = (job / info.filename).resolve()
                if (
                    not destination.is_relative_to(job)
                    or (info.external_attr >> 16) & 0o170000 == 0o120000
                ):
                    raise ValueError("Unsafe archive member")
            z.extractall(job)
    pilots = verify_snapshot(job)
    versions = {
        p: importlib.metadata.version(p)
        for p in ("numpy", "pandas", "pyarrow", "torch", "jsbsim", "gymnasium")
    }
    versions["python"] = sys.version.split()[0]
    config = {
        "schema_version": "nn-duels-v1",
        "flights": flights,
        "seed": seed,
        "episode_limit_s": seconds,
        "sample_interval_s": DT,
        "decision_interval_s": 1,
        "deterministic_policy": deterministic,
        "pilots": pilots,
        "cases": scenarios(flights, seed),
        "parent_archive_sha256": expected,
        "parent_manifest_sha256": sha256(PARENT / "manifest.json"),
        "source_experiment": read(job / "experiment.json")["source_experiment"],
        "collector_hashes": {
            p.name: sha256(p)
            for p in (Path(__file__), Path(__file__).with_name("tactical_corpus_schema.py"))
        },
        "software": versions,
        "inference_device": "cpu",
        "inference_precision": "fp32",
        "native_throttle_override": True,
    }
    if (work / "config.json").exists():
        if read(work / "config.json") != config:
            raise ValueError(
                "Resume configuration/source/software differs; use a new work directory"
            )
    else:
        write(work / "config.json", config)
    return config


def initialize_worker(work):
    work = Path(work).resolve()
    job = work / "tactical_snapshot"
    source = job / "source"
    sys.path[:0] = [str(source), str(source / "bvr_sim_source")]
    os.environ["BVR_SIM_RESOURCE_DIR"] = str(source / "bvr_sim_source/bvr_sim/resources")
    os.environ["JSBSIM_DIR"] = str(source / "bvr_sim_source/bvr_sim/resources/jsbsim")
    import bvr_sim.bvr_env as sim_module
    import torch

    import bvr_behavior_prediction.rl.pilot as pilot_module
    from bvr_behavior_prediction.rl.multi_pilot import load_trained_pilot

    if not Path(pilot_module.__file__).resolve().is_relative_to(source):
        raise ValueError("Wrong policy source imported; run the collector in a fresh subprocess")
    if not Path(sim_module.__file__).resolve().is_relative_to(source):
        raise ValueError("Wrong simulator source imported")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    STATE.update(work=work, config=read(work / "config.json"), models={}, loader=load_trained_pilot)


def completed(work, case, verify=True):
    directory = Path(work) / "episodes" / case["episode_id"]
    path = directory / "summary.json"
    if not path.exists():
        return None
    summary = read(path)
    if (
        summary["case_sha256"]
        != hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()
    ):
        raise ValueError("Resume case changed")
    if verify:
        for name, expected in summary["files"].items():
            if sha256(directory / name) != expected:
                raise ValueError(
                    f"Damaged episode {case['episode_id']}/{name}; restore from backup"
                )
    return summary


def collect_one(case):
    import bvr_sim.bvr_env as sim_module
    import torch
    from bvr_sim.agents.skill_manager import SkillManager

    work, config = STATE["work"], STATE["config"]
    previous = completed(work, case)
    if previous:
        return previous
    started = time.perf_counter()
    directory = work / "episodes" / case["episode_id"]
    directory.mkdir(parents=True, exist_ok=True)
    random.seed(case["seed"])
    np.random.seed(case["seed"])
    torch.manual_seed(case["seed"])
    for pilot_id in sorted(set(case["pilots"].values())):
        if pilot_id not in STATE["models"]:
            run = work / "tactical_snapshot/training" / pilot_id
            if sha256(run / "best_model.pt") != config["pilots"][pilot_id]["checkpoint_sha256"]:
                raise ValueError("Checkpoint changed")
            STATE["models"][pilot_id] = STATE["loader"](run, device="cpu")

    class DuelEnvironment(sim_module.BVR3DEnv):
        # Unused scripted expert labels and rewards are expensive diagnostics.
        # Aircraft/sensors/weapons/physics/termination remain in the frozen source.
        def _gen_expert_action(self):
            return np.zeros((1, 2, 4))

        def _get_rewards(self, info):
            return {uid: 0.0 for uid in self.agents}

        def _baseline_policy(self, agent):
            raise RuntimeError(f"Missing NN action for {agent.uid}; scripted fallback forbidden")

    native_factory = sim_module.create_aircraft

    def create_at_requested_state(model, **kwargs):
        state = case["initial"][kwargs["uid"]]
        angle = math.radians(state["heading_nwu_deg"])
        kwargs["position"] = np.array(state["position"], dtype=float)
        kwargs["velocity"] = state["speed_mps"] * np.array([math.cos(angle), math.sin(angle), 0.0])
        return native_factory(model, **kwargs)

    sim_config = {
        "dt": DT,
        "max_steps": round(config["episode_limit_s"] / DT),
        "red_fighters": {"A01": {"model": "F16", "record": False}},
        "blue_fighters": {"B01": {"model": "F16", "record": False}},
        "blue_opponent_type": None,
        "obs_type": "compact",
        "ground_units": {},
    }
    frames = {s: [] for s in SIDES}
    tracks = {s: [] for s in SIDES}
    decisions = {s: [] for s in SIDES}
    histories = {
        s: deque(maxlen=config["pilots"][case["pilots"][s]]["history_steps"]) for s in SIDES
    }
    managers = {s: SkillManager() for s in SIDES}
    skills, active = {}, {}

    def snapshot(env):
        step = env.current_step
        for side, target in (("A01", "B01"), ("B01", "A01")):
            frame = observer_frame(env, side, target)
            frames[side].append(frame)
            if not histories[side]:
                histories[side].extend([frame] * histories[side].maxlen)
            else:
                histories[side].append(frame)
            row = {"episode_id": case["episode_id"], "step": step, "time_s": round(step * DT, 8)}
            for role, uid in (("observer", side), ("target", target)):
                aircraft = env.agents[uid]
                for name, value in zip(
                    ("x", "y", "z", "vx", "vy", "vz"),
                    [*aircraft.position, *aircraft.velocity],
                    strict=True,
                ):
                    row[f"{role}_{name}"] = float(value)
                row[f"{role}_alive"] = bool(aircraft.is_alive)
                row[f"{role}_launches"] = len(aircraft.launched_missiles)
                row[f"{role}_missiles_remaining"] = aircraft.num_left_missiles
                row[f"{role}_incoming_missiles"] = sum(m.is_alive for m in aircraft.under_missiles)
            tracks[side].append(row)

    with (directory / "simulator.log").open("w", encoding="utf-8") as log:  # noqa: SIM117
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            env = DuelEnvironment(sim_config, logdir=str(directory / "logs"))
            try:
                # The patch exists only during reset in this isolated process.
                # It sets the actual JSBSim initial conditions, not metadata alone.
                with patch.object(sim_module, "create_aircraft", create_at_requested_state):
                    env.reset(seed=case["seed"])
                if env.ego_ids != list(SIDES) or env.blue_opponent_type is not None:
                    raise ValueError("Both aircraft must accept NN actions")
                for side in SIDES:
                    aircraft = env.agents[side]
                    requested = case["initial"][side]
                    if not np.allclose(aircraft.position, requested["position"], atol=1e-5, rtol=0):
                        raise ValueError("Actual reset position differs from requested geometry")
                    heading = math.radians(requested["heading_nwu_deg"])
                    expected_v = requested["speed_mps"] * np.array(
                        [math.cos(heading), math.sin(heading), 0]
                    )
                    if not np.allclose(aircraft.velocity, expected_v, atol=1e-5, rtol=0):
                        raise ValueError("Actual reset orientation/speed differs")
                snapshot(env)
                for step in range(sim_config["max_steps"]):
                    if step % 10 == 0:
                        # Batch both sides when the same pilot flies against itself.
                        for pilot_id in sorted(set(case["pilots"].values())):
                            sides = [s for s in SIDES if case["pilots"][s] == pilot_id]
                            observations = np.stack([np.asarray(histories[s]) for s in sides])
                            actions = STATE["models"][pilot_id].act_batch(
                                observations, deterministic=config["deterministic_policy"]
                            )
                            for side, action in zip(sides, actions, strict=True):
                                name, parameters = action[:2]
                                choice = (name, parameters)
                                if active.get(side) != choice:
                                    skills[side] = managers[side].create_skill(name, parameters)
                                    active[side] = choice
                                decisions[side].append(
                                    {
                                        "episode_id": case["episode_id"],
                                        "aircraft_id": side,
                                        "step": step,
                                        "time_s": round(step * DT, 8),
                                        "skill": name,
                                        "parameters_json": json.dumps(parameters, sort_keys=True),
                                    }
                                )
                    actions = {}
                    for side in SIDES:
                        a = env.agents[side]
                        guidance, _ = skills[side].execute(
                            {
                                "time": step * DT,
                                "self_status": {
                                    "position": {
                                        "heading_deg": float(np.degrees(a.get_heading()) % 360),
                                        "altitude_m": float(a.get_altitude()),
                                    },
                                    "performance": {"speed_mps": float(a.get_speed())},
                                },
                            }
                        )
                        actions[side] = env.act_manager.legacy_norm_campus_action(
                            guidance, to_std1=True
                        )
                    env.step(actions)
                    snapshot(env)
                    # Native terminal waits for missiles; requested contract stops at first defeat.
                    if any(not env.agents[s].is_alive for s in SIDES):
                        break
                alive = {s: bool(env.agents[s].is_alive) for s in SIDES}
                kills = {
                    s: any(
                        m.is_success and m.target is env.agents[t]
                        for m in env.agents[s].launched_missiles
                    )
                    for s, t in (("A01", "B01"), ("B01", "A01"))
                }
            finally:
                env.close()
    files, support_counts = {}, {}
    classes = read(work / "tactical_snapshot/training/pilot_001/config.json")["skill_names"]
    for side, target in (("A01", "B01"), ("B01", "A01")):
        track = pd.DataFrame(tracks[side])
        observation = pd.concat(
            [
                track[list(KEYS)],
                pd.DataFrame(np.array(frames[side], dtype=np.float32), columns=OBSERVATION_COLUMNS),
            ],
            axis=1,
        )
        validate_episode(observation, track)
        validate_observer_geometry(observation, track)
        support_counts[side] = forecast_counts(
            track, pd.DataFrame(decisions[target]), classes
        ).tolist()
        for part, table in zip(
            PARTS, (observation, track, pd.DataFrame(decisions[target])), strict=True
        ):
            name = f"{side}/{part}.parquet"
            save_table(directory / name, table)
            files[name] = sha256(directory / name)
    summary = {
        "episode_id": case["episode_id"],
        "scenario_group": case["scenario_group"],
        "seed": case["seed"],
        "family": case["family"],
        "pilots": case["pilots"],
        "duration_s": round((len(frames["A01"]) - 1) * DT, 8),
        "frames": len(frames["A01"]),
        "termination_reason": "time_limit" if all(alive.values()) else "aircraft_defeated",
        "alive": alive,
        "missile_kills": kills,
        "support_counts": support_counts,
        "initial_state": {s: tracks[s][0] for s in SIDES},
        "case_sha256": hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest(),
        "wall_seconds": time.perf_counter() - started,
        "files": files,
    }
    write(directory / "summary.json", summary)  # Completion marker is always written last.
    return summary


@contextlib.contextmanager
def campaign_lock(work):
    lock = Path(work) / "collection.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as e:
        raise RuntimeError(
            f"Campaign is locked: {lock}. After a crash, confirm its PID is dead before removing it."
        ) from e
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        lock.unlink()


def collect(work, config, workers=1, limit=None):
    if workers < 1 or (limit is not None and limit < 1):
        raise ValueError("Positive workers/limit required")
    with campaign_lock(work):
        done = [completed(work, c) for c in config["cases"]]
        pending = [c for c, d in zip(config["cases"], done) if d is None]
        pending = pending[:limit] if limit else pending
        already = sum(d is not None for d in done)
        failures, results = [], []
        start = time.perf_counter()
        # Spawn avoids inherited torch/CUDA state. One BLAS thread per process.
        # Separate processes also isolate simulator RNG and native JSBSim state.
        for key in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[key] = "1"
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
            initializer=initialize_worker,
            initargs=(str(work),),
        ) as pool:
            futures = {pool.submit(collect_one, c): c for c in pending}
            for future in as_completed(futures):
                case = futures[future]
                try:
                    results.append(future.result())
                except BrokenProcessPool as e:
                    write(
                        Path(work) / "failure.json",
                        {"episode_id": case["episode_id"], "error": repr(e)},
                    )
                    raise RuntimeError(
                        "Simulator process pool failed; inspect worker traceback and failure.json"
                    ) from e
                except Exception as e:  # noqa: BLE001 -- retain failed seeds and report the batch failure
                    failures.append({"episode_id": case["episode_id"], "error": repr(e)})
                elapsed = time.perf_counter() - start
                progress = {
                    "completed": already + len(results),
                    "requested": config["flights"],
                    "failed_this_invocation": failures,
                    "new_completed": len(results),
                    "wall_seconds": elapsed,
                    "workers": workers,
                    "flights_per_hour": len(results) * 3600 / max(elapsed, 0.001),
                }
                write(Path(work) / "progress.json", progress)
                if failures or (len(results) % 10 == 0) or len(results) == len(pending):
                    print(json.dumps(progress), flush=True)
        if failures:
            raise RuntimeError(
                "Collection failures retained with original seeds; inspect progress.json and resume"
            )
    return results


def forecast_counts(track, decisions, classes):
    """Exact +1/+2/+5-second targets with 09b's 56-frame alive-history rule."""
    if decisions.step.duplicated().any() or not np.allclose(decisions.time_s, decisions.step * DT):
        raise ValueError("Misaligned decisions")
    labels = pd.Series(
        decisions.skill.map({s: i for i, s in enumerate(classes)}).to_numpy(), index=decisions.step
    )
    if labels.isna().any():
        raise ValueError("Unknown skill")
    origins = np.arange(60, len(track), 10)
    invalid = np.r_[0, np.cumsum(~(track.observer_alive & track.target_alive).to_numpy())]
    origins = origins[invalid[origins + 1] == invalid[origins - 55]]
    counts = []
    for horizon in (1, 2, 5):
        target = labels.reindex(origins + horizon * 10, fill_value=-1).to_numpy(dtype=int)
        counts.append(np.bincount(target[target >= 0], minlength=len(classes)))
    return np.array(counts)


def package(work, output, config, shard_flights=100):
    """Publish only closed immutable shards. A partial manifest never claims completion."""
    work, output = Path(work), Path(output)
    if shard_flights < 1:
        raise ValueError("Positive shard size required")
    with campaign_lock(work):
        output.mkdir(parents=True, exist_ok=True)
        config_path = output / "config.json"
        if config_path.exists() and read(config_path) != config:
            raise ValueError("Publication belongs to a different campaign")
        previous_files = (
            read(output / "manifest.json")["files"] if (output / "manifest.json").exists() else {}
        )
        write(config_path, config)
        if (output / "schema.json").exists() and read(output / "schema.json")[
            "shard_flights"
        ] != shard_flights:
            raise ValueError("Publication shard size differs")
        write(
            output / "schema.json",
            {
                "schema_version": "nn-duels-v1",
                "shard_flights": shard_flights,
                "observation_columns": list(OBSERVATION_COLUMNS),
                "key_columns": list(KEYS),
                "coordinate_frame": "North/West/Up; metres, metres/second; 10 Hz",
                "perspectives": {
                    "A01": {"observer": "A01", "target": "B01"},
                    "B01": {"observer": "B01", "target": "A01"},
                },
                "decision_role": "privileged target choices at 1 Hz; excluded from input features",
                "split_group": "scenario_group shared by both perspectives; never split a physical flight",
                "forecasting": "Raw data; see load_forecasting_view for 09b-compatible per-episode frames",
            },
        )
        classes = read(work / "tactical_snapshot/training/pilot_001/config.json")["skill_names"]
        records, support, published, inventory = (
            [],
            np.zeros((3, len(classes)), dtype=np.int64),
            [],
            {},
        )
        episode_support = np.zeros_like(support)
        support_groups = {}
        for offset in range(0, config["flights"], shard_flights):
            cases = config["cases"][offset : offset + shard_flights]
            summaries = [completed(work, c) for c in cases]
            if any(s is None for s in summaries):
                continue
            shard = f"part-{offset // shard_flights:05d}"
            for side, target in (("A01", "B01"), ("B01", "A01")):
                for part in PARTS:
                    relative = f"{side}/{part}/{shard}.parquet"
                    path = output / relative
                    if relative in previous_files:
                        expected = previous_files[relative]
                        if not path.exists() or sha256(path) != expected["sha256"]:
                            raise ValueError(f"Immutable published shard damaged: {relative}")
                        inventory[relative] = expected
                        continue
                    tables = [
                        pq.read_table(
                            work / "episodes" / c["episode_id"] / side / f"{part}.parquet"
                        )
                        for c in cases
                    ]
                    if path.exists():
                        # Compare every row group to verified local originals before reusing a shard.
                        old = pq.ParquetFile(path)
                        if old.num_row_groups != len(tables) or any(
                            not old.read_row_group(i).equals(t) for i, t in enumerate(tables)
                        ):
                            raise ValueError(f"Immutable shard differs: {relative}")
                    else:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        temp = path.with_suffix(".tmp")
                        with pq.ParquetWriter(
                            temp, tables[0].schema, compression="zstd", compression_level=3
                        ) as writer:
                            for table in tables:
                                writer.write_table(table, row_group_size=len(table))
                        durable_replace(temp, path)
                    if path.stat().st_size >= 1_000_000_000:
                        raise ValueError("Shard exceeds conservative 1 GB limit")
                    inventory[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
                for c, summary in zip(cases, summaries):
                    directory = work / "episodes" / c["episode_id"] / side
                    if "support_counts" in summary:
                        count = np.asarray(summary["support_counts"][side], dtype=np.int64)
                    else:
                        count = forecast_counts(
                            pd.read_parquet(directory / "tracks.parquet"),
                            pd.read_parquet(directory / "target_decisions.parquet"),
                            classes,
                        )
                    support += count
                    episode_support += count > 0
                    group_key = (side, c["pilots"][target], c["family"])
                    totals = support_groups.setdefault(
                        group_key, [np.zeros_like(count), np.zeros_like(count)]
                    )
                    totals[0] += count
                    totals[1] += count > 0
                    records.append(
                        {
                            "episode_id": c["episode_id"],
                            "scenario_group": c["scenario_group"],
                            "seed": c["seed"],
                            "family": c["family"],
                            "observer_id": side,
                            "target_id": target,
                            "observer_pilot_id": c["pilots"][side],
                            "pilot_id": c["pilots"][target],
                            "checkpoint_sha256": config["pilots"][c["pilots"][target]][
                                "checkpoint_sha256"
                            ],
                            "frames": summary["frames"],
                            "duration_s": summary["duration_s"],
                            "target_survived": summary["alive"][target],
                            "observer_survived": summary["alive"][side],
                            "termination_reason": summary["termination_reason"],
                            "shard": shard,
                            "initial_state_json": json.dumps(
                                summary["initial_state"][side], sort_keys=True
                            ),
                        }
                    )
            published.extend(c["episode_id"] for c in cases)
        save_table(output / "episodes.parquet", pd.DataFrame(records))
        support_frame = pd.DataFrame(
            [
                {
                    "horizon_s": h,
                    "skill": skill,
                    "windows": int(support[i, j]),
                    "perspective_episodes": int(episode_support[i, j]),
                }
                for i, h in enumerate((1, 2, 5))
                for j, skill in enumerate(classes)
            ]
        )
        save_table(output / "skill_support.parquet", support_frame)
        breakdown = [
            {
                "observer_id": side,
                "pilot_id": pilot,
                "family": family,
                "horizon_s": h,
                "skill": skill,
                "windows": int(totals[0][i, j]),
                "perspective_episodes": int(totals[1][i, j]),
            }
            for (side, pilot, family), totals in sorted(support_groups.items())
            for i, h in enumerate((1, 2, 5))
            for j, skill in enumerate(classes)
        ]
        save_table(output / "skill_support_by_pilot_family.parquet", pd.DataFrame(breakdown))
        # Include the exact code/assets/weights used; the archive is already hash verified.
        if not (output / "tactical_reproduction.zip").exists():
            shutil.copyfile(ARCHIVE, output / "tactical_reproduction.zip")
        if sha256(output / "tactical_reproduction.zip") != config["parent_archive_sha256"]:
            raise ValueError("Published reproduction archive differs")
        for source in (Path(__file__), Path(__file__).with_name("tactical_corpus_schema.py")):
            destination = output / "collector" / source.name
            destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, destination)
        readme = (
            "# NN versus NN collection\n\n"
            "Two recorded perspectives per physical engagement, using frozen notebook-07 tactical pilots.\n"
            "manifest.json reports the actual flight count; complete is true only at the requested count.\n"
            "Both perspectives share scenario_group. Keep them in the same future split.\n"
            "Observations are idealised 51-feature controller inputs at 10 Hz. Target decisions are privileged.\n"
            "Decisions occur at 1 Hz before the next physics step. Terminal samples have no new decision.\n"
            "No forward filling, fitted scaling, splits, or forecast windows are stored.\n"
            "The native forced-throttle setting is preserved; catalogue speed labels may not change throttle.\n"
            "A01/ and B01/ each contain observations, tracks and target_decisions, sharded by 100 flights by default.\n"
            "episodes.parquet has one row per perspective (twice the physical flight count).\n"
            "Use scripts/generate_nn_duel_corpus.py load_forecasting_view for a 09b-compatible episode.\n"
            "The existing 09b loader is fixed to its original corpus; it does not auto-discover this dataset.\n"
            "See notebooks/11_nn_vs_nn_training_data.ipynb for collection, verification, skill gaps and publication.\n"
        )
        (output / "README.md").write_text(readme, encoding="utf-8", newline="\n")
        for path in output.rglob("*"):
            if path.is_file() and path.name != "manifest.json" and path.suffix != ".tmp":
                inventory[path.relative_to(output).as_posix()] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
        manifest = {
            "name": "nn-duels-v1",
            "complete": len(published) == config["flights"],
            "requested_flights": config["flights"],
            "physical_flights": len(published),
            "perspective_episodes": len(records),
            "files": inventory,
            "config_sha256": sha256(config_path),
        }
        write(output / "manifest.json", manifest)  # Last: snapshot completeness marker.
        return manifest


def verify(output, require_complete=False):
    output = Path(output)
    manifest = read(output / "manifest.json")
    config = read(output / "config.json")
    if sha256(output / "config.json") != manifest["config_sha256"]:
        raise ValueError("Configuration checksum differs")
    if require_complete and not manifest["complete"]:
        raise ValueError("Full requested flight count has not been acquired")
    for name, item in manifest["files"].items():
        path = output / name
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Publication checksum mismatch: {name}")
    episodes = pd.read_parquet(output / "episodes.parquet")
    if (
        len(episodes) != manifest["perspective_episodes"]
        or len(episodes) != 2 * manifest["physical_flights"]
    ):
        raise ValueError("Physical flight/perspective count mismatch")
    if len(episodes) and (
        episodes.duplicated(["episode_id", "observer_id"]).any()
        or episodes.episode_id.nunique() != manifest["physical_flights"]
        or (episodes.groupby("episode_id").observer_id.nunique() != 2).any()
    ):
        raise ValueError("A physical flight is missing a perspective")
    if manifest["complete"] != (manifest["physical_flights"] == config["flights"]):
        raise ValueError("Incorrect completeness flag")
    case_by_id = {c["episode_id"]: c for c in config["cases"]}
    for row in episodes.itertuples():
        obs, track, decisions = _load_shard_episode(
            output, row.episode_id, row.observer_id, row.shard
        )
        validate_episode(obs, track)
        validate_observer_geometry(obs, track)
        expected_decisions = np.arange(0, len(track) - 1, 10)
        if not np.array_equal(decisions.step, expected_decisions) or set(decisions.aircraft_id) != {
            row.target_id
        }:
            raise ValueError("Invalid target decision cadence/identity")
        if len(obs) != row.frames or float(track.time_s.iloc[-1]) != row.duration_s:
            raise ValueError("Index disagrees with frames")
        if row.duration_s > config["episode_limit_s"] or (
            row.termination_reason == "time_limit" and row.duration_s != config["episode_limit_s"]
        ):
            raise ValueError("Incorrect time limit")
        live = track.observer_alive & track.target_alive
        if not live.iloc[:-1].all() or (
            not live.iloc[-1] and row.termination_reason != "aircraft_defeated"
        ):
            raise ValueError("Flight continued after defeat")
        initial = case_by_id[row.episode_id]["initial"][row.observer_id]
        if not np.allclose(
            track.iloc[0][["observer_x", "observer_y", "observer_z"]].to_numpy(dtype=float),
            initial["position"],
            atol=1e-5,
            rtol=0,
        ):
            raise ValueError("Published geometry differs from requested reset")
    return manifest


def load_forecasting_view(output, episode_id, observer_id="B01", canonical=True):
    """Load one chronological 09b-style episode; never mix both perspectives.

    For A01 observers, canonical=True renames the decision's target aircraft to
    A01 as required by 09b's forecast_targets. Physical ID remains in a separate
    actual_aircraft_id column. No observational columns or track axes change.
    """
    output = Path(output)
    if observer_id not in SIDES:
        raise ValueError("Unknown observer")
    index = pd.read_parquet(output / "episodes.parquet")
    rows = index[index.episode_id.eq(episode_id) & index.observer_id.eq(observer_id)]
    if len(rows) != 1:
        raise ValueError("Expected exactly one indexed perspective")
    shard = rows.iloc[0].shard
    tables = _load_shard_episode(output, episode_id, observer_id, shard)
    if canonical:
        tables[2]["actual_aircraft_id"] = tables[2].aircraft_id
        tables[2]["aircraft_id"] = "A01"
    return tuple(tables)


def _load_shard_episode(output, episode_id, observer_id, shard):
    # Verification already has the episode index; do not reload it 20,000 times.
    return [
        pd.read_parquet(
            output / observer_id / part / f"{shard}.parquet",
            filters=[("episode_id", "==", episode_id)],
        )
        .sort_values("step")
        .reset_index(drop=True)
        for part in PARTS
    ]


def publish(output, branch="forecasting_ed_261104", allow_partial=False):
    """Commit only this verified package; the Git push pre-push hook sends LFS."""
    output = Path(output).resolve()
    manifest = verify(output, require_complete=not allow_partial)
    relative = output.relative_to(ROOT).as_posix()
    if not relative.startswith("shared/corpora/"):
        raise ValueError("Publication must be under shared/corpora")

    def git(*args, capture=False):
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=True, text=True, capture_output=capture
        )

    if git("branch", "--show-current", capture=True).stdout.strip() != branch:
        raise ValueError(f"Check out {branch} before publishing")
    if git("diff", "--cached", "--name-only", capture=True).stdout.strip():
        raise ValueError("Unrelated staged changes; commit or unstage them before publication")
    git("lfs", "install", "--local")
    for path in output.rglob("*.parquet"):
        attr = git(
            "check-attr", "filter", "--", path.relative_to(ROOT).as_posix(), capture=True
        ).stdout
        if not attr.rstrip().endswith(": lfs"):
            raise ValueError("Parquet files must be tracked by LFS")
    git("add", "--", relative)
    if git("diff", "--cached", "--name-only", capture=True).stdout.strip():
        git(
            "commit",
            "-m",
            f"Archive {manifest['physical_flights']} NN duel flights with both perspectives",
        )
    # Existing, unrelated corpora may be pointers-only in a selective checkout.
    git("lfs", "fsck", "--objects", "HEAD^..HEAD")
    git("lfs", "fsck", "--pointers", "HEAD")
    git("push", "origin", f"HEAD:refs/heads/{branch}")
    # A second invocation is a cheap check that no LFS uploads remain pending.
    git("lfs", "push", "origin", "HEAD")
    local = git("rev-parse", "HEAD", capture=True).stdout.strip()
    remote = git("ls-remote", "origin", f"refs/heads/{branch}", capture=True).stdout.split()[0]
    if local != remote:
        raise ValueError("Remote branch does not match the published commit")
    return {
        "commit": local,
        "physical_flights": manifest["physical_flights"],
        "remote_confirmed": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "collect", "package", "verify", "publish", "status")
    )
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--flights", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=26_110_600)
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--workers", type=int, default=max(1, int((os.cpu_count() or 1) * 0.8)))
    parser.add_argument(
        "--limit", type=int, help="Maximum NEW flights this invocation; manifest stays partial"
    )
    parser.add_argument("--shard-flights", type=int, default=100)
    parser.add_argument(
        "--stochastic", action="store_true", help="Separate non-argmax campaign; recorded in config"
    )
    parser.add_argument(
        "--allow-partial", action="store_true", help="Publish a clearly marked incremental backup"
    )
    args = parser.parse_args()
    if args.command in ("verify", "publish"):
        result = (
            verify(args.output)
            if args.command == "verify"
            else publish(args.output, allow_partial=args.allow_partial)
        )
    elif args.command == "status":
        config = read(args.work / "config.json")
        done = sum(completed(args.work, c) is not None for c in config["cases"])
        result = {"completed": done, "requested": config["flights"]}
    else:
        # Prepare/resume is locked too, so two notebook kernels cannot initialize the same campaign.
        with campaign_lock(args.work):
            config = prepare(args.work, args.flights, args.seed, args.seconds, not args.stochastic)
        if args.command == "collect":
            result = {"new_flights": len(collect(args.work, config, args.workers, args.limit))}
        elif args.command == "package":
            result = package(args.work, args.output, config, args.shard_flights)
        else:
            result = {"prepared_flights": config["flights"], "pilots": config["pilots"]}
    if isinstance(result, dict) and "files" in result:
        result = {k: v for k, v in result.items() if k != "files"}
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
