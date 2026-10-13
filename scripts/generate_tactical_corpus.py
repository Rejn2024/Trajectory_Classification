"""Generate a resumable, inference-only corpus from frozen tactical pilots.

Run --help for collection and packaging commands. The simulator snapshot and
checkpoint files are read only. No PPO update or model training occurs here.
"""

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tactical_corpus_schema import DT, FEATURES, HORIZONS, LABEL_MAP, derive_tables

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JOB = ROOT / "artifacts/rl_pilot_experiments/20261010T200223_142625Z_tactical"
DEFAULT_WORK = ROOT / "artifacts/tactical_forecasting_v1_collection"
DEFAULT_OUTPUT = ROOT / "shared/corpora/tactical-forecasting-v1"
PARTS = ("observations", "labels", "privileged", "decisions")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def parquet(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    frame.to_parquet(temp, index=False, compression="zstd", compression_level=6)
    temp.replace(path)


def activate(job):
    source = job / "source"
    sys.path[:0] = [str(source), str(source / "bvr_sim_source")]
    # Use the versioned assets, never an unrelated environment override.
    resources = source / "bvr_sim_source/bvr_sim/resources"
    os.environ["BVR_SIM_RESOURCE_DIR"] = str(resources)
    os.environ["JSBSIM_DIR"] = str(resources / "jsbsim")
    return source


def prepare(args):
    source = activate(args.job)
    from bvr_behavior_prediction.rl.scenarios import ScenarioSampler

    original = read(args.job / "experiment.json")
    hashes = original["source_hashes"]
    # Verify every recorded snapshot file, not just Python modules.
    for relative, expected in hashes.items():
        if digest(source / relative) != expected:
            raise ValueError(f"Snapshot mismatch: {relative}")
    pilots = []
    for directory in sorted((args.job / "training").glob("pilot_*")):
        if not directory.is_dir():
            continue
        actual = digest(directory / "best_model.pt")
        if actual != read(directory / "manifest.json")["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint mismatch: {directory}")
        pilots.append({"pilot_id": directory.name, "checkpoint_sha256": actual,
                       "config_sha256": digest(directory / "config.json")})
    if len(pilots) != 10:
        raise ValueError("Expected the complete population of ten pilots")
    scenarios = ScenarioSampler(seed=args.seed).sample_batch(args.episodes_per_pilot)
    permutation = np.random.default_rng(args.split_seed).permutation(len(scenarios))
    split_by_index = {}
    for rank, index in enumerate(permutation):
        split_by_index[int(index)] = ("train" if rank < int(0.7 * len(scenarios)) else
                                     "validation" if rank < int(0.85 * len(scenarios)) else "test")
    cases = [{"scenario_index": i, "scenario_group": f"case_{i:04d}",
              "seed": args.seed + i, "split": split_by_index[i], "scenario": s.as_dict()}
             for i, s in enumerate(scenarios)]
    config = {
        "schema_version": "tactical-forecasting-v1", "seed": args.seed,
        "split_seed": args.split_seed, "episodes_per_pilot": len(cases),
        "device": args.device, "deterministic_policy": True,
        "sample_interval_s": DT, "episode_limit_s": 120,
        "observer_id": "A01", "target_id": "B01", "target_policy": "tactical",
        "observation_model": "idealised exact kinematic track in observer course frame",
        "pilots": pilots, "cases": cases, "source_hashes": hashes,
        "collector_hashes": {p.name: digest(p) for p in
                             (Path(__file__), Path(__file__).with_name("tactical_corpus_schema.py"))},
    }
    path = args.work / "config.json"
    if path.exists() and read(path) != config:
        raise ValueError("Existing corpus config differs; use a new work directory")
    write(path, config)
    return config


_WORKER = {}


def initialize_worker(job, device):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    job = Path(job)
    source = activate(job)
    import torch

    import bvr_behavior_prediction.rl.pilot as pilot_module
    from bvr_behavior_prediction.rl.multi_pilot import load_trained_pilot

    if not Path(pilot_module.__file__).resolve().is_relative_to(source):
        raise ValueError("Inference imported the wrong source tree")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    _WORKER.update(job=job, device=device, models={}, loader=load_trained_pilot)


def collect_one(task):
    import torch

    from bvr_behavior_prediction.rl.bvr_jsbsim_backend import BVRJSBSimBackend
    from bvr_behavior_prediction.rl.environment import BluePilotEnvironment
    from bvr_behavior_prediction.rl.scenarios import EngagementScenario

    case, pilot, work = task
    pilot_id = pilot["pilot_id"]
    episode_id = f"{pilot_id}_{case['scenario_group']}"
    directory = Path(work) / "episodes" / episode_id
    directory.mkdir(parents=True, exist_ok=True)
    finished = directory / "summary.json"
    if finished.exists():
        result = read(finished)
        for part, expected in result["files"].items():
            if digest(directory / f"{part}.parquet") != expected:
                raise ValueError(f"Resume file mismatch: {episode_id}/{part}")
        return result
    started = time.perf_counter()
    run_dir = _WORKER["job"] / "training" / pilot_id
    if pilot_id not in _WORKER["models"]:
        if digest(run_dir / "best_model.pt") != pilot["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed during generation")
        _WORKER["models"][pilot_id] = _WORKER["loader"](run_dir, device=_WORKER["device"])
    model = _WORKER["models"][pilot_id]
    training = read(run_dir / "config.json")["training"]
    random.seed(case["seed"])
    np.random.seed(case["seed"])
    torch.manual_seed(case["seed"])
    records, decisions = [], []

    class RecordingBackend(BVRJSBSimBackend):
        def snapshot(self):
            row = {"episode_id": episode_id, "step": self.env.current_step,
                   "time_s": round(self.env.current_step * self.env.dt, 8)}
            for role, uid in (("observer", "A01"), ("target", "B01")):
                aircraft = self.env.agents[uid]
                for key, value in zip(("x", "y", "z", "vx", "vy", "vz"),
                                      [*aircraft.position, *aircraft.velocity], strict=True):
                    row[f"{role}_{key}"] = float(value)
                row[f"{role}_alive"] = bool(aircraft.is_alive)
                row[f"{role}_launches"] = len(aircraft.launched_missiles)
            records.append(row)

        def reset(self, seed):
            result = super().reset(seed)
            self.snapshot()
            return result

        def step(self, blue_action, red_action):
            result = super().step(blue_action, red_action)
            self.snapshot()
            return result

    def factory(scenario, recording_path=None):
        return RecordingBackend(scenario, log_dir=directory / "logs", opponent_type="tactical")

    with (directory / "simulator.log").open("w", encoding="utf-8") as log:  # noqa: SIM117
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            env = BluePilotEnvironment(
                factory, EngagementScenario(**case["scenario"]),
                planning_horizon_s=training["planning_horizon_s"], episode_duration_s=120,
                history_duration_s=training["history_duration_s"], sample_interval_s=DT,
            )
            try:
                observation = env.reset(case["seed"])
                for _ in range(round(120 / training["planning_horizon_s"])):
                    skill, parameters, *_ = model.act(observation, deterministic=True)
                    decisions.append({"episode_id": episode_id, "time_s": round(env.elapsed, 8),
                                      "skill": skill, "parameters_json": json.dumps(parameters, sort_keys=True)})
                    observation, terminated, truncated, info = env.step(skill, parameters)
                    if terminated or truncated:
                        break
            finally:
                env.close()
    if info["red_policy"] != "tactical":
        raise ValueError("Wrong opponent controller")
    raw = pd.DataFrame(records)
    observations, labels = derive_tables(raw)
    tables = {"observations": observations, "labels": labels, "privileged": raw,
              "decisions": pd.DataFrame(decisions)}
    for name, frame in tables.items():
        parquet(frame, directory / f"{name}.parquet")
    survived, eliminated = bool(info["blue_alive"]), bool(info["opponent_eliminated"])
    destroyed = bool(info["opponent_destroyed"])
    result = {
        "episode_id": episode_id, "pilot_id": pilot_id,
        "scenario_group": case["scenario_group"], "seed": case["seed"], "split": case["split"],
        "checkpoint_sha256": pilot["checkpoint_sha256"], "frames": len(raw),
        "duration_s": float(raw.time_s.iloc[-1]), "survived": survived, "eliminated": eliminated,
        "target_destroyed": destroyed,
        "outcome": ("clean_win" if survived and eliminated else "survived_without_elimination"
                    if survived else "mutual_destruction" if destroyed else "loss_without_elimination"),
        "termination_reason": "simulator_terminal" if terminated else "time_limit",
        "observer_launches": int(raw.observer_launches.iloc[-1]),
        "target_launches": int(raw.target_launches.iloc[-1]),
        "initial_state": {k: v for k, v in records[0].items() if k not in ("episode_id", "step", "time_s")},
        "valid_inputs": int(observations.input_valid.sum()),
        "valid_forecasts": {f"{h}s": int(labels[f"valid_{h}s"].sum()) for h in HORIZONS},
        "wall_seconds": round(time.perf_counter() - started, 3),
        "files": {name: digest(directory / f"{name}.parquet") for name in tables},
    }
    write(finished, result)
    return result


def collect(args, config):
    tasks = [(case, pilot, str(args.work)) for case in config["cases"]
             for pilot in config["pilots"]]
    if args.limit:
        tasks = tasks[:args.limit]
    # Completed episodes are fully verified in the workers when resuming.
    started = time.perf_counter()
    completed, failures = [], []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker,
                             initargs=(str(args.job), args.device)) as pool:
        pending = {pool.submit(collect_one, task): task for task in tasks}
        for future in as_completed(pending):
            task = pending[future]
            try:
                completed.append(future.result())
            except Exception as exc:  # noqa: BLE001 -- record worker failures; never replace seeds
                failures.append({"pilot_id": task[1]["pilot_id"], "seed": task[0]["seed"],
                                 "error": repr(exc)})
            progress = {"completed": len(completed), "requested": len(tasks),
                        "failed": len(failures), "wall_seconds": round(time.perf_counter() - started, 1),
                        "updated_utc": datetime.now(timezone.utc).isoformat(), "failures": failures}
            write(args.work / "progress.json", progress)
            if (len(completed) + len(failures)) % 25 == 0 or failures or len(completed) == len(tasks):
                print(json.dumps(progress), flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} episodes failed; inspect progress.json and resume unchanged")


def package(args, config):
    """Validate every episode, then write one Zstandard shard per pilot per table."""
    output = args.output
    if (output / "manifest.json").exists():
        raise ValueError("Publication already exists; validate it instead of overwriting")
    rows, counts = [], {name: {} for name in ("lateral", "vertical", "energy")}
    all_initial = {}
    for pilot in config["pilots"]:
        pilot_id = pilot["pilot_id"]
        writers = {}
        try:
            for case in config["cases"]:
                episode_id = f"{pilot_id}_{case['scenario_group']}"
                directory = args.work / "episodes" / episode_id
                summary = read(directory / "summary.json")
                if (summary["checkpoint_sha256"] != pilot["checkpoint_sha256"] or
                        summary["seed"] != case["seed"] or summary["split"] != case["split"]):
                    raise ValueError(f"Episode provenance mismatch: {episode_id}")
                initial = summary["initial_state"]
                if (case["scenario_group"] in all_initial and
                        initial != all_initial[case["scenario_group"]]):
                    raise ValueError("Matched scenario spawned different initial states")
                all_initial[case["scenario_group"]] = initial
                frames = {}
                for part in PARTS:
                    path = directory / f"{part}.parquet"
                    if digest(path) != summary["files"][part]:
                        raise ValueError(f"Corrupt episode: {path}")
                    table = pq.read_table(path)
                    frames[part] = table.to_pandas()
                    if part not in writers:
                        dest = output / part / f"{pilot_id}.parquet"
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        writers[part] = pq.ParquetWriter(dest, table.schema, compression="zstd",
                                                        compression_level=6)
                    writers[part].write_table(table)
                recalculated = derive_tables(frames["privileged"])
                pd.testing.assert_frame_equal(frames["observations"], recalculated[0])
                pd.testing.assert_frame_equal(frames["labels"], recalculated[1])
                if len(frames["observations"]) != summary["frames"]:
                    raise ValueError("Frame count mismatch")
                for name, frequencies in counts.items():
                    for label, number in frames["labels"][f"current_{name}"].value_counts().items():
                        frequencies[str(label)] = frequencies.get(str(label), 0) + int(number)
                rows.append({k: v for k, v in summary.items()
                             if k not in ("files", "initial_state", "valid_forecasts")})
                rows[-1].update({f"valid_forecasts_{k}": v for k, v in summary["valid_forecasts"].items()})
                rows[-1]["initial_state_json"] = json.dumps(initial, sort_keys=True)
        finally:
            for writer in writers.values():
                writer.close()
        print(f"Verified and packaged {pilot_id}", flush=True)
    episodes = pd.DataFrame(rows)
    expected = len(config["pilots"]) * len(config["cases"])
    if len(episodes) != expected or episodes.episode_id.nunique() != expected:
        raise ValueError("Incomplete episode population")
    if episodes.groupby("scenario_group").split.nunique().max() != 1:
        raise ValueError("Scenario leakage between splits")
    parquet(episodes, output / "episodes.parquet")
    parquet(episodes[["episode_id", "scenario_group", "pilot_id", "split", "seed"]],
            output / "splits.parquet")
    write(output / "config.json", config)
    write(output / "label_map.json", LABEL_MAP)
    write(output / "schema.json", {
        "features": list(FEATURES), "key_columns": ["episode_id", "step", "time_s"],
        "valid_input_column": "input_valid", "labels": LABEL_MAP,
        "coordinate_frame": "World North/West/Up; relative forward/left/up aligned to observer ground course.",
        "exclusions": "Episode metadata, identity, split, labels, privileged state and decisions are not model features.",
    })
    by_pilot = {}
    for pilot_id, frame in episodes.groupby("pilot_id"):
        by_pilot[pilot_id] = {"episodes": len(frame), "frames": int(frame.frames.sum()),
                              "outcomes": frame.outcome.value_counts().to_dict(),
                              "mean_duration_s": float(frame.duration_s.mean())}
    metrics = {"episodes": len(episodes), "scenario_groups": len(config["cases"]),
               "frames": int(episodes.frames.sum()), "simulated_hours": float(episodes.duration_s.sum() / 3600),
               "splits": episodes.split.value_counts().to_dict(), "by_pilot": by_pilot,
               "current_label_counts": counts,
               "valid_forecasts": {f"{h}s": int(episodes[f"valid_forecasts_{h}s"].sum()) for h in HORIZONS},
               "checks": ["every episode file SHA-256", "checkpoint and source provenance",
                          "labels/features recomputed from raw states", "finite kinematics and regular timestamps",
                          "identical initial state for each matched scenario", "no scenario group crosses splits"]}
    write(output / "metrics.json", metrics)
    # Portable inference snapshot: the receiving PC need not have local training artifacts.
    with zipfile.ZipFile(output / "reproduction.zip", "w", zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for relative, expected_hash in config["source_hashes"].items():
            path = args.job / "source" / relative
            if digest(path) != expected_hash:
                raise ValueError(f"Snapshot changed: {relative}")
            archive.write(path, f"source/{relative}")
        for pilot in config["pilots"]:
            for name in ("best_model.pt", "config.json", "manifest.json"):
                relative = f"training/{pilot['pilot_id']}/{name}"
                archive.write(args.job / relative, relative)
        archive.writestr("experiment.json", json.dumps({
            "source_hashes": config["source_hashes"], "source_experiment": args.job.name,
            "note": "Inference reproduction subset; no optimizer state or training history."}, indent=2))
    with zipfile.ZipFile(output / "reproduction.zip") as archive:
        if archive.testzip() is not None:
            raise ValueError("Reproduction archive failed CRC validation")
    files = {p.relative_to(output).as_posix(): {"bytes": p.stat().st_size, "sha256": digest(p)}
             for p in sorted(output.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    if any(v["bytes"] >= 1_000_000_000 for v in files.values()):
        raise ValueError("Shard exceeds conservative 1 GB publication cap")
    write(output / "manifest.json", {
        "name": "tactical-forecasting-v1", "complete": True, "episodes": expected,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": files, "compressed_bytes": sum(v["bytes"] for v in files.values()),
        "software": {p: importlib.metadata.version(p) for p in
                     ("numpy", "pandas", "pyarrow", "torch", "jsbsim", "gymnasium")},
        "source_experiment": args.job.name, "collector_commit": subprocess.check_output(
            ["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "collector_file_hashes": config["collector_hashes"],
        "limitations": ["Idealised perfect tracks, no sensor noise or occlusion.",
                        "300 matched scenarios produce 3000 encounters, not 3000 independent geometries.",
                        "Scripted tactical F16 targets only; no real-world generalisation claim.",
                        "Native spawn is approximately head-on with fixed initial speed; nominal scenario bearings/headings/speeds are not applied by this adapter.",
                        "Labels describe measured motion, not the target controller's intent or learned pilot skill IDs.",
                        "Initial one-second history and censored forecast horizons are invalid, not straight/steady examples.",
                        "Deterministic action selection is not a guarantee of bitwise reproducibility across hardware."]})
    print(json.dumps(metrics, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("collect", "package", "verify"))
    parser.add_argument("--job", type=Path, default=DEFAULT_JOB)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes-per-pilot", type=int, default=300)
    parser.add_argument("--seed", type=int, default=3000000)
    parser.add_argument("--split-seed", type=int, default=3100000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--limit", type=int, help="Smoke collection of the first N pilot/scenario pairs")
    args = parser.parse_args()
    if args.command == "verify":
        manifest = read(args.output / "manifest.json")
        for relative, expected in manifest["files"].items():
            path = args.output / relative
            if path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
                raise ValueError(f"Publication checksum mismatch: {relative}")
        print(f"Verified {len(manifest['files'])} files, {manifest['episodes']} episodes")
        return
    config = prepare(args)
    if args.command == "collect":
        collect(args, config)
    else:
        package(args, config)


if __name__ == "__main__":
    main()
