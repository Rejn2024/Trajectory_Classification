"""Run a matched PPO experiment without modifying the baseline batch.

Prepare an immutable source snapshot first, then launch its copy of this script:
    python scripts/run_pilot_experiment.py --prepare BASELINE_DIRECTORY
    python JOB/source/scripts/run_pilot_experiment.py --run-job JOB

Training preserves the baseline's coefficients, seeds and settings, changing only
the selected intervention (plus output/MLflow destinations). Both populations face one
predeclared fresh confirmation suite. A JSON and HTML comparison are saved automatically.
"""

import argparse
import hashlib
import html
import json
import math
import os
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf8")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def baseline_inventory(baseline, pilots):
    paths = [baseline / "pilots.json"]
    paths += [baseline / p["pilot_id"] / name for p in pilots for name in (
        "config.json", "manifest.json", "best_model.pt", "training_metrics.jsonl",
        "test_episodes.json",
    )]
    return {p.relative_to(baseline).as_posix(): digest(p) for p in paths}


def prepare(args):
    root = Path(__file__).resolve().parents[1]
    baseline = args.prepare.resolve()
    batch = read(baseline / "pilots.json")
    pilots = batch["pilots"]
    if not pilots or any(p["status"] != "completed" for p in pilots):
        raise ValueError("the baseline must be a completed population")
    configs = [read(baseline / p["pilot_id"] / "config.json") for p in pilots]
    rates = {c["training"]["learning_rate"] for c in configs}
    if len(rates) != 1:
        raise ValueError("baseline learning rates must match")
    baseline_rate = next(iter(rates))
    baseline_selected = configs[0]["training"].get("selected_skill_parameters", False)
    if args.intervention == "selected_skill_parameters":
        if baseline_selected:
            raise ValueError("the baseline already uses selected-skill parameters")
        if args.learning_rate is not None and args.learning_rate != baseline_rate:
            raise ValueError("selected-skill experiment must preserve the baseline learning rate")
        learning_rate = baseline_rate
        selected_parameters = True
        label = "Selected-skill parameters"
        change = "Only selected-skill log-probabilities and expected parameter entropy changed."
        suffix = "selected_skill_parameters"
    else:
        learning_rate = 3e-5 if args.learning_rate is None else args.learning_rate
        if learning_rate == baseline_rate:
            raise ValueError("the experiment learning rate must differ from the baseline")
        selected_parameters = baseline_selected
        label = "Learning rate experiment"
        change = f"Only the learning rate changed: {baseline_rate} to {learning_rate}."
        suffix = f"lr{learning_rate:g}"
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning rate must be finite and positive")
    if args.confirmation_scenarios < 3:
        raise ValueError("confirmation requires at least three scenarios")
    end = args.confirmation_seed + args.confirmation_scenarios
    if not 0 <= args.confirmation_seed < end <= 2**32:
        raise ValueError("confirmation seed range is invalid")
    settings = configs[0]["training"]
    # Previous confirmation suites have already informed later experiments.
    # Reject reuse anywhere in this baseline's recorded experiment ancestry.
    ancestor = baseline
    seen = set()
    previous_suites = []
    while (ancestor.parent / "experiment.json").exists():
        ancestor_plan_path = (ancestor.parent / "experiment.json").resolve()
        if ancestor_plan_path in seen:
            raise ValueError("cyclic experiment ancestry")
        seen.add(ancestor_plan_path)
        ancestor_plan = read(ancestor_plan_path)
        previous_suites.append(ancestor_plan["confirmation"])
        ancestor = Path(ancestor_plan["baseline"])
    if any(max(args.confirmation_seed, suite["seed"]) < min(
            end, suite["seed"] + suite["scenarios"]) for suite in previous_suites):
        raise ValueError("confirmation overlaps a previously inspected confirmation suite")
    for pilot, config in zip(pilots, configs):
        training = config["training"]
        changed = {key for key in settings if training.get(key) != settings[key]}
        if changed - {"seed", "output_dir"}:
            raise ValueError(f"baseline training settings differ for {pilot['pilot_id']}: {changed}")
        if config["reward"] != pilot["reward"]:
            raise ValueError("baseline reward metadata is inconsistent")
        if digest(baseline / pilot["checkpoint"]) != pilot["checkpoint_sha256"]:
            raise ValueError("baseline checkpoint hash does not match")
        periods = [
            (training["seed"], training["scenarios_per_epoch"] * (
                training["epochs"] if training["resample_training_scenarios"] else 1)),
            (training["evaluation_seed"], training["evaluation_scenarios_per_epoch"]),
            (training["test_seed"], training["test_scenarios_per_pilot"]),
        ]
        if any(max(args.confirmation_seed, seed) < min(end, seed + count)
               for seed, count in periods):
            raise ValueError("confirmation overlaps a previous training/evaluation seed range")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    job = root / "artifacts/rl_pilot_experiments" / f"{stamp}_{suffix}"
    job.mkdir(parents=True, exist_ok=False)
    source = job / "source"
    ignored = shutil.ignore_patterns("__pycache__", "*.pyc", "build", "install", ".git")
    for name in ("bvr_behavior_prediction", "bvr_sim_source/bvr_sim"):
        shutil.copytree(root / name, source / name, ignore=ignored)
    (source / "scripts").mkdir()
    shutil.copy2(__file__, source / "scripts" / Path(__file__).name)
    (source / "notebooks").mkdir()
    shutil.copy2(root / "notebooks/05_train_multiple_pilots.ipynb", source / "notebooks")
    source_hashes = {p.relative_to(source).as_posix(): digest(p)
                     for p in source.rglob("*") if p.is_file()}
    previous_seconds = sum(
        json.loads(line)["epoch_seconds"]
        for pilot in pilots
        for line in (baseline / pilot["pilot_id"] / "training_metrics.jsonl").read_text().splitlines()
    )
    plan = {
        "created_utc": now(), "repository": str(root), "job": str(job),
        "source": str(source), "baseline": str(baseline),
        "training_output": str(job / "training"),
        "intervention": args.intervention, "baseline_learning_rate": baseline_rate,
        "learning_rate": learning_rate, "pilot_count": len(pilots),
        "baseline_selected_skill_parameters": baseline_selected,
        "selected_skill_parameters": selected_parameters,
        "experiment_label": label, "change_description": change,
        "epochs_per_pilot": settings["epochs"], "device": "cuda",
        "torch_threads": min(8, settings["simulator_workers"]),
        "simulator_workers": settings["simulator_workers"],
        "baseline_training_hours": previous_seconds / 3600,
        "confirmation": {"seed": args.confirmation_seed, "scenarios": args.confirmation_scenarios},
        "primary_measure": "mean clean win rate across matched reward profiles",
        "existing_test_role": "development comparison; these cases have already been inspected",
        "confirmation_role": "predeclared fresh paired evaluation; never used to select checkpoints",
        "baseline_hashes": baseline_inventory(baseline, pilots), "source_hashes": source_hashes,
    }
    write(job / "experiment.json", plan)
    write(job / "status.json", {"status": "prepared", "updated_utc": now()})
    print(json.dumps({key: value for key, value in plan.items()
                      if key not in ("source_hashes", "baseline_hashes")}, indent=2))


def make_environment_factory(config):
    from bvr_behavior_prediction.rl.bvr_jsbsim_backend import BVRJSBSimBackend
    from bvr_behavior_prediction.rl.environment import BluePilotEnvironment

    def backend(scenario, recording_path=None):
        return BVRJSBSimBackend(scenario, recording_path=recording_path,
                               log_dir=config.output_dir / "simulator_logs")

    def environment(scenario, recording_path=None):
        return BluePilotEnvironment(
            backend, scenario, planning_horizon_s=config.planning_horizon_s,
            episode_duration_s=config.episode_duration_s,
            history_duration_s=config.history_duration_s,
            sample_interval_s=config.sample_interval_s, recording_path=recording_path,
        )

    return environment


def prepare_recovery(previous_job):
    """Freeze an operational repair; preserve the failed attempt and its study plan."""
    root = Path(__file__).resolve().parents[1]
    previous_job = previous_job.resolve()
    old_plan = read(previous_job / "experiment.json")
    status = read(previous_job / "status.json")
    if status["status"] != "failed":
        raise ValueError("recovery requires a failed job; do not duplicate an active run")
    old_source = Path(old_plan["source"])
    for name, expected in old_plan["source_hashes"].items():
        if digest(old_source / name) != expected:
            raise ValueError(f"original source snapshot changed: {name}")
    baseline = Path(old_plan["baseline"])
    if baseline_inventory(baseline, read(baseline / "pilots.json")["pilots"]) != old_plan["baseline_hashes"]:
        raise ValueError("baseline changed since the original experiment")
    previous_output = Path(old_plan["training_output"])
    pilots = read(previous_output / "pilots.json")["pilots"]
    completed = [p for p in pilots if p["status"] == "completed"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    job = root / "artifacts/rl_pilot_experiments" / f"{stamp}_{old_plan['intervention']}_recovery"
    source = job / "source"
    shutil.copytree(old_source, source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # Keep all simulation, model, loss, config and notebook code from the original
    # snapshot. Only the collector's transport recovery and batch reuse change.
    repair_files = (
        "bvr_behavior_prediction/rl/simulation_workers.py",
        "bvr_behavior_prediction/rl/trainer.py",
        "bvr_behavior_prediction/rl/multi_pilot.py",
        "scripts/run_pilot_experiment.py",
    )
    for name in repair_files:
        shutil.copy2(root / name, source / name)
    recovery = {
        "previous_job": str(previous_job), "previous_training": str(previous_output),
        "previous_plan_sha256": digest(previous_job / "experiment.json"),
        "failure": status.get("error"),
        "reused_pilots": [p["pilot_id"] for p in completed],
        "restart_from_scratch": [p["pilot_id"] for p in pilots if p["status"] != "completed"],
        "previous_completed_hashes": baseline_inventory(previous_output, completed),
        "operational_repair_files": list(repair_files),
        "note": "Completed pilots copied byte-for-byte; unfinished pilots restart at their original seeds. "
                "No optimiser resume or change to rewards, PPO settings or evaluation suites.",
    }
    plan = {**old_plan, "created_utc": now(), "job": str(job), "source": str(source),
            "training_output": str(job / "training"), "recovery": recovery,
            "source_hashes": {p.relative_to(source).as_posix(): digest(p)
                              for p in source.rglob("*") if p.is_file()}}
    write(job / "experiment.json", plan)
    write(job / "status.json", {"status": "prepared", "updated_utc": now()})
    print(json.dumps({"job": str(job), "recovery": {
        k: v for k, v in recovery.items() if k != "previous_completed_hashes"
    }}, indent=2))
    return job


def paired_comparison(old, new):
    """Match every pilot and case; bootstrap scenarios jointly across pilots."""
    import numpy as np

    if old.keys() != new.keys():
        raise ValueError("comparison populations differ")
    rows, differences = [], []
    for pilot_id in old:
        before, after = old[pilot_id], new[pilot_id]
        key = lambda row: (row["seed"], row["scenario"])
        if [key(r) for r in before["episodes"]] != [key(r) for r in after["episodes"]]:
            raise ValueError(f"unmatched scenarios for {pilot_id}")
        a = np.asarray([r["outcome"] == "surviving_elimination" for r in before["episodes"]])
        b = np.asarray([r["outcome"] == "surviving_elimination" for r in after["episodes"]])
        differences.append(b.astype(float) - a.astype(float))
        rows.append({
            "pilot_id": pilot_id, "scenarios": len(a),
            "baseline_clean_win_rate": float(a.mean()), "experiment_clean_win_rate": float(b.mean()),
            "difference_percentage_points": float(100 * (b.mean() - a.mean())),
            "gained_clean_wins": int((b & ~a).sum()), "lost_clean_wins": int((a & ~b).sum()),
            "baseline_survival_rate": float(np.mean([r["survived"] for r in before["episodes"]])),
            "experiment_survival_rate": float(np.mean([r["survived"] for r in after["episodes"]])),
            "baseline_mutual_destruction_rate": float(np.mean([
                r["outcome"] == "mutual_destruction" for r in before["episodes"]
            ])),
            "experiment_mutual_destruction_rate": float(np.mean([
                r["outcome"] == "mutual_destruction" for r in after["episodes"]
            ])),
        })
    per_scenario_difference = np.mean(differences, axis=0)
    rng = np.random.default_rng(8128)
    indices = rng.integers(0, len(per_scenario_difference), size=(2000, len(per_scenario_difference)))
    interval = np.quantile(per_scenario_difference[indices].mean(axis=1), [0.025, 0.975]) * 100
    return {
        "pilots": rows,
        "baseline_mean_clean_win_rate": float(np.mean([r["baseline_clean_win_rate"] for r in rows])),
        "experiment_mean_clean_win_rate": float(np.mean([r["experiment_clean_win_rate"] for r in rows])),
        "difference_percentage_points": float(100 * per_scenario_difference.mean()),
        "paired_scenario_bootstrap_ci95_percentage_points": interval.tolist(),
        "uncertainty_note": "Scenarios resampled jointly across pilots; training-seed variation is not measured.",
    }


def save_report(job, comparison):
    write(job / "comparison.json", comparison)
    sections = []
    for key, title in (("existing_test", "Previously inspected comparison cases"),
                       ("fresh_confirmation", "Predeclared fresh confirmation cases")):
        result = comparison.get(key)
        if result is None:
            continue
        rows = "".join(
            f"<tr><td>{html.escape(r['pilot_id'])}</td>"
            f"<td>{r['baseline_clean_win_rate']:.0%}</td><td>{r['experiment_clean_win_rate']:.0%}</td>"
            f"<td>{r['difference_percentage_points']:+.1f}</td>"
            f"<td>{r['baseline_survival_rate']:.0%} / {r['experiment_survival_rate']:.0%}</td>"
            f"<td>{r['baseline_mutual_destruction_rate']:.0%} / {r['experiment_mutual_destruction_rate']:.0%}</td></tr>"
            for r in result["pilots"]
        )
        sections.append(
            f"<h2>{title}</h2><p>Population mean clean win: "
            f"{result['baseline_mean_clean_win_rate']:.1%} to {result['experiment_mean_clean_win_rate']:.1%}; "
            f"change {result['difference_percentage_points']:+.1f} percentage points.</p>"
            "<table><tr><th>Pilot</th><th>Baseline win</th><th>New win</th><th>Change (pp)</th>"
            "<th>Survival: old / new</th><th>Mutual destruction: old / new</th></tr>"
            f"{rows}</table><p>{html.escape(result['uncertainty_note'])}</p>"
            f"<p>Paired scenario bootstrap interval (percentage points): "
            f"{result['paired_scenario_bootstrap_ci95_percentage_points']}</p>"
        )
    document = (
        "<!doctype html><meta charset='utf-8'><title>PPO experiment comparison</title>"
        "<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px}"
        "table{border-collapse:collapse}td,th{padding:9px;border:1px solid #ccc;text-align:right}"
        "th:first-child,td:first-child{text-align:left}</style>"
        f"<h1>{html.escape(comparison['experiment_label'])}</h1>"
        f"<p>Status: {html.escape(comparison['status'])}. "
        f"{html.escape(comparison['change_description'])} "
        "All pilots trained from scratch, with matched rewards and scenario seeds.</p>"
        + "".join(sections)
        + "<p>Lower training loss is not the outcome measure. The baseline and all new models "
        "are retained regardless of the result. One training seed cannot establish robust improvement.</p>"
    )
    (job / "comparison.html").write_text(document, encoding="utf8")


def run(job):
    job = job.resolve()
    plan = read(job / "experiment.json")
    source = Path(plan["source"])
    if Path(__file__).resolve() != source / "scripts" / Path(__file__).name:
        raise ValueError("run the frozen script inside JOB/source/scripts")
    for name, expected in plan["source_hashes"].items():
        if digest(source / name) != expected:
            raise ValueError(f"source snapshot changed: {name}")
    baseline = Path(plan["baseline"])
    batch = read(baseline / "pilots.json")
    if baseline_inventory(baseline, batch["pilots"]) != plan["baseline_hashes"]:
        raise ValueError("baseline changed since experiment preparation")
    recovery = plan.get("recovery")
    if recovery:
        previous_job = Path(recovery["previous_job"])
        if digest(previous_job / "experiment.json") != recovery["previous_plan_sha256"]:
            raise ValueError("previous experiment plan changed since recovery preparation")
        previous_output = Path(recovery["previous_training"])
        for name, expected in recovery["previous_completed_hashes"].items():
            if digest(previous_output / name) != expected:
                raise ValueError(f"previous completed result changed: {name}")
    for path in (source, source / "bvr_sim_source"):
        sys.path.insert(0, str(path))
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    import torch
    from dataclasses import fields, replace
    from bvr_behavior_prediction.rl.config import PilotTrainingConfig, RewardWeights
    from bvr_behavior_prediction.rl.evaluation import summarize_test_episodes
    from bvr_behavior_prediction.rl.multi_pilot import PilotSpec, load_trained_pilot, train_pilots
    from bvr_behavior_prediction.rl.reward import combat_reward_definition
    from bvr_behavior_prediction.rl.scenarios import ScenarioSampler
    from bvr_behavior_prediction.rl.trainer import PPOTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing an unexpected CPU-only experiment")
    torch.set_num_threads(plan["torch_threads"])
    config_fields = {f.name for f in fields(PilotTrainingConfig)}

    def config_for(path, output):
        settings = read(path / "config.json")["training"]
        kwargs = {k: v for k, v in settings.items() if k in config_fields}
        kwargs["reward"] = RewardWeights(**kwargs["reward"])
        kwargs["output_dir"] = output
        return PilotTrainingConfig(**kwargs)

    def definition(metadata):
        if metadata["version"] != "2" or set(metadata["parameters"]) != {"weights", "safe_altitude_m"}:
            raise ValueError("this runner supports the recorded CombatReward v2 recipe only")
        return combat_reward_definition(metadata["name"],
                                        weights=RewardWeights(**metadata["parameters"]["weights"]),
                                        safe_altitude_m=metadata["parameters"]["safe_altitude_m"])

    # Exclusive marker prevents accidental duplicate training against this job.
    with (job / "started.json").open("x", encoding="utf8") as stream:
        json.dump({"pid": os.getpid(), "started_utc": now()}, stream)
    status = {"status": "training", "pid": os.getpid(), "started_utc": now()}
    write(job / "status.json", status)
    try:
        output = Path(plan["training_output"])
        config = replace(config_for(baseline / batch["pilots"][0]["pilot_id"], output),
                         learning_rate=plan["learning_rate"],
                         selected_skill_parameters=plan["selected_skill_parameters"],
                         mlflow_experiment=f"jsbsim-combat-{plan['intervention']}",
                         mlflow_tracking_uri=f"sqlite:///{(job / 'mlflow.db').as_posix()}")
        pilots = [PilotSpec(p["pilot_id"], p["seed"], definition(p["reward"])) for p in batch["pilots"]]
        benchmark = definition(batch["benchmark_reward"])
        config_diffs = {}
        for pilot in pilots:
            original = config_for(baseline / pilot.pilot_id, output).as_dict()
            experiment = replace(config, seed=pilot.seed).as_dict()
            changes = {k: {"old": original[k], "new": experiment[k]}
                       for k in original if original[k] != experiment[k]}
            changed_setting = ("selected_skill_parameters" if plan["intervention"] ==
                               "selected_skill_parameters" else "learning_rate")
            if set(changes) != {changed_setting, "mlflow_experiment", "mlflow_tracking_uri"}:
                raise ValueError(f"unexpected configuration difference: {changes}")
            config_diffs[pilot.pilot_id] = changes
        write(job / "configuration_differences.json", config_diffs)
        print(f"Starting {len(pilots)} pilots, {config.epochs} epochs each: {plan['change_description']}", flush=True)
        results = train_pilots(
            pilots, config, make_environment_factory,
            read(baseline / pilots[0].pilot_id / "config.json")["observation_size"],
            device=plan["device"], benchmark_reward=benchmark,
            population_metadata={**batch["population_metadata"], "experiment": {
                "intervention": plan["intervention"], "baseline": str(baseline),
                "baseline_learning_rate": plan["baseline_learning_rate"],
                "learning_rate": plan["learning_rate"], "source_snapshot": str(source),
                "selected_skill_parameters": plan["selected_skill_parameters"],
            }},
            reuse_completed_from=Path(recovery["previous_training"]) if recovery else None,
        )
        old_test = {p.pilot_id: read(baseline / p.pilot_id / "test_episodes.json") for p in pilots}
        new_test = {p.pilot_id: read(output / p.pilot_id / "test_episodes.json") for p in pilots}
        comparison = {
            "status": "confirmation in progress", "baseline": str(baseline), "experiment": str(output),
            "baseline_learning_rate": plan["baseline_learning_rate"], "learning_rate": plan["learning_rate"],
            "intervention": plan["intervention"], "experiment_label": plan["experiment_label"],
            "change_description": plan["change_description"],
            "existing_test": paired_comparison(old_test, new_test),
        }
        if recovery:
            comparison["recovery"] = {k: recovery[k] for k in (
                "previous_job", "reused_pilots", "restart_from_scratch", "note",
            )}
        save_report(job, comparison)
        suite = plan["confirmation"]
        scenarios = ScenarioSampler(suite["seed"]).sample_batch(suite["scenarios"])
        seeds = tuple(range(suite["seed"], suite["seed"] + suite["scenarios"]))
        confirmations = {"baseline": {}, "experiment": {}}
        for population, root in (("baseline", baseline), ("experiment", output)):
            for pilot in pilots:
                status.update(status="confirmation", population=population,
                              pilot_id=pilot.pilot_id, updated_utc=now())
                write(job / "status.json", status)
                print(f"Fresh confirmation: {population}/{pilot.pilot_id}", flush=True)
                directory = job / "confirmation" / population / pilot.pilot_id
                run_config = config_for(root / pilot.pilot_id, directory)
                model_config = read(root / pilot.pilot_id / "config.json")
                trainer = PPOTrainer(make_environment_factory(run_config), model_config["observation_size"],
                                     run_config, plan["device"], definition(pilot.reward.as_dict()))
                try:
                    trainer.pilot.load_state_dict(load_trained_pilot(root / pilot.pilot_id, plan["device"]).state_dict())
                    episodes = trainer.evaluate_scenarios(scenarios, seeds, reward_definition=benchmark)
                    summary, records = summarize_test_episodes(episodes, scenarios, seeds, trainer.pilot.skill_names)
                    data = {"summary": summary, "episodes": records,
                            "checkpoint_sha256": digest(root / pilot.pilot_id / "best_model.pt")}
                    write(directory / "evaluation.json", data)
                    confirmations[population][pilot.pilot_id] = data
                finally:
                    trainer.close()
                    del trainer
                    torch.cuda.empty_cache()
        comparison["fresh_confirmation"] = paired_comparison(confirmations["baseline"], confirmations["experiment"])
        comparison["confirmation_suite"] = suite
        if baseline_inventory(baseline, batch["pilots"]) != plan["baseline_hashes"]:
            raise RuntimeError("baseline files changed during the experiment")
        comparison.update(status="completed", completed_utc=now(), baseline_unchanged=True)
        save_report(job, comparison)
        status.update(status="completed", updated_utc=now(), trained_pilots=len(results),
                      comparison=str(job / "comparison.html"), baseline_unchanged=True)
        write(job / "status.json", status)
        print(json.dumps({k: v for k, v in comparison.items() if k not in ("existing_test", "fresh_confirmation")}), flush=True)
    except BaseException as error:
        status.update(status="failed", updated_utc=now(), error=f"{type(error).__name__}: {error}")
        write(job / "status.json", status)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", type=Path)
    mode.add_argument("--run-job", type=Path)
    mode.add_argument("--recover", type=Path, help="prepare a fresh job reusing completed pilots from a failed job")
    parser.add_argument("--intervention", choices=("learning_rate_only", "selected_skill_parameters"),
                        default="learning_rate_only")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--confirmation-scenarios", type=int, default=100)
    parser.add_argument("--confirmation-seed", type=int)
    options = parser.parse_args()
    if options.prepare:
        if options.confirmation_seed is None:
            options.confirmation_seed = (
                620000 if options.intervention == "selected_skill_parameters" else 420000
            )
        prepare(options)
    elif options.recover:
        prepare_recovery(options.recover)
    else:
        run(options.run_job)
