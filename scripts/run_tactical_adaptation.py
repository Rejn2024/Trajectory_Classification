"""Evaluate a frozen population against tactical, then fine-tune separate copies."""

import argparse
import csv
import html
import importlib.util
import json
import os
import shutil
import sys
import traceback
from dataclasses import fields, replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "bvr_sim_source"):
    sys.path.insert(0, str(directory))

_spec = importlib.util.spec_from_file_location(
    "pilot_experiment_helpers", Path(__file__).with_name("run_pilot_experiment.py")
)
_shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared)
baseline_inventory, definition = _shared.baseline_inventory, _shared.definition
digest, paired_comparison = _shared.digest, _shared.paired_comparison
from bvr_behavior_prediction.rl.config import PilotTrainingConfig, RewardWeights
from bvr_behavior_prediction.rl.multi_pilot import _write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, value)


def now():
    return datetime.now(timezone.utc).isoformat()


def config_from(directory, output, **overrides):
    saved = read(Path(directory) / "config.json")["training"]
    known = {f.name for f in fields(PilotTrainingConfig)}
    settings = {k: v for k, v in saved.items() if k in known}
    settings["reward"] = RewardWeights(**settings["reward"])
    return PilotTrainingConfig(**{**settings, "output_dir": Path(output), **overrides})


def check_seed_ranges(configs, epochs, assessment, validation_seed, test_seed, reserved=(),
                      training_seed_offset=1_000_000):
    """Different pilots intentionally share cases; different data roles must not."""
    for config in configs:
        current = [
            ("training", config.seed + training_seed_offset, epochs * config.scenarios_per_epoch),
            ("validation", validation_seed, config.evaluation_scenarios_per_epoch),
            ("assessment", assessment["seed"], assessment["scenarios"]),
            ("test", test_seed, config.test_scenarios_per_pilot),
        ]
        previous = [
            ("parent training", config.seed, config.training_seed_count),
            ("parent validation", config.resolved_evaluation_seed, config.evaluation_scenarios_per_epoch),
            ("parent test", config.test_seed, config.test_scenarios_per_pilot),
        ] + [("previous confirmation", x["seed"], x["scenarios"]) for x in reserved]
        for i, (name, seed, count) in enumerate(current):
            if not isinstance(seed, int) or not 0 <= seed < seed + count <= 2**32:
                raise ValueError(f"invalid {name} seed range")
            for other, start, size in current[:i] + previous:
                if size and max(seed, start) < min(seed + count, start + size):
                    raise ValueError(f"{name} overlaps {other}")


def prepare(baseline, epochs=150, assessment_scenarios=300, job_root=None, seed_offset=1_000_000):
    baseline = Path(baseline).resolve()
    batch = read(baseline / "pilots.json")
    pilots = batch["pilots"]
    if not pilots or any(p["status"] != "completed" for p in pilots):
        raise ValueError("tactical adaptation requires a completed source population")
    if epochs < 1 or assessment_scenarios < 3:
        raise ValueError("provide positive epochs and at least three assessment scenarios")
    configs = [config_from(baseline / p["pilot_id"], baseline) for p in pilots]
    reference = configs[0].as_dict()
    for pilot, config in zip(pilots, configs):
        if not config.selected_skill_parameters or config.skill_conditioned_parameters:
            raise ValueError("use the shared-head selected-skill population")
        metadata = read(baseline / pilot["pilot_id"] / "config.json")
        if pilot["reward"] != metadata["reward"] or pilot["reward"]["version"] != "2":
            raise ValueError("use the original event rewards without the clean-win bonus")
        if digest(baseline / pilot["checkpoint"]) != pilot["checkpoint_sha256"]:
            raise ValueError("source checkpoint hash differs from manifest")
        differences = {k for k, v in config.as_dict().items() if reference[k] != v}
        if differences - {"seed", "output_dir", "mlflow_tracking_uri"}:
            raise ValueError(f"source pilot settings differ: {differences}")
        if not config.resample_training_scenarios or not config.test_scenarios_per_pilot:
            raise ValueError("adaptation requires fresh training batches and post-selection tests")
    assessment = {"seed": seed_offset + 120_000, "scenarios": assessment_scenarios}
    validation_seed, test_seed = seed_offset + 100_000, seed_offset + 220_000
    reserved = []
    registry = ROOT / "configs/pilot_experiments.json"
    if registry.exists():
        for entry in read(registry)["experiments"].values():
            path = ROOT / entry["job"] / "experiment.json"
            if path.exists() and "confirmation" in read(path):
                reserved.append(read(path)["confirmation"])
    check_seed_ranges(configs, epochs, assessment, validation_seed, test_seed, reserved, seed_offset)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    job = Path(job_root or ROOT / "artifacts/rl_pilot_experiments").resolve() / f"{stamp}_tactical"
    job.mkdir(parents=True, exist_ok=False)
    source = job / "source"
    ignored = shutil.ignore_patterns("__pycache__", "*.pyc", "build", "install", ".git")
    for name in ("bvr_behavior_prediction", "bvr_sim_source/bvr_sim"):
        shutil.copytree(ROOT / name, source / name, ignore=ignored)
    (source / "scripts").mkdir()
    for name in ("run_tactical_adaptation.py", "run_pilot_experiment.py"):
        shutil.copy2(ROOT / "scripts" / name, source / "scripts" / name)
    plan = {
        "created_utc": now(), "job": str(job), "source": str(source),
        "baseline": str(baseline), "training_output": str(job / "training"),
        "intervention": "tactical_adaptation", "opponent_type": "tactical",
        "experiment_label": "Tactical adaptation", "pilot_count": len(pilots),
        "epochs_per_pilot": epochs, "learning_rate": configs[0].learning_rate,
        "device": "cuda", "torch_threads": min(8, configs[0].resolved_simulator_workers),
        "simulator_workers": configs[0].resolved_simulator_workers,
        "assessment": assessment, "validation_seed": validation_seed,
        "confirmation": {"seed": test_seed, "scenarios": configs[0].test_scenarios_per_pilot},
        "training_seed_offset": seed_offset,
        "initialization": "Each pilot loads its own policy and critic; fresh Adam optimizer.",
        "checkpoint_selection": "Own tactical validation reward; initial checkpoint included.",
        "assessment_role": "Shared development benchmark before and after adaptation; never used for checkpoint selection.",
        "confirmation_role": "Separate matched post-selection cases; never used for training or selection.",
        "reward_intervention": None, "baseline_hashes": baseline_inventory(baseline, pilots),
        "source_hashes": {p.relative_to(source).as_posix(): digest(p)
                          for p in source.rglob("*") if p.is_file()},
    }
    write(job / "experiment.json", plan)
    write(job / "status.json", {"status": "prepared", "updated_utc": now()})
    print(json.dumps({k: v for k, v in plan.items() if k not in ("source_hashes", "baseline_hashes")}, indent=2))
    return job


def make_environment_factory(config, opponent_type="tactical"):
    from bvr_behavior_prediction.rl.bvr_jsbsim_backend import BVRJSBSimBackend
    from bvr_behavior_prediction.rl.environment import BluePilotEnvironment

    def backend(scenario, recording_path=None):
        return BVRJSBSimBackend(scenario, recording_path=recording_path,
                               log_dir=config.output_dir / "simulator_logs",
                               opponent_type=opponent_type)

    def environment(scenario, recording_path=None):
        return BluePilotEnvironment(
            backend, scenario, config.planning_horizon_s, config.episode_duration_s,
            recording_path, history_duration_s=config.history_duration_s,
            sample_interval_s=config.sample_interval_s,
        )
    return environment


def save_progress_report(job):
    """Publish comparable tactical outcomes only; never mix in simple-opponent scores."""
    plan = read(job / "experiment.json")
    rows = []
    for population, directory in (("Before fine-tuning", "baseline_assessment"),
                                  ("After fine-tuning", "adapted_assessment")):
        for file in sorted((job / directory).glob("*/evaluation.json")):
            data = read(file)
            rows.append({"population": population, "pilot": file.parent.name,
                         **{key: data["summary"][key] for key in (
                             "scenarios", "clean_win_rate", "survival_rate",
                             "mutual_destruction_rate", "loss_without_elimination_rate", "outcome_score")}})
    if rows:
        with (job / "tactical_results.csv").open("w", newline="", encoding="utf8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    headings = ("Population", "Pilot", "Scenarios", "Clean wins", "Survival", "Mutual loss", "Own loss/no elimination", "Outcome score")
    table = []
    for row in rows:
        cells = [row["population"], row["pilot"], str(row["scenarios"])]
        cells += [f"{100 * row[k]:.1f}%" for k in ("clean_win_rate", "survival_rate", "mutual_destruction_rate", "loss_without_elimination_rate")]
        cells.append(f"{row['outcome_score']:.1f}")
        table.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>")
    status = read(job / "status.json")
    progress = []
    if (job / "training/pilots.json").exists():
        for p in read(job / "training/pilots.json")["pilots"]:
            file = job / "training" / p["pilot_id"] / "training_metrics.jsonl"
            lines = file.read_text().splitlines() if file.exists() else []
            epoch = json.loads(lines[-1])["epoch"] if lines else 0
            progress.append(f"{p['pilot_id']}: {p['status']}, {epoch}/{plan['epochs_per_pilot']} epochs")
    doc = ("<!doctype html><meta charset='utf-8'><title>Tactical adaptation</title>"
           "<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:16px}"
           "table{border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:right}"
           "td:first-child,th:first-child{text-align:left}</style><h1>Tactical adaptation</h1>"
           f"<p>Updated {html.escape(now())}. Status: {html.escape(status['status'])}.</p>"
           f"<p>{plan['pilot_count']} separate pilots, {plan['epochs_per_pilot']} additional epochs each. "
           "Original rewards and architecture; policy and critic initialized from the selected-skill population.</p>"
           f"<p>All rows face the tactical opponent on the same {plan['assessment']['scenarios']} "
           "development scenarios. This table does not control checkpoint selection. "
           "Partial results are not a population-wide improvement claim.</p><table><thead><tr>"
           + "".join(f"<th>{h}</th>" for h in headings) + "</tr></thead><tbody>"
           + "".join(table) + "</tbody></table><p>" + "<br>".join(progress) + "</p>"
           "<p>During training, live epoch progress is in each pilot's training_metrics.jsonl and stderr.log. "
           "This report updates at evaluation-stage boundaries.</p>")
    (job / "tactical_report.html").write_text(doc, encoding="utf8")


def evaluate_population(job, population, directory, suite, stage, status):
    import torch

    from bvr_behavior_prediction.rl.evaluation import summarize_test_episodes
    from bvr_behavior_prediction.rl.multi_pilot import load_trained_pilot
    from bvr_behavior_prediction.rl.scenarios import ScenarioSampler
    from bvr_behavior_prediction.rl.trainer import PPOTrainer

    plan = read(job / "experiment.json")
    batch = read(Path(plan["baseline"]) / "pilots.json")
    benchmark = definition(batch["benchmark_reward"])
    scenarios = ScenarioSampler(suite["seed"]).sample_batch(suite["scenarios"])
    seeds = tuple(range(suite["seed"], suite["seed"] + suite["scenarios"]))
    result = {}
    for record in batch["pilots"]:
        pilot_id = record["pilot_id"]
        status.update(status=stage, pilot_id=pilot_id, updated_utc=now())
        write(job / "status.json", status)
        print(f"{stage}: {pilot_id}, {len(scenarios)} tactical scenarios", flush=True)
        output = job / directory / pilot_id
        config = config_from(population / pilot_id, output)
        metadata = read(population / pilot_id / "config.json")
        trainer = PPOTrainer(make_environment_factory(config), metadata["observation_size"],
                             config, device=plan["device"], reward_definition=definition(metadata["reward"]))
        try:
            model = load_trained_pilot(population / pilot_id, plan["device"])
            trainer.pilot.load_state_dict(model.state_dict())
            del model
            episodes = trainer.evaluate_scenarios(scenarios, seeds, reward_definition=benchmark)
            if any(info.get("red_policy") != "tactical" for _, _, info in episodes):
                raise RuntimeError("evaluation did not use the tactical opponent")
            summary, records = summarize_test_episodes(episodes, scenarios, seeds, trainer.pilot.skill_names)
            summary["role"] = plan["assessment_role"] if "assessment" in directory else plan["confirmation_role"]
            data = {"summary": summary, "episodes": records, "opponent_type": "tactical",
                    "checkpoint_sha256": digest(population / pilot_id / "best_model.pt"),
                    "benchmark_reward": batch["benchmark_reward"], "suite": suite}
            write(output / "evaluation.json", data)
            result[pilot_id] = data
            print(f"{pilot_id}: clean wins {summary['clean_win_rate']:.1%}, survival {summary['survival_rate']:.1%}", flush=True)
            save_progress_report(job)
        finally:
            trainer.close()
            del trainer
            torch.cuda.empty_cache()
    return result


def run(job):
    job = Path(job).resolve()
    plan = read(job / "experiment.json")
    if Path(__file__).resolve() != Path(plan["source"]) / "scripts/run_tactical_adaptation.py":
        raise ValueError("run the frozen script in JOB/source/scripts")
    for name, expected in plan["source_hashes"].items():
        if digest(Path(plan["source"]) / name) != expected:
            raise ValueError(f"source snapshot changed: {name}")
    baseline = Path(plan["baseline"])
    batch = read(baseline / "pilots.json")
    if baseline_inventory(baseline, batch["pilots"]) != plan["baseline_hashes"]:
        raise ValueError("source population changed since preparation")
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    import torch

    from bvr_behavior_prediction.rl.multi_pilot import PilotSpec, train_pilots
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing unexpected CPU-only execution")
    torch.set_num_threads(plan["torch_threads"])
    with (job / "started.json").open("x", encoding="utf8") as stream:
        json.dump({"pid": os.getpid(), "started_utc": now()}, stream)
    status = {"status": "baseline_assessment", "pid": os.getpid(), "started_utc": now()}
    write(job / "status.json", status)
    try:
        before = evaluate_population(job, baseline, "baseline_assessment", plan["assessment"], "baseline_assessment", status)
        write(job / "baseline_assessment_complete.json", {"completed_utc": now(), "pilots": len(before), "suite": plan["assessment"]})
        output = Path(plan["training_output"])
        first = batch["pilots"][0]
        config = config_from(baseline / first["pilot_id"], output,
                             epochs=plan["epochs_per_pilot"], evaluation_seed=plan["validation_seed"],
                             test_seed=plan["confirmation"]["seed"],
                             mlflow_experiment="jsbsim-tactical-adaptation",
                             mlflow_tracking_uri=f"sqlite:///{(job / 'mlflow.db').as_posix()}")
        pilots = [PilotSpec(p["pilot_id"], p["seed"] + plan["training_seed_offset"], definition(p["reward"]))
                  for p in batch["pilots"]]
        differences = {}
        for pilot in pilots:
            old = config_from(baseline / pilot.pilot_id, output).as_dict()
            new = replace(config, seed=pilot.seed).as_dict()
            changes = {k: {"old": old[k], "new": new[k]} for k in old if old[k] != new[k]}
            if set(changes) - {"seed", "epochs", "evaluation_seed", "test_seed", "mlflow_experiment", "mlflow_tracking_uri"}:
                raise ValueError(f"unexpected training configuration changes: {changes}")
            differences[pilot.pilot_id] = changes
        write(job / "configuration_differences.json", differences)
        status.update(status="training", pilot_id=first["pilot_id"], updated_utc=now())
        write(job / "status.json", status)
        save_progress_report(job)
        train_pilots(pilots, config, make_environment_factory,
                     read(baseline / first["pilot_id"] / "config.json")["observation_size"],
                     device=plan["device"], benchmark_reward=definition(batch["benchmark_reward"]),
                     population_metadata={**batch.get("population_metadata", {}),
                                          "adaptation": {"opponent_type": "tactical", "parent": str(baseline),
                                                         "source": plan["source"], "initialization": plan["initialization"]}},
                     initialize_from=baseline)
        after = evaluate_population(job, output, "adapted_assessment", plan["assessment"], "adapted_assessment", status)
        old_test = evaluate_population(job, baseline, "baseline_confirmation", plan["confirmation"], "confirmation", status)
        new_test = {p.pilot_id: read(output / p.pilot_id / "test_episodes.json") for p in pilots}
        comparison = {"status": "completed", "opponent_type": "tactical", "baseline": str(baseline),
                      "experiment": str(output), "development": paired_comparison(before, after),
                      "fresh_confirmation": paired_comparison(old_test, new_test),
                      "confirmation_suite": plan["confirmation"], "completed_utc": now()}
        if baseline_inventory(baseline, batch["pilots"]) != plan["baseline_hashes"]:
            raise RuntimeError("source population changed during adaptation")
        write(job / "comparison.json", comparison)
        status.update(status="completed", updated_utc=now(), baseline_unchanged=True)
        write(job / "status.json", status)
        save_progress_report(job)
        print(json.dumps(comparison["fresh_confirmation"], indent=2), flush=True)
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
    mode.add_argument("--report", type=Path)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--assessment-scenarios", type=int, default=300)
    parser.add_argument("--job-root", type=Path)
    parser.add_argument("--seed-offset", type=int, default=1_000_000,
                        help="use a separate offset for diagnostic runs")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare, args.epochs, args.assessment_scenarios, args.job_root, args.seed_offset)
    elif args.report:
        save_progress_report(args.report)
    else:
        run(args.run_job)
