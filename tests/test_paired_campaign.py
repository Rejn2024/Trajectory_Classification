"""All ten paired pilots, real JSBSim, both stages, assessments and verified corpus."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
for dependency in ("mlflow", "jsbsim", "gymnasium", "cv2", "pyarrow", "loky", "psutil"):
    pytest.importorskip(dependency)

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_varied_pilot_weekend as weekend
from paired_pilot_training import PairProgress, worker_budgets

from bvr_behavior_prediction.rl.config import PilotTrainingConfig
from bvr_behavior_prediction.rl.dogfight_reward import hybrid_dogfight_reward_definition


def population(source):
    reward = hybrid_dogfight_reward_definition()
    records = []
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for index, pilot in enumerate(weekend.PILOTS):
            directory = source / pilot
            config = PilotTrainingConfig(
                hidden_size=8, transformer_heads=4 if index % 2 else 8,
                transformer_layers=1 + index % 2, epochs=2, episode_duration_s=3,
                seed=10_000 + index * 100, evaluation_seed=100_000_000,
                scenarios_per_epoch=3, evaluation_scenarios_per_epoch=3,
                simulator_workers=1, simulator_executor="process", update_epochs=2,
                minibatch_size=4, resample_training_scenarios=True, output_dir=directory,
            )
            trainer = weekend.PPOTrainer(lambda *_: None, 91, config, "cpu", reward)
            weekend.write(directory / "config.json", {
                "training": config.as_dict(), "observation_size": 91,
                "reward": reward.as_dict(), "skill_names": trainer.pilot.skill_names,
                "parameter_names": trainer.pilot.parameter_names,
                "skill_catalogue": trainer.pilot.manager.get_catalogue(),
            })
            torch.save(trainer.pilot.state_dict(), directory / "best_model.pt")
            record = {"pilot_id": pilot, "status": "completed", "output_dir": pilot,
                      "checkpoint_sha256": weekend.sha256(directory / "best_model.pt"),
                      "reward": reward.as_dict(), "seed": config.seed}
            weekend.write(directory / "manifest.json", record)
            records.append(record)
            trainer.close()
    finally:
        torch.set_num_threads(previous)
    weekend.write(source / "pilots.json", {"pilots": records, "population_metadata": {
        "observation": {"schema": "compact_hostile_missiles_energy_v1", "frame_size": 91}}})


def test_worker_budget_is_shared_and_one_worker_requires_serial():
    assert worker_budgets(12) == (6, 6)
    assert worker_budgets(14) == (7, 7)
    assert worker_budgets(3) == (2, 1)
    with pytest.raises(ValueError, match="serial"):
        worker_budgets(1)


def test_pair_progress_counts_both_pilots_without_double_counting(tmp_path):
    settings = {"simple_epochs": 2, "tactical_epochs": 2}
    pair = weekend.PILOTS[:2]
    for pilot, count in zip(pair, (1, 2)):
        weekend.write(tmp_path / "simple" / pilot / "progress.json", {
            "completed_epochs": count, "phase": "PPO update"})
    progress = PairProgress(tmp_path, settings, "simple", pair, weekend.PILOTS, weekend.write)
    progress.refresh()
    record = weekend.read(tmp_path / "training_progress.json")
    assert record["overall_completed_epochs"] == record["completed_epochs"] == 3
    assert record["overall_target_epochs"] == 40 and record["target_epochs"] == 4
    assert len(record["active_pilots"]) == 2


def test_all_up_paired_campaign_and_completed_restart(tmp_path):
    source, job, output = tmp_path / "source", tmp_path / "job", tmp_path / "corpus"
    population(source)
    command = [sys.executable, "-u", str(ROOT / "scripts/run_varied_pilot_weekend.py"),
               "--source", str(source), "--job", str(job), "--output", str(output),
               "--pilot-schedule", "paired", "--workers", "2", "--torch-threads", "1",
               "--device", "cpu", "--simple-epochs", "2", "--tactical-epochs", "2",
               "--assessment-scenarios", "3", "--scenarios-per-epoch", "3",
               "--evaluation-scenarios-per-epoch", "3", "--update-epochs", "2",
               "--episode-duration-s", "3", "--flights", "3", "--flight-seconds", "3",
               "--benchmark-flights", "3", "--inference-backend", "local_cpu"]
    environment = os.environ.copy()
    environment.update(MLFLOW_DISABLE_AGENT_HINT="1", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

    def run(extra=()):
        log = tmp_path / f"run_{len(list(tmp_path.glob('run_*.log')))}.log"
        with log.open("w") as stream:
            result = subprocess.run(command + list(extra), env=environment, cwd=ROOT,
                                    stdout=stream, stderr=subprocess.STDOUT, timeout=600, check=False)
        assert result.returncode == 0, log.read_text()[-8000:]

    run(["--preflight-only"])
    run()
    assert weekend.read(job / "status.json")["stage"] == "completed"
    pairs = [weekend.PILOTS[i:i + 2] for i in range(0, 10, 2)]
    events = weekend.metric_rows(job / "pair_events.jsonl")
    starts = [event for event in events if event["event"] == "started"]
    ends = [event for event in events if event["event"] == "completed"]
    assert [(event["stage"], event["pilots"]) for event in starts] == [
        (stage, pair) for stage in ("simple", "tactical") for pair in pairs]
    assert len(ends) == 10
    assert all(event["new_epochs"] == 4 for event in ends)
    for index, (started, ended) in enumerate(zip(starts, ends)):
        if index:
            assert started["time"] >= ends[index - 1]["time"]
        assert sum(started["workers"].values()) == 2
        executions = [weekend.read(job / started["stage"] / pid / "execution.json")
                      for pid in started["pilots"]]
        assert len({record["pid"] for record in executions}) == 2
        assert max(r["started_at"] for r in executions) < min(r["finished_at"] for r in executions)
        assert all(record["workers"] == 1 for record in executions)
    for stage in ("simple", "tactical"):
        assert len(weekend.read(job / stage / "pilots.json")["pilots"]) == 10
        for pid in weekend.PILOTS:
            rows = weekend.metric_rows(job / stage / pid / "training_metrics.jsonl")
            assert [row["epoch"] for row in rows] == [1, 2]
            assert all(row["evaluation_performed"] == 1 for row in rows)
            assert (job / ("before_tactical" if stage == "simple" else "after_tactical")
                    / pid / "evaluation.json").exists()
    manifest = weekend.read(output / "manifest.json")
    assert manifest["complete"] and manifest["physical_flights"] == 3
    assert manifest["perspective_episodes"] == 6
    assert weekend.read(job / "training_progress.json")["overall_completed_epochs"] == 40
    saved = {str(path): weekend.sha256(path) for path in job.glob("*/pilot_*/best_model.pt")}
    saved.update({str(path): weekend.sha256(path) for path in output.rglob("*.parquet")})
    run()
    assert all(weekend.sha256(Path(path)) == checksum for path, checksum in saved.items())
    assert all(len(weekend.metric_rows(job / stage / pid / "training_metrics.jsonl")) == 2
               for stage in ("simple", "tactical") for pid in weekend.PILOTS)


def test_pair_failure_stops_the_peer_and_does_not_publish_completion(tmp_path, monkeypatch):
    import paired_pilot_training as paired

    launched, stopped = [], []

    class Process:
        def __init__(self, command, **kwargs):
            self.pid = 9000 + len(launched)
            self.returncode = 7 if not launched else None
            launched.append(command[command.index("--pilot-worker") + 1])

        def poll(self):
            return self.returncode

    monkeypatch.setattr(paired.subprocess, "Popen", Process)
    monkeypatch.setattr(paired, "stop_worker", lambda process: stopped.append(process.pid))
    settings = {"workers": 14, "simple_epochs": 2, "tactical_epochs": 2,
                "output": str(tmp_path / "corpus")}
    with pytest.raises(RuntimeError, match="exited 7"):
        paired.run_pair(tmp_path, "simple", settings, weekend.PILOTS[:2],
                        weekend.PILOTS, weekend.write)
    assert launched == weekend.PILOTS[:2] and len(stopped) == 2
    events = weekend.metric_rows(tmp_path / "pair_events.jsonl")
    assert [event["event"] for event in events] == ["started"]
