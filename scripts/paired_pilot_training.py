"""Bounded, process-isolated pilot pairs; only the coordinator writes campaign progress."""

import json
import os
import signal
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

from weekend_training_progress import metric_rows, read_json


def worker_budgets(total):
    if total < 2:
        raise ValueError("Paired training requires --workers >= 2 (a total budget); use serial for one")
    return ((total + 1) // 2, total // 2)


def stop_worker(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       capture_output=True, check=False)
    else:
        process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        # The graceful path lets train_one close nested simulators. On timeout,
        # terminate descendants too; killing just the pilot would orphan them.
        import psutil

        try:
            descendants = psutil.Process(process.pid).children(recursive=True)
        except psutil.NoSuchProcess:
            descendants = []
        for child in reversed(descendants):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        process.kill()
        process.wait()


class PairProgress:
    def __init__(self, job, settings, stage, pair, pilots, write):
        self.job, self.settings, self.stage = Path(job), settings, stage
        self.pair, self.pilots, self.write = pair, pilots, write
        self.started = time.monotonic()
        self.initial = self.counts()
        self.fixed_counts = self.initial.copy()
        self.last_state = None
        self.last_written = float("-inf")

    def counts(self):
        result = getattr(self, "fixed_counts", {}).copy()
        for stage in ("simple", "tactical"):
            for pilot in self.pilots:
                if (stage, pilot) in result and not (stage == self.stage and pilot in self.pair):
                    continue
                directory = self.job / stage / pilot
                manifest = read_json(directory / "manifest.json")
                progress = (read_json(directory / "progress.json")
                            if stage == self.stage and pilot in self.pair else {})
                if manifest.get("status") == "completed":
                    count = manifest["additional_epochs"]
                elif "completed_epochs" in progress:
                    count = progress["completed_epochs"]
                else:
                    rows = metric_rows(directory / "training_metrics.jsonl")
                    count = rows[-1]["epoch"] if rows else 0
                result[stage, pilot] = min(count, self.settings[f"{stage}_epochs"])
        return result

    def refresh(self):
        counts = self.counts()
        records = []
        for pilot in self.pair:
            record = read_json(self.job / self.stage / pilot / "progress.json")
            records.append({"pilot": pilot, "phase": record.get("phase", "starting"),
                            "completed_epochs": counts[self.stage, pilot]})
            if counts[self.stage, pilot] == self.settings[f"{self.stage}_epochs"]:
                records[-1]["phase"] = "pilot complete"
        completed = sum(counts.values())
        gained = completed - sum(self.initial.values())
        elapsed = time.monotonic() - self.started
        rate = gained / elapsed if gained > 0 else None
        if rate is None:
            samples = [row for row in metric_rows(self.job / "pair_events.jsonl")
                       if row.get("event") == "completed" and row.get("new_epochs", 0) > 0]
            if samples:
                rate = sum(row["new_epochs"] for row in samples) / sum(
                    row["wall_seconds"] for row in samples)
        state = [(r["pilot"], r["phase"], r["completed_epochs"]) for r in records]
        if state == self.last_state and elapsed - self.last_written < 2:
            return counts
        self.last_state, self.last_written = state, elapsed
        pair_done = sum(counts[self.stage, pilot] for pilot in self.pair)
        pair_target = len(self.pair) * self.settings[f"{self.stage}_epochs"]
        overall_target = len(self.pilots) * sum(
            self.settings[f"{stage}_epochs"] for stage in ("simple", "tactical"))
        self.write(self.job / "training_progress.json", {
            "stage": self.stage, "pilot": " + ".join(self.pair),
            "phase": "; ".join(f"{r['pilot']}: {r['phase']}" for r in records),
            "completed_epochs": pair_done, "target_epochs": pair_target,
            "overall_completed_epochs": completed, "overall_target_epochs": overall_target,
            "pilot_eta_seconds": (pair_target - pair_done) / rate if rate else None,
            "training_eta_seconds": (overall_target - completed) / rate if rate else None,
            "epoch_started_at": None, "updated_at": time.time(),
            "loss": None, "validation_return": None, "active_pilots": records,
        })
        return counts


def run_pair(job, stage, settings, pair, pilots, write):
    job = Path(job)
    quotas = dict(zip(pair, worker_budgets(settings["workers"])))
    logs = job / "logs"
    logs.mkdir(exist_ok=True)
    processes = {}
    token = str(time.time_ns())
    released = False
    with ExitStack() as stack:
        for pilot in pair:
            directory = job / stage / pilot
            if (directory / "manifest.json").exists():
                continue
            # Discard stale live phase records. Committed counts are recovered
            # from the checkpoint by the worker before its next progress event.
            rows = metric_rows(directory / "training_metrics.jsonl")
            write(directory / "progress.json", {"phase": "starting",
                  "completed_epochs": rows[-1]["epoch"] if rows else 0})
        progress = PairProgress(job, settings, stage, pair, pilots, write)
        event = {"event": "started", "stage": stage, "pilots": pair,
                 "workers": quotas, "time": time.time()}

        def record(value):
            with (job / "pair_events.jsonl").open("a", encoding="utf8") as stream:
                stream.write(json.dumps(value) + "\n")

        try:
            for pilot in pair:
                if (job / stage / pilot / "manifest.json").exists():
                    continue
                log = logs / f"{stage}_{pilot}_{time.time_ns()}.log"
                stream = stack.enter_context(log.open("w", encoding="utf8"))
                environment = os.environ.copy()
                for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                             "NUMEXPR_NUM_THREADS"):
                    environment[name] = "1"
                command = [sys.executable, "-u", str(Path(__file__).with_name(
                    "run_varied_pilot_weekend.py")), "--job", str(job),
                    "--output", settings["output"], "--pilot-worker", pilot,
                    "--worker-stage", stage, "--pair-token", token]
                # Inherit the campaign process group so notebook interruption
                # also reaches both pilot subprocess trees on POSIX.
                processes[pilot] = (subprocess.Popen(command, env=environment,
                    stdout=stream, stderr=subprocess.STDOUT), log)
            event["pids"] = {p: value[0].pid for p, value in processes.items()}
            record(event)
            while True:
                progress.refresh()
                for pilot, (process, log) in processes.items():
                    if process.poll() not in (None, 0):
                        raise RuntimeError(f"{stage} {pilot} exited {process.returncode}; log: {log}")
                if not released and all(
                    read_json(job / stage / pilot / "ready.json").get("token") == token
                    for pilot in processes
                ):
                    write(job / "pair_release.json", {"token": token})
                    released = True
                if all(process.poll() is not None for process, _ in processes.values()):
                    break
                time.sleep(.5)
            counts = progress.refresh()
            record({**event, "event": "completed", "time": time.time(),
                    "wall_seconds": time.monotonic() - progress.started,
                    "new_epochs": sum(counts.values()) - sum(progress.initial.values())})
        finally:
            for process, _ in processes.values():
                stop_worker(process)
