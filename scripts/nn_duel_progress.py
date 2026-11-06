"""Low-overhead notebook progress: observe saved counters, never call a simulator.

Child output goes directly to local log files. The notebook process wakes only
when a command exits or the next report is due. No worker RPCs, directory scans,
checksums or GPU queries are added to the rollout path. One
blocked waiter thread in the notebook process allows immediate command completion
without a busy loop; it never touches simulator state.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def duration(seconds):
    seconds = max(0, int(seconds))
    if seconds >= 3600:
        return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds}s"


def read_counter(path):
    """Atomic collector writes can still meet temporary Windows sharing locks."""
    try:
        path = Path(path)
        modified = path.stat().st_mtime
        data = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(data, dict)
            or type(data.get("completed")) is not int
            or type(data.get("requested")) is not int
            or not 0 <= data["completed"] <= data["requested"]
            or not isinstance(data.get("failed_this_invocation", []), list)
        ):
            return {}, None
        return data, modified
    except (OSError, ValueError):
        return {}, None


def log_tail(path, max_bytes=6000):
    with Path(path).open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell() - max_bytes))
        return "\n".join(stream.read().decode("utf-8", errors="replace").splitlines()[-16:])


class NotebookProgress:
    def __init__(self, work, total, interval_seconds=300, emit=None, clock=None):
        if total < 1 or not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("Positive total and finite positive reporting interval required")
        self.work = Path(work).resolve()
        self.total = total
        self.interval = interval_seconds
        self.emit = emit or (lambda message: print(message, flush=True))
        self.clock = clock or time.monotonic
        self.next_report = self.clock() + self.interval
        self.started = None
        self.initial_completed = self.completed = self.failures = 0
        self.backend, self.workers = "not selected", None
        self.snapshot_completed, self.snapshot_at, self.snapshot_pushed = 0, None, False
        self.current = None
        self.latest_record_at = None
        self.last_log = None
        self.logs = self.work / "notebook_logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.session = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
        )
        self.sequence = 0
        self.events = self.logs / f"{self.session}_progress.log"
        self.log_error = None

    def _say(self, message):
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        text = f"[{stamp}] {message}"
        # Keep the human-readable transcript even after notebook outputs are cleared.
        try:
            with self.events.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(text + "\n")
        except OSError as error:
            self.log_error = repr(error)  # Optional progress logging must not stop a rollout.
        try:
            self.emit(text)
        except OSError:
            pass

    def start_collection(self, completed, backend, workers):
        self.started = self.clock()
        self.initial_completed = self.completed = completed
        self.backend, self.workers = backend, workers
        self._say(
            f"Starting collection: {completed:,}/{self.total:,} flights already saved; "
            f"{backend}, {workers} simulator workers. Updates about every "
            f"{duration(self.interval)}; detailed command logs: {self.logs}"
        )
        # Preserve a single wall-clock cadence across all chunks and command stages.
        self.next_report = self.clock() + self.interval

    def snapshot(self, completed, pushed):
        self.completed = max(self.completed, completed)
        self.snapshot_completed = completed
        self.snapshot_pushed = pushed
        self.snapshot_at = self.clock()

    def _observe(self):
        if not self.current or self.current["command"] != "collect":
            return {}, None
        data, modified = read_counter(self.current["work"] / "progress.json")
        if data.get("requested") != self.current["target"]:
            return {}, None
        if self.current["work"] == self.work:
            value = data.get("completed", self.completed)
            if isinstance(value, int) and 0 <= value <= self.total:
                self.completed = max(self.completed, value)
            if modified is not None and modified >= self.current["started_wall"]:
                self.failures = len(data.get("failed_this_invocation", []))
                self.latest_record_at = modified
        return data, modified

    def report(self, force=False):
        now = self.clock()
        if not force and now < self.next_report:
            return False
        data, modified = self._observe()
        phase = self.current["stage"] if self.current else "Between collection stages"
        lines = [phase]
        if self.started is None:
            lines.append(f"Main campaign has not started; target {self.total:,} flights.")
            if self.current and self.current["command"] == "collect" and data:
                lines.append(
                    f"Benchmark probe: {data.get('completed', 0):,}/{self.current['target']:,} flights; "
                    f"failed in this probe: {len(data.get('failed_this_invocation', []))}."
                )
        else:
            elapsed = max(0.0, now - self.started)
            acquired = self.completed - self.initial_completed
            lines.append(
                f"Flights saved: {self.completed:,}/{self.total:,} "
                f"({100 * self.completed / self.total:.1f}%); "
                f"{2 * self.completed:,} aircraft perspectives; collection elapsed {duration(elapsed)}."
            )
            if acquired > 0 and elapsed > 0 and self.completed < self.total:
                rate = acquired * 3600 / elapsed
                remaining = (self.total - self.completed) * elapsed / acquired
                lines.append(
                    f"Observed rate: {rate:,.0f} flights/hour; estimated remaining "
                    f"{duration(remaining)} including observed backup time."
                )
            elif self.completed >= self.total:
                lines.append("All flights recorded; finishing verification and backup.")
            else:
                lines.append("Rate/ETA will be available after the first completed flights.")
            label = {
                "central_cuda": "GPU batched inference",
                "local_cpu": "CPU inference",
                "central_cpu": "shared CPU inference (test mode)",
            }.get(self.backend, self.backend)
            lines.append(
                f"Backend: {label}; configured simulator workers: {self.workers}; "
                f"failed flights in latest collection batch: {self.failures}."
            )
            if self.snapshot_at is not None:
                location = (
                    "verified and pushed to Git LFS"
                    if self.snapshot_pushed
                    else "verified locally (push disabled)"
                )
                lines.append(
                    f"Last backup: {self.snapshot_completed:,} flights {location}, "
                    f"{duration(now - self.snapshot_at)} ago."
                )
            else:
                lines.append("No backup has completed during this notebook execution yet.")
            if self.current and self.current["command"] == "collect":
                if modified is not None and modified >= self.current["started_wall"]:
                    lines.append(
                        f"Latest completed-flight counter updated {duration(time.time() - modified)} ago."
                    )
                else:
                    lines.append(
                        "Waiting for the first new completed-flight record from this command; "
                        "worker startup may still be in progress."
                    )
        self._say("\n".join(lines))
        self.next_report = now + self.interval
        return True

    def run(self, args, *, cwd, env, command, work, stage, target):
        self.sequence += 1
        self.last_log = self.logs / f"{self.session}_{self.sequence:04d}_{command}.log"
        self.current = {
            "command": command,
            "work": Path(work).resolve(),
            "stage": stage,
            "target": target,
            "started_wall": time.time(),
        }
        if command == "collect" and self.current["work"] == self.work:
            self.failures = 0
        self.report()
        flags = (
            {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
            if os.name == "nt"
            else {"start_new_session": True}
        )
        with self.last_log.open("w", encoding="utf-8") as log:
            log.write(f"Stage: {stage}\nWorking directory: {cwd}\n\n")
            log.flush()
            process = subprocess.Popen(
                args, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, **flags
            )
            done, wait_errors = threading.Event(), []

            def wait_for_exit():
                try:
                    process.wait()  # Blocking waitpid/WaitForSingleObject, without polling.
                except BaseException as error:  # noqa: BLE001 -- forward waiter failures to the notebook
                    wait_errors.append(error)
                finally:
                    done.set()

            waiter = threading.Thread(target=wait_for_exit, daemon=True)
            waiter.start()
            try:
                while not done.wait(timeout=max(0.001, self.next_report - self.clock())):
                    self.report(force=True)
                if wait_errors:
                    raise wait_errors[0]
                code = process.returncode
            except BaseException:
                self._stop(process, log)
                self._say(f"Interrupted {stage}. Logs: {self.last_log}")
                raise
            finally:
                waiter.join(timeout=1)
        self._observe()
        self.report()
        if code:
            tail = log_tail(self.last_log)
            self._say(
                f"FAILED: {stage} (exit {code}). Detailed log: {self.last_log}\n{tail}"
            )
            raise subprocess.CalledProcessError(code, args, output=f"Log: {self.last_log}\n{tail}")
        return code

    @staticmethod
    def _stop(process, log):
        """Interrupt only this command's process group, including its workers."""
        if process.poll() is not None:
            return
        try:
            if os.name == "nt":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=10)
            return
        except (OSError, subprocess.TimeoutExpired):
            pass
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=10)

    def finish(self):
        if self.completed != self.total:
            raise ValueError("Cannot report completion before all requested flights are saved")
        self.current = None
        elapsed = self.clock() - self.started if self.started is not None else 0
        backup = (
            "Git LFS push confirmed"
            if self.snapshot_pushed
            else "verified local data; push disabled"
        )
        self._say(
            f"COMPLETE: {self.completed:,}/{self.total:,} flights, "
            f"{2 * self.completed:,} aircraft perspectives; final full verification passed; "
            f"{backup}; collection elapsed {duration(elapsed)}. Logs: {self.logs}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Final complete verification, observed by the notebook reporter"
    )
    parser.add_argument("command", choices=("verify-complete",))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from generate_nn_duel_corpus import verify

    manifest = verify(args.output, require_complete=True)
    print(
        f"Final full verification passed: {manifest['physical_flights']:,} physical flights",
        flush=True,
    )


if __name__ == "__main__":
    main()
