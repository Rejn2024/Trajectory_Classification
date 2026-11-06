"""Wall-clock reporting stays outside the flight workers and survives quiet phases."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import nn_duel_progress as progress


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


def test_five_minute_reports_read_only_one_existing_counter(tmp_path, monkeypatch):
    clock, messages = Clock(), []
    reporter = progress.NotebookProgress(tmp_path, 1000, emit=messages.append, clock=clock)
    reporter.start_collection(100, "central_cuda", 12)
    reporter.snapshot(200, pushed=True)
    reporter.current = {
        "stage": "Collecting flights",
        "command": "collect",
        "work": tmp_path,
        "target": 1000,
        "started_wall": 0,
    }
    (tmp_path / "progress.json").write_text(
        json.dumps(
            {
                "completed": 250,
                "requested": 1000,
                "failed_this_invocation": [{"episode_id": "failed-example"}],
            }
        )
    )
    calls, native_read = [], progress.read_counter

    def observed_read(path):
        calls.append(path)
        return native_read(path)

    monkeypatch.setattr(progress, "read_counter", observed_read)
    for tick in (1, 100, 299.99):
        clock.now = tick
        assert not reporter.report()
    assert calls == []
    clock.now = 300
    assert reporter.report()
    assert calls == [tmp_path / "progress.json"]
    text = messages[-1]
    for expected in (
        "250/1,000",
        "25.0%",
        "500 aircraft perspectives",
        "1,800 flights/hour",
        "25m 00s",
        "GPU batched inference",
        "workers: 12",
        "batch: 1",
        "200 flights verified and pushed to Git LFS",
    ):
        assert expected in text
    clock.now = 599
    assert not reporter.report()
    clock.now = 600
    reporter.current["command"] = "publish"
    reporter.current["stage"] = "Verifying and uploading Git LFS backup"
    assert reporter.report()
    assert "uploading Git LFS" in messages[-1]
    assert len(calls) == 1  # Upload keepalives do not scan trajectory or progress files.


@pytest.mark.parametrize(
    "contents", [None, "{", "[]", '{"requested": 100, "completed": "unknown"}']
)
def test_missing_or_busy_counter_keeps_last_known_counts(tmp_path, contents):
    messages = []
    reporter = progress.NotebookProgress(tmp_path, 100, emit=messages.append)
    reporter.start_collection(10, "local_cpu", 2)
    reporter.current = {
        "stage": "Collecting flights",
        "command": "collect",
        "work": tmp_path,
        "target": 100,
        "started_wall": time.time(),
    }
    if contents is not None:
        (tmp_path / "progress.json").write_text(contents)
    reporter.report(force=True)
    assert "10/100" in messages[-1]
    assert "Waiting for the first new completed-flight record" in messages[-1]
    assert "Rate/ETA will be available" in messages[-1]


def test_benchmark_counts_never_count_as_training_flights(tmp_path):
    probe = tmp_path / "benchmark"
    probe.mkdir()
    (probe / "progress.json").write_text(json.dumps({"requested": 30, "completed": 12}))
    messages = []
    reporter = progress.NotebookProgress(tmp_path, 10000, emit=messages.append)
    reporter.current = {
        "stage": "Benchmark",
        "command": "collect",
        "work": probe,
        "target": 30,
        "started_wall": 0,
    }
    reporter.report(force=True)
    assert "Main campaign has not started" in messages[-1]
    assert "Benchmark probe: 12/30" in messages[-1]
    assert reporter.completed == 0


def test_quiet_child_gets_periodic_keepalives_and_raw_output_goes_to_log(tmp_path):
    messages = []
    reporter = progress.NotebookProgress(tmp_path, 100, interval_seconds=0.07, emit=messages.append)
    reporter.start_collection(0, "local_cpu", 2)
    reporter.run(
        [sys.executable, "-c", "import time; print('RAW_JSON_SENTINEL'); time.sleep(.25)"],
        cwd=tmp_path,
        env=os.environ.copy(),
        command="publish",
        work=tmp_path,
        stage="Uploading backup",
        target=100,
    )
    assert any("Uploading backup" in m for m in messages)
    assert all("RAW_JSON_SENTINEL" not in m for m in messages)
    assert "RAW_JSON_SENTINEL" in reporter.last_log.read_text()
    assert reporter.events.exists()


def test_child_exit_returns_immediately_and_failure_reports_log_tail(tmp_path):
    messages = []
    reporter = progress.NotebookProgress(tmp_path, 100, emit=messages.append)
    start = time.monotonic()
    with pytest.raises(subprocess.CalledProcessError) as error:
        reporter.run(
            [sys.executable, "-c", "print('EXAMPLE_FAILURE'); raise SystemExit(7)"],
            cwd=tmp_path,
            env=os.environ.copy(),
            command="verify",
            work=tmp_path,
            stage="Verifying data",
            target=100,
        )
    assert error.value.returncode == 7
    assert "EXAMPLE_FAILURE" in error.value.output
    assert str(reporter.last_log) in error.value.output
    assert time.monotonic() - start < 10  # Do not wait five minutes after a command has exited.
    assert "FAILED: Verifying data" in messages[-1] and "EXAMPLE_FAILURE" in messages[-1]


def test_optional_progress_transcript_failure_does_not_interrupt_collection(tmp_path):
    reporter = progress.NotebookProgress(tmp_path, 100, emit=lambda _: None)
    reporter.events = tmp_path / "missing-directory" / "progress.log"
    reporter.start_collection(0, "local_cpu", 2)
    assert reporter.log_error is not None
    reporter.report(force=True)


def test_completion_does_not_claim_unperformed_upload(tmp_path):
    messages = []
    reporter = progress.NotebookProgress(tmp_path, 100, emit=messages.append)
    with pytest.raises(ValueError, match="all requested"):
        reporter.finish()
    reporter.start_collection(0, "local_cpu", 2)
    reporter.snapshot(100, pushed=False)
    reporter.finish()
    assert "COMPLETE: 100/100" in messages[-1]
    assert "push disabled" in messages[-1]
    assert "Git LFS push confirmed" not in messages[-1]


def test_keyboard_interrupt_stops_the_command_process_group(tmp_path, monkeypatch):
    reporter = progress.NotebookProgress(tmp_path, 100, interval_seconds=0.05, emit=lambda _: None)
    native_stop = reporter._stop
    stopped = []

    def stop_and_check(process, log):
        native_stop(process, log)
        stopped.append(process.poll())

    def interrupt_report(force=False):
        if force:
            raise KeyboardInterrupt
        return False

    monkeypatch.setattr(reporter, "_stop", stop_and_check)
    monkeypatch.setattr(reporter, "report", interrupt_report)
    with pytest.raises(KeyboardInterrupt):
        reporter.run(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            cwd=tmp_path,
            env=os.environ.copy(),
            command="collect",
            work=tmp_path,
            stage="Collecting flights",
            target=100,
        )
    assert len(stopped) == 1 and stopped[0] is not None
