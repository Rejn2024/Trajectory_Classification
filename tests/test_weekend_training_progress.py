"""Committed epoch counts and measured-time forecasts drive notebook bars."""

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from weekend_training_progress import (
    CampaignProgress,
    EpochProgressReporter,
    TextProgress,
    duration,
    read_json,
    timing_windows,
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class FakeBar:
    def __init__(self, **kwargs):
        self.n = kwargs.get("initial", 0)
        self.total = kwargs["total"]
        self.closed = False
        self.postfix = ""
        self.refreshes = 0

    def update(self, delta):
        self.n += delta

    def set_postfix_str(self, value, refresh=False):
        self.postfix = value

    def refresh(self):
        self.refreshes += 1

    def close(self):
        self.closed = True


def test_progress_json_skips_a_windows_sharing_lock_and_recovers(tmp_path, monkeypatch):
    path = tmp_path / "training_progress.json"
    write(path, {"completed_epochs": 2})
    native_read = Path.read_text

    def locked_read(self, *args, **kwargs):
        if self == path:
            raise PermissionError(13, "Permission denied", str(self))
        return native_read(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_text", locked_read)
        assert read_json(path) == {}
    assert read_json(path) == {"completed_epochs": 2}


@pytest.mark.parametrize("locked_file", [
    "status.json", "training_progress.json", "collection/progress.json",
])
def test_progress_refresh_and_close_tolerate_locked_snapshots(tmp_path, monkeypatch, locked_file):
    stage = "corpus" if locked_file.startswith("collection/") else "simple"
    write(tmp_path / "status.json", {"stage": stage})
    reporter = EpochProgressReporter(
        tmp_path, {"simple_epochs": 3, "tactical_epochs": 3},
        "simple", "pilot_001", write,
    )
    row = {"epoch": 1, "epoch_seconds": 10, "loss": 0.25, "mean_return": 15}
    reporter("PPO update", 1, [row])
    write(tmp_path / "collection/progress.json", {"completed": 1, "flights_per_hour": 3600})
    progress = CampaignProgress(tmp_path, 10, FakeBar)
    progress.refresh()
    bars = [bar for bar in (progress.overall, progress.pilot, progress.collection) if bar]
    native_read = Path.read_text

    def locked_read(self, *args, **kwargs):
        if self == tmp_path / locked_file:
            raise PermissionError(13, "Permission denied", str(self))
        return native_read(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_text", locked_read)
        progress.refresh()
        progress.close()
    assert all(bar.closed for bar in bars)

    # A later poll sees new committed work once the sharing lock disappears.
    reporter("PPO update", 2, [row, {**row, "epoch": 2}])
    write(tmp_path / "collection/progress.json", {"completed": 2, "flights_per_hour": 3600})
    recovered = CampaignProgress(tmp_path, 10, FakeBar)
    recovered.refresh()
    assert recovered.overall.n == 2
    assert (recovered.collection if stage == "corpus" else recovered.pilot).n == 2
    recovered.close()


@pytest.mark.parametrize("returncode", [0, 7])
def test_notebook_progress_lock_does_not_kill_child_or_mask_its_exit(tmp_path, monkeypatch, returncode):
    notebook = Path(__file__).parents[1] / "notebooks/31_continue_varied_pilots_and_generate_corpus.ipynb"
    cells = json.loads(notebook.read_text())["cells"]
    source = next("".join(cell["source"]) for cell in cells
                  if cell["cell_type"] == "code" and "def run_logged(" in "".join(cell["source"]))
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "run_logged")
    polls = iter([None, returncode])
    child = SimpleNamespace(returncode=None, pid=123)

    def poll():
        child.returncode = next(polls)
        return child.returncode

    def start_child(*args, **kwargs):
        kwargs["stdout"].write("worker diagnostic\n")
        return child

    def unexpected_stop(*args, **kwargs):
        pytest.fail("An optional progress read must not terminate the campaign")

    child.poll = poll
    namespace = {
        "JOB": tmp_path, "ROOT": tmp_path, "FLIGHTS": 10,
        "CampaignProgress": lambda *args: CampaignProgress(*args, bar_factory=FakeBar),
        "os": SimpleNamespace(name="posix", environ={}, killpg=unexpected_stop),
        "signal": SimpleNamespace(SIGTERM=15),
        "subprocess": SimpleNamespace(Popen=start_child, STDOUT=-2),
        "time": SimpleNamespace(time_ns=lambda: 1, monotonic=lambda: 0, sleep=lambda _: None),
    }
    # Execute only the trusted notebook's runner, without setup or training cells.
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(notebook), "exec"), namespace)  # noqa: S102
    native_read = Path.read_text
    reads = []

    def locked_read(self, *args, **kwargs):
        if self == tmp_path / "training_progress.json":
            reads.append(self)
            raise PermissionError(13, "Permission denied", str(self))
        return native_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", locked_read)
    run = namespace["run_logged"]
    if returncode:
        with pytest.raises(RuntimeError, match="campaign exited with 7") as error:
            run(["child"], "campaign")
        assert "worker diagnostic" in str(error.value)
    else:
        assert run(["child"], "campaign") == tmp_path / "logs/campaign_1.log"
    assert len(reads) == 2  # Normal polling and the final refresh in close().


def test_epoch_forecasts_resume_counts_and_collection_bar(tmp_path):
    clock = [100.0]
    row = {"epoch": 1, "epoch_seconds": 10.0, "loss": 0.25, "mean_return": 15.0}
    previous = tmp_path / "simple/pilot_002/training_metrics.jsonl"
    previous.parent.mkdir(parents=True)
    previous.write_text(json.dumps(row) + "\n" + json.dumps({**row, "epoch": 2}) + "\n")
    reporter = EpochProgressReporter(
        tmp_path,
        {"simple_epochs": 4, "tactical_epochs": 5},
        "simple",
        "pilot_001",
        write,
        clock=lambda: clock[0],
    )
    write(tmp_path / "status.json", {"stage": "simple"})
    reporter("collecting flights", 1, [row])
    clock[0] = 105.0
    reporter("PPO update", 1, [row])
    clock[0] = 110.0
    bars = CampaignProgress(tmp_path, 10, FakeBar, clock=lambda: clock[0])
    bars.refresh()
    assert bars.overall.n == 3 and bars.overall.total == 90
    assert bars.pilot.n == 1 and bars.pilot.total == 4
    assert "PPO update" in bars.pilot.postfix
    assert "0:00:20" in bars.pilot.postfix
    bars.refresh()
    assert bars.overall.n == 3 and bars.pilot.n == 1
    record = json.loads((tmp_path / "training_progress.json").read_text())
    assert record["overall_completed_epochs"] == 3
    assert record["epoch_started_at"] == 100.0
    write(tmp_path / "status.json", {"stage": "before_tactical"})
    bars.refresh()
    assert bars.pilot is None and "before_tactical" in bars.overall.postfix
    write(tmp_path / "status.json", {"stage": "corpus"})
    write(tmp_path / "collection/progress.json", {"completed": 2, "flights_per_hour": 3600})
    bars.refresh()
    assert bars.collection.n == 2
    assert "0:00:08" in bars.collection.postfix
    bars.close()
    assert bars.overall.closed and bars.collection.closed


def test_first_epoch_estimates_wait_for_real_timing_and_stale_eta_is_labelled(tmp_path):
    clock = [100.0]
    reporter = EpochProgressReporter(
        tmp_path,
        {"simple_epochs": 2, "tactical_epochs": 2},
        "simple",
        "pilot_001",
        write,
        clock=lambda: clock[0],
    )
    reporter("collecting flights", 0, [])
    write(tmp_path / "status.json", {"stage": "simple"})
    bars = CampaignProgress(tmp_path, 3, FakeBar, clock=lambda: clock[0])
    bars.refresh()
    assert "estimating" in bars.pilot.postfix
    assert duration(3665) == "1:01:05"
    reporter(
        "collecting flights",
        1,
        [{"epoch": 1, "epoch_seconds": 10.0, "loss": 1.0, "mean_return": 2.0}],
    )
    clock[0] = 120.0
    bars.refresh()
    assert "re-estimating" in bars.pilot.postfix
    bars.close()


def test_unchanged_polls_and_phase_changes_do_not_redraw_overall_training(tmp_path):
    clock = [100.0]
    reporter = EpochProgressReporter(
        tmp_path, {"simple_epochs": 200, "tactical_epochs": 300},
        "simple", "pilot_002", write, clock=lambda: clock[0],
    )
    row = {"epoch": 1, "epoch_seconds": 60, "loss": 22.88, "mean_return": 67.57}
    write(tmp_path / "status.json", {"stage": "simple"})
    reporter("collecting flights", 1, [row])
    bars = CampaignProgress(tmp_path, 10_000, FakeBar, clock=lambda: clock[0])
    bars.refresh()
    for _ in range(1000):
        clock[0] += 2
        bars.refresh()
    assert bars.overall.refreshes == bars.pilot.refreshes == 1
    reporter("PPO update", 1, [row])
    bars.refresh()
    assert bars.overall.refreshes == 1
    assert bars.pilot.refreshes == 2
    reporter("epoch complete", 2, [row, {**row, "epoch": 2}])
    bars.refresh()
    assert bars.overall.n == bars.pilot.n == 2
    assert bars.overall.refreshes == 2
    bars.close()


def test_widget_can_refresh_eta_without_appending_text_events(tmp_path):
    class WidgetBar(FakeBar):
        pass

    WidgetBar.__module__ = "tqdm.notebook"
    write(tmp_path / "status.json", {"stage": "simple"})
    reporter = EpochProgressReporter(
        tmp_path, {"simple_epochs": 2, "tactical_epochs": 2},
        "simple", "pilot_001", write,
    )
    reporter("collecting flights", 0, [])
    bars = CampaignProgress(tmp_path, 3, WidgetBar)
    bars.refresh()
    bars.refresh()
    assert bars.pilot.refreshes == bars.overall.refreshes == 2


def test_text_fallback_has_no_terminal_control_codes_or_unchanged_lines(tmp_path, capsys):
    write(tmp_path / "status.json", {"stage": "simple"})
    reporter = EpochProgressReporter(
        tmp_path, {"simple_epochs": 2, "tactical_epochs": 2},
        "simple", "pilot_001", write,
    )
    reporter("collecting flights", 0, [])
    bars = CampaignProgress(tmp_path, 3, TextProgress)
    for _ in range(20):
        bars.refresh()
    reporter("validation flights", 0, [])
    bars.refresh()
    bars.close()
    output = capsys.readouterr().out
    assert len(output.splitlines()) == 3  # Overall, pilot, then changed pilot phase.
    assert output.count("Overall training:") == 1
    assert "validation flights" in output
    assert "\r" not in output and "\x1b" not in output


def test_timing_windows_distinguish_more_work_from_slower_minibatches(tmp_path):
    rows = [
        {"epoch": i + 1, "epoch_seconds": count * .05, "update_seconds": count * .005,
         "rollout_transitions": count * 64, "update_minibatches": count,
         "update_passes_completed": 60}
        for i, count in enumerate([100, 100, 1000, 1000])
    ]
    path = tmp_path / "metrics.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows) + '{"epoch":')
    early, late = timing_windows(path, size=2)
    assert late["PPO_s"] == early["PPO_s"] * 10
    assert late["PPO_ms/minibatch"] == early["PPO_ms/minibatch"] == 5
    assert early["PPO_passes"] == late["PPO_passes"] == 60
    assert late["first_epoch"] == 3 and late["last_epoch"] == 4
    assert "validation_decisions" not in late and "CUDA_peak_MB" not in late
