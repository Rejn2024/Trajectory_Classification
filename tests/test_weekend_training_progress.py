"""Committed epoch counts and measured-time forecasts drive notebook bars."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from weekend_training_progress import CampaignProgress, EpochProgressReporter, duration


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class FakeBar:
    def __init__(self, **kwargs):
        self.n = kwargs.get("initial", 0)
        self.total = kwargs["total"]
        self.closed = False
        self.postfix = ""

    def update(self, delta):
        self.n += delta

    def set_postfix_str(self, value, refresh=False):
        self.postfix = value

    def refresh(self):
        pass

    def close(self):
        self.closed = True


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
