"""Actual optimizer rates, cosine endpoints and exact recovery after interruption."""

import numpy as np
import pandas as pd
import pytest
import torch

from bvr_behavior_prediction import paired_skill_forecasting as paired
from tests.test_paired_skill_forecasting import controlled_training


def recipe():
    return dict(epochs=6, width=6, heads=6, layers=1, batch_size=1, dropout=.1,
                cpu_threads=1, precision="fp32", data_backend="direct", early_stopping=False,
                lr_schedule="warmup_cosine", warmup_fraction=.25, warmup_start_ratio=.1, min_lr_ratio=.05)


def test_used_learning_rates_and_weights_match_after_resume(tmp_path, monkeypatch):
    data, _ = controlled_training(tmp_path, monkeypatch, lambda epoch: 1.)
    data.indices["train"] = np.arange(3)
    fit_batch = paired.fit_batch
    observed = []

    def record_rate(model, optimizer, *args, **kwargs):
        observed.append(optimizer.param_groups[0]["lr"])
        return fit_batch(model, optimizer, *args, **kwargs)

    monkeypatch.setattr(paired, "fit_batch", record_rate)
    uninterrupted = paired.train_model(data, tmp_path / "full", recipe(), device="cpu")
    reference_rates = observed.copy()
    observed.clear()

    def interrupt(row, *args):
        if row["epoch"] == 2:
            raise RuntimeError("Simulated interruption")

    with pytest.raises(RuntimeError, match="Simulated"):
        paired.train_model(data, tmp_path / "resumed", recipe(), device="cpu", callback=interrupt)
    checkpoint = torch.load(tmp_path / "resumed/last.pt", weights_only=False)
    assert checkpoint["optimizer_steps"] == checkpoint["scheduler_state"]["last_epoch"] == 6
    resumed = paired.train_model(data, tmp_path / "resumed", recipe(), device="cpu", resume=True)
    assert observed == reference_rates
    assert len(reference_rates) == 18
    assert reference_rates[0] == pytest.approx(3e-5)
    assert max(reference_rates) == pytest.approx(3e-4)
    assert reference_rates[-1] == pytest.approx(1.5e-5)
    assert all(a <= b for a, b in zip(reference_rates[:3], reference_rates[1:4]))
    assert all(a >= b for a, b in zip(reference_rates[4:-1], reference_rates[5:]))
    pd.testing.assert_frame_equal(uninterrupted.drop(columns="seconds"), resumed.drop(columns="seconds"))
    full = torch.load(tmp_path / "full/last.pt", weights_only=False)
    recovered = torch.load(tmp_path / "resumed/last.pt", weights_only=False)
    assert recovered["scheduler_state"] == full["scheduler_state"]
    for key, value in full["model_state"].items():
        torch.testing.assert_close(value, recovered["model_state"][key], atol=0, rtol=0)
    # Completed recovery does not restart the schedule or perform extra optimizer updates.
    paired.train_model(data, tmp_path / "resumed", recipe(), device="cpu", resume=True)
    assert observed == reference_rates


@pytest.mark.parametrize("alteration", ["missing", "count", "schedule"])
def test_resume_rejects_incomplete_or_inconsistent_scheduler_state(tmp_path, monkeypatch, alteration):
    data, _ = controlled_training(tmp_path, monkeypatch, lambda epoch: 1.)
    destination = tmp_path / "run"
    paired.train_model(data, destination, recipe(), device="cpu")
    checkpoint = torch.load(destination / "last.pt", weights_only=False)
    if alteration == "missing":
        checkpoint.pop("scheduler_state")
    elif alteration == "count":
        checkpoint["optimizer_steps"] -= 1
    else:
        checkpoint["learning_rate_schedule"]["total_updates"] += 1
    torch.save(checkpoint, destination / "last.pt")
    with pytest.raises(ValueError, match="matching scheduler"):
        paired.train_model(data, destination, recipe(), device="cpu", resume=True)


def test_stopping_rules_allow_the_configured_optimization_period(tmp_path, monkeypatch):
    data, _ = controlled_training(tmp_path, monkeypatch, lambda epoch: 1 + .1 * epoch,
                                  lambda epoch: 1 / epoch)
    config = {**recipe(), "epochs": 8, "early_stopping": True, "early_stopping_min_epochs": 4,
              "early_stopping_patience": 1, "overtraining_shutdown": True,
              "overtraining_min_epochs": 4, "overtraining_patience": 1}
    history = paired.train_model(data, tmp_path / "run", config, device="cpu")
    assert len(history) == 5
    assert not history.overtraining_detected.iloc[:4].any()
    assert history.iloc[-1].stop_reason == "overtraining"


@pytest.mark.parametrize("settings", [dict(lr_schedule="unknown"), dict(warmup_fraction=1.),
                                     dict(warmup_start_ratio=-.1), dict(min_lr_ratio=float("nan")),
                                     dict(overtraining_min_epochs=True), dict(learning_rate=0)])
def test_invalid_schedule_fails_before_creating_run(tmp_path, settings):
    with pytest.raises(ValueError):
        paired.train_model(None, tmp_path / "unused", settings, device="cpu")
    assert not (tmp_path / "unused").exists()
