"""Notebook setup starts fresh even when an older campaign is pinned in the shell."""

# Execute only the trusted, checked-in setup cell against an isolated temporary checkout.
# ruff: noqa: S102

import json
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).parents[1] / "notebooks/31_continue_varied_pilots_and_generate_corpus.ipynb"
SETUP = "".join(json.loads(NOTEBOOK.read_text())["cells"][2]["source"])


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts/run_varied_pilot_weekend.py").touch()
    monkeypatch.setenv("BVR_WEEKEND_ROOT", str(root))
    monkeypatch.delenv("BVR_WEEKEND_MODE", raising=False)
    monkeypatch.delenv("BVR_WEEKEND_SOURCE", raising=False)
    return root


def test_default_ignores_stale_job_and_output_and_preserves_old_files(checkout, monkeypatch):
    old_job = checkout / "old-job"
    old_output = checkout / "old-corpus"
    old_job.mkdir()
    old_output.mkdir()
    plan = old_job / "campaign.json"
    plan.write_text('{"source_hashes": "old pinned source"}')
    marker = old_output / "saved-flight"
    marker.write_bytes(b"original data")
    monkeypatch.setenv("BVR_WEEKEND_JOB", str(old_job))
    monkeypatch.setenv("BVR_WEEKEND_OUTPUT", str(old_output))
    namespace = {}
    exec(compile(SETUP, str(NOTEBOOK), "exec"), namespace)
    job, output = namespace["JOB"], namespace["OUTPUT"]
    assert job != old_job and output != old_output
    assert job.is_dir() and output.is_dir()
    assert not (job / "campaign.json").exists()
    assert not namespace["ACCEPT_PERFORMANCE_FIX"]
    assert "--accept-performance-fix" not in namespace["ARGS"]
    assert plan.read_text() == '{"source_hashes": "old pinned source"}'
    assert marker.read_bytes() == b"original data"
    # Reexecuting setup in this kernel resumes the newly selected campaign.
    exec(compile(SETUP, str(NOTEBOOK), "exec"), namespace)
    assert (namespace["JOB"], namespace["OUTPUT"]) == (job, output)
    # A fresh kernel gets a distinct campaign and corpus, never a previous plan.
    another = {}
    exec(compile(SETUP, str(NOTEBOOK), "exec"), another)
    assert another["JOB"] != job and another["OUTPUT"] != output


def test_resume_uses_explicit_paths(checkout, monkeypatch):
    job, output = checkout / "chosen-job", checkout / "chosen-corpus"
    monkeypatch.setenv("BVR_WEEKEND_MODE", "resume")
    monkeypatch.setenv("BVR_WEEKEND_JOB", str(job))
    monkeypatch.setenv("BVR_WEEKEND_OUTPUT", str(output))
    namespace = {}
    exec(compile(SETUP, str(NOTEBOOK), "exec"), namespace)
    assert namespace["JOB"] == job
    assert namespace["OUTPUT"] == output


@pytest.mark.parametrize("mode", ["resume", "invalid"])
def test_invalid_resume_selection_fails_before_launch(checkout, monkeypatch, mode):
    monkeypatch.setenv("BVR_WEEKEND_MODE", mode)
    monkeypatch.delenv("BVR_WEEKEND_JOB", raising=False)
    with pytest.raises(ValueError):
        exec(compile(SETUP, str(NOTEBOOK), "exec"), {})


@pytest.mark.parametrize("cpus,expected", [(16, 14), (8, 6), (3, 2), (64, 14)])
def test_paired_defaults_reserve_driver_capacity(checkout, monkeypatch, cpus, expected):
    import os

    monkeypatch.setattr(os, "cpu_count", lambda: cpus)
    for key in ("BVR_WEEKEND_WORKERS", "BVR_WEEKEND_TORCH_THREADS", "BVR_WEEKEND_PILOT_SCHEDULE"):
        monkeypatch.delenv(key, raising=False)
    namespace = {}
    exec(compile(SETUP, str(NOTEBOOK), "exec"), namespace)
    assert namespace["WORKERS"] == expected
    assert namespace["TORCH_THREADS"] == 2
    assert namespace["PILOT_SCHEDULE"] == "paired"
