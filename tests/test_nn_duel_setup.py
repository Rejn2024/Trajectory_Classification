"""Temporary setup must survive switching kernels without relaxing campaign checks."""

import hashlib
import importlib.metadata
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import generate_nn_duel_corpus as duel
import nn_duel_paths as duel_paths


def select_setup(namespace):
    path = Path(__file__).parents[1] / "notebooks/11_nn_vs_nn_training_data.ipynb"
    notebook = json.loads(path.read_text())
    source = "".join(notebook["cells"][3]["source"])
    source = source[source.index("setup_sources =") : source.index('run("prepare",')]
    exec(compile(source, str(path), "exec"), namespace)  # noqa: S102 -- trusted repository cell
    return namespace["SETUP_WORK"]


@pytest.fixture
def setup_environment(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    parent.mkdir()
    archive = parent / "reproduction.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("experiment.json", json.dumps({"source_experiment": "test-tactical"}))
    duel.write(
        parent / "manifest.json",
        {
            "files": {
                "reproduction.zip": {
                    "sha256": duel.sha256(archive),
                }
            }
        },
    )
    monkeypatch.setattr(duel, "ARCHIVE", archive)
    monkeypatch.setattr(duel, "PARENT", parent)
    monkeypatch.setattr(duel, "verify_snapshot", lambda _: {})
    versions = {p: "1.0" for p in ("numpy", "pandas", "pyarrow", "torch", "jsbsim", "gymnasium")}
    monkeypatch.setattr(importlib.metadata, "version", lambda name: versions[name])
    namespace = {
        "duel": duel,
        "duel_paths": duel_paths,
        "importlib": importlib,
        "hashlib": hashlib,
        "json": json,
        "sys": SimpleNamespace(version=sys.version),
        "WORK": tmp_path / "work",
        "FLIGHTS": 3,
        "SEED": 42,
        "EPISODE_SECONDS": 2.0,
        "DETERMINISTIC": True,
        "FORCE_SPEED_COMMAND": False,
        "BATCH_WAIT_MS": 2.0,
        "BATCH_MAX_REQUESTS": 32,
        "WORKER_CANDIDATES": [1],
    }
    return namespace, versions


def test_switching_environment_prepares_new_setup_and_preserves_old_config(setup_environment):
    namespace, versions = setup_environment
    work = namespace["WORK"]
    duel.write(work / "config.json", {"main_campaign": "must remain unchanged"})
    old_setup = select_setup(namespace)
    original = duel.prepare(old_setup, flights=3, seed=42, seconds=2.0)
    assert select_setup(namespace) == old_setup
    assert duel.prepare(old_setup, flights=3, seed=42, seconds=2.0) == original

    versions["torch"] = "2.10+cu128"
    # Reproduce the actual exception when the old directory is reused.
    with pytest.raises(ValueError, match="Resume configuration/source/software differs"):
        duel.prepare(old_setup, flights=3, seed=42, seconds=2.0)
    new_setup = select_setup(namespace)
    assert new_setup != old_setup
    current = duel.prepare(new_setup, flights=3, seed=42, seconds=2.0)
    assert current["software"]["torch"] == versions["torch"]
    assert duel.read(old_setup / "config.json") == original
    assert duel.read(work / "config.json") == {"main_campaign": "must remain unchanged"}


@pytest.mark.parametrize(
    "package", ["numpy", "pandas", "pyarrow", "torch", "jsbsim", "gymnasium", "python"]
)
def test_setup_key_changes_with_each_recorded_software_version(setup_environment, package):
    namespace, versions = setup_environment
    original = select_setup(namespace)
    if package == "python":
        namespace["sys"].version = "3.99.0 test"
    else:
        versions[package] = "changed"
    assert select_setup(namespace) != original


@pytest.mark.parametrize(
    "option,value",
    [
        ("FLIGHTS", 4),
        ("SEED", 43),
        ("EPISODE_SECONDS", 3.0),
        ("DETERMINISTIC", False),
        ("FORCE_SPEED_COMMAND", True),
        ("BATCH_WAIT_MS", 5.0),
        ("BATCH_MAX_REQUESTS", 8),
    ],
)
def test_setup_key_changes_with_preparation_settings(setup_environment, option, value):
    namespace, _ = setup_environment
    original = select_setup(namespace)
    namespace[option] = value
    assert select_setup(namespace) != original


def test_speed_mode_is_recorded_and_cannot_change_on_resume(setup_environment):
    namespace, _ = setup_environment
    work = select_setup(namespace)
    config = duel.prepare(work, flights=3, seed=42, seconds=2.0)
    assert config["native_throttle_override"] is False
    with pytest.raises(ValueError, match="Resume configuration/source/software differs"):
        duel.prepare(work, flights=3, seed=42, seconds=2.0, native_throttle_override=True)
    assert duel.read(work / "config.json") == config
    namespace["FORCE_SPEED_COMMAND"] = True
    legacy = duel.prepare(
        select_setup(namespace), flights=3, seed=42, seconds=2.0, native_throttle_override=True
    )
    assert legacy["native_throttle_override"] is True


@pytest.mark.parametrize("invalid", ["false", 0, 1, None])
def test_speed_mode_rejects_non_boolean_preparation(tmp_path, invalid):
    with pytest.raises(TypeError, match="native_throttle_override must be a boolean"):
        duel.prepare(tmp_path, native_throttle_override=invalid)


@pytest.mark.parametrize("legacy", [False, True])
def test_cli_passes_speed_mode_to_preparation(tmp_path, monkeypatch, legacy):
    calls = []

    def prepare(*args, **kwargs):
        calls.append(kwargs["native_throttle_override"])
        return {"flights": 3, "pilots": {}}

    monkeypatch.setattr(duel, "prepare", prepare)
    argv = ["collector", "prepare", "--work", str(tmp_path / "work"), "--flights", "3"]
    if legacy:
        argv.append("--force-speed-command")
    monkeypatch.setattr(sys, "argv", argv)
    duel.main()
    assert calls == [legacy]
