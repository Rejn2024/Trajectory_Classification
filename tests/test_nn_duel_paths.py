"""Catch Windows archive path overflow before any partial snapshot is extracted."""

import functools
import json
import sys
import zipfile
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from nn_duel_paths import check_snapshot_paths

MEMBER = (
    "source/bvr_sim_source/bvr_sim/resources/jsbsim/aircraft/"
    "Submarine_Scout/Systems/Submarine_Scout_crew_ap.xml"
)
WINDOWS_ROOT = PureWindowsPath(r"C:\Users\theon\Trajectory_Classification")


@pytest.fixture
def archive(tmp_path):
    path = tmp_path / "reproduction.zip"
    with zipfile.ZipFile(path, "w") as source:
        source.writestr("experiment.json", "{}")
        source.writestr(MEMBER, "<xml/>")
    return path


def test_reported_directory_exceeds_windows_path_limit(archive):
    work = WINDOWS_ROOT / (
        "artifacts/nn_duels_20k_v1/20261107T235219_567783Z_8f886e25/"
        "benchmarks/0cee00de7b2a/local_cpu_6"
    )
    assert len(str(work / "tactical_snapshot" / MEMBER)) == 261
    with pytest.raises(RuntimeError, match="261 characters") as error:
        check_snapshot_paths(archive, [work], windows=True)
    assert "BVR_NN_WORK" in str(error.value)
    assert MEMBER.rsplit("/", 1)[1] in str(error.value)


def test_short_series_fits_setup_main_and_all_benchmark_candidates(archive):
    work = WINDOWS_ROOT / "artifacts/nn_duels_20k_v1/0123456789ab"
    directories = [work, work / "setup/0cee00de7b2a"] + [
        work / "b/0cee00de7b2a" / f"{backend}_{workers}"
        for backend in ("local_cpu", "central_cpu", "central_cuda")
        for workers in (6, 12)
    ]
    assert check_snapshot_paths(archive, directories, windows=True) == 236


def test_long_custom_work_root_fails_before_notebook_preparation(archive):
    notebook = json.loads((ROOT / "notebooks/11_nn_vs_nn_training_data.ipynb").read_text())
    source = "".join(notebook["cells"][3]["source"])
    check = source[source.index("# Check every archive member") : source.index('config = duel.read(')]
    work = WINDOWS_ROOT / ("long_checkout_path_" * 5) / "0123456789ab"
    commands = []
    scope = {
        "WORK": work,
        "SETUP_WORK": work / "setup/0cee00de7b2a",
        "WORKER_CANDIDATES": [6, 12],
        "FLIGHTS": 20000,
        "SEED": 42,
        "duel": SimpleNamespace(ARCHIVE=archive),
        "duel_paths": SimpleNamespace(
            check_snapshot_paths=functools.partial(check_snapshot_paths, windows=True)
        ),
        "run": lambda *args, **kwargs: commands.append(args),
    }
    with pytest.raises(RuntimeError, match="BVR_NN_WORK"):
        exec(compile(check, "notebook snapshot preflight", "exec"), scope)  # noqa: S102
    assert commands == []


def test_path_budget_checks_every_member_and_counts_windows_utf16_units(archive):
    # A later non-ASCII member is longer in UTF-16 units than Python character count.
    member = "source/" + "\U0001f680" * 60 + "/asset.xml"
    with zipfile.ZipFile(archive, "a") as source:
        source.writestr(member, "<xml/>")
    work = WINDOWS_ROOT / "artifacts/nn_duels_20k_v1/0123456789ab/b/0cee00de7b2a/central_cuda_12"
    assert len(str(work / "tactical_snapshot" / member)) < 260
    with pytest.raises(RuntimeError, match="Windows snapshot path"):
        check_snapshot_paths(archive, [work], windows=True)
