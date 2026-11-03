"""The primary uncertainty interval must preserve cross-horizon correlation."""

import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from report_sequence_experiments import export_source_archive, mean_horizon_interval, sha


def test_shared_scenario_draws_cancel_opposing_horizon_gains():
    # Each individual horizon has large uncertainty, but gains cancel exactly
    # within every scenario. Independent horizon draws/averaged CI ends fail.
    old = np.eye(2)[np.array([[1, 0, 0], [0, 1, 0]])]
    new = np.eye(2)[np.array([[0, 1, 0], [1, 0, 0]])]
    targets = np.zeros((2, 3), dtype=int)
    interval = mean_horizon_interval(old, new, targets, np.array(["a", "b"]), draws=1000)
    assert interval["mean_accuracy_gain"] == 0
    assert interval["lower_95"] == 0
    assert interval["upper_95"] == 0


def test_missing_targets_cannot_create_a_spurious_correct_prediction():
    old = np.eye(2)[np.array([[1, 0, 0], [0, 1, 0]])]
    new = np.eye(2)[np.array([[0, 1, 0], [1, 0, 0]])]
    targets = np.array([[0, 0, -1], [0, 0, -1]])
    with pytest.raises(ValueError, match="supported targets"):
        mean_horizon_interval(old, new, targets, np.array(["a", "b"]), draws=1000)


def test_source_archive_preserves_line_endings_and_has_reproducible_bytes(tmp_path):
    experiment = tmp_path / "experiment"
    source = experiment / "source/pkg/example.py"
    source.parent.mkdir(parents=True)
    contents = b"# Frozen Windows source\r\nvalue = 1\r\n"
    source.write_bytes(contents)
    hashes = {"pkg/example.py": sha(source)}
    first, second = tmp_path / "first.zip", tmp_path / "second.zip"
    export_source_archive(experiment, first, hashes)
    export_source_archive(experiment, second, hashes)
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        assert archive.read("pkg/example.py") == contents
        assert archive.getinfo("pkg/example.py").date_time == (1980, 1, 1, 0, 0, 0)
    source.write_bytes(contents.replace(b"\r\n", b"\n"))
    with pytest.raises(ValueError, match="snapshot changed"):
        export_source_archive(experiment, tmp_path / "invalid.zip", hashes)
