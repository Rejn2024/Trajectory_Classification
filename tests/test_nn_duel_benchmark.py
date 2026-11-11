"""Exercise the notebook's retry and error reporting without running simulators."""

# Execute the repository's actual benchmark cell, with simulator commands replaced.
# ruff: noqa: S102

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import generate_nn_duel_corpus as duel


@pytest.mark.parametrize("initial_failures", [{1, 2}, {2}])
def test_benchmark_retries_failures_and_reuses_successes(tmp_path, initial_failures):
    notebook = json.loads(
        (Path(__file__).parents[1] / "notebooks/11_nn_vs_nn_training_data.ipynb").read_text()
    )
    source = next(
        "".join(cell["source"]).removeprefix("%%time\n")
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "benchmark_path = WORK /" in "".join(cell["source"])
    )
    failures = set(initial_failures)
    collected = []
    case = {"episode_id": "probe"}

    def run(command, *, work=tmp_path, workers=None, **kwargs):
        if command == "prepare":
            duel.write(
                work / "config.json",
                {
                    "inference_backend": "local_cpu",
                    "collector_hashes": {},
                    "software": {},
                },
            )
        if command != "collect":
            return
        collected.append(workers)
        if workers in failures:
            raise subprocess.CalledProcessError(
                1, ["collector"], output="Log: example_collect.log\nWrong simulator source imported"
            )
        duel.write(work / "config.json", {"cases": [case]})
        duel.write(work / "progress.json", {"wall_seconds": 1, "cumulative_wall_seconds": 1})
        directory = work / "episodes/probe"
        directory.mkdir(parents=True)
        payload = directory / "observations.parquet"
        payload.write_bytes(b"same deterministic data")
        duel.write(
            directory / "summary.json",
            {
                "episode_id": "probe",
                "files": {payload.name: duel.sha256(payload)},
                "case_sha256": hashlib.sha256(
                    json.dumps(case, sort_keys=True).encode()
                ).hexdigest(),
                "duration_s": 1,
                "wall_seconds": 1,
                "timing": {"decision_seconds": 0.1},
            },
        )

    namespace = {
        "WORK": tmp_path,
        "BACKEND": "auto",
        "CUDA_AVAILABLE": False,
        "DETERMINISTIC": True,
        "FORCE_SPEED_COMMAND": False,
        "config": {"collector_hashes": {}, "software": {}},
        "runtime": {},
        "WORKER_CANDIDATES": [1, 2],
        "BENCHMARK_FLIGHTS": 1,
        "EPISODE_SECONDS": 1,
        "BATCH_WAIT_MS": 2,
        "BATCH_MAX_REQUESTS": 32,
        "RUN_BENCHMARK": True,
        "SEED": 42,
        "FLIGHTS": 10,
        "SHARD_FLIGHTS": 100,
        "UPLOAD_MBPS": 20,
        "CHECKSUM_READ_MBPS": 400,
        "GIT_SECONDS_PER_SNAPSHOT": 10,
        "duel": duel,
        "pd": pd,
        "np": np,
        "hashlib": hashlib,
        "json": json,
        "subprocess": subprocess,
        "time": time,
        "shutil": shutil,
        "run": run,
        "display": lambda _: None,
    }
    code = compile(source, "notebook benchmark", "exec")
    if initial_failures == {1, 2}:
        with pytest.raises(RuntimeError, match="Wrong simulator source imported") as error:
            exec(code, namespace)
        assert "example_collect.log" in str(error.value)
    else:
        exec(code, namespace)
    assert collected == [1, 2]
    collected.clear()
    failures.clear()
    exec(code, namespace)
    assert collected == sorted(initial_failures)
    assert all(
        row["status"] == "ok" for row in duel.read(tmp_path / "benchmark_batched.json")["results"]
    )
    collected.clear()
    exec(code, namespace)
    assert collected == []  # A wholly successful cached benchmark needs no new rollouts.
    namespace["FORCE_SPEED_COMMAND"] = True
    exec(code, namespace)
    assert collected == [1, 2]  # Changing speed mode invalidates every cached candidate.
    assert duel.read(tmp_path / "benchmark_batched.json")["identity"]["native_throttle_override"]
