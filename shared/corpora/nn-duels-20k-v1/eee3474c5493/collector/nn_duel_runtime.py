"""Initialize native libraries consistently and probe them outside the kernel."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import uuid
from pathlib import Path


def bootstrap_native_runtime():
    # On Windows, loading MKL-backed NumPy/the simulator before Torch can load
    # Conda's Intel OpenMP DLL before Torch loads its bundled copy (OMP Error 15).
    # Let Torch select/initialize its runtime first in every spawned process.
    if sys.platform == "win32":
        import torch

        torch.set_num_threads(1)


def probe(numpy_first=False):
    print(f"Native runtime probe: {sys.executable}", flush=True)
    # These are candidate DLLs, not a claim that all of them are loaded. Print
    # before importing native packages so even a native abort leaves evidence.
    roots = {Path(sys.prefix) / "Library/bin", Path(sys.base_prefix) / "Library/bin"}
    spec = importlib.util.find_spec("torch")
    if spec and spec.origin:
        roots.add(Path(spec.origin).parent / "lib")
    for root in sorted(roots):
        for dll in sorted(root.glob("*iomp*.dll")):
            print(f"OpenMP DLL candidate: {dll}", flush=True)

    if numpy_first:
        print("Kernel already loaded NumPy; checking NumPy-first initialization", flush=True)
        import numpy as np

        array = np.ones((32, 32))
        array @ array

    print("Importing/initializing Torch CPU runtime", flush=True)
    import torch

    torch.set_num_threads(1)
    torch.ones((32, 32)).mm(torch.ones((32, 32)))
    print("Torch CPU operation passed; importing rollout dependencies", flush=True)
    import cv2
    import gymnasium  # noqa: F401 -- exercise the same native dependency mix as rollouts
    import jsbsim  # noqa: F401
    import numpy as np
    import pandas  # noqa: F401
    import pyarrow  # noqa: F401
    from scipy.linalg.blas import dgemm

    array = np.ones((32, 32))
    if not (np.allclose(array @ array, 32) and np.allclose(dgemm(1, array, array), 32)):
        raise RuntimeError("NumPy/SciPy BLAS probe returned incorrect values")
    cv2.resize(array, (16, 16))
    if not torch.all(torch.ones((32, 32)).mm(torch.ones((32, 32))) == 32).item():
        raise RuntimeError("Torch CPU probe returned incorrect values")
    print("Torch, NumPy, SciPy BLAS and OpenCV operations passed", flush=True)
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        pass
    else:
        print(f"Loaded native thread pools: {threadpool_info()}", flush=True)


def run_preflight(log_directory):
    """A native OpenMP abort cannot be caught by Python in the failing process."""
    directory = Path(log_directory)
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / f"native_runtime_{uuid.uuid4().hex[:12]}.log"
    env = os.environ.copy()
    args = [sys.executable, "-u", str(Path(__file__).resolve()), "check"]
    if sys.platform == "win32" and "numpy" in sys.modules:
        # IDE/kernel extensions can import NumPy before the notebook's first cell.
        # Validate that order before initializing Torch inside this kernel.
        args.append("--numpy-first")
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[key] = "1"
    print(f"Checking native rollout libraries in a fresh process. Log: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            args,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=120,
        )
    if result.returncode:
        with log_path.open("r", encoding="utf-8", errors="replace") as stream:
            output = stream.read(12000)
        raise RuntimeError(
            f"Native runtime preflight failed (exit {result.returncode}) before rollout workers "
            f"were started. Log: {log_path}\n{output}\n"
            "For OMP Error #15 / libiomp5md.dll, repair the mixed native-library environment; "
            "see docs/nn_duel_gpu_batching.md (Windows OpenMP). Restart the kernel afterward."
        )
    print(f"Native runtime preflight passed. Log: {log_path}", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] not in (["check"], ["check", "--numpy-first"]):
        raise SystemExit("Usage: python scripts/nn_duel_runtime.py check [--numpy-first]")
    probe(numpy_first="--numpy-first" in sys.argv)
