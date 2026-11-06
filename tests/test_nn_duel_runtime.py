"""Native aborts must surface before launching multiple rollout processes."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import nn_duel_runtime as runtime


@pytest.mark.parametrize("windows", [False, True])
def test_spawn_header_initializes_torch_before_numerical_imports(tmp_path, windows):
    (tmp_path / "torch.py").write_text(
        "import builtins\n"
        "def set_num_threads(n): builtins.native_events.append(('torch_threads', n))\n",
        encoding="utf-8",
    )
    prefix = (SCRIPTS / "generate_nn_duel_corpus.py").read_text().split("import numpy as np")[0]
    driver = """
import builtins
import subprocess
import sys
import types
sys.path[:0] = sys.argv[1:3]
import nn_duel_runtime
nn_duel_runtime.sys = types.SimpleNamespace(platform="win32" if sys.argv[3] == "1" else "linux")
builtins.native_events = []
exec(sys.argv[4])
assert builtins.native_events == ([("torch_threads", 1)] if sys.argv[3] == "1" else [])
assert "numpy" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", driver, str(tmp_path), str(SCRIPTS), str(int(windows)), prefix],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_preflight_reports_native_abort_with_original_openmp_error(tmp_path, monkeypatch):
    child = tmp_path / "fatal_runtime.py"
    child.write_text(
        "import os\nprint('OMP: Error #15: libiomp5md.dll already initialized', flush=True)\n"
        "os._exit(3)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime, "__file__", str(child))
    with pytest.raises(RuntimeError, match="OMP: Error #15") as error:
        runtime.run_preflight(tmp_path / "logs")
    assert "before rollout workers were started" in str(error.value)
    assert "repair the mixed native-library environment" in str(error.value)
    logs = list((tmp_path / "logs").glob("*.log"))
    assert len(logs) == 1 and "libiomp5md.dll" in logs[0].read_text()


def test_preflight_child_uses_worker_thread_limits_without_duplicate_override(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("KMP_DUPLICATE_LIB_OK", raising=False)
    child = tmp_path / "healthy_runtime.py"
    child.write_text(
        "import os\n"
        "for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):\n"
        "    assert os.environ[name] == '1'\n"
        "assert 'KMP_DUPLICATE_LIB_OK' not in os.environ\n"
        "print('NATIVE_RUNTIME_OK')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime, "__file__", str(child))
    runtime.run_preflight(tmp_path / "logs")
    assert "NATIVE_RUNTIME_OK" in next((tmp_path / "logs").glob("*.log")).read_text()


@pytest.mark.parametrize("numpy_loaded", [False, True])
def test_preflight_checks_existing_windows_kernel_import_order(tmp_path, monkeypatch, numpy_loaded):
    child = tmp_path / "check_import_order.py"
    child.write_text(
        f"import sys\nassert ('--numpy-first' in sys.argv) == {numpy_loaded!r}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime, "__file__", str(child))
    monkeypatch.setattr(
        runtime,
        "sys",
        SimpleNamespace(
            executable=sys.executable,
            platform="win32",
            modules={"numpy": object()} if numpy_loaded else {},
        ),
    )
    runtime.run_preflight(tmp_path / "logs")


def test_notebook_preflight_precedes_heavy_kernel_imports(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "generate_nn_duel_corpus.py").touch()
    (scripts / "nn_duel_runtime.py").write_text(
        "def run_preflight(log_directory): raise RuntimeError('PREFLIGHT_SENTINEL')\n",
        encoding="utf-8",
    )
    notebook = json.loads(
        (SCRIPTS.parent / "notebooks/11_nn_vs_nn_training_data.ipynb").read_text()
    )
    source = "".join(notebook["cells"][1]["source"]).removeprefix("%%time\n")
    driver = """
import importlib.abc
import os
import sys
os.environ["BVR_NN_ROOT"] = sys.argv[1]
class ScientificImportGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("numpy", "pandas", "torch"):
            raise AssertionError("Kernel native import happened before the probe")
sys.meta_path.insert(0, ScientificImportGuard())
try:
    exec(sys.argv[2])
except RuntimeError as error:
    assert str(error) == "PREFLIGHT_SENTINEL"
else:
    raise AssertionError("Notebook ignored a failed native preflight")
"""
    result = subprocess.run(
        [sys.executable, "-c", driver, str(tmp_path), source],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
