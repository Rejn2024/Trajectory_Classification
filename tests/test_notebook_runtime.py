"""Missing packages and Windows policy blocks need different repair instructions."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bvr_behavior_prediction import notebook_runtime as runtime


def fail_import(monkeypatch, error):
    def blocked(name):
        assert name == "pyarrow.parquet"
        raise error

    metadata = SimpleNamespace(version=lambda name: "26.0.1",
                               PackageNotFoundError=runtime.importlib.metadata.PackageNotFoundError)
    monkeypatch.setattr(runtime, "importlib", SimpleNamespace(import_module=blocked, metadata=metadata))


def test_application_control_preserves_cause_and_identifies_active_environment(monkeypatch):
    original = ImportError(
        "DLL load failed while importing lib: An Application Control policy has blocked this file."
    )
    fail_import(monkeypatch, original)
    monkeypatch.setattr(runtime, "sys", SimpleNamespace(
        executable=r"C:\envs\trajectory-nn-clean\python.exe", prefix=r"C:\envs\trajectory-nn-clean"))
    with pytest.raises(RuntimeError, match="Windows Application Control blocked") as caught:
        runtime.require_pyarrow()
    message = str(caught.value)
    assert "CodeIntegrity > Operational" in message
    assert "policy administrator" in message
    assert "26.0.1" in message and r"C:\envs\trajectory-nn-clean\python.exe" in message
    assert caught.value.__cause__ is original


def test_missing_package_reports_installation_not_policy(monkeypatch):
    fail_import(monkeypatch, ModuleNotFoundError("No module named 'pyarrow'", name="pyarrow"))

    def missing_version(name):
        raise runtime.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(runtime.importlib.metadata, "version", missing_version)
    with pytest.raises(RuntimeError, match="PyArrow is missing") as caught:
        runtime.require_pyarrow()
    assert "Installed PyArrow version: not installed" in str(caught.value)
    assert "Windows Application Control blocked" not in str(caught.value)


def test_missing_dll_is_not_misdiagnosed_as_a_policy_block(monkeypatch):
    fail_import(monkeypatch, ImportError("DLL load failed: The specified module could not be found."))
    with pytest.raises(RuntimeError, match="native dependencies") as caught:
        runtime.require_pyarrow()
    assert "Windows Application Control blocked" not in str(caught.value)


def test_healthy_parquet_import_is_returned_without_changes(monkeypatch):
    parquet = object()
    monkeypatch.setattr(runtime, "importlib", SimpleNamespace(import_module=lambda name: parquet))
    assert runtime.require_pyarrow() is parquet


@pytest.mark.parametrize("name", ["12_prepare_nn_duel_skill_forecasting.ipynb",
                                 "13_nn_duel_skill_forecasting.ipynb"])
def test_notebooks_check_parquet_before_importing_forecasting_modules(name, monkeypatch):
    path = Path(__file__).parents[1] / "notebooks" / name
    notebook = json.loads(path.read_text())
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if cell["cell_type"] == "code").removeprefix("%%time\n")
    original = ImportError("An Application Control policy has blocked this file.")
    fail_import(monkeypatch, original)
    monkeypatch.chdir(path.parent)
    with pytest.raises(RuntimeError, match="Windows Application Control blocked") as caught:
        exec(compile(source, str(path), "exec"), {})
    assert caught.value.__cause__ is original
