import json
import tomllib
from pathlib import Path


ROOT = Path(__file__).parents[1]
NOTEBOOK = ROOT / "notebooks" / "03_train_jsbsim_skill_classifier.ipynb"


def _notebook_source():
    notebook = json.loads(NOTEBOOK.read_text())
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def test_pyarrow_dependency_includes_native_hotfix_release():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]

    assert "pyarrow>=14.0.1" in project["dependencies"]


def test_parquet_registry_error_has_actionable_recovery_message():
    source = _notebook_source()

    assert "except pa.ArrowKeyError as error:" in source
    assert "restart the " in source
    assert "notebook kernel, and run all cells again." in source


def test_training_uses_a_safe_cpu_default_and_allows_explicit_cuda_opt_in():
    source = _notebook_source()

    assert 'os.getenv("BVR_TRAIN_DEVICE", "cpu")' in source
    assert 'if REQUESTED_DEVICE.startswith("cuda") and not torch.cuda.is_available():' in source
    assert 'torch.device(REQUESTED_DEVICE)' in source


def test_training_limits_torch_threads_to_avoid_notebook_resource_exhaustion():
    source = _notebook_source()

    assert 'BVR_TORCH_THREADS' in source
    assert "torch.set_num_threads(TORCH_THREADS)" in source


def test_cuda_gru_uses_safe_aten_fallback_by_default():
    source = _notebook_source()

    assert 'BVR_GRU_USE_CUDNN' in source
    assert 'os.getenv("BVR_GRU_USE_CUDNN", "0") == "1"' in source
    assert "with torch.backends.cudnn.flags(enabled=GRU_USE_CUDNN):" in source


def test_cuda_configuration_runs_a_synchronized_device_probe():
    source = _notebook_source()

    assert "probe = torch.ones(1, device=DEVICE)" in source
    assert "torch.cuda.synchronize(DEVICE)" in source
    assert '"device_name": torch.cuda.get_device_name(DEVICE)' in source
