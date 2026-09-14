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


def test_training_notebook_links_windows_torch_dll_recovery():
    notebook = json.loads(NOTEBOOK.read_text())
    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    )
    guide = (ROOT / "notebooks" / "README.md").read_text()

    assert "DLL load failed" in markdown
    assert "importing _C" in markdown
    assert "Windows PyTorch DLL recovery" in guide
    assert "python -c \"import torch;" in guide
    assert "Microsoft Visual C++" in guide
    assert "Redistributable (x64)" in guide
