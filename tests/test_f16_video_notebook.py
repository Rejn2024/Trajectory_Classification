import ast
import json
from pathlib import Path


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "f16_scripted_manoeuvre_video.ipynb"


def test_f16_video_notebook_is_valid_and_code_compiles():
    notebook = json.loads(NOTEBOOK.read_text())

    assert notebook["nbformat"] == 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    for index, cell in enumerate(code_cells):
        source = "".join(cell["source"])
        ast.parse(source, filename=f"{NOTEBOOK.name}:code-cell-{index}")


def test_f16_video_notebook_covers_simulation_and_video_output():
    contents = NOTEBOOK.read_text()

    assert "jsbsim.FGFDMExec" in contents
    assert "fcs/aileron-cmd-norm" in contents
    assert "f16_scripted_manoeuvre.mp4" in contents
    assert "PillowWriter" in contents
