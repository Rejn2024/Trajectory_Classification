import ast
import json
from pathlib import Path


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "04_train_rl_pilot.ipynb"


def _source_cells():
    notebook = json.loads(NOTEBOOK.read_text())
    return ["".join(cell.get("source", [])) for cell in notebook["cells"]]


def test_rl_pilot_notebook_is_valid_and_code_compiles():
    notebook = json.loads(NOTEBOOK.read_text())
    assert notebook["nbformat"] == 4
    for source in _source_cells():
        if source and source in [
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]:
            ast.parse(source)


def test_rl_pilot_notebook_uses_real_pipeline_and_visualizes_after_training():
    cells = _source_cells()
    source = "\n".join(cells)
    training_index = next(i for i, cell in enumerate(cells) if "trained_pilot = trainer.train()" in cell)
    plot_index = next(i for i, cell in enumerate(cells) if 'axes[0].plot(metrics["epoch"]' in cell)

    assert "PPOTrainer(" in source
    assert "BluePilotEnvironment(" in source
    assert "PilotTrainingConfig(" in source
    assert "class LearningSmokeBackend:" in source
    assert "OBSERVATION_SIZE = 8" in source
    assert 'pd.read_json(OUTPUT_DIR / "training_metrics.jsonl", lines=True)' in source
    assert training_index < plot_index
