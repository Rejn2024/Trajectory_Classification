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
    replay_index = next(
        i for i, cell in enumerate(cells) if "final_evaluation_manifest = []" in cell
    )
    plot_index = next(i for i, cell in enumerate(cells) if 'axes[0].plot(epochs, evaluation_mean' in cell)

    assert "PPOTrainer(" in source
    assert "BluePilotEnvironment(" in source
    assert "PilotTrainingConfig(" in source
    assert "BVRJSBSimBackend" in source
    assert "bvr_jsbsim_backend_factory" in source
    assert "LearningSmokeBackend" not in source
    assert "OBSERVATION_SIZE = BASE_OBSERVATION_SIZE + ENERGY_FEATURES" in source
    assert "history_duration_s=2.0" in source
    assert "sample_interval_s=0.1" in source
    assert 'pd.read_json(OUTPUT_DIR / "training_metrics.jsonl", lines=True)' in source
    assert 'BVR_PILOT_SCENARIOS' in source
    assert 'BVR_PILOT_EVALUATION_SCENARIOS' in source
    assert 'simulator_workers=int(os.getenv("BVR_PILOT_WORKERS", "1"))' in source
    assert "Every epoch reuses those set-ups and their fixed simulator seeds" in source
    assert "generated once from the configured seed" in source
    assert 'metrics["evaluation_return_std"]' in source
    assert 'metrics["training_mean_return"]' in source
    assert 'FINAL_EVALUATION_DIR = OUTPUT_DIR / "final_evaluation"' in source
    assert "for index, scenario in enumerate(trainer.evaluation_scenarios):" in source
    assert "episode_seed = config.seed + index" in source
    assert 'f"scenario_{index:03d}_seed_{episode_seed}.txt.acmi"' in source
    assert "recording_path=acmi_path" in source
    assert "deterministic=True" in source
    assert 'FINAL_EVALUATION_DIR / "manifest.json"' in source
    assert training_index < replay_index < plot_index
