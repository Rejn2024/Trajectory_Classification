import ast
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

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
    assert 'simulator_workers=int(os.getenv("BVR_PILOT_WORKERS",' in source
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


def test_single_pilot_notebook_trains_and_replays_with_original_combat_rewards(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("mlflow")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from bvr_behavior_prediction.rl.reward import CombatReward

    for key, value in {
        "BVR_PILOT_EPOCHS": "1", "BVR_PILOT_EPISODE_SECONDS": "2",
        "BVR_PILOT_SCENARIOS": "3", "BVR_PILOT_EVALUATION_SCENARIOS": "3",
        "BVR_PILOT_WORKERS": "2", "BVR_PILOT_DEVICE": "cpu",
        "BVR_PILOT_OUTPUT": str(tmp_path / "single_pilot"),
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(NOTEBOOK.parents[1])
    monkeypatch.setattr(plt, "show", lambda: None)
    instances = []

    class Backend:
        OBSERVATION_SIZE = 47

        def __init__(self, scenario_dict, recording_path=None, log_dir=None):
            self.path, self.closed = recording_path, False
            instances.append(self)

        def reset(self, seed):
            self.tick = 0
            if self.path:
                self.path.write_text("fixture replay", encoding="utf8")
            return np.zeros(47, np.float32), {}

        def step(self, *_):
            self.tick += 1
            # Deliberately omit all new dogfight-only signals. Legacy notebook 04
            # must still train using its original lock/launch/evasion/kill recipe.
            return np.zeros(47, np.float32), 0.0, self.tick == 10, {
                "blue_alive": True, "blue_altitude_m": 6000,
                "target_locked": self.tick < 4,
                "fired": self.tick == 2, "fired_with_lock": self.tick == 2,
                "missile_avoided": self.tick == 5, "opponent_destroyed": self.tick == 10,
            }

        def close(self):
            self.closed = True

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf8"))
    namespace = {"display": lambda *_: None}
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            exec(  # noqa: S102 - Execute the notebook to verify its actual legacy workflow.
                compile("".join(cell["source"]), f"{NOTEBOOK.name}:cell-{index}", "exec"), namespace
            )
            if index == 2:
                namespace["BVRJSBSimBackend"] = Backend
            if index == 7:
                namespace["config"] = replace(
                    namespace["config"], hidden_size=8, transformer_heads=2,
                    transformer_layers=1, update_epochs=1,
                    mlflow_tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
                )
    finally:
        torch.set_num_threads(old_threads)
        plt.close("all")
    trainer = namespace["trainer"]
    assert not trainer.config.cuda_graph_updates
    assert trainer.config.resolved_simulator_workers == 2
    assert isinstance(trainer.reward_definition.factory(), CombatReward)
    assert trainer.reward_definition.name == "combat"
    assert (trainer.output / "best_model.pt").is_file()
    assert len(namespace["final_evaluation_manifest"]) == 3
    for flight in namespace["final_evaluation_manifest"]:
        assert flight["score"] == pytest.approx(280.02)
        parts = flight["final_info"]["reward_component_totals"]
        assert parts["target_locked"] == 2
        assert parts["fired_with_lock"] == 8
        assert parts["missile_avoided"] == 20
        assert parts["opponent_destroyed"] == 250
        assert "combat_elimination" not in parts
    assert all(instance.closed for instance in instances)
