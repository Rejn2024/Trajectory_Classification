import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mlflow")
pytest.importorskip("pandas")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

NOTEBOOK = Path(__file__).parents[1] / "notebooks/05_train_multiple_pilots.ipynb"


def test_notebook_executes_configurable_roster_and_replays_selected_pilot(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    for key, value in {
        "BVR_PILOT_COUNT": "4",
        "BVR_PILOT_EPOCHS": "1",
        "BVR_PILOT_EPISODE_SECONDS": "1",
        "BVR_PILOT_SCENARIOS": "3",
        "BVR_PILOT_EVALUATION_SCENARIOS": "3",
        "BVR_PILOT_UPDATE_EPOCHS": "1",
        "BVR_PILOT_WORKERS": "1",
        "BVR_PILOT_EXECUTOR": "thread",
        "BVR_PILOT_TORCH_THREADS": "1",
        "BVR_PILOT_OUTPUT": str(tmp_path / "batch"),
        "BVR_PILOT_DEVICE": "cpu",
        "BVR_PILOT_VIEW": "pilot_004",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(NOTEBOOK.parents[1])
    monkeypatch.setattr(plt, "show", lambda: None)
    instances = []

    class Backend:
        OBSERVATION_SIZE = 47

        def __init__(self, scenario_dict, recording_path=None, log_dir=None):
            self.path = recording_path
            self.closed = False
            self.info = {
                "blue_speed_mps": 250.0,
                "red_speed_mps": 250.0,
                "blue_altitude_m": 6000.0,
                "red_altitude_m": 6000.0,
                "blue_alive": True,
                "target_locked": True,
                "opponent_destroyed": False,
            }
            instances.append(self)

        def reset(self, seed):
            self.steps = 0
            if self.path:
                self.path.write_text("notebook fixture replay", encoding="utf8")
            return np.zeros(47, np.float32), self.info

        def step(self, blue_action, red_action):
            self.steps += 1
            return np.zeros(47, np.float32), 0.0, self.steps >= 10, self.info

        def close(self):
            self.closed = True

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf8"))
    assert notebook["nbformat"] == 4
    namespace = {"display": lambda *_: None}
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            exec(  # noqa: S102 - Execute the repository notebook to test the complete workflow.
                compile("".join(cell["source"]), f"{NOTEBOOK.name}:cell-{index}", "exec"), namespace
            )
            if index == 2:
                namespace["BVRJSBSimBackend"] = Backend
            if index == 7:
                namespace["config"] = replace(
                    namespace["config"],
                    hidden_size=8,
                    transformer_heads=2,
                    transformer_layers=1,
                    mlflow_tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
                )
    finally:
        torch.set_num_threads(old_threads)
        plt.close("all")
    assert len(namespace["results"]) == 4
    assert len(namespace["metrics_by_pilot"]) == 4
    replays = namespace["final_evaluation_manifest"]
    assert len(replays) == 3
    evaluation_seed = namespace["results"][0]["evaluation_seed"]
    assert all(result["evaluation_seed"] == evaluation_seed for result in namespace["results"])
    evaluation_seeds = set(range(evaluation_seed, evaluation_seed + 3))
    assert {replay["seed"] for replay in replays} == evaluation_seeds
    assert all(
        not evaluation_seeds.intersection(range(pilot.seed, pilot.seed + 3))
        for pilot in namespace["PILOTS"]
    )
    assert all(replay["pilot_id"] == "pilot_004" for replay in replays)
    assert all(replay["reward"]["name"] == "objective_004" for replay in replays)
    assert len(list((tmp_path / "batch/pilot_004/final_evaluation").glob("*.acmi"))) == 3
    assert all(instance.closed for instance in instances)
