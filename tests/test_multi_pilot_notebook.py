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

NOTEBOOKS = Path(__file__).parents[1] / "notebooks"


@pytest.mark.parametrize("seed_stride", [0, 1000])
@pytest.mark.parametrize("number", ["05", "06"])
def test_notebook_executes_configurable_roster_and_replays_selected_pilot(
    tmp_path, monkeypatch, seed_stride, number,
):
    import matplotlib.pyplot as plt
    notebook_path = NOTEBOOKS / f"{number}_train_multiple_pilots.ipynb"

    for key, value in {
        "BVR_PILOT_COUNT": "4",
        "BVR_PILOT_SEED_STRIDE": str(seed_stride),
        "BVR_PILOT_EPOCHS": "1",
        "BVR_PILOT_EPISODE_SECONDS": "1",
        "BVR_PILOT_SCENARIOS": "3",
        "BVR_PILOT_EVALUATION_SCENARIOS": "3",
        "BVR_PILOT_UPDATE_EPOCHS": "1",
        "BVR_PILOT_WORKERS": "1",
        "BVR_PILOT_EXECUTOR": "thread",
        "BVR_PILOT_TORCH_THREADS": "1",
        "BVR_PILOT_CUDA_GRAPHS": "1",
        "BVR_PILOT_OUTPUT": str(tmp_path / "batch"),
        "BVR_PILOT_DEVICE": "cpu",
        "BVR_PILOT_VIEW": "pilot_004",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(NOTEBOOKS.parent)
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
                "opponent_eliminated": False,
                "target_range_m": 30_000.0,
                "target_alignment": 1.0,
                "initial_target_range_m": 30_000.0,
                "initial_target_alignment": 1.0,
                "missiles_avoided_total": 0,
                "incoming_missiles": 0,
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

    notebook = json.loads(notebook_path.read_text(encoding="utf8"))
    assert notebook["nbformat"] == 4
    namespace = {"display": lambda *_: None}
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            exec(  # noqa: S102 - Execute the repository notebook to test the complete workflow.
                compile("".join(cell["source"]), f"{notebook_path.name}:cell-{index}", "exec"), namespace
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
    assert namespace["config"].cuda_graph_updates
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
    prefix = "combat" if number == "05" else "hybrid"
    assert all(replay["reward"]["name"] == f"{prefix}_objective_004" for replay in replays)
    pilots = namespace["PILOTS"]
    assert [pilot.seed for pilot in pilots] == [pilots[0].seed + seed_stride * i for i in range(4)]
    assert len({tuple(pilot.reward.parameters["weights"].values()) for pilot in pilots}) == 4
    if number == "05":
        from bvr_behavior_prediction.rl.reward import CombatReward
        assert all(isinstance(pilot.reward.factory(), CombatReward) for pilot in pilots)
        assert all(result["benchmark_reward"]["name"] == "common_combat"
                   for result in namespace["results"])
        for result, pilot in zip(namespace["results"], pilots):
            weights = pilot.reward.parameters["weights"]
            assert result["selection_return"] == pytest.approx(
                weights["target_lock_acquired"] + weights["ground_clearance"],
            )
            assert result["benchmark_return"] == pytest.approx(2.02)
            assert "pursue" not in result["benchmark_component_means"]
            assert result["benchmark_component_means"]["target_locked"] == 2
    else:
        assert all(sum(pilot.reward.parameters["weights"].values()) == pytest.approx(1)
                   for pilot in pilots)
        assert all(result["benchmark_reward"]["name"] == "common_hybrid_dogfight"
                   for result in namespace["results"])
        assert all(set(result["benchmark_component_means"]) == {
            "evade", "pursue", "eliminate", "loss", "combat_elimination",
            "lock_acquired", "locked_launch", "missile_support",
        } for result in namespace["results"])
        # A fixture lock pays during training, but not in the hybrid benchmark.
        assert all(result["selection_return"] == 2 for result in namespace["results"])
        assert all(result["benchmark_return"] == 0 for result in namespace["results"])
    assert len(list((tmp_path / "batch/pilot_004/final_evaluation").glob("*.acmi"))) == 3
    assert all(instance.closed for instance in instances)


@pytest.mark.parametrize("number,epochs,family", [("05", 150, "combat"), ("06", 80, "hybrid")])
def test_population_defaults_keep_acceleration_and_separate_outputs(
    monkeypatch, number, epochs, family,
):
    import os

    for key in list(os.environ):
        if key.startswith("BVR_PILOT_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("BVR_PILOT_COUNT", "13")
    monkeypatch.chdir(NOTEBOOKS.parent)
    notebook = json.loads((NOTEBOOKS / f"{number}_train_multiple_pilots.ipynb").read_text())
    namespace = {"display": lambda *_: None}
    old_threads = torch.get_num_threads()
    try:
        for index in (2, 6, 7):
            exec(  # noqa: S102 - Verify the executable notebook defaults without training.
                compile("".join(notebook["cells"][index]["source"]), "setup", "exec"), namespace,
            )
    finally:
        torch.set_num_threads(old_threads)
    config = namespace["config"]
    assert config.epochs == epochs
    assert len(namespace["PILOTS"]) == 13
    assert config.output_dir.parent.name == family
    assert config.mlflow_experiment == f"jsbsim-{family}-population"
    assert config.simulator_executor == "process"
    assert config.simulator_workers == max(1, ((os.cpu_count() or 1) * 80) // 100)
    assert config.cuda_graph_updates
    assert config.scenarios_per_epoch == config.evaluation_scenarios_per_epoch == 50
    assert config.update_epochs == 50
    assert config.minibatch_size == 64
