import threading
import time
from pathlib import Path

import numpy as np
import pytest

from bvr_behavior_prediction.rl.config import PilotTrainingConfig, RewardWeights
from bvr_behavior_prediction.rl.reward import CombatReward
from bvr_behavior_prediction.rl.scenarios import ScenarioSampler


def test_config_enforces_episode_and_scenario_requirements():
    config = PilotTrainingConfig()
    assert config.decisions_per_episode == 90
    assert config.scenarios_per_epoch == 50
    assert config.evaluation_scenarios_per_epoch == 50
    assert config.evaluation_interval == 5
    assert 1 <= config.resolved_simulator_workers <= 8
    with pytest.raises(ValueError):
        PilotTrainingConfig(episode_duration_s=121)
    with pytest.raises(ValueError):
        PilotTrainingConfig(scenarios_per_epoch=2)
    with pytest.raises(ValueError):
        PilotTrainingConfig(evaluation_scenarios_per_epoch=2)
    with pytest.raises(ValueError):
        PilotTrainingConfig(simulator_workers=-1)
    with pytest.raises(ValueError):
        PilotTrainingConfig(evaluation_interval=0)


def test_evaluation_reuses_fixed_setups_seeds_and_deterministic_actions():
    torch = pytest.importorskip("torch")
    from bvr_behavior_prediction.rl.trainer import PPOTrainer

    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = PilotTrainingConfig(seed=23)
    trainer.evaluation_scenarios = ["first", "second", "third"]
    trainer.pilot = torch.nn.Linear(1, 1)
    calls = []

    def episodes(scenarios, seeds, deterministic=False):
        calls.append((tuple(scenarios), tuple(seeds), deterministic, trainer.pilot.training))
        return [(None, float(seed), None) for seed in seeds]

    trainer._episodes = episodes
    first = trainer._evaluate()
    second = trainer._evaluate()

    np.testing.assert_array_equal(first, [23.0, 24.0, 25.0])
    np.testing.assert_array_equal(second, first)
    assert calls == [
        (("first", "second", "third"), (23, 24, 25), True, False),
        (("first", "second", "third"), (23, 24, 25), True, False),
    ]
    assert trainer.pilot.training


def test_trainer_establishes_fixed_training_setups_and_episode_seeds(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("mlflow")
    from bvr_behavior_prediction.rl.trainer import PPOTrainer

    config = PilotTrainingConfig(
        seed=23,
        scenarios_per_epoch=50,
        evaluation_scenarios_per_epoch=3,
        hidden_size=16,
        output_dir=tmp_path,
    )
    trainer = PPOTrainer(lambda *_: None, observation_size=1, config=config, device="cpu")

    assert len(trainer.training_scenarios) == 50
    assert trainer.training_scenarios == ScenarioSampler(23).sample_batch(50)
    assert trainer.training_episode_seeds == tuple(range(23, 73))

    calls = []
    trainer._episodes = lambda scenarios, seeds: calls.extend(zip(scenarios, seeds)) or []
    trainer._training_episodes()
    trainer._training_episodes()
    expected = list(zip(trainer.training_scenarios, trainer.training_episode_seeds))
    assert calls == expected * 2


def test_episode_collection_parallelizes_simulators_and_batches_policy(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("mlflow")
    from bvr_behavior_prediction.rl.trainer import PPOTrainer

    lock = threading.Lock()
    concurrent = 0
    peak_concurrent = 0

    class Environment:
        def reset(self, seed):
            return np.full((1, 1), seed, dtype=np.float32)

        def step(self, _name, _parameters):
            nonlocal concurrent, peak_concurrent
            with lock:
                concurrent += 1
                peak_concurrent = max(peak_concurrent, concurrent)
            time.sleep(0.03)
            with lock:
                concurrent -= 1
            return np.zeros((1, 1), dtype=np.float32), True, False, {}

        def close(self):
            pass

    config = PilotTrainingConfig(
        episode_duration_s=1,
        scenarios_per_epoch=3,
        evaluation_scenarios_per_epoch=3,
        simulator_workers=3,
        hidden_size=16,
        history_duration_s=1,
        sample_interval_s=1,
        output_dir=tmp_path,
    )
    trainer = PPOTrainer(lambda *_: Environment(), 1, config, device="cpu")
    batch_sizes = []

    def act_batch(observations, deterministic=False):
        batch_sizes.append(len(observations))
        raw = np.zeros(len(trainer.pilot.parameter_names), dtype=np.float32)
        return tuple(("maintain_heading", {}, 0.0, 0.0, raw.copy()) for _ in observations)

    trainer.pilot.act_batch = act_batch
    episodes = trainer._episodes([object()] * 3, (1, 2, 3))

    assert len(episodes) == 3
    assert batch_sizes == [3]
    assert peak_concurrent == 3
    assert trainer._last_collection_profile["simulator_work_seconds"] >= 0.08
    assert trainer._last_collection_profile["simulator_wall_seconds"] > 0
    assert trainer._last_collection_profile["policy_seconds"] >= 0
    assert trainer._last_collection_profile["simulator_cpu_seconds"] >= 0
    assert "simulator_parallel_speedup" not in trainer._last_collection_profile


def test_cpp_simulator_releases_gil_while_advancing():
    bindings = Path("bvr_sim_source/bvr_sim/src_cxx/pybind11_bindings.cxx").read_text()
    assert '.def("step", &SimCore::step, py::call_guard<py::gil_scoped_release>())' in bindings
    assert 'py::arg("steps"), py::call_guard<py::gil_scoped_release>())' in bindings


def test_trainer_reports_epoch_timing_and_overall_eta():
    source = Path("bvr_behavior_prediction/rl/trainer.py").read_text()
    assert 'desc="Training"' in source
    assert 'unit="epoch"' in source
    assert "elapsed {elapsed} < ETA {remaining}" in source
    assert 'epoch=f"{metrics[\'epoch_seconds\']:.1f}s"' in source


def test_jsbsim_fdm_does_not_print_routine_startup_diagnostics():
    source = Path(
        "bvr_sim_source/bvr_sim/src_py/simulator/aircraft/fdm/jsbsim_fdm.py"
    ).read_text()
    assert 'print_red(f"JSBSim dt:' not in source
    assert 'print_green(f"JSBSim {self.aircraft_model} reset at' not in source


def test_scenarios_are_reproducible_diverse_and_safe():
    first = ScenarioSampler(42).sample_batch(3)
    second = ScenarioSampler(42).sample_batch(3)
    assert first == second
    assert len({item.range_m for item in first}) == 3
    assert len({item.bearing_deg for item in first}) == 3
    assert all(item.red_policy == "constant_course" for item in first)
    assert all(item.blue_altitude_m > 2500 and item.red_altitude_m > 2500 for item in first)


def test_reward_is_escalating_and_events_are_edge_triggered():
    reward = CombatReward(RewardWeights())
    safe, _ = reward({"blue_altitude_m": 6000})
    locked, _ = reward({"blue_altitude_m": 6000, "target_locked": True})
    repeated, _ = reward({"blue_altitude_m": 6000, "target_locked": True})
    evaded, _ = reward({"blue_altitude_m": 6000, "missile_avoided": True})
    killed, _ = reward({"blue_altitude_m": 6000, "opponent_destroyed": True})
    assert safe < locked < evaded < killed
    assert repeated == safe


def test_hybrid_policy_exposes_every_skill_and_nn_parameters():
    torch = pytest.importorskip("torch")
    from bvr_behavior_prediction.rl.pilot import HybridSkillPilot

    pilot = HybridSkillPilot(12, hidden_size=16)
    assert isinstance(pilot.encoder, torch.nn.TransformerEncoder)
    assert len(pilot.skill_names) == 33
    assert pilot._skill_indices == {name: i for i, name in enumerate(pilot.skill_names)}
    assert len(pilot._skill_contracts) == len(pilot.skill_names)
    name, params, _, _, raw = pilot.act(np.zeros(12, dtype=np.float32), deterministic=True)
    assert name in pilot.skill_names
    assert len(raw) == len(pilot.parameter_names)
    contract = pilot.manager.get_contract(name)["parameter_schema"]["properties"]
    assert set(params) == set(contract)
    for key, value in params.items():
        if contract[key]["type"] == "number":
            assert contract[key].get("minimum", -np.inf) <= value
            assert value <= contract[key].get("maximum", np.inf)


def test_environment_builds_10_hz_history_with_separate_energy_features():
    from bvr_behavior_prediction.rl.environment import BluePilotEnvironment

    class Backend:
        def __init__(self, *_):
            self.tick = 0

        def reset(self, seed):
            return np.array([seed], dtype=np.float32), {
                "blue_speed_mps": 200,
                "blue_altitude_m": 6_000,
                "red_speed_mps": 400,
                "red_altitude_m": 12_000,
            }

        def step(self, blue, red):
            self.tick += 1
            return (
                np.array([self.tick], dtype=np.float32),
                0.0,
                False,
                {
                    "blue_speed_mps": 200 + self.tick,
                    "blue_altitude_m": 6_000,
                    "red_speed_mps": 400,
                    "red_altitude_m": 12_000,
                },
            )

        def close(self):
            pass

    scenario = type("Scenario", (), {"as_dict": lambda self: {}})()
    env = BluePilotEnvironment(Backend, scenario, planning_horizon_s=1.0)
    initial = env.reset(3)
    assert initial.shape == (20, 5)
    np.testing.assert_allclose(initial[-1, 1:], [0.25, 0.5, 1.0, 1.0])
    history, _, _, info = env.step("maintain_heading", {})
    assert history.shape == (20, 5)
    np.testing.assert_array_equal(history[-10:, 0], np.arange(1, 11))
    assert info["elapsed_game_s"] == pytest.approx(1.0)


def test_evaluation_cadence_keeps_first_periodic_and_final_epochs(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("mlflow")
    from bvr_behavior_prediction.rl.trainer import PPOTrainer

    config = PilotTrainingConfig(
        epochs=12,
        evaluation_interval=5,
        scenarios_per_epoch=3,
        evaluation_scenarios_per_epoch=3,
        hidden_size=16,
        output_dir=tmp_path,
    )
    trainer = PPOTrainer(lambda *_: None, 1, config, device="cpu")

    assert [epoch for epoch in range(1, 13) if trainer._should_evaluate(epoch)] == [1, 5, 10, 12]


def test_estimated_epoch_speedup_uses_measured_phase_times(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("mlflow")
    from bvr_behavior_prediction.rl.trainer import PPOTrainer

    config = PilotTrainingConfig(
        evaluation_interval=5,
        scenarios_per_epoch=3,
        evaluation_scenarios_per_epoch=3,
        hidden_size=16,
        output_dir=tmp_path,
    )
    trainer = PPOTrainer(lambda *_: None, 1, config, device="cpu")

    # A 10 s rollout, 2 s update and 10 s evaluation becomes 14 s amortized.
    assert trainer._estimated_epoch_speedup(10.0, 2.0, 10.0) == pytest.approx(22.0 / 14.0)
