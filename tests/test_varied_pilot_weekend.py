"""Continuation recovery and the exact 91-input recording/inference contract."""

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_varied_pilot_weekend as weekend
from bvr_sim.src_py.observation_space import CompactObsSpace
from nn_duel_inference import GroupedActors
from tactical_corpus_schema import (
    observation_feature_schema,
    observer_frame,
    reconstruct_policy_history,
)

from bvr_behavior_prediction.rl.bvr_jsbsim_backend import BVRJSBSimBackend
from bvr_behavior_prediction.rl.config import PilotTrainingConfig
from bvr_behavior_prediction.rl.dogfight_reward import hybrid_dogfight_reward_definition
from bvr_behavior_prediction.rl.environment import ENERGY_FEATURE_NAMES, BluePilotEnvironment
from bvr_behavior_prediction.rl.pilot import HybridSkillPilot


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_grouped_91_input_actors_preserve_architectures_order_and_actions():
    torch.manual_seed(19)
    models = {
        name: HybridSkillPilot(
            91,
            hidden_size=width,
            history_steps=20,
            transformer_heads=heads,
            transformer_layers=layers,
            selected_skill_parameters=True,
        ).eval()
        for name, width, layers, heads in (
            ("a", 8, 1, 2),
            ("b", 12, 2, 2),
            ("c", 8, 1, 2),
            ("d", 8, 1, 4),
        )
    }
    actors = GroupedActors(models, "cpu")
    assert len(actors.groups) == 3
    assert actors.parity_check(histories=2)["passed"]
    x = np.random.default_rng(42).normal(0, 0.2, (20, 91)).astype(np.float32)
    requests = [(name, x) for name in ("b", "a", "d", "b", "c")]
    actions, stats = actors.act(requests)
    assert stats["rows"] == len(requests)
    for (name, history), (skill, parameters) in zip(requests, actions, strict=True):
        reference = models[name].act(history, deterministic=True)
        assert skill == reference[0]
        assert parameters == pytest.approx(reference[1])


def test_recorded_frames_equal_training_frames_for_both_sides_with_live_threats():
    aircraft = {}
    for side, color, position, velocity in (
        ("A01", "Red", [100, 200, 6000], [250, 10, 1]),
        ("B01", "Blue", [20000, -300, 7000], [-280, 2, -1]),
    ):
        a = SimpleNamespace(
            uid=side,
            color=color,
            is_alive=True,
            position=np.array(position, float),
            velocity=np.array(velocity, float),
            partners=[],
            under_missiles=[],
        )
        a.get_altitude = lambda a=a: a.position[2]
        a.get_speed = lambda a=a: np.linalg.norm(a.velocity)
        a.get_heading = lambda: 0.4
        aircraft[side] = a
    aircraft["A01"].enemies = [aircraft["B01"]]
    aircraft["B01"].enemies = [aircraft["A01"]]
    missiles = {}
    for side, target in (("A01", "B01"), ("B01", "A01")):
        missile = SimpleNamespace(
            uid=side + "_m",
            is_alive=True,
            color=aircraft[side].color,
            target=aircraft[target],
            position=aircraft[side].position + 200,
            velocity=aircraft[side].velocity * 2,
            radar_on=True,
        )
        missiles[missile.uid] = missile
        aircraft[target].under_missiles = [missile]
    sim = SimpleNamespace(agents=aircraft, missiles=missiles, obs_manager=CompactObsSpace())
    for side, target in (("A01", "B01"), ("B01", "A01")):
        backend = BVRJSBSimBackend.__new__(BVRJSBSimBackend)
        backend.env, backend.CONTROLLED_ID, backend.include_hostile_missiles = sim, side, True
        compact = sim.obs_manager.extract_obs(aircraft[side], aircraft, missiles)
        env = BluePilotEnvironment.__new__(BluePilotEnvironment)
        env._energy_measurements = dict.fromkeys(ENERGY_FEATURE_NAMES, 0.0)
        env.speed_scale_mps, env.altitude_scale_m = 400.0, 12000.0
        info = {
            f"{team}_{kind}": float(getattr(aircraft[uid], method)())
            for team, uid in (("blue", side), ("red", target))
            for kind, method in (("speed_mps", "get_speed"), ("altitude_m", "get_altitude"))
        }
        expected = env._frame(backend._policy_observation(compact), info)
        actual = observer_frame(sim, side, target, observation_size=91)
        np.testing.assert_array_equal(actual, expected)
        assert actual[47] == actual[56] == 1  # present and seeker-active
        frames = np.repeat(actual[None], 30, axis=0)
        history = reconstruct_policy_history(frames, 10)
        assert history.shape == (20, 91)
        assert (
            hashlib.sha256(history.tobytes()).hexdigest()
            == hashlib.sha256(frames[:20].tobytes()).hexdigest()
        )
    schema = observation_feature_schema(91)
    assert len(schema) == 91
    assert schema[47]["name"] == "hostile_0_present"
    assert schema[87]["name"] == "own_kinetic_energy"


class ToyEnvironment:
    """A deterministic simulator fixture; policy sampling and PPO are real."""

    def reset(self, seed):
        self.rng = np.random.default_rng(seed)
        self.observation = self.rng.normal(0, 0.1, (20, 91)).astype(np.float32)
        self.steps = 0
        return self.observation.copy()

    def step(self, name, parameters):
        self.steps += 1
        self.observation = np.roll(self.observation, -1, axis=0)
        self.observation[-1] = self.rng.normal(0, 0.1, 91)
        info = {
            "blue_alive": True,
            "blue_altitude_m": 6000,
            "red_altitude_m": 6000,
            "blue_speed_mps": 250,
            "red_speed_mps": 250,
            "target_range_m": 30000,
            "target_alignment": 1,
            "initial_target_range_m": 30000,
            "initial_target_alignment": 1,
            "incoming_missiles": 0,
            "opponent_eliminated": False,
            "opponent_destroyed": False,
            "target_locked": True,
            "fired_with_lock": name == "launch",
            "elapsed_game_s": self.steps,
            "missile_support_time_s": 0.0,
            "threatened_time_s": 0.0,
            "missiles_avoided_total": 0,
        }
        return self.observation.copy(), self.steps == 2, False, info

    def close(self):
        pass


def test_interrupted_epoch_restores_optimizer_rng_selection_and_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(weekend, "make_environment", lambda *_: lambda *_: ToyEnvironment())
    source = tmp_path / "source"
    source.mkdir()
    config = PilotTrainingConfig(
        hidden_size=8,
        transformer_heads=2,
        transformer_layers=1,
        epochs=4,
        episode_duration_s=2,
        scenarios_per_epoch=3,
        evaluation_scenarios_per_epoch=3,
        evaluation_interval=1,
        resample_training_scenarios=True,
        update_epochs=1,
        minibatch_size=4,
        simulator_workers=1,
        simulator_executor="thread",
        evaluation_seed=100,
        output_dir=source,
    )
    reward = hybrid_dogfight_reward_definition()
    trainer = weekend.PPOTrainer(lambda *_: ToyEnvironment(), 91, config, "cpu", reward)
    metadata = {
        "training": config.as_dict(),
        "observation_size": 91,
        "reward": reward.as_dict(),
        "skill_catalogue": trainer.pilot.manager.get_catalogue(),
        "skill_names": trainer.pilot.skill_names,
        "parameter_names": trainer.pilot.parameter_names,
    }
    weekend.write(source / "config.json", metadata)
    torch.save(trainer.pilot.state_dict(), source / "best_model.pt")
    weekend.write(
        source / "manifest.json", {"checkpoint_sha256": weekend.sha256(source / "best_model.pt")}
    )
    trainer.close()

    reference = tmp_path / "reference"
    interrupted = tmp_path / "interrupted"
    weekend.train_one(source, reference, replace(config, output_dir=reference), "simple", "cpu")
    original_save = weekend.save_state

    def crash_after_commit(trainer, epoch, metrics):
        original_save(trainer, epoch, metrics)
        if epoch == 1:
            raise RuntimeError("injected process failure after epoch commit")

    monkeypatch.setattr(weekend, "save_state", crash_after_commit)
    with pytest.raises(RuntimeError, match="injected"):
        weekend.train_one(
            source, interrupted, replace(config, output_dir=interrupted), "simple", "cpu"
        )
    monkeypatch.setattr(weekend, "save_state", original_save)
    weekend.train_one(source, interrupted, replace(config, output_dir=interrupted), "simple", "cpu")
    actual = torch.load(interrupted / "resume.pt", weights_only=False)
    expected = torch.load(reference / "resume.pt", weights_only=False)
    for section in ("pilot", "best_pilot"):
        for name in actual[section]:
            torch.testing.assert_close(
                actual[section][name], expected[section][name], rtol=0, atol=0
            )
    for parameter, values in actual["optimizer"]["state"].items():
        for key, value in values.items():
            torch.testing.assert_close(
                value, expected["optimizer"]["state"][parameter][key], rtol=0, atol=0
            )
    assert actual["epoch"] == actual["training_batch_index"] == 4
    assert actual["best_score"] == expected["best_score"]
    assert torch.equal(actual["torch_rng"], expected["torch_rng"])
    rows = [
        json.loads(line)
        for line in (interrupted / "training_metrics.jsonl").read_text().splitlines()
    ]
    assert [row["epoch"] for row in rows] == [1, 2, 3, 4]
    assert [row["evaluation_performed"] for row in rows] == [1, 1, 1, 1]


def test_stage_seeds_and_varied_objectives_survive_both_transitions(tmp_path):
    source, simple = tmp_path / "source", tmp_path / "simple"
    config = PilotTrainingConfig(
        hidden_size=12,
        transformer_heads=2,
        transformer_layers=3,
        value_coefficient=0.537,
        entropy_coefficient=0.0093,
    )
    weekend.write(source / "config.json", {"training": config.as_dict()})
    settings = {"simple_epochs": 200, "tactical_epochs": 300, "workers": 2}
    continued = weekend.stage_config(source, simple, settings, "simple")
    weekend.write(
        simple / "config.json",
        {"training": continued.as_dict(), "continuation_origin_seed": config.seed},
    )
    tactical = weekend.stage_config(simple, tmp_path / "tactical", settings, "tactical")
    assert continued.seed == config.seed + 1_000_000
    assert tactical.seed == config.seed + 2_000_000
    assert continued.epochs == 200 and tactical.epochs == 300
    assert continued.episode_duration_s == tactical.episode_duration_s == 300
    assert continued.evaluation_interval == tactical.evaluation_interval == 1
    for key in (
        "hidden_size",
        "transformer_heads",
        "transformer_layers",
        "value_coefficient",
        "entropy_coefficient",
        "learning_rate",
        "training_reward_scale",
        "update_epochs",
    ):
        assert getattr(continued, key) == getattr(tactical, key) == getattr(config, key)


@pytest.mark.parametrize("opponent", ["simple", "tactical"])
def test_every_epoch_validates_and_only_improvements_write_selected_weights(
    tmp_path, monkeypatch, opponent
):
    monkeypatch.setattr(weekend, "make_environment", lambda *_: lambda *_: ToyEnvironment())
    source, output = tmp_path / "source", tmp_path / "trained"
    config = PilotTrainingConfig(
        hidden_size=8,
        transformer_heads=2,
        transformer_layers=1,
        epochs=4,
        episode_duration_s=2,
        scenarios_per_epoch=3,
        evaluation_scenarios_per_epoch=3,
        update_epochs=1,
        evaluation_interval=1,
        minibatch_size=4,
        simulator_workers=1,
        simulator_executor="thread",
        output_dir=source,
    )
    reward = hybrid_dogfight_reward_definition()
    trainer = weekend.PPOTrainer(lambda *_: ToyEnvironment(), 91, config, "cpu", reward)
    weekend.write(
        source / "config.json",
        {
            "training": config.as_dict(),
            "observation_size": 91,
            "reward": reward.as_dict(),
            "skill_catalogue": trainer.pilot.manager.get_catalogue(),
            "skill_names": trainer.pilot.skill_names,
            "parameter_names": trainer.pilot.parameter_names,
        },
    )
    torch.save(trainer.pilot.state_dict(), source / "best_model.pt")
    weekend.write(
        source / "manifest.json", {"checkpoint_sha256": weekend.sha256(source / "best_model.pt")}
    )
    trainer.close()
    scores = iter([10.0, 11.0, 11.0, 9.0, 12.0])  # Initialization, then every epoch.
    monkeypatch.setattr(weekend.PPOTrainer, "_evaluate", lambda _: np.repeat(next(scores), 3))
    writes, epochs = [], []
    original = weekend.save_best_weights

    def save_best(trainer):
        writes.append(trainer.weekend_best_epoch)
        original(trainer)

    def progress(phase, epoch, metrics):
        if phase == "epoch complete":
            epochs.append((epoch, weekend.sha256(output / "best_model.pt")))

    monkeypatch.setattr(weekend, "save_best_weights", save_best)
    weekend.train_one(source, output, replace(config, output_dir=output), opponent, "cpu", progress)
    assert writes == [1, 4]
    assert epochs[0][1] == epochs[1][1] == epochs[2][1] != epochs[3][1]
    state = torch.load(output / "resume.pt", weights_only=False)
    assert state["epoch"] == state["best_epoch"] == 4
    assert [row["evaluation_performed"] for row in state["metrics"]] == [1] * 4
    assert [row["best_model_improved"] for row in state["metrics"]] == [1, 0, 0, 1]


@pytest.mark.parametrize("conditioned", [False, True])
def test_score_only_actions_and_returns_match_full_validation_without_critic(tmp_path, conditioned):
    config = PilotTrainingConfig(
        hidden_size=8,
        transformer_heads=2,
        transformer_layers=1,
        episode_duration_s=2,
        scenarios_per_epoch=3,
        evaluation_scenarios_per_epoch=3,
        simulator_workers=1,
        simulator_executor="thread",
        skill_conditioned_parameters=conditioned,
        output_dir=tmp_path,
    )
    trainer = weekend.PPOTrainer(
        lambda *_: ToyEnvironment(), 91, config, "cpu", hybrid_dogfight_reward_definition()
    )
    trainer._training_active = True
    try:
        expected = trainer.evaluate_episodes()
        x = np.random.default_rng(29).normal(0.0, 0.2, (5, 20, 91)).astype(np.float32)
        trainer.pilot.eval()
        expected_actions = [(a[0], a[1]) for a in trainer.pilot.act_batch(x, deterministic=True)]
        rng = torch.get_rng_state()

        def unused_critic(*_):
            raise AssertionError("score-only validation evaluated the critic")

        trainer.pilot.value_head.register_forward_pre_hook(unused_critic)
        actual_actions = trainer.pilot.act_deterministic_batch(x)
        assert list(actual_actions) == expected_actions
        actual = trainer.evaluate_episodes(retain_rollouts=False)
        assert all(row[0] is None for row in actual)
        assert [row[1:] for row in actual] == [row[1:] for row in expected]
        assert torch.equal(rng, torch.get_rng_state())
    finally:
        trainer.close()


def test_combined_rollout_transfer_preserves_all_action_fields():
    indices = torch.tensor([0, 1, 0])
    raw = torch.arange(9, dtype=torch.float32).reshape(3, 3) * 0.123
    log_probs = torch.tensor([-1.2, -0.3, -4.5])
    values = torch.tensor([17.1, -34.0, 0.123])
    got = HybridSkillPilot._rollout_outputs_to_cpu(indices, raw, log_probs, values)
    assert got[0] == indices.tolist()
    np.testing.assert_array_equal(got[1], raw.numpy())
    assert got[2] == log_probs.tolist()
    assert got[3] == values.tolist()


def test_campaign_lock_recovers_after_process_is_killed(tmp_path):
    program = (
        "import sys; from pathlib import Path; "
        "sys.path.insert(0, sys.argv[1]); "
        "from generate_nn_duel_corpus import campaign_lock\n"
        "with campaign_lock(Path(sys.argv[2])):\n"
        " print('ready', flush=True)\n"
        " sys.stdin.read()\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", program, str(Path(weekend.__file__).parent), str(tmp_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        child.kill()
        child.wait(timeout=10)
        assert (tmp_path / "collection.lock").exists()
        with weekend.duel.campaign_lock(tmp_path):
            assert (tmp_path / "collection.lock").read_text() != str(child.pid)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
