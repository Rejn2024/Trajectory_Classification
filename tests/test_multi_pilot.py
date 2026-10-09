import json
from dataclasses import replace
from functools import partial

import numpy as np
import pytest

torch = pytest.importorskip("torch")
mlflow = pytest.importorskip("mlflow")

from bvr_behavior_prediction.rl.config import PilotTrainingConfig
from bvr_behavior_prediction.rl.multi_pilot import PilotSpec, load_trained_pilot, train_pilots
from bvr_behavior_prediction.rl.reward import RewardDefinition
from bvr_behavior_prediction.rl.trainer import PPOTrainer


class CountingReward:
    """State would leak across pilots/episodes if the factory were not honored."""

    def __init__(self, scale):
        self.scale = scale
        self.steps = 0

    def __call__(self, info):
        self.steps += 1
        value = self.steps * self.scale
        return value, {"custom": value}


class TinyEnvironment:
    def __init__(self, recording_path=None):
        self.recording_path = recording_path
        self.closed = False

    def reset(self, seed):
        self.steps = 0
        if self.recording_path:
            self.recording_path.write_text("fixture replay", encoding="utf8")
        return np.zeros((2, 1), dtype=np.float32)

    def step(self, name, parameters):
        self.steps += 1
        return (
            np.full((2, 1), self.steps, np.float32),
            self.steps == 2,
            False,
            {
                "blue_alive": True,
                "blue_altitude_m": 6000.0,
                "target_locked": True,
                "opponent_destroyed": False,
                "opponent_eliminated": False,
            },
        )

    def close(self):
        self.closed = True


@pytest.fixture
def config(tmp_path):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield PilotTrainingConfig(
        seed=7,
        evaluation_seed=100_000,
        epochs=1,
        episode_duration_s=2,
        scenarios_per_epoch=3,
        evaluation_scenarios_per_epoch=3,
        simulator_workers=1,
        history_duration_s=2,
        sample_interval_s=1,
        hidden_size=8,
        transformer_heads=2,
        transformer_layers=1,
        update_epochs=1,
        minibatch_size=6,
        checkpoint_interval=1,
        output_dir=tmp_path / "batch",
        mlflow_tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
    )
    torch.set_num_threads(old_threads)


def definition(scale):
    return RewardDefinition(
        f"counter_{scale}", partial(CountingReward, scale), {"scale": scale}, version="2"
    )


def test_arbitrary_roster_trains_distinct_models_and_preserves_reward_identity(config):
    pilots = [
        PilotSpec(f"pilot_{index}", 7 + 100 * index, definition(index + 1)) for index in range(4)
    ]
    environments = []
    received_configs = []

    def factory_builder(run_config):
        received_configs.append(run_config)

        def factory(scenario, recording_path):
            env = TinyEnvironment(recording_path)
            environments.append(env)
            return env

        return factory

    results = train_pilots(pilots, config, factory_builder, 1, device="cpu")
    assert len(results) == 4
    assert len({result["checkpoint_sha256"] for result in results}) == 4
    assert len({result["mlflow_run_id"] for result in results}) == 4
    assert all(env.closed for env in environments)
    assert [run_config.seed for run_config in received_configs] == [7, 107, 207, 307]
    assert all(run_config.evaluation_seed == 100_000 for run_config in received_configs)
    manifest = json.loads((config.output_dir / "pilots.json").read_text(encoding="utf8"))
    assert all(record["status"] == "completed" for record in manifest["pilots"])
    for index, result in enumerate(results):
        # Two reward calls, with an independent counter for EVERY episode.
        assert result["selection_return"] == 3 * (index + 1)
        # All pilots use the same benchmark despite different training rewards.
        assert result["benchmark_return"] == pytest.approx(2.04)
        assert result["survival_rate"] == 1.0
        run_dir = config.output_dir / result["output_dir"]
        metadata = json.loads((run_dir / "config.json").read_text(encoding="utf8"))
        assert metadata["reward"] == pilots[index].reward.as_dict()
        demo = json.loads((run_dir / "demonstrations/epoch_0001/summary.json").read_text())
        assert demo["score"] == result["selection_return"]
        policy = load_trained_pilot(run_dir)
        saved = torch.load(run_dir / "best_model.pt", weights_only=True)
        for name, value in policy.state_dict().items():
            torch.testing.assert_close(value, saved[name])
        assert not policy.training
        run = mlflow.get_run(result["mlflow_run_id"])
        assert run.data.tags["pilot_id"] == pilots[index].pilot_id
        assert run.data.params["reward.name"] == pilots[index].reward.name
        assert run.data.metrics["benchmark_return"] == pytest.approx(2.04)

    before = (config.output_dir / "pilots.json").read_bytes()
    with pytest.raises(FileExistsError):
        train_pilots(pilots, config, factory_builder, 1, device="cpu")
    assert (config.output_dir / "pilots.json").read_bytes() == before
    checkpoint = config.output_dir / results[0]["checkpoint"]
    with checkpoint.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        load_trained_pilot(checkpoint.parent)


def test_seed_reinitializes_weights_independently_of_other_pilots(config):
    def make(seed):
        return PPOTrainer(lambda *_: None, 1, replace(config, seed=seed), device="cpu")

    first = make(7)
    first_weights = {key: value.clone() for key, value in first.pilot.state_dict().items()}
    unrelated = make(103)
    assert not torch.equal(first_weights["skill_head.weight"], unrelated.pilot.skill_head.weight)
    repeated = make(7)
    for name, value in repeated.pilot.state_dict().items():
        torch.testing.assert_close(first_weights[name], value)
    assert first.optimizer is not repeated.optimizer
    assert first.evaluation_scenarios == unrelated.evaluation_scenarios


def test_failed_batch_keeps_completed_checkpoint_and_status(config):
    def builder(run_config):
        if run_config.output_dir.name == "second":
            raise RuntimeError("simulator unavailable")
        return lambda scenario, path: TinyEnvironment(path)

    with pytest.raises(RuntimeError, match="simulator unavailable"):
        train_pilots([PilotSpec("first", 7), PilotSpec("second", 107)], config, builder, 1, "cpu")
    manifest = json.loads((config.output_dir / "pilots.json").read_text())
    assert [record["status"] for record in manifest["pilots"]] == ["completed", "failed"]
    assert (config.output_dir / "first/best_model.pt").is_file()
    assert load_trained_pilot(config.output_dir / "first") is not None


def test_recovery_reuses_completed_pilots_and_restarts_only_unfinished(config):
    config = replace(config, test_scenarios_per_pilot=3)
    pilots = [PilotSpec("first", 7), PilotSpec("second", 107)]

    def fail_second(settings):
        if settings.output_dir.name == "second":
            raise RuntimeError("worker failed")
        return lambda scenario, path: TinyEnvironment(path)

    with pytest.raises(RuntimeError, match="worker failed"):
        train_pilots(pilots, config, fail_second, 1, "cpu")
    failed_dir = config.output_dir / "second"
    failed_dir.mkdir()
    (failed_dir / "partial.txt").write_text("preserve the incomplete attempt")
    original = {p.relative_to(config.output_dir): p.read_bytes()
                for p in config.output_dir.rglob("*") if p.is_file()}
    recovery_config = replace(config, output_dir=config.output_dir.parent / "recovery")
    trained = []

    def builder(settings):
        trained.append(settings.output_dir.name)
        return lambda scenario, path: TinyEnvironment(path)

    results = train_pilots(pilots, recovery_config, builder, 1, "cpu",
                           reuse_completed_from=config.output_dir)
    assert trained == ["second"]
    assert len(results) == 2
    assert all((config.output_dir / name).read_bytes() == value for name, value in original.items())
    for path in (config.output_dir / "first").rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (recovery_config.output_dir / path.relative_to(config.output_dir)).read_bytes()
    assert not (recovery_config.output_dir / "second/partial.txt").exists()
    manifest = json.loads((recovery_config.output_dir / "pilots.json").read_text())
    assert all(p["status"] == "completed" for p in manifest["pilots"])
    assert manifest["pilots"][0]["reused_from"] == str(config.output_dir / "first")
    assert load_trained_pilot(recovery_config.output_dir / "first") is not None

    bad_config = replace(recovery_config, output_dir=config.output_dir.parent / "bad", learning_rate=0.001)
    with pytest.raises(ValueError, match="training settings differ"):
        train_pilots(pilots, bad_config, builder, 1, "cpu", reuse_completed_from=config.output_dir)
    assert not bad_config.output_dir.exists()
    checkpoint = config.output_dir / "first/best_model.pt"
    with checkpoint.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        train_pilots(pilots, replace(bad_config, learning_rate=config.learning_rate), builder, 1,
                     "cpu", reuse_completed_from=config.output_dir)
    assert not bad_config.output_dir.exists()


@pytest.mark.parametrize("pilots", [[], [PilotSpec("same", 1), PilotSpec("Same", 2)]])
def test_invalid_roster_does_not_create_output(config, pilots):
    with pytest.raises(ValueError):
        train_pilots(pilots, config, lambda *_: None, 1, "cpu")
    assert not config.output_dir.exists()


def test_validation_seed_overlap_is_rejected_before_training(config):
    with pytest.raises(ValueError, match="overlap"):
        train_pilots([PilotSpec("example", 100_000)], config, lambda *_: None, 1, "cpu")
    assert not config.output_dir.exists()


def test_fresh_test_is_shared_disjoint_and_cannot_select_checkpoints(config):
    config = replace(config, epochs=2, resample_training_scenarios=True,
                     test_scenarios_per_pilot=3)
    pilots = [PilotSpec("first", 7, definition(1)), PilotSpec("second", 7, definition(2))]
    results = train_pilots(pilots, config, lambda _: lambda scenario, path: TinyEnvironment(path),
                          1, "cpu", population_metadata={"reward_seed": 42})
    records = [json.loads((config.output_dir / p.pilot_id / "test_episodes.json").read_text())
               for p in pilots]
    for result, data in zip(results, records):
        assert result["test"]["clean_win_rate"] == 0
        assert result["test"]["survival_rate"] == 1
        assert result["test"]["outcome_score"] == 0
        assert result["population_metadata"] == {"reward_seed": 42}
        assert result["checkpoint_sha256"] == data["checkpoint_sha256"]
        seeds = {row["seed"] for row in data["episodes"]}
        assert not seeds.intersection(range(7, 13))
        assert not seeds.intersection(range(config.evaluation_seed, config.evaluation_seed + 3))
    assert results[0]["selection_return"] == 3
    assert results[1]["selection_return"] == 6
    assert [(r["seed"], r["scenario"]) for r in records[0]["episodes"]] == [
        (r["seed"], r["scenario"]) for r in records[1]["episodes"]
    ]


@pytest.mark.parametrize("test_seed", [8, 100_001])
def test_test_seed_overlap_is_rejected_before_writing(config, test_seed):
    config = replace(config, resample_training_scenarios=True, epochs=2,
                     test_scenarios_per_pilot=3, test_seed=test_seed)
    with pytest.raises(ValueError, match="overlap"):
        train_pilots([PilotSpec("pilot", 7)], config, lambda *_: None, 1, "cpu")
    assert not config.output_dir.exists()


def test_validation_overlap_in_later_training_epoch_is_rejected(config):
    config = replace(config, evaluation_seed=11, resample_training_scenarios=True, epochs=2)
    with pytest.raises(ValueError, match="overlap"):
        train_pilots([PilotSpec("pilot", 7)], config, lambda *_: None, 1, "cpu")
    assert not config.output_dir.exists()


@pytest.mark.parametrize("pilot_id", ["../escape", "pilot/one", "CON", "", "pilot one"])
def test_pilot_identifiers_cannot_escape_the_batch(pilot_id):
    with pytest.raises(ValueError):
        PilotSpec(pilot_id, 1)


def test_environments_close_when_a_later_factory_fails(config):
    environments = []

    def factory(*_):
        if environments:
            raise RuntimeError("reset construction failed")
        env = TinyEnvironment()
        environments.append(env)
        return env

    trainer = PPOTrainer(factory, 1, config, "cpu")
    with pytest.raises(RuntimeError, match="construction failed"):
        trainer._training_episodes()
    assert environments[0].closed
