"""Exercise real subprocess isolation, state affinity, and failure cleanup."""

import os
import json
import random
import sys
import time

import numpy as np
import pytest

from bvr_behavior_prediction.rl.simulation_workers import (
    ProcessSimulatorPool, SimulatorWorkerError, ThreadSimulatorPool,
)


def environment_factory(close_dir=None, fail=None):
    # A closure, like the notebook's factory, rather than a module-level callable.
    class Environment:
        def __init__(self, scenario):
            self.scenario = scenario
            self.seed = None
            self.ticks = 0

        def reset(self, seed):
            self.seed = seed
            if fail == "reset":
                raise ValueError("deliberate reset failure")
            return np.array([[seed]], dtype=np.float32)

        def step(self, name, parameters):
            if fail == "step":
                raise ValueError("deliberate step failure")
            if fail == "exit":
                os._exit(5)
            self.ticks += 1
            return (
                np.array([[self.seed + self.ticks]], dtype=np.float32),
                self.ticks >= self.seed % 3 + 1,
                False,
                {"pid": os.getpid(), "ticks": self.ticks, "seed": self.seed,
                 "torch_loaded": "torch" in sys.modules},
            )

        def close(self):
            if close_dir:
                (close_dir / f"closed_{self.scenario}_{self.seed}").touch()

    return lambda scenario, path: Environment(scenario)


def test_thread_resets_do_not_race_global_numpy_seeds():
    class SeededEnvironment:
        def reset(self, seed):
            np.random.seed(seed)
            time.sleep(0.01)  # Simulate native reset work that releases the GIL.
            return np.random.random()

        def close(self):
            pass

    seeds = (1, 2, 3)
    expected = [np.random.RandomState(seed).random() for seed in seeds]
    pool = ThreadSimulatorPool(lambda *_: SeededEnvironment(), 3)
    try:
        assert pool.reset(range(3), seeds) == expected
        assert pool.reset(range(3), seeds) == expected
    finally:
        pool.close()


def test_processes_keep_environment_affinity_and_reuse_workers(tmp_path):
    pool = ProcessSimulatorPool(environment_factory(tmp_path), 2)
    processes = [process for process, _ in pool.workers]
    try:
        seeds = (10, 11, 12, 13, 14)
        observations = pool.reset(range(5), seeds)
        assert [obs.item() for obs in observations] == list(seeds)
        first = pool.step([(index, "maintain_heading", {}) for index in range(5)])
        infos = [step[0][3] for step in first]
        pids = [info["pid"] for info in infos]
        assert set(pids) == {process.pid for process in processes}
        assert os.getpid() not in pids
        assert pids[0] == pids[2] == pids[4]
        assert pids[1] == pids[3]
        assert all(not info["torch_loaded"] for info in infos)
        assert all(wall >= 0 and cpu >= 0 for _, wall, cpu in first)

        # Reordering/subsetting actions must still reach their original environment.
        subset = pool.step([(3, "maintain_heading", {}), (0, "maintain_heading", {})])
        assert [step[0][3]["seed"] for step in subset] == [13, 10]
        assert [step[0][3]["ticks"] for step in subset] == [2, 2]
        assert [step[0][3]["pid"] for step in subset] == [pids[3], pids[0]]
        pool.release()
        assert len(list(tmp_path.glob("closed_*"))) == 5
        pool.reset(range(5), seeds)
        second = pool.step([(index, "maintain_heading", {}) for index in range(5)])
        assert [step[0][3]["pid"] for step in second] == pids
        assert [step[0][3]["ticks"] for step in second] == [1] * 5
    finally:
        pool.close()
    assert all(not process.is_alive() for process in processes)
    pool.close()  # Idempotent cleanup is needed when training also handles a failure.


@pytest.mark.parametrize("failure", ["reset", "step", "exit", "interrupt", "timeout"])
def test_worker_failure_or_interrupt_stops_every_process(failure, monkeypatch):
    pool = ProcessSimulatorPool(environment_factory(fail=failure), 2)
    processes = [process for process, _ in pool.workers]
    try:
        if failure == "reset":
            with pytest.raises(RuntimeError, match="deliberate reset failure"):
                pool.reset(range(3), (1, 2, 3))
        else:
            pool.reset(range(3), (1, 2, 3))
            expected = RuntimeError
            match = "deliberate step failure" if failure == "step" else "worker"
            if failure in ("interrupt", "timeout"):
                expected = KeyboardInterrupt if failure == "interrupt" else TimeoutError
                match = "injected wait failure"

                def interrupted_receive(worker):
                    raise expected("injected wait failure")

                monkeypatch.setattr(pool, "_receive", interrupted_receive)
            with pytest.raises(expected, match=match):
                pool.step([(index, "maintain_heading", {}) for index in range(3)])
        assert pool.closed
        assert all(not process.is_alive() for process in processes)
    finally:
        pool.close(force=True)


@pytest.mark.parametrize("interrupt", [False, True])
def test_trainer_preserves_episode_state_and_closes_persistent_pool(tmp_path, interrupt):
    torch = pytest.importorskip("torch")
    pytest.importorskip("mlflow")
    from bvr_behavior_prediction.rl.config import PilotTrainingConfig
    from bvr_behavior_prediction.rl.reward import RewardDefinition
    from bvr_behavior_prediction.rl.trainer import PPOTrainer

    class CounterReward:
        def __init__(self):
            self.count = 0

        def __call__(self, info):
            self.count += 1
            return self.count, {"count": self.count}

    config = PilotTrainingConfig(
        episode_duration_s=3, simulator_executor="process", simulator_workers=2,
        scenarios_per_epoch=3, evaluation_scenarios_per_epoch=3, hidden_size=8,
        history_duration_s=1, sample_interval_s=1, output_dir=tmp_path,
    )
    trainer = PPOTrainer(
        environment_factory(), 1, config, device="cpu",
        reward_definition=RewardDefinition("counter", CounterReward),
    )
    batch_sizes = []

    def act_batch(observations, deterministic=False):
        batch_sizes.append(len(observations))
        raw = np.zeros(len(trainer.pilot.parameter_names), dtype=np.float32)
        return [("maintain_heading", {}, 0.0, 0.0, raw.copy()) for _ in observations]

    trainer.pilot.act_batch = act_batch
    pools = []

    def collect_twice(run_name, tags):
        for _ in range(2):
            episodes = trainer._episodes(range(3), (1, 2, 3))
            pools.append(trainer._simulator_pool)
            assert [len(rollout.rewards) for rollout, _, _ in episodes] == [2, 3, 1]
            assert [score for _, score, _ in episodes] == [3, 6, 1]
            assert [rollout.observations[0].item() for rollout, _, _ in episodes] == [1, 2, 3]
            assert all(rollout.dones[-1] for rollout, _, _ in episodes)
        assert pools[0] is pools[1]
        if interrupt:
            raise KeyboardInterrupt
        return trainer.pilot

    trainer._train = collect_twice
    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            trainer.train()
    else:
        assert isinstance(trainer.train(), torch.nn.Module)
    assert batch_sizes == [3, 2, 1] * 2
    assert trainer._simulator_pool is None
    assert all(not process.is_alive() for process, _ in pools[0].workers)


@pytest.mark.parametrize("phase", ["reset", "step", "release"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_worker_death_replays_same_batch_and_random_state(tmp_path, monkeypatch, phase, device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from dataclasses import replace
    from bvr_behavior_prediction.rl.config import PilotTrainingConfig
    from bvr_behavior_prediction.rl.reward import RewardDefinition
    from bvr_behavior_prediction.rl.trainer import PPOTrainer

    config = PilotTrainingConfig(
        episode_duration_s=3, simulator_executor="process", simulator_workers=2,
        scenarios_per_epoch=3, evaluation_scenarios_per_epoch=3, hidden_size=8,
        history_duration_s=1, sample_interval_s=1, output_dir=tmp_path / "healthy",
        resample_training_scenarios=True,
    )
    reward = RewardDefinition("ticks", lambda: lambda info: (info["ticks"], {"ticks": info["ticks"]}))

    def collect(settings):
        trainer = PPOTrainer(environment_factory(), 1, settings, device, reward)
        trainer._training_active = True
        # Consume all parent RNG types, including the real stochastic policy.
        act = trainer.pilot.act_batch

        def stochastic_actions(*args, **kwargs):
            random.random()
            np.random.random()
            return act(*args, **kwargs)

        trainer.pilot.act_batch = stochastic_actions
        try:
            episodes = trainer._training_episodes()
            state_sample = (random.random(), np.random.random(), torch.rand(5),
                            torch.rand(5, device=device).cpu())
            return episodes, state_sample, trainer
        finally:
            trainer.close()

    expected, expected_state, _ = collect(config)
    pools = []

    def faulty_pool(*args, **kwargs):
        pool = ProcessSimulatorPool(*args, **kwargs)
        pools.append(pool)
        if len(pools) == 1:
            original = getattr(pool, phase)
            calls = 0

            def disconnect(*items):
                nonlocal calls
                calls += 1
                if calls == (2 if phase == "step" else 1):
                    process = pool.workers[0][0]
                    process.terminate()
                    process.join(timeout=2)
                return original(*items)

            setattr(pool, phase, disconnect)
        return pool

    monkeypatch.setattr("bvr_behavior_prediction.rl.trainer.ProcessSimulatorPool", faulty_pool)
    with pytest.warns(RuntimeWarning, match="replaying the entire flight batch"):
        actual, actual_state, trainer = collect(replace(config, output_dir=tmp_path / "recovered"))
    assert len(pools) == 2
    assert trainer._training_batch_index == 1
    assert trainer._last_collection_profile["simulator_batch_retries"] == 1
    for (a, score_a, info_a), (b, score_b, info_b) in zip(expected, actual):
        for key in vars(a):
            np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
        assert score_a == score_b
        assert {k: v for k, v in info_a.items() if k != "pid"} == {
            k: v for k, v in info_b.items() if k != "pid"
        }
    for a, b in zip(expected_state, actual_state):
        np.testing.assert_array_equal(a, b)
    events = [json.loads(line) for line in
              (trainer.output / "simulator_recovery.jsonl").read_text().splitlines()]
    assert len(events) == 1
    assert events[0]["scenario_seeds"] == [7, 8, 9]
    assert events[0]["action"] == "retry_batch"
    assert "exit code" in events[0]["error"]
    assert all(not process.is_alive() for pool in pools for process, _ in pool.workers)


@pytest.mark.parametrize("failure", ["exit", "step"])
def test_retries_are_bounded_and_python_errors_are_not_retried(tmp_path, monkeypatch, failure):
    pytest.importorskip("torch")
    from bvr_behavior_prediction.rl.config import PilotTrainingConfig
    from bvr_behavior_prediction.rl.trainer import PPOTrainer

    config = PilotTrainingConfig(
        episode_duration_s=1, simulator_executor="process", simulator_workers=1,
        scenarios_per_epoch=3, evaluation_scenarios_per_epoch=3, hidden_size=8,
        history_duration_s=1, sample_interval_s=1, output_dir=tmp_path,
    )
    trainer = PPOTrainer(environment_factory(fail=failure), 1, config, "cpu")
    pools = []

    def pool_factory(*args, **kwargs):
        pool = ProcessSimulatorPool(*args, **kwargs)
        pools.append(pool)
        return pool

    monkeypatch.setattr("bvr_behavior_prediction.rl.trainer.ProcessSimulatorPool", pool_factory)
    if failure == "exit":
        with pytest.warns(RuntimeWarning) as warnings, pytest.raises(SimulatorWorkerError):
            trainer._training_episodes()
        assert len(warnings) == 2
        assert len(pools) == 3
        events = [json.loads(line) for line in (tmp_path / "simulator_recovery.jsonl").read_text().splitlines()]
        assert [e["action"] for e in events] == ["retry_batch", "retry_batch", "abort"]
    else:
        with pytest.raises(RuntimeError, match="deliberate step failure"):
            trainer._training_episodes()
        assert len(pools) == 1
        assert not (tmp_path / "simulator_recovery.jsonl").exists()
    assert all(not process.is_alive() for pool in pools for process, _ in pool.workers)
