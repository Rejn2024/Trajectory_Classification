"""Exercise real subprocess isolation, state affinity, and failure cleanup."""

import json
import os
import random
import sys
import time
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pytest

from bvr_behavior_prediction.rl.simulation_workers import (
    ProcessSimulatorPool,
    SimulatorWorkerError,
    ThreadSimulatorPool,
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
        first = pool.step([(index, "change_trajectory", {}) for index in range(5)])
        infos = [step[0][3] for step in first]
        pids = [info["pid"] for info in infos]
        assert set(pids) == {process.pid for process in processes}
        assert os.getpid() not in pids
        assert pids[0] == pids[2] == pids[4]
        assert pids[1] == pids[3]
        assert all(not info["torch_loaded"] for info in infos)
        assert all(wall >= 0 and cpu >= 0 for _, wall, cpu in first)

        # Reordering/subsetting actions must still reach their original environment.
        subset = pool.step([(3, "change_trajectory", {}), (0, "change_trajectory", {})])
        assert [step[0][3]["seed"] for step in subset] == [13, 10]
        assert [step[0][3]["ticks"] for step in subset] == [2, 2]
        assert [step[0][3]["pid"] for step in subset] == [pids[3], pids[0]]
        pool.release()
        assert len(list(tmp_path.glob("closed_*"))) == 5
        pool.reset(range(5), seeds)
        second = pool.step([(index, "change_trajectory", {}) for index in range(5)])
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
    memory_name = None
    try:
        if failure == "reset":
            with pytest.raises(RuntimeError, match="deliberate reset failure"):
                pool.reset(range(3), (1, 2, 3))
        else:
            pool.reset(range(3), (1, 2, 3))
            memory_name = pool.shared_memory_name
            expected = RuntimeError
            match = "deliberate step failure" if failure == "step" else "worker"
            if failure in ("interrupt", "timeout"):
                expected = KeyboardInterrupt if failure == "interrupt" else TimeoutError
                match = "injected wait failure"

                def interrupted_receive(worker):
                    raise expected("injected wait failure")

                monkeypatch.setattr(pool, "_receive", interrupted_receive)
            with pytest.raises(expected, match=match):
                pool.step([(index, "change_trajectory", {}) for index in range(3)])
        assert pool.closed
        assert all(not process.is_alive() for process in processes)
        assert pool.shared_memory_name is None
        if memory_name is not None:
            with pytest.raises(FileNotFoundError):
                SharedMemory(name=memory_name)
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
        return [("change_trajectory", {}, 0.0, 0.0, raw.copy()) for _ in observations]

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


@pytest.mark.parametrize("transport", ["pipe", "shared_memory"])
def test_observation_snapshots_survive_later_steps_reset_resize_and_close(transport):
    def factory(scenario, path):
        class Environment:
            def reset(self, seed):
                self.observation = np.full((20, 91), seed, np.float32)
                return self.observation

            def step(self, name, parameters):
                self.observation += 1
                return self.observation, False, False, {}

            def close(self):
                pass

        return Environment()

    pool = ProcessSimulatorPool(factory, 2, transport=transport)
    try:
        initial = pool.reset(range(5), (1, 2, 3, 4, 5))
        assert pool.observation_transport == transport
        name = pool.shared_memory_name
        request = pool._request

        def checked_request(operation, groups):
            result = request(operation, groups)
            if operation == "step":
                for batch in result:
                    for _, item in batch:
                        assert (item[0][0] is None) == (transport == "shared_memory")
            return result

        pool._request = checked_request
        earlier = pool.step([(4, "skill", {}), (0, "skill", {})])
        pool.step([(4, "skill", {}), (0, "skill", {})])
        np.testing.assert_array_equal(initial[4], np.full((20, 91), 5, np.float32))
        np.testing.assert_array_equal(earlier[0][0][0], np.full((20, 91), 6, np.float32))
        np.testing.assert_array_equal(earlier[1][0][0], np.full((20, 91), 2, np.float32))
        pool.reset(range(5), (5, 4, 3, 2, 1))
        assert pool.shared_memory_name == name  # The same layout reuses its mapping.
        pool.reset(range(3), (8, 9, 10))
        if name is not None:
            assert pool.shared_memory_name != name
            with pytest.raises(FileNotFoundError):
                SharedMemory(name=name)
        name = pool.shared_memory_name
        latest = pool.step([(2, "skill", {})])[0][0][0]
        np.testing.assert_array_equal(latest, np.full((20, 91), 11, np.float32))
    finally:
        pool.close()
    np.testing.assert_array_equal(latest, np.full((20, 91), 11, np.float32))
    if name is not None:
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=name)


def test_auto_transport_falls_back_once_when_shared_memory_is_unavailable(monkeypatch):
    import bvr_behavior_prediction.rl.simulation_workers as workers

    def unavailable(*args, **kwargs):
        raise OSError("No shared memory available")

    pool = ProcessSimulatorPool(environment_factory(), 1)
    try:
        monkeypatch.setattr(workers, "SharedMemory", unavailable)
        with pytest.warns(RuntimeWarning, match="using pipes"):
            pool.reset(range(3), (1, 2, 3))
        assert pool.observation_transport == "pipe"
        pool.reset(range(3), (1, 2, 3))
        assert pool.step([(0, "skill", {})])[0][0][0].item() == 2
    finally:
        pool.close()


@pytest.mark.parametrize("kind", ["scalar", "mapping"])
def test_auto_transport_preserves_unsupported_observation_types(kind):
    def factory(scenario, path):
        class Environment:
            def reset(self, seed):
                return np.array(seed, np.float32) if kind == "scalar" else {"seed": seed}

            def close(self):
                pass

        return Environment()

    pool = ProcessSimulatorPool(factory, 1)
    try:
        observations = pool.reset(range(3), (1, 2, 3))
        assert pool.observation_transport == "pipe"
        if kind == "scalar":
            assert observations[0].shape == ()
            assert observations[0].item() == 1
        else:
            assert observations == [{"seed": 1}, {"seed": 2}, {"seed": 3}]
    finally:
        pool.close()


@pytest.mark.parametrize("changed", ["shape", "dtype"])
def test_midflight_observation_layout_change_is_rejected_and_unlinked(changed):
    def factory(scenario, path):
        class Environment:
            def reset(self, seed):
                return np.zeros((20, 91), np.float32)

            def step(self, *args):
                observation = (np.zeros((1, 91), np.float32) if changed == "shape"
                               else np.zeros((20, 91), np.float64))
                return observation, False, False, {}

            def close(self):
                pass

        return Environment()

    pool = ProcessSimulatorPool(factory, 1, transport="shared_memory")
    pool.reset(range(3), (1, 2, 3))
    name = pool.shared_memory_name
    try:
        with pytest.raises(RuntimeError, match="shape/dtype changed"):
            pool.step([(0, "skill", {})])
        assert pool.closed and pool.shared_memory_name is None
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=name)
    finally:
        pool.close()


def test_shared_and_pipe_training_have_identical_policy_optimizer_and_rng(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("mlflow")
    from bvr_behavior_prediction.rl.config import PilotTrainingConfig
    from bvr_behavior_prediction.rl.reward import RewardDefinition
    from bvr_behavior_prediction.rl.trainer import PPOTrainer

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    trainers, traces = [], []
    try:
        for transport in ("pipe", "shared_memory"):
            config = PilotTrainingConfig(
                episode_duration_s=3, hidden_size=8, transformer_heads=2, transformer_layers=1,
                history_duration_s=1, sample_interval_s=1, scenarios_per_epoch=3,
                evaluation_scenarios_per_epoch=3, evaluation_seed=1000,
                simulator_executor="process", simulator_workers=2,
                simulator_transport=transport, update_epochs=2, minibatch_size=4,
                output_dir=tmp_path / transport,
            )
            trainer = PPOTrainer(environment_factory(), 1, config, "cpu", RewardDefinition(
                "ticks", lambda: lambda info: (float(info["ticks"]), {}),
            ))
            trainer._training_active = True
            trainers.append(trainer)
            trace = []
            for _ in range(2):
                episodes = trainer._training_episodes()
                loss = trainer._update([episode[0] for episode in episodes])
                trace.append((loss, trainer._evaluate().copy(), torch.get_rng_state().clone()))
            traces.append(trace)
        for left, right in zip(traces[0], traces[1]):
            assert left[0] == right[0]
            np.testing.assert_array_equal(left[1], right[1])
            assert torch.equal(left[2], right[2])
        for name, tensor in trainers[0].pilot.state_dict().items():
            torch.testing.assert_close(tensor, trainers[1].pilot.state_dict()[name], rtol=0, atol=0)
        for left, right in zip(trainers[0].optimizer.state.values(),
                               trainers[1].optimizer.state.values()):
            for name, tensor in left.items():
                torch.testing.assert_close(tensor, right[name], rtol=0, atol=0)
    finally:
        for trainer in trainers:
            trainer.close()
        torch.set_num_threads(previous_threads)
