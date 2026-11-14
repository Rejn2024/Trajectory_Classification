"""Measure observation transport only; this is not a flight/PPO speed benchmark."""

import argparse
import json
import statistics
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "bvr_sim_source")]

from bvr_behavior_prediction.rl.simulation_workers import ProcessSimulatorPool


def benchmark(workers=2, scenarios=60, decisions=100, repeats=5):
    if min(workers, scenarios, decisions, repeats) < 1:
        raise ValueError("Benchmark budgets must be positive")

    def factory(scenario, recording_path):
        class Environment:
            def reset(self, seed):
                self.observation = np.full((20, 91), seed, dtype=np.float32)
                return self.observation

            def step(self, name, parameters):
                self.observation += 1
                return self.observation, False, False, {}

            def close(self):
                pass

        return Environment()

    samples = {"pipe": [], "shared_memory": []}
    actions = [(index, "change_trajectory", {}) for index in range(scenarios)]
    with ExitStack() as stack:
        pools = {}
        for transport in samples:
            pool = ProcessSimulatorPool(factory, min(workers, scenarios), transport=transport)
            stack.callback(pool.close)
            pools[transport] = pool
        # Alternate order to reduce bias from a changing CPU load/frequency.
        for repeat in range(repeats):
            order = list(samples) if repeat % 2 == 0 else list(reversed(samples))
            for transport in order:
                pool = pools[transport]
                pool.reset(range(scenarios), tuple(range(scenarios)))
                for _ in range(5):
                    pool.step(actions)
                started = time.perf_counter()
                for _ in range(decisions):
                    result = pool.step(actions)
                elapsed = time.perf_counter() - started
                for index, item in enumerate(result):
                    np.testing.assert_array_equal(item[0][0], index + 5 + decisions)
                samples[transport].append(elapsed)
    medians = {name: statistics.median(values) for name, values in samples.items()}
    return {
        "workers": min(workers, scenarios), "scenarios": scenarios,
        "decisions": decisions, "repeats": repeats, "history_shape": [20, 91],
        "observation_bytes_per_batch": scenarios * 20 * 91 * 4,
        "samples_seconds": samples, "median_seconds": medians,
        "pipe_to_shared_speed_ratio": medians["pipe"] / medians["shared_memory"],
        "scope": "Synthetic observation transport, including independent parent snapshots; "
                 "excludes worker startup, simulator physics, inference and PPO.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--scenarios", type=int, default=60)
    parser.add_argument("--decisions", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    print(json.dumps(benchmark(**vars(parser.parse_args())), indent=2))
