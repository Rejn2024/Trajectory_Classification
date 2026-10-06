"""Stateful simulator workers; policy inference stays in the parent process."""

import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter, process_time, thread_time


def _close_environments(environments):
    pending = list(environments.values())
    environments.clear()
    error = None
    for environment in pending:
        try:
            environment.close()
        except Exception as exc:  # noqa: BLE001 - close every environment before re-raising
            error = error or exc
    if error is not None:
        raise error


def _step(environment, name, parameters, cpu_clock):
    started, cpu_started = perf_counter(), cpu_clock()
    result = environment.step(name, parameters)
    return result, perf_counter() - started, cpu_clock() - cpu_started


class ThreadSimulatorPool:
    def __init__(self, factory, workers):
        self.factory = factory
        self.environments = {}
        self.executor = ThreadPoolExecutor(max_workers=workers)

    def reset(self, scenarios, seeds):
        self.release()
        for index, scenario in enumerate(scenarios):
            self.environments[index] = self.factory(scenario, None)
        # The BVR backend seeds NumPy's process-global RNG during reset. Resetting
        # concurrently in threads races those seeds and changes initial geometries.
        return [self.environments[index].reset(seed) for index, seed in enumerate(seeds)]

    def step(self, actions):
        return list(self.executor.map(
            lambda item: _step(self.environments[item[0]], item[1], item[2], thread_time),
            actions,
        ))

    def release(self):
        _close_environments(self.environments)

    def close(self, force=False):
        try:
            self.executor.shutdown(wait=True, cancel_futures=True)
        finally:
            self.release()


def _simulator_worker(connection, factory_bytes):
    """Importable loky entry point; never constructs a policy or CUDA context."""
    import cloudpickle

    environments = {}
    try:
        factory = cloudpickle.loads(factory_bytes)
        connection.send((True, os.getpid()))
        while True:
            operation, payload = connection.recv()
            if operation == "reset":
                _close_environments(environments)
                observations = []
                for index, scenario, seed in payload:
                    environments[index] = factory(scenario, None)
                    observations.append((index, environments[index].reset(seed)))
                connection.send((True, observations))
            elif operation == "step":
                results = [
                    (index, _step(environments[index], name, parameters, process_time))
                    for index, name, parameters in payload
                ]
                connection.send((True, results))
            elif operation in ("release", "close"):
                _close_environments(environments)
                connection.send((True, None))
                if operation == "close":
                    break
            else:
                raise ValueError(f"Unknown simulator operation: {operation}")
    except EOFError:
        pass
    except BaseException:  # noqa: BLE001 - propagate worker failures across the pipe
        try:
            connection.send((False, traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        try:
            _close_environments(environments)
        finally:
            connection.close()


class ProcessSimulatorPool:
    """Assign each environment to one persistent, notebook-compatible CPU process.

    Processes live across a pilot's training epochs; environments are reconstructed
    for each batch, matching the original collector's reset and cleanup behaviour.
    Only observations/actions cross pipes. Notebook closures are serialized once.
    """

    def __init__(self, factory, workers, timeout=120):
        import cloudpickle
        from loky.backend.context import get_context

        self.workers = []
        self.timeout = timeout
        self.closed = False
        factory_bytes = cloudpickle.dumps(factory)
        context = get_context("loky")
        worker_env = {
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "",
        }
        try:
            for _ in range(workers):
                parent, child = context.Pipe()
                process = context.Process(
                    target=_simulator_worker, args=(child, factory_bytes), env=worker_env,
                    daemon=True,
                )
                try:
                    process.start()
                except BaseException:
                    parent.close()
                    raise
                finally:
                    child.close()
                self.workers.append((process, parent))
            for worker in self.workers:
                self._receive(worker)
        except BaseException:
            self.close(force=True)
            raise

    def _receive(self, worker):
        process, connection = worker
        deadline = perf_counter() + self.timeout
        while not connection.poll(0.1):
            if not process.is_alive():
                raise RuntimeError(f"Simulator worker {process.pid} exited unexpectedly")
            if perf_counter() > deadline:
                raise TimeoutError(f"Simulator worker {process.pid} did not respond")
        try:
            succeeded, value = connection.recv()
        except (EOFError, OSError) as error:
            raise RuntimeError(f"Simulator worker {process.pid} disconnected") from error
        if not succeeded:
            raise RuntimeError(f"Simulator worker {process.pid} failed:\n{value}")
        return value

    def _request(self, operation, groups):
        if self.closed:
            raise RuntimeError("Simulator pool is closed")
        try:
            for worker, group in groups:
                worker[1].send((operation, group))
            return [self._receive(worker) for worker, _ in groups]
        except BaseException:
            self.close(force=True)
            raise

    def reset(self, scenarios, seeds):
        groups = [[] for _ in self.workers]
        for index, (scenario, seed) in enumerate(zip(scenarios, seeds)):
            groups[index % len(groups)].append((index, scenario, seed))
        batches = self._request("reset", list(zip(self.workers, groups)))
        observations = dict(item for batch in batches for item in batch)
        return [observations[index] for index in range(len(scenarios))]

    def step(self, actions):
        groups = [[] for _ in self.workers]
        for action in actions:
            groups[action[0] % len(groups)].append(action)
        batches = self._request("step", [
            (worker, group) for worker, group in zip(self.workers, groups) if group
        ])
        results = dict(item for batch in batches for item in batch)
        return [results[action[0]] for action in actions]

    def release(self):
        if not self.closed:
            self._request("release", [(worker, None) for worker in self.workers])

    def close(self, force=False):
        if self.closed:
            return
        self.closed = True
        if not force:
            try:
                for _, connection in self.workers:
                    connection.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                force = True
        for process, connection in self.workers:
            try:
                if not force:
                    process.join(timeout=2)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2)
            finally:
                connection.close()
