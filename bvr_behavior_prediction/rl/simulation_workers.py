"""Stateful simulator workers; policy inference stays in the parent process."""

import os
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.shared_memory import SharedMemory
from time import perf_counter, process_time, thread_time

import numpy as np


class SimulatorWorkerError(RuntimeError):
    """A lost worker/pipe; the parent may replay the entire uncommitted batch."""


class SimulatorWorkerTimeout(SimulatorWorkerError, TimeoutError):
    """A worker exceeded its response deadline."""


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
    observation_memory = observation_array = None
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
                results = []
                for index, name, parameters in payload:
                    step, wall, cpu = _step(environments[index], name, parameters, process_time)
                    if observation_array is not None:
                        observation = np.asarray(step[0])
                        if (observation.shape != observation_array.shape[1:]
                                or observation.dtype != observation_array.dtype):
                            raise ValueError("Shared observation shape/dtype changed during a flight")
                        np.copyto(observation_array[index], observation, casting="no")
                        step = (None, *step[1:])
                    results.append((index, (step, wall, cpu)))
                connection.send((True, results))
            elif operation == "observations":
                observation_array = None
                if observation_memory is not None:
                    observation_memory.close()
                    observation_memory = None
                if payload is not None:
                    name, shape, dtype = payload
                    observation_memory = SharedMemory(name=name)
                    observation_array = np.ndarray(
                        shape, dtype=np.dtype(dtype), buffer=observation_memory.buf,
                    )
                connection.send((True, None))
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
            observation_array = None
            try:
                if observation_memory is not None:
                    observation_memory.close()
            finally:
                connection.close()


class ProcessSimulatorPool:
    """Assign each environment to one persistent, notebook-compatible CPU process.

    Processes live across a pilot's training epochs; environments are reconstructed
    for each batch, matching the original collector's reset and cleanup behaviour.
    Actions and step metadata cross pipes. Numeric observations use a shared
    buffer when available; the parent copies snapshots before the next command.
    Notebook closures are serialized once. Only the parent unlinks shared memory.
    """

    def __init__(self, factory, workers, timeout=120, *, transport="auto"):
        import cloudpickle
        from loky.backend.context import get_context

        self.workers = []
        self.timeout = timeout
        self.closed = False
        if transport not in ("auto", "shared_memory", "pipe"):
            raise ValueError("Unknown simulator observation transport")
        self.transport = transport
        self._observation_memory = self._observation_array = None
        self._shared_memory_failed = False
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
        try:
            while not connection.poll(0.1):
                if not process.is_alive():
                    raise SimulatorWorkerError(
                        f"Simulator worker {process.pid} exited unexpectedly "
                        f"(exit code {process.exitcode})"
                    )
                if perf_counter() > deadline:
                    raise SimulatorWorkerTimeout(f"Simulator worker {process.pid} did not respond")
            succeeded, value = connection.recv()
        except SimulatorWorkerError:
            raise
        except (EOFError, OSError) as error:
            # A pipe can close just before the process exit code becomes available.
            process.join(timeout=0.1)
            raise SimulatorWorkerError(
                f"Simulator worker {process.pid} disconnected (exit code {process.exitcode})"
            ) from error
        if not succeeded:
            raise RuntimeError(f"Simulator worker {process.pid} failed:\n{value}")
        return value

    def _request(self, operation, groups):
        if self.closed:
            raise RuntimeError("Simulator pool is closed")
        try:
            for worker, group in groups:
                try:
                    worker[1].send((operation, group))
                except (EOFError, OSError) as error:
                    process = worker[0]
                    process.join(timeout=0.1)
                    raise SimulatorWorkerError(
                        f"Simulator worker {process.pid} disconnected during {operation} "
                        f"(exit code {process.exitcode})"
                    ) from error
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
        ordered = [observations[index] for index in range(len(scenarios))]
        try:
            self._configure_observations(ordered)
        except BaseException:
            self.close(force=True)
            raise
        return ordered

    @property
    def observation_transport(self):
        return "shared_memory" if self._observation_memory is not None else "pipe"

    @property
    def shared_memory_name(self):
        return self._observation_memory.name if self._observation_memory is not None else None

    def _dispose_observations(self):
        self._observation_array = None
        memory, self._observation_memory = self._observation_memory, None
        if memory is not None:
            try:
                memory.close()
            finally:
                try:
                    memory.unlink()
                except FileNotFoundError:
                    pass

    def _configure_observations(self, observations):
        supported = bool(observations) and all(
            isinstance(obs, np.ndarray) and obs.dtype.kind in "biuf" and obs.size > 0 and obs.ndim > 0
            and obs.shape == observations[0].shape and obs.dtype == observations[0].dtype
            for obs in observations
        )
        enabled = self.transport != "pipe" and not self._shared_memory_failed and supported
        if not supported and self.transport == "shared_memory":
            raise ValueError("Shared observations require matching nonempty numeric array histories")
        shape = (len(observations), *observations[0].shape) if enabled else None
        dtype = observations[0].dtype if enabled else None
        if self._observation_array is not None and (
            not enabled or shape != self._observation_array.shape
            or dtype != self._observation_array.dtype
        ):
            self._request("observations", [(worker, None) for worker in self.workers])
            self._dispose_observations()
        if not enabled:
            return
        if self._observation_memory is None:
            try:
                self._observation_memory = SharedMemory(
                    create=True, size=int(np.prod(shape)) * dtype.itemsize,
                )
            except OSError:
                if self.transport == "shared_memory":
                    raise
                self._shared_memory_failed = True
                warnings.warn("Shared observation memory unavailable; using pipes", RuntimeWarning)
                return
            self._observation_array = np.ndarray(
                shape, dtype=dtype, buffer=self._observation_memory.buf,
            )
            specification = (self._observation_memory.name, shape, dtype.str)
            self._request("observations", [(worker, specification) for worker in self.workers])
        for index, observation in enumerate(observations):
            np.copyto(self._observation_array[index], observation, casting="no")

    def step(self, actions):
        groups = [[] for _ in self.workers]
        for action in actions:
            groups[action[0] % len(groups)].append(action)
        batches = self._request("step", [
            (worker, group) for worker, group in zip(self.workers, groups) if group
        ])
        results = dict(item for batch in batches for item in batch)
        ordered = []
        for index, *_ in actions:
            step, wall, cpu = results[index]
            if self._observation_array is not None:
                # Rollouts must never retain a view into mutable shared storage.
                step = (self._observation_array[index].copy(), *step[1:])
            ordered.append((step, wall, cpu))
        return ordered

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
        try:
            for process, connection in self.workers:
                try:
                    if not force:
                        process.join(timeout=2)
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=2)
                finally:
                    connection.close()
        finally:
            self._dispose_observations()
