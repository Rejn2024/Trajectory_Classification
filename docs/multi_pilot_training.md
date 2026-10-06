# Training independent pilots with different rewards

Use [`05_train_multiple_pilots.ipynb`](../notebooks/05_train_multiple_pilots.ipynb).
The original `04_train_rl_pilot.ipynb` remains the single-pilot example.

The new notebook accepts an arbitrary-length list of `PilotSpec` entries. Each entry
has a unique ID, a seed, and a `RewardDefinition`. Names are identifiers; there is no
fixed classification into aggressive, defensive, or standard pilots. The notebook's
default ten pilots demonstrate varying weights, and `BVR_PILOT_COUNT` can generate
any positive number. Edit `make_reward_definition(index)` or supply an explicit list
to choose each pilot's actual objective.

Each pilot gets a fresh network, optimizer, training scenarios, MLflow run, metrics,
and checkpoint. Pilots run sequentially so GPU allocations can be released between
runs. `BVR_PILOT_WORKERS` controls simulator concurrency within each pilot.
The multi-pilot notebook uses separate CPU **processes** for flight simulation and
opponent control. Each process owns its environments and is reused across epochs;
batched policy inference and PPO updates stay in the parent process on the GPU.
This lets Python control code run on multiple cores despite Python's
[global interpreter lock](https://docs.python.org/3/library/threading.html#gil-and-performance-considerations).
Factories defined inside the notebook are supported through `loky` and `cloudpickle`.
The notebook defaults to 80% of the machine's logical CPU count, rounded down with
a minimum of one worker. Active workers are capped by the number of flights in each
batch. PyTorch CPU operations default to at most 8 threads, also bounded by the
simulator worker budget. These settings leave capacity for interactive work without reserving specific
cores or enforcing an exact CPU percentage. Set `BVR_PILOT_WORKERS` and
`BVR_PILOT_TORCH_THREADS` before setup to override them. More workers do not guarantee
proportional speedups; compare the recorded rollout and epoch times.
Training more pilots increases total work; it does not divide a fixed epoch budget.
These are separate 1-v-1 training runs, each against the existing BVR Sim baseline.

## Installation and a small run

In VS Code, open this repository folder and the notebook, then choose **Select Kernel
> Select Another Kernel > Jupyter Kernels > Trajectory pilots (.venv)**. The notebook
requests this named kernel. Register it once from the repository root if missing:

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name trajectory-pilots --display-name "Trajectory pilots (.venv)"
```

Alternatively, select the repository's `.venv/Scripts/python.exe` under **Python
Environments**. The environment must have `ipykernel` installed. If the picker is
stale, run **Developer: Reload Window** from the Command Palette and try again.
Restart the kernel after changing environments and run the notebook from the top.
The setup prints the PyTorch thread count; the configuration cell prints simulator
workers. Environment variables set in an integrated terminal do not change an
already running notebook kernel; set overrides in a notebook cell before setup.

For NVIDIA GPU training, install CUDA-enabled PyTorch into that same environment.
Shut down kernels using it before replacing PyTorch on Windows, then restart the
notebook after installation. The CUDA 13.0 build below requires a compatible NVIDIA
driver (580 or newer); consult the [PyTorch installation guide](https://pytorch.org/get-started/locally/)
and [NVIDIA compatibility table](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
when setting up another machine:

```powershell
.\.venv\Scripts\python.exe -m pip --use-feature=truststore install "torch==2.14.1+cu130" --index-url https://download.pytorch.org/whl/cu130
```

The notebook automatically selects CUDA when `torch.cuda.is_available()` is true.
An explicit `BVR_PILOT_DEVICE=cpu` overrides automatic selection. Check that setup
prints `device: cuda` and `cuda_available: True` before starting a GPU run. Policy
inference and PPO updates run on the GPU; flight dynamics, opponent control, and
reward calculation remain on the CPU. The simulator worker budget still applies.

After updating this code, **restart the notebook kernel and run from the top** so it
loads the new collector. Check that the printed configuration includes
`"simulator_executor": "process"`. The first collection includes process startup;
subsequent collections reuse the workers. A running kernel keeps its previously
imported code until restarted. `BVR_PILOT_EXECUTOR=thread` selects the previous
collector for comparisons; the standalone `PilotTrainingConfig` keeps that default.

CPU load varies by phase: flight collection uses the CPU workers, while policy
updates use the GPU. An 80% worker budget does not guarantee 80% sustained CPU load.
`training_metrics.jsonl` records `rollout_seconds`, `update_seconds`, and
`evaluation_seconds` separately. `simulator_effective_cores` divides measured worker
CPU time by simulator wall time; `simulator_cpu_utilization_percent` normalizes it
by the logical CPU count. These describe simulator steps only, excluding startup,
policy work, and other applications. The former `simulator_parallel_speedup` and
`estimated_total_speedup` metrics were removed because summing threaded wall times
includes waiting and is not a valid measurement of speedup.

Install the project with `python -m pip install -e '.[ml,video]'`. For a short
PowerShell installation check, set these before launching the notebook:

```powershell
$env:BVR_PILOT_COUNT = "2"
$env:BVR_PILOT_EPOCHS = "1"
$env:BVR_PILOT_EPISODE_SECONDS = "1"
$env:BVR_PILOT_SCENARIOS = "3"
$env:BVR_PILOT_EVALUATION_SCENARIOS = "3"
$env:BVR_PILOT_UPDATE_EPOCHS = "1"
$env:BVR_PILOT_WORKERS = "1"
```

This checks the pipeline; one epoch does not establish that a pilot learned useful
behaviour. The notebook defaults to a fresh timestamped output directory. An explicit
`BVR_PILOT_OUTPUT` must also name a new directory. Reusing an existing batch raises
`FileExistsError` before training, instead of overwriting models or mixing metrics.

## Choose weights or supply a different formula

The existing event-based combat reward is available with arbitrary weights:

```python
from dataclasses import replace
from bvr_behavior_prediction.rl import (
    PilotSpec, RewardWeights, combat_reward_definition,
)

pilots = [
    PilotSpec("experiment_001", 7, combat_reward_definition(
        "objective_001", weights=replace(RewardWeights(), crashed=-200.0),
    )),
    PilotSpec("experiment_002", 1007, combat_reward_definition(
        "objective_002", weights=replace(RewardWeights(), target_lock_acquired=5.0),
    )),
]
```

The reward interface also accepts entirely different formulas. A factory returns a
fresh callable for each episode, including evaluation and demonstrations. The callable
accepts the environment's `info` dictionary and returns a scalar reward and a dictionary
of named reward components. Reward state must belong to that episode, not to a shared
singleton. For example:

```python
from functools import partial
from bvr_behavior_prediction.rl import CombatReward, RewardDefinition

class SpeedReward:
    def __init__(self, target_mps, penalty):
        self.events = CombatReward()
        self.target_mps = target_mps
        self.penalty = penalty

    def __call__(self, info):
        _, components = self.events(info)
        if info.get("blue_alive", False):
            error = abs(info["blue_speed_mps"] - self.target_mps) / self.target_mps
            components["speed_error"] = -self.penalty * error
        return sum(components.values()), components

pilots.append(PilotSpec(
    "experiment_003", 2007,
    RewardDefinition(
        name="speed_objective",
        factory=partial(SpeedReward, target_mps=250.0, penalty=0.1),
        parameters={"target_mps": 250.0, "penalty": 0.1},
        version="1",
    ),
))
```

Keep all tunable settings in `parameters`, increment `version` when the formula
changes, and retain the implementation in version-controlled source. The manifest
records the recipe metadata, not executable Python or a serialized closure. Reward
signals can use simulator information during training; do not add privileged reward
inputs to the deployed policy's observations.

The current backend supplies own/opponent speed and altitude, own survival, target
lock, firing, evasion, opponent destruction, crash, and shot-down indicators. New
objectives needing other signals must also add those signals to the backend.

## Calling the pipeline from Python

```python
from bvr_behavior_prediction.rl import train_pilots

results = train_pilots(
    pilots=pilots,
    config=config,                       # PilotTrainingConfig; output_dir is a new batch
    env_factory_builder=make_environment_factory,
    observation_size=OBSERVATION_SIZE,
    device="cuda",
)
```

`make_environment_factory(run_config)` returns the existing two-argument environment
factory `(scenario, recording_path)`. Use `run_config.output_dir` for simulator logs
and other per-pilot outputs. The notebook provides the complete JSBSim implementation.
The existing `PPOTrainer` also accepts `reward_definition=` for a standalone run.

## Comparing rewards fairly

Each pilot selects its best checkpoint using its own reward. Its `selection_return`
cannot be compared directly with a different reward formula or scale. After training,
all selected checkpoints are scored with the same common combat reward on exactly
the same deterministic evaluation geometries and seeds. Compare `benchmark_return`,
`survival_rate`, and `opponent_destroyed_rate`; standard deviations describe variation
across scenarios, not confidence intervals. You can pass a different common
`benchmark_reward=RewardDefinition(...)` to `train_pilots`.

`evaluation_seed` is shared across pilots. The pipeline rejects overlap with training
episode seed ranges. If unspecified, it chooses a separate range above the training
seeds. These scenarios also select checkpoints, so they are validation data, not a
final held-out test set. Use another scenario/seed range for final assessment.
Pilots may share a training seed for a controlled reward comparison. Seeds initialize
Python, NumPy, and PyTorch, but identical results across hardware/library versions or
simulator threading configurations are not guaranteed.

## Saved files and reuse

```text
<batch>/
  pilots.json
  <pilot-id>/
    best_model.pt
    config.json
    manifest.json
    training_metrics.jsonl
    demonstrations/
    final_evaluation/       # generated for the selected pilot by the replay cell
```

`pilots.json` records pending, training, completed, failed, or interrupted states.
Completed results remain available when a later pilot fails. Start remaining pilots
in a new batch; this pipeline does not resume optimizer state.

Each pilot manifest records its objective, seed, checkpoint SHA-256, common benchmark,
outcomes, and MLflow run ID. `config.json` also saves network dimensions and the exact
skill and parameter ordering. Load a completed model after restarting Python:

```python
from bvr_behavior_prediction.rl import load_trained_pilot
pilot = load_trained_pilot("artifacts/my_batch/experiment_001", device="cpu")
action = pilot.act(observation_history, deterministic=True)
```

The loader verifies the checkpoint checksum and action catalogue before returning
an evaluation-mode model. Publish its checkpoint, configuration, and manifest together
under `shared/models/<pilot-version>/` using the [LFS workflow](../shared/README.md).
Record both pilot ID and checkpoint hash in any flight corpus derived from that model.

References: [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)
and [MLflow run tracking](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html#mlflow.start_run).
