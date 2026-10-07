# Training independent pilots with different rewards

Choose one of two independent population experiments:

| Notebook | Reward | Default population / epochs per pilot | Output subfolder / MLflow experiment |
| --- | --- | --- | --- |
| [05](../notebooks/05_train_multiple_pilots.ipynb) | Original combat events with varied coefficients | 10 / **150** | `combat/` / `jsbsim-combat-population` |
| [06](../notebooks/06_train_multiple_pilots.ipynb) | Hybrid evade/pursue/eliminate and shared combat guidance | 10 / 80 | `hybrid/` / `jsbsim-hybrid-population` |

Both output subfolders live under `artifacts/rl_pilot_notebook/`, with a new timestamp
per batch. `BVR_PILOT_OUTPUT` overrides the location, but an existing batch is rejected.
Notebook 06 preserves the former notebook 05 reward experiment. Notebook 04 remains
the original single-pilot example. `BVR_PILOT_EPOCHS` overrides either epoch default.

Both population notebooks accept an arbitrary-length list of `PilotSpec` entries. Each entry
has a unique ID, a seed, and a `RewardDefinition`. Names are identifiers; there is no
fixed classification into aggressive, defensive, or standard pilots. `BVR_PILOT_COUNT`
can generate any positive number. Edit `REWARD_WEIGHTS` in 05, `STYLE_WEIGHTS` and
`REWARD_SETTINGS` in 06, or `make_reward_definition(index)` to choose objectives.

Each pilot gets a fresh network, optimizer, training scenarios, MLflow run, metrics,
and checkpoint. Pilots run sequentially so GPU allocations can be released between
runs. `BVR_PILOT_WORKERS` controls simulator concurrency within each pilot.
Both notebooks use separate CPU **processes** for flight simulation and
opponent control. Each process owns its environments and is reused across epochs;
batched policy inference and PPO updates stay in the parent process on the GPU.
This lets Python control code run on multiple cores despite Python's
[global interpreter lock](https://docs.python.org/3/library/threading.html#gil-and-performance-considerations).
Factories defined inside the notebook are supported through `loky` and `cloudpickle`.
Both notebooks default to 80% of the machine's logical CPU count, rounded down with
a minimum of one worker. Active workers are capped by the number of flights in each
batch. PyTorch CPU operations default to at most 8 threads, also bounded by the
simulator worker budget. These settings leave capacity for interactive work without reserving specific
cores or enforcing an exact CPU percentage. Set `BVR_PILOT_WORKERS` and
`BVR_PILOT_TORCH_THREADS` before setup to override them. More workers do not guarantee
proportional speedups; compare the recorded rollout and epoch times.
Training more pilots increases total work; it does not divide a fixed epoch budget.
These are separate 1-v-1 training runs, each against BVR Sim's existing `simple`
opponent with JSBSim flight dynamics. The same 50 training scenarios are reused every
epoch by default; 50 separate validation scenarios run on the first, every fifth,
and last epoch. More epochs add optimization, not new scenarios.

## Notebook 05: original combat-event population

Notebook 05 uses `CombatReward`, with the same event rules as notebook 04. Locks,
locked launches, individual avoided missiles, and opponent destruction earn rewards;
crashes, shot-down events, and unlocked launches incur penalties. Safe airborne
decisions earn a small positive reward. There is no new pursuit, exposure-time, or
missile-support term. Original event repetition rules are retained, alongside the
shared fixes that preserve substep events and count resolved missiles individually.

`combat_population_weights(n)` begins with the exact original `RewardWeights()`:
crash -100, shot down -150, safe airborne +0.02, lock +2, locked launch +8, evasion +20,
opponent destruction +250, unlocked launch -4. Subsequent pilots vary five groups:

1. Crash and shot-down penalties together.
2. Opponent-destruction reward.
3. Avoided-missile reward.
4. Lock and locked-launch rewards together.
5. Unlocked-launch penalty.

Independent coordinates in a deterministic Halton sequence select log-spaced
multipliers between 0.5 and 2. Each event coefficient is then scaled by the same
normalization factor to retain the original absolute coefficient sum of **534**.
The ground-clearance coefficient remains +0.02. Relative preferences therefore vary
without a monotonic increase in all rewards. Final normalized coefficients can fall
outside 0.5–2 times the original. Extending the roster preserves earlier entries.
Normalization does not equalize episode returns: event frequencies still depend on
the pilot, and different coefficients do not guarantee distinct behaviour.

The notebook displays every coefficient before training and saves each recipe in
the pilot manifests. Override individual entries with `replace(RewardWeights(), ...)`
or define an explicit roster. Each checkpoint is selected using that pilot's own
reward, then scored using `common_combat` with the original weights for comparison.
Opponent destruction retains its original any-cause meaning; `elimination_rate`
separately reports confirmed missile kills. The default budget is **150 epochs per
pilot**, adjustable with `BVR_PILOT_EPOCHS`; use fresh test scenarios to assess whether
the larger budget improves generalization.

## Notebook 06: tactical objectives and style diversity

Notebook 06 uses `HybridDogfightReward`. Its default reward is
`250 * (w_evade * E + w_pursue * P + w_eliminate * K) + 100*K - 150*L + G`.
The three weights vary by pilot; the shared outcome and guidance terms stay fixed:

| Term | Measurement | Episode budget |
| --- | --- | --- |
| E: evade | Newly defeated, unique incoming missiles divided by 4, minus newly accumulated time under active missile threat divided by the episode time limit | Positive missile credit capped at 1; exposure cost capped at 1 |
| P: pursue | Improvement over the best closeness-and-alignment value already achieved, starting from the actual initial geometry | Total progress at most 1 |
| K: eliminate | One confirmed successful friendly missile against the destroyed opponent | At most 1 |
| L: loss | One penalty when the controlled aircraft dies, regardless of cause | -150 points, once |
| Shared elimination | The same confirmed elimination event K | +100 points, once, in addition to the weighted objective |
| G: shared guidance | First lock +2, first locked launch +8, and missile-support time up to +5 | At most +15 points total |

Closeness-and-alignment is `max(0, cos(angle)) * exp(-max(0, range - 30000) / 30000)`, where the angle
is between the controlled aircraft's actual velocity and the direction to its
opponent. Closeness stops paying once within the nominal 30 km engagement range;
the pilot can still improve alignment. This avoids rewarding ever-closer passes
solely for proximity. Staying still or repeatedly retreating and returning to the
same position cannot keep collecting progress.
It is a deliberately bounded pursuit objective, not a complete measure of tactical
advantage or a reward-shaping rule that preserves the original optimal policy.

The shared elimination reward gives every pilot a reason to complete the engagement.
With the default weight floor, an elimination earns 125 to 300 points across the
roster; the loss cost matches the original shot-down penalty of 150. These settings
are experiment defaults, not demonstrated optimal values.

Lock acquisition and a locked launch pay only once per flight, so reacquiring the
lock or firing repeatedly cannot multiply these bonuses. Support earns 0.25 points
per second up to 20 seconds total. It requires a live friendly missile targeting the
live opponent, an inactive missile seeker, and the controlled aircraft maintaining
target lock while alive. Multiple missiles do not multiply the time reward. Support
credit stops once the seeker activates, the target lock is lost, or the missile or
target dies. Support time is integrated at the simulator substep rate, so briefly
providing support inside a policy decision is still recorded.

The guidance terms are bounded heuristics, not potential-based shaping with a policy
invariance guarantee. They remain small relative to the elimination reward. There
are no separate altitude, speed, or named-skill bonuses. The complete 33-skill action catalogue and current missile-based combat
remain available. An elimination requires a successful friendly missile, so an
unrelated opponent crash does not pay. An evasion means a targeted missile failed
while the controlled aircraft remained alive; this alone does not prove a particular
manoeuvre caused the failure. Exposure counts time with at least one live incoming
missile, rather than multiplying by the number of missiles.

`dogfight_population_weights(n)` spreads a roster across a triangular grid of
preferences, beginning with equal weights and then selecting separated combinations.
The default 0.1 floor keeps all three objectives relevant, including near each
extreme (0.8/0.1/0.1). Increasing `n` can change the grid and therefore the roster:
the saved manifest, not the numeric pilot ID, is the authoritative style definition.
You can specify exact ratios using `DogfightWeights(6, 3, 1)`, which normalizes to
0.6/0.3/0.1. All scaling and cap settings are recorded in each reward definition.

```python
from bvr_behavior_prediction.rl import DogfightWeights, PilotSpec, hybrid_dogfight_reward_definition

pilots = [
    PilotSpec("experiment_001", 7, hybrid_dogfight_reward_definition(
        "objective_001", weights=DogfightWeights(evade=6, pursue=3, eliminate=1),
    )),
    PilotSpec("experiment_002", 7, hybrid_dogfight_reward_definition(
        "objective_002", weights=DogfightWeights(evade=1, pursue=3, eliminate=6),
    )),
]
```

The notebook defaults to matched seeds, initialization, and training scenarios for
all independently trained pilots. Set `BVR_PILOT_SEED_STRIDE=1000` to give each pilot
a different seed, or repeat the complete preference roster with a different
`BVR_PILOT_SEED` to assess training variability. Shared validation scenarios still
come from a separate seed range. Equal initial seeds do not share models or optimizers.

Different reward preferences can yield the same policy when objectives agree or
training does not find different solutions. The component bounds reduce scale
confounding, but reward frequency and PPO optimization still affect influence;
the weights are not measured percentages of behaviour. Evaluate on fresh scenarios
and additional opponents, and compare replays before assigning style labels.
This is the usual distinction between reward preferences and a learned policy in
[multi-objective reinforcement learning](https://arxiv.org/abs/2103.09568).

The common benchmark uses equal hybrid style weights and the same outcome rewards
for every pilot, with `lock_bonus`, `launch_bonus`, and `support_bonus` set to zero.
Training guidance therefore does not inflate the comparison score. Manifests
include `benchmark_component_means`, `elimination_rate`, `mean_missiles_avoided`,
and `mean_threatened_seconds`, alongside survival and any-cause opponent destruction.
Each rollout's final info includes `reward_component_totals` for the entire flight.
Compare evasion counts with exposure and survival: pilots choosing different
engagement patterns may encounter different numbers of threats.

The shared simulator adapter now counts resolved missiles individually, preserves
transient events across policy substeps, and reports the actual `simple` opponent
policy. This also corrects reward events when running notebook 04. Existing saved
artifacts are preserved; scores from runs before these fixes are not directly
comparable to newly computed scores. Notebook 04 still selects its legacy combat
reward, as does notebook 05 with a varied weight population. Notebook 06 explicitly
selects the hybrid reward. The notebook 04 file
and its original reward weights are unchanged. The pipeline's default reward remains
`combat_reward_definition`; the hybrid is enabled only by explicitly supplying it.

Three recipes remain available for comparisons: `combat_reward_definition` for the
original event-based reward, `dogfight_reward_definition` for the first pure
three-objective experiment, and `hybrid_dogfight_reward_definition` for notebook 06.
Their defaults and saved reward names distinguish the formulas. The pure recipe
keeps its original proximity formula and loss penalty; switching recipes requires
a fresh training run to evaluate its effects. Existing checkpoints remain usable.

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
The progress bar labels collection, PPO updates, validation, and demonstration saving.
Notebooks 05 and 06 enable `cuda_graph_updates` for CUDA: full minibatches replay a captured
forward/backward pass, avoiding repeated Python and GPU launch overhead. The optimizer,
gradient clipping, and AMP scaler still execute once per minibatch. The 50 PPO passes,
batch size of 64, architecture, rewards, scenarios, and evaluation cadence are unchanged.
Warmup computes gradients without applying extra optimizer steps. A single graph is
captured per update; the final partial minibatch and CPU training use the eager path.
Notebook 04 and standalone configurations retain the eager path by default.
Set `BVR_PILOT_CUDA_GRAPHS=0` before setup to compare eager execution or disable capture.
This follows PyTorch's [CUDA graph guidance for AMP](https://docs.pytorch.org/docs/stable/notes/cuda.html#usage-with-torch-cuda-amp).

On a fixed batch of 3,764 real flight transitions, 50 passes (2,950 minibatches) took
20.02 seconds eagerly versus 2.76 seconds with capture, averaged over two runs each.
Final model tensors were identical in that comparison. These are update timings,
not a guarantee of whole-run speedup; flight simulation and validation still take time.
In a separate two-epoch workflow with 50 training and 50 validation flights per epoch,
elapsed time fell from 145.7 to 74.4 seconds, including startup and saving. Selected
checkpoints were bit-for-bit identical. PPO averaged 37.4 versus 5.1 seconds per epoch;
simulation and validation were essentially unchanged. This short benchmark validates
execution speed and equivalence, not final pilot quality or a full-roster duration.
The local reproducible benchmark and its cached batch are under
`artifacts/training_performance_20261006/`, with the runner at `artifacts/profile_training.py`.

`training_metrics.jsonl` records `rollout_seconds`, `update_seconds`, and
`evaluation_seconds` separately, plus `checkpoint_seconds`, `rollout_transitions`,
`update_minibatches`, `update_transitions_per_second` (including repeated PPO passes),
`cuda_graph_updates` (1 if used), and `cuda_graph_capture_seconds` (included in update time).
`simulator_effective_cores` divides measured worker
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

## Legacy combat weights or a different formula

The original event-based combat reward used by notebook 05 accepts arbitrary weights:

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
all selected checkpoints are scored with the same benchmark reward on exactly
the same deterministic evaluation geometries and seeds. Notebook 05 uses the original
combat benchmark. Notebook 06 uses equal hybrid weights with guidance disabled;
`train_pilots` also defaults to the original combat benchmark when none is supplied.
Scores are comparable within each notebook's population. To compare populations from
05 and 06, evaluate both under one common reward and scenario set; their default
`benchmark_return` formulas differ. Compare component means, `survival_rate`,
and `elimination_rate`; standard deviations describe variation across scenarios,
not confidence intervals. You can pass a different common
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

The [published notebook 05 combat population](../shared/models/notebook05-combat-20261006T140858_362365Z/README.md)
contains all 10 selected checkpoints from the completed 150-epoch run with 75
training scenarios. Its release README provides selective LFS download and loading
commands. Configurations, manifests, complete epoch logs, and publication provenance
are included; the total LFS payload is 5.68 MiB. These checkpoints support inference,
not optimizer-state resumption. Their reported scores use validation scenarios, so
use a new scenario set for a final assessment.

References: [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)
and [MLflow run tracking](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html#mlflow.start_run).
