# BVR behaviour prediction

An implementation of the revised project plan for target-centric F-16 manoeuvre
recognition, tactical-transition forecasting, and BVR Sim native-action forecasting.

The package keeps observable features separate from privileged labels, groups splits
by episode, and exposes the simulator through a small adapter. BVR Sim itself is an
external dependency and must be pinned to commit
`db4f95657c081d120442022a339bed62d340c93b`.

```bash
python -m pip install -e '.[ml,dev]'
pytest
bvr-build-dataset --help
```

The lightweight tactical skill catalogue is included in this installation and
can be imported without installing the full simulator runtime or native backend:

```python
from bvr_sim.agents.skill_manager import SkillManager
```

See `configs/` for the initial F-16 experiment and `examples/train_pipeline.py` for
an end-to-end training skeleton.

## Sharing training results and flight datasets

Publish selected model checkpoints, training results, and flight corpora under
`shared/` using Git LFS. Large files are tracked by `.gitattributes`; small manifests
and configuration stay readable in Git. See the [sharing guide](shared/README.md)
for setup on each developer's machine, publishing and downloading, dataset layout,
and recording the exact trained agent behind a corpus.

## Reinforcement-learning 1-v-1 pilot

Two notebooks train independent populations, each with its own models and checkpoints:

- [`05_train_multiple_pilots.ipynb`](notebooks/05_train_multiple_pilots.ipynb) uses the
  original combat-event reward with varied relative coefficients. Defaults: 10 pilots,
  **150 epochs per pilot**, with the original notebook 04 weights as pilot 001.
- [`06_train_multiple_pilots.ipynb`](notebooks/06_train_multiple_pilots.ipynb) preserves
  the hybrid evade/pursue/eliminate experiment and shared lock/launch/support guidance.
  Defaults: 10 pilots, 80 epochs per pilot.

Both support arbitrary population sizes and editable reward factories, matched seeds,
persistent simulator processes, an 80% CPU worker budget, and CUDA-graph PPO updates.
They use separate output folders and MLflow experiments. Each has a common benchmark
within its population; benchmark formulas differ between notebooks.
See the [multi-pilot training guide](docs/multi_pilot_training.md) for configuration,
custom rewards, and loading saved pilots.

`bvr_behavior_prediction.rl` contains a PPO training path for a **blue-controlled**
pilot against the simulator's simple baseline opponent. The hybrid policy chooses among
all 33 `SkillManager` skills and predicts the bounded numeric parameters of the chosen
skill. A transformer processes a rolling two-second, 10 Hz history containing the normal
flight observations plus separate normalized kinetic- and potential-energy estimates for
blue and red. `PilotTrainingConfig` centralizes that sampling/history setup, the one-second
planning horizon, 90-second episode default, number of randomized geometries per epoch,
checkpoint cadence, and total epochs. Training writes JSONL diagnostics, MLflow
parameters/metrics, the improving best checkpoint, and seeded ACMI demonstration
directories. Supply
`PPOTrainer` with a `BluePilotEnvironment` factory backed by JSBSim; its narrow backend
contract is documented in `bvr_behavior_prediction/rl/environment.py`.

Training throughput is improved without reducing training scenario or epoch counts:
independent simulators advance concurrently, policy inference is batched and uses PyTorch
inference mode, skill contracts are cached instead of reparsed on every action, and CUDA
updates use mixed precision, fused Adam, and TF32 matrix multiplication. Full deterministic
evaluation runs on the first, last, and every fifth epoch by default (configure
`evaluation_interval=1` to restore every-epoch evaluation). If rollout, update, and
evaluation take `R`, `U`, and `E` seconds, respectively, this cadence estimates steady-state
acceleration as `(R + U + E) / (R + U + E / evaluation_interval)`. For similarly sized
training and evaluation batches where simulation dominates, the default approaches
`2 / 1.2 = 1.67x`; the measured phase times and `estimated_epoch_speedup` are written to
the per-epoch JSONL/MLflow diagnostics rather than attributing a hardware-independent
number to mixed precision. The multi-pilot notebook uses persistent CPU processes for
simulation, avoiding the Python interpreter lock in CPU-heavy flight control code.
The diagnostics separate policy time, simulator wall time, and actual worker CPU time;
`simulator_effective_cores` reports worker CPU seconds divided by simulator wall seconds.
Summed worker elapsed times include waiting and must not be interpreted as speedup.
PPO updates shuffle each training tensor once per update pass and then use
zero-copy contiguous minibatch views; the mean loss is transferred to the CPU only once,
eliminating five GPU gathers and one device synchronization per minibatch.

For a comparison of downloadable 1-v-1 policy resources, the bundled JSBSim PPO
training path, and a concrete plan for producing classifier-compatible RL trajectories,
see [`docs/jsbsim_1v1_ai_options.md`](docs/jsbsim_1v1_ai_options.md).

The executable notebook
[`notebooks/01_generate_smoke_dataset.ipynb`](notebooks/01_generate_smoke_dataset.ipynb)
walks through the first dataset milestone with a deterministic local dynamics stub. It
samples controlled scenarios, runs scripted policies through the same adapter used for
BVR Sim, writes the canonical Parquet layout, and performs schema and leakage checks.
The notebook marks the single environment-factory seam that must be replaced to run
the workflow against a pinned BVR Sim checkout.
## F-16 scripted-manoeuvre video

The runnable [`f16_scripted_manoeuvre_video.ipynb`](notebooks/f16_scripted_manoeuvre_video.ipynb)
uses JSBSim to fly a scripted F-16 demonstration and export an MP4 or GIF. See the
[setup, BVR Sim integration, validation, and troubleshooting guide](docs/f16_video_guide.md)
before running it.

To inspect a generated ACMI replay and turn it into a shareable presentation, follow the
step-by-step [Tacview visualisation and video-recording guide](docs/tacview_video_guide.md).
