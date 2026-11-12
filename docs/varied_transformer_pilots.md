# Train ten pilots with small network and loss variations

[Notebook 30](../notebooks/30_train_varied_transformer_pilots.ipynb) follows notebook
06's simple-opponent JSBSim experiment and trains ten independent catalogue-v4
pilots (`change_trajectory` and `launch`). It retains the existing hybrid reward
preferences and common comparison benchmark, and adds random architecture/loss
variations to each pilot.

| Setting | Sampling rule |
| --- | --- |
| Transformer width | Uniform choice from 56, 64, 72 (64 ±12.5%) |
| Transformer layers | Uniform choice from 1, 2, 3 (2 ±1 layer) |
| Attention heads | Fixed at 4 |
| Value-loss coefficient | Uniform 0.45–0.55 (0.5 ±10%) |
| Entropy coefficient | Uniform 0.009–0.011 (0.01 ±10%) |
| Policy-loss multiplier | Fixed at 1 |
| Reward preferences | Notebook 06's existing `dogfight_population_weights(10)` |

The PPO objective remains `policy_loss + value_coefficient * value_loss -
entropy_coefficient * entropy`. Each sampled coefficient is applied to the
actual trainer configuration. The existing `PilotSpec` now accepts validated
`training_overrides` for the four varied fields; it cannot override seeds,
evaluation suites or output paths. Width/head compatibility is checked before
output is reserved or simulation starts. Checkpoint loading reconstructs each
pilot's saved width and layer count, including the selected-pilot replay cell.

## Hostile missile observations

Notebook 30 uses **91 features per frame** and a chronological **20-frame history**
sampled every 0.1 seconds. `BVRJSBSimBackend(include_hostile_missiles=True)` returns
87 features, and `BluePilotEnvironment` appends four energy features. The input
projection in `HybridSkillPilot` is constructed with this 91-feature size for both
collection and PPO updates; replay reconstructs it from the saved checkpoint metadata.

The first 47 values retain the original compact observation, including enemy
aircraft state and four friendly missile slots. Four hostile missile slots follow,
each with ten values. Only live hostile missiles targeting the controlled aircraft
are included. They are sorted nearest first, with UID as a deterministic tie-break;
at most four are shown. Each new sample recomputes the slots, so slot identity may
change as ranges cross. Missing slots are all zero. Resolved missiles are excluded
even though historical entries remain in `under_missiles`.

For slot `i` (zero-based), its first index is `47 + 10*i`:

| Offset | Feature | Normalization |
| --- | --- | --- |
| 0 | Presence | 1 for a populated slot; 0 for padding |
| 1–3 | Missile position minus own position, North/West/Up | / 50,000 m, / 50,000 m, / 10,000 m |
| 4–6 | Missile velocity minus own velocity, North/West/Up | / 1,200 m/s |
| 7 | Three-dimensional range | / 50,000 m |
| 8 | Signed closing speed | / 1,200 m/s; positive when approaching |
| 9 | Active missile seeker (`radar_on`) | 0 or 1 |

Closing speed is `-dot(relative_position, relative_velocity) / range`, defined as
zero at zero range. It is a geometric range rate, not an estimated time to impact.
Values are scaled without clipping. Slots are zero when the controlled aircraft is dead.
The four original energy features now occupy indices **87–90**: own kinetic,
own potential, opponent kinetic, opponent potential. Energy formulas and sampling
remain unchanged.

Missile information comes directly from **simulator truth**, consistent with the
existing compact enemy observation. It is not gated by radar/MWS detection or
seeker activation: a missile is visible before its seeker turns on. This does not
change the sensor models or make the policy's information sensor-realistic.

Train fresh pilots for this observation contract. Old 51-feature checkpoints
remain usable with the default `BVRJSBSimBackend(include_hostile_missiles=False)`
and their original environment. They cannot initialize the new 91-input models.
The notebook records the observation schema, slot count, field order, source and
frame size in `pilots.json` population metadata; each pilot's `config.json` records
`observation_size: 91`. Other training notebooks retain their existing input sizes.

## Run

Use the current branch and a fresh notebook kernel. Install dependencies as in
notebook 06 (`python -m pip install -e '.[ml,video]'`). Run notebook 30 from the top.
Defaults remain **10 pilots, 80 epochs each, 50 training and 50 validation
scenarios, 120 simulated seconds per flight**. Flights run in CPU worker
processes, while policy/PPO computation uses CUDA when available. The opponent
uses the existing simple baseline with JSBSim flight dynamics.

The architecture choices and coefficient ranges are editable in the roster cell.
`BVR_PILOT_VARIATION_SEED` defaults to 42 and controls only these draws; network
and scenario seeds retain their separate `BVR_PILOT_SEED`/`BVR_PILOT_SEED_STRIDE`
settings. Identical options and variation seed reproduce the same draws.
Increasing the population count preserves the architecture/loss sample prefix;
the reward preference grid retains notebook 06's existing count-dependent rule.

For an execution check covering all ten variants, set these before setup:

```python
import os
os.environ.update({
    "BVR_PILOT_EPOCHS": "1", "BVR_PILOT_EPISODE_SECONDS": "1",
    "BVR_PILOT_SCENARIOS": "3", "BVR_PILOT_EVALUATION_SCENARIOS": "3",
    "BVR_PILOT_UPDATE_EPOCHS": "1", "BVR_PILOT_WORKERS": "1",
    "BVR_PILOT_TORCH_THREADS": "1", "BVR_PILOT_DEVICE": "cpu",
})
```

Keep `BVR_PILOT_COUNT` unset for ten pilots. Clear these environment overrides
and restart the kernel before a production run. A short check verifies execution;
it does not train competent pilots.

## Recorded outputs

A fresh batch is created under
`artifacts/rl_pilot_simplified/varied_transformers/<timestamp>/`, with MLflow
experiment `jsbsim-simplified-varied-transformers`. `BVR_PILOT_OUTPUT` may select
another fresh directory.

- `pilots.json` records the sampler version, seed, choices/ranges, complete planned
  roster, exact reward definitions and each pilot's overrides before training starts.
  Thus interrupted training still retains the requested settings.
- `<pilot>/config.json` and MLflow parameters record the effective training
  configuration, including the sampled width, layers and loss coefficients.
- `<pilot>/manifest.json` records the overrides, parameter count, reward definition,
  checkpoint hash and benchmark results.
- `variant_roster.csv` exports the exact sampled settings and reward weights after
  the training call completes or raises an exception. The JSON roster is the
  authoritative record if a kernel/process is killed before CSV export.
- Existing JSONL learning curves, checkpoints, benchmark tables and selected-pilot
  ACMI replays are retained. Replay manifests include the selected overrides.

Common scenario seeds match the conditions across pilots; different architectures
have different initial weights even with the same seed. Network, loss and reward
variations are combined, so outcomes cannot isolate their individual causal effects.
Compare common benchmark outcomes. Own-reward returns and weighted PPO losses have
different meanings between pilots. Run separate controlled experiments to attribute
changes to one setting.

## Validation

The hostile-missile observation update passed **153 regression tests**, with
**12 CUDA checks skipped** on a CPU environment. Coverage includes threat filtering,
nearest-four ordering and tie-breaking, geometry/scaling, seeker state, zero range,
resolved/dead missile padding, history sampling, PPO, and checkpoint reloads.
A real JSBSim flight with an incoming AIM-120 updated the new slot for ten steps.
The updated notebook also ran end to end for all ten variants using one epoch,
three one-second training and validation flights, one PPO pass and one process
worker. All ten 91-input checkpoints were saved and benchmarked; pilot 010 was
reloaded and three replays were generated. These are execution checks, not evidence
of learning quality; the full production experiment was not run.

Before the hostile-missile update, the focused sampling, per-pilot configuration, replay, reuse and original population
notebook suite passed 61 tests. After final notebook guards and failure-recording
checks, the broader RL/reward/experiment regression run passed **144 tests**, with
**8 CUDA checks skipped** on this CPU environment. Notebook 30 also executed
end to end against real JSBSim for **all ten sampled variants**, using one epoch,
three one-second training and validation scenarios per pilot, and one PPO pass.
All ten checkpoints and recorded settings were verified; the selected tenth pilot
was reloaded and three replay flights were produced. This does not validate
production training quality. The default ten-by-80-epoch experiment was not started.
