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

The focused sampling, per-pilot configuration, replay, reuse and original population
notebook suite passed 61 tests. After final notebook guards and failure-recording
checks, the broader RL/reward/experiment regression run passed **144 tests**, with
**8 CUDA checks skipped** on this CPU environment. Notebook 30 also executed
end to end against real JSBSim for **all ten sampled variants**, using one epoch,
three one-second training and validation scenarios per pilot, and one PPO pass.
All ten checkpoints and recorded settings were verified; the selected tenth pilot
was reloaded and three replay flights were produced. This does not validate
production training quality. The default ten-by-80-epoch experiment was not started.
