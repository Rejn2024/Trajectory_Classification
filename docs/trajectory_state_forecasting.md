# Two-skill pilots and continuous flight-state forecasts

The current pilot catalogue is version **4.0.0**, with exactly two skills:

| Skill | Parameters |
| --- | --- |
| `change_trajectory` | `delta_heading_deg` (±180°), `delta_altitude_m` (±80 m), `delta_speed_mps` (±80 m/s) |
| `launch` | None |

`change_trajectory` emits all three relative requests together on each substep.
Each defaults to zero. There are no separate heading, altitude or speed skills.
The NN categorical head has two outputs and its continuous head has three values.
PPO scores all three parameters when trajectory control is chosen; launch has no
continuous action probability. Existing simulator normalization and control limits
remain in force, including steering-request saturation beyond ±45°.

Pilots from earlier catalogues must be retrained. Inference, initialization and
completed-run reuse validate the full versioned catalogue before loading weights.
Train with notebooks 04, 05 or 06; they print the actual two-skill catalogue and
create fresh timestamped run directories.

## Forecasting target

The current forecasting workflow predicts **future measured states**, not the
pilot's catalogue choice or commanded deltas. Both GRU and Transformer encoders
predict the target aircraft's speed, heading and altitude at **+1, +2 and +5 seconds**.
Notebook [29](../notebooks/29_trajectory_state_forecasting.ipynb) runs the workflow.

- Speed is the magnitude of world NWU velocity, in m/s.
- Heading is horizontal course, clockwise from north, in degrees [0, 360).
  It is derived from NWU velocity, not nose attitude; stationary/vertical tracks
  with undefined course are excluded from the targets.
- Altitude is world vertical position in metres.

Each horizon has four neural outputs: speed residual, sine of heading change,
cosine of heading change minus one, and altitude residual. Residuals reference
only the current observed state. A zero network predicts constant speed, course
and altitude. Heading decoding uses `atan2`; circular error treats 359° and 1°
as two degrees apart. Training balances speed, the two heading coordinates as
one quantity, and altitude using masked Huber loss. Speed/altitude scales and
input normalization are fitted on training groups only.

Inputs are 51 chronological observer frames spanning five seconds, plus backward
acceleration features. Streaming inference requires 56 consecutive raw 10 Hz
frames to compute acceleration at the oldest token. Target actions, skill labels,
pilot IDs and future observations are excluded from inputs. Future physical
tracks provide labels only. Complete matched scenario groups are assigned to
70/15/15 train/validation/test partitions. Reserved test episodes are not loaded
into preparation and no test performance is calculated during model selection.

## Collect data from new pilots

The collector accepts either its historical frozen experiment job or a completed
notebook 05/06 population of any size. A current population uses the current source
tree, whose source and resource hashes are recorded and checked on reuse and
packaging. Use new collection and publication directories for the new pilots.

```bash
python scripts/generate_tactical_corpus.py collect --perspective tactical \
  --job artifacts/rl_pilot_simplified/combat/NEW_BATCH \
  --work artifacts/trajectory_pilot_episodes_v4 --device cpu --workers 1
python scripts/generate_tactical_corpus.py package --perspective tactical \
  --job artifacts/rl_pilot_simplified/combat/NEW_BATCH \
  --work artifacts/trajectory_pilot_episodes_v4 \
  --output shared/corpora/trajectory-pilots-v4 --device cpu
```

This records B01 observing the learned A01 aircraft. The existing raw corpus format
is retained. Physical tracks from historical corpora can also supply the new
regression targets, but do not measure the newly trained two-skill population.

## Train, report and infer

The config is `configs/training/trajectory_forecasting.json` (CPU by default;
`--device cuda` enables GPU training). The standard skill-forecast command now
routes to this continuous forecasting workflow. Historical classification studies
require `--legacy-skill-labels` and their original datasets/checkpoints.

```bash
python scripts/train_skill_forecaster.py prepare --corpus shared/corpora/trajectory-pilots-v4
python scripts/train_skill_forecaster.py train --architecture gru
python scripts/train_skill_forecaster.py train --architecture transformer
python scripts/train_skill_forecaster.py report
```

The equivalent direct entry point is `scripts/train_trajectory_forecaster.py`.
Preparation writes `artifacts/trajectory_forecasting/data_v1`; model runs use
`artifacts/trajectory_forecasting/runs_v1/{gru,transformer}`. Change `--data` and
`--output` for a new study, or use `--resume` with identical source/config/data.
Validation selects checkpoints using the mean error ratio to constant-state
prediction across all three quantities and horizons. Reports include equal-episode
MAE, sample MAE, RMSE and percentiles, separately in m/s, degrees and metres. The report command also writes a standalone
`report.html` with error and learning-curve plots, plus
constant-state and constant-acceleration baselines in `baseline_metrics.csv`.

```python
from bvr_behavior_prediction.forecasting.flight_state import FlightStatePredictor

predictor = FlightStatePredictor("artifacts/trajectory_forecasting/runs_v1/transformer/best.pt")
future = predictor.predict(raw_b01_observation_frames)
# horizons_s, speed_mps, heading_deg, altitude_m: length-3 arrays
```

Regression datasets and checkpoints record `future-speed-course-heading-altitude-v1`.
Old classification or position/velocity checkpoints are rejected by this predictor.
Saved historical forecasting reports and notebooks retain their original task;
new runs need fresh prepared targets and forecast checkpoints.

## Validation scope

Tests exercise simultaneous three-axis commands, PPO masks, checkpoint rejection,
causal targets, exact horizons, heading wraparound, terminal masking, train-only
scaling, grouped test reservation and both neural architectures through training,
validation, checkpoint reload, inference and resume. A small synthetic forecasting
run verifies the pipeline; it does not establish forecast accuracy on newly trained
pilots. Full pilot-population and forecasting experiments must be run separately.

Validation for this revision: the full suite recorded **937 passed, 17 skipped,
and seven previously confirmed historical notebook failures**. Follow-up checks
also passed for the new CLI, three-axis normalization and physical-unit reports.
A real JSBSim two-skill pilot completed a short PPO training/save/reload run, and
both regressors completed a synthetic preparation/train/validation/report/reload
and resume check. No full-population accuracy experiment was run.
