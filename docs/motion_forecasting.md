# Trained pilot position and velocity forecasting

The first forecasting experiment predicts a trained pilot's position and velocity
1, 2 and 5 seconds ahead from the scripted tactical aircraft's observations.
It is supervised learning on recorded encounters. The ten pilot policies and the
simulator remain fixed.

## Initial results

These are **development validation** results on 450 episodes from 45 held-out
starting scenario groups. The final test partition remains reserved. Values are
mean 3D Euclidean error within each episode, averaged equally across episodes.

| Model | Position 1 s (m) | Position 2 s (m) | Position 5 s (m) | Velocity 1 s (m/s) | Velocity 2 s (m/s) | Velocity 5 s (m/s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Constant velocity | 9.146 | 36.177 | 221.456 | 17.914 | 35.706 | 86.968 |
| Constant acceleration | 0.867 | 5.227 | 63.943 | 2.181 | 7.026 | 35.642 |
| GRU | 0.440 | 1.927 | 20.655 | 0.906 | 2.402 | 12.113 |
| Transformer | 0.484 | 2.022 | 19.092 | 0.976 | 2.369 | 10.937 |

The GRU has lower short-horizon position error. The transformer has lower
five-second position and velocity error. Both improve on constant acceleration at
every reported horizon and quantity. The transformer has a slightly better
combined validation selection score; this is one split and one training seed,
not evidence that transformers are universally superior.

The transformer ran 29 epochs with early stopping, selecting epoch 22. The GRU
ran 30 epochs, selecting epoch 28. The initial one-epoch smoke run is stored
separately and is not included in this comparison.

The local, self-contained report is
`artifacts/motion_forecasting/initial_v1/report.html`. It contains comparison
tables, training curves and per-pilot metrics. The accompanying CSV files also
include window-weighted means, RMSE, medians and 90th-percentile errors.

## Data and information boundary

Source: [trained-pilot-episodes-v1](../shared/corpora/trained-pilot-episodes-v1/README.md).
The observer is scripted tactical **B01**; the prediction target is trained **A01**.
Both are F-16s. All ten learned policies contribute to one population model per
architecture. No pilot identifier is supplied to either model.

The encoder receives 51 observation frames at 10 Hz, spanning five seconds.
Each token has the original 51 observation values and three causal acceleration
features. Acceleration is calculated from the change in observed target velocity
over the preceding 0.5 seconds. Consequently the complete raw input dependency is
5.5 seconds: **56 raw frames produce 51 model tokens**. This extra lookback is
needed to calculate acceleration for the oldest token.

Absolute target velocity is reconstructed from observer velocity plus relative
target velocity. Target position is reconstructed in the same way. These inputs
come exclusively from the recorded observation interface. The physical track
table supplies future supervision, never model input. Internal target skills,
continuous action parameters, pilot identity and final outcomes are excluded.

Windows stay within one episode. The reset frame is omitted and a complete live
observation history is required. Forecast origins are sampled every second,
starting at six seconds. Each horizon is independently masked if the future
sample is unavailable or the target is dead. For example, a late window may have
a valid one-second target but no five-second target. No values are interpolated
across episode endings or aircraft destruction.

Coordinates are world North/West/Up; positions are in metres and velocities in
metres per second. They are not latitude/longitude or aircraft-body coordinates.
The source observations retain idealised exact simulator tracks, with no radar
noise, occlusion or track loss.

## Partitions and scaling

| Partition | Scenario groups | Episodes | Forecast origins | Valid 1 s / 2 s / 5 s targets |
| --- | ---: | ---: | ---: | --- |
| Training | 210 | 2,100 | 147,718 | 147,718 / 146,191 / 141,544 |
| Validation | 45 | 450 | 31,029 | 31,029 / 30,693 / 29,685 |
| Reserved test | 45 | 450 | 32,457 | 32,457 / 32,099 / 31,025 |

Split seed 1729 deterministically assigns 70% / 15% / 15% of starting scenario
groups, stratified by requested initial range. All ten pilots' encounters from
one scenario group stay together. A random split of individual windows or
individual pilot episodes would expose matching encounters across partitions.

Feature means and standard deviations are fitted using the union of frames used
by training histories. Target residual scales are fitted using valid training
targets only. Validation and test values never contribute to those fits. The
saved `scenario_splits.json`, `split_assignments.csv` and `scaling.json` specify
the partition and transformation exactly.

The test partition is prepared but not evaluated. Validation selects checkpoints
and supports development comparisons. Final test evaluation should occur after
architecture and preprocessing choices are frozen. These partitions assess new
starting scenarios for the known ten-pilot population; unseen-pilot evaluation
requires a different holdout design.

## Models and objective

Both models forecast corrections to the constant-velocity baseline:

```text
baseline position(t+h) = observed position(t) + h * observed velocity(t)
baseline velocity(t+h) = observed velocity(t)
predicted state(t+h)   = baseline state(t+h) + learned residual(t+h)
```

The network outputs 18 numbers: three position corrections and three velocity
corrections for each of the three horizons. Its final layer starts at zero, so
the initial prediction is exactly constant velocity. The stronger constant-
acceleration comparison uses the same backward acceleration estimate available
to the neural models.

| Architecture | Encoder | Width | Attention heads per block | Parameters |
| --- | --- | ---: | ---: | ---: |
| Transformer | Two transformer blocks, learned positional embeddings, last-token summary | 128 | 4 | 429,202 |
| GRU | Two recurrent layers, final hidden state | 128 | Not applicable | 224,274 |

Both use a 54-to-128 input projection, layer normalisation, a small regression
head and dropout 0.1. The transformer attends within the observed history only;
there are no future tokens. Bidirectional attention within that past-only
window does not expose future simulation states.

The loss is Huber loss on scaled residuals. One training-fitted RMS scale is used
for position and one for velocity at each horizon, with a lower bound of one in
the corresponding physical units. The three axes have equal weight. Valid
examples are averaged within each horizon before averaging across horizons;
missing future labels do not contribute. There is no critic, PPO ratio or reward
function in this supervised objective.

Training uses AdamW, initial learning rate 0.0003, weight decay 0.01, gradient
clipping at 1.0, batch size 1,024 and BF16 GPU autocasting. The budget is 30 epochs,
with learning-rate reduction on a plateau and early stopping after seven epochs
without a validation-score improvement of at least 0.001. Seed 42 is shared by
both initial runs. Histories are gathered from GPU-resident frame arrays rather
than materialised repeatedly or generated by live simulator workers.

Checkpoint selection averages six validation error ratios to constant velocity:
position and velocity at each of the three horizons. Errors first average within
episodes, then across episodes. A score of 1 equals the baseline; smaller is
better. The initial zero-output model is retained if no trained epoch improves
on that baseline.

## Files and commands

- `bvr_behavior_prediction/forecasting/data.py`: grouped partitions, causal
  features, horizon masks, training-only scaling and compressed-source checksums.
- `bvr_behavior_prediction/forecasting/models.py`: transformer/GRU regression
  models and masked Huber loss.
- `bvr_behavior_prediction/forecasting/train.py`: training, validation selection,
  source snapshots, progress and checkpoint resume.
- `bvr_behavior_prediction/forecasting/evaluation.py`: physical-unit metrics,
  population/per-pilot comparisons and portable plots/report.
- `bvr_behavior_prediction/forecasting/inference.py`: checkpoint loading and
  position/velocity predictions from a raw observation history.
- `configs/training/motion_forecasting.json`: recorded initial experiment settings.
- `scripts/train_motion_forecaster.py`: command-line entry point.

Inspect the completed results without training again:

```powershell
.venv\Scripts\python.exe -X utf8 scripts/train_motion_forecaster.py status
.venv\Scripts\python.exe -X utf8 scripts/train_motion_forecaster.py report
```

The initial prepared data is in `artifacts/motion_forecasting/data_v1/`. The two
model runs are in `artifacts/motion_forecasting/initial_v1/transformer/` and
`artifacts/motion_forecasting/initial_v1/gru/`. Each contains `best.pt`, `last.pt`,
`run.json`, `history.csv`, `progress.json`, `summary.json`, validation metrics and
a frozen source snapshot. These generated files are local ignored artifacts.

To create another experiment while retaining these runs:

```powershell
.venv\Scripts\python.exe -X utf8 scripts/train_motion_forecaster.py train --architecture transformer --output artifacts/motion_forecasting/another_run
.venv\Scripts\python.exe -X utf8 scripts/train_motion_forecaster.py train --architecture gru --output artifacts/motion_forecasting/another_run
.venv\Scripts\python.exe -X utf8 scripts/train_motion_forecaster.py report --output artifacts/motion_forecasting/another_run
```

Use `--resume` with the same settings to resume from a saved epoch. If source
files have changed, run the entry point in the run's `source/scripts/` directory
with explicit original `--data`, `--output` and `--config` paths. Source and
prepared-data hashes prevent silently resuming a different experiment. The
`prepare` command refuses to overwrite a completed preprocessing directory.

For inference, supply consecutive 10 Hz frames ordered oldest to newest:

```python
from bvr_behavior_prediction.forecasting.inference import MotionPredictor

predictor = MotionPredictor(
    "artifacts/motion_forecasting/initial_v1/transformer/best.pt", device="cpu"
)
# observed_history has shape (at least 56, 51), using the original observation scales.
forecast = predictor.predict(observed_history)
print(forecast["horizons_s"])          # [1, 2, 5]
print(forecast["position_nwu_m"])      # shape (3, 3)
print(forecast["velocity_nwu_mps"])    # shape (3, 3)
```

## Verification and limitations

Tests cover deterministic grouped splits, exact horizon alignment, future-input
independence, terminal masks, training-only scaling, model output dimensions,
masked gradients, episode-weighted metrics and streaming/offline feature parity.
Both selected checkpoints were reloaded in a fresh GPU process and reproduced
every saved validation prediction exactly; their checksums also matched.

The GRU process returned a Windows native exit error after saving its completed
30-epoch results. The portable report was generated separately. The independent
checkpoint reload and full validation reproduction completed successfully; no
training data or model outputs had to be regenerated. This runtime event is
recorded separately from model accuracy.

These are point forecasts, not calibrated uncertainty estimates. They do not
classify named manoeuvres or impose cross-horizon physical consistency. Nearby
windows are correlated, so the window count is not a count of independent
experiments. Generalisation to other aircraft, unseen policies, realistic noisy
sensors and different engagement distributions remains untested.
