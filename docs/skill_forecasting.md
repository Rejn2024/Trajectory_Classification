# Forecasting trained-pilot skill-catalogue selections

This supervised experiment forecasts A01's **selected catalogue label** at
exactly +1, +2 and +5 seconds, using scripted tactical observer B01's observation
history. It uses `shared/corpora/trained-pilot-episodes-v1/` and the ten frozen
pilots; no simulator or RL training changes are required. The existing
[motion experiment](motion_forecasting.md) and its artifacts are retained.

## Initial validation results

These are development results on the 450 validation episodes, using one training
seed. All scores below are percentages except F1, which is on a 0-1 scale.

| Model | Horizon | Top-1 | Top-3 | Macro-F1, all 33 | Macro-F1, 26 supported |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training frequency | 1 s | 17.84 | 50.03 | 0.0092 | 0.0116 |
| Training frequency | 2 s | 17.99 | 50.41 | 0.0092 | 0.0117 |
| Training frequency | 5 s | 18.46 | 51.48 | 0.0094 | 0.0120 |
| GRU | 1 s | 54.40 | 86.95 | 0.2374 | 0.3013 |
| GRU | 2 s | 54.01 | 86.72 | 0.2309 | 0.2931 |
| GRU | 5 s | 51.56 | 85.04 | 0.2123 | 0.2694 |
| Transformer | 1 s | 58.97 | 88.40 | 0.3010 | 0.3820 |
| Transformer | 2 s | 58.38 | 88.22 | 0.2953 | 0.3748 |
| Transformer | 5 s | 55.90 | 86.51 | 0.2739 | 0.3476 |

The transformer leads the population metrics in this experiment. The GRU stopped
after 22 epochs, selecting epoch 15 (mean validation NLL 1.2912); the transformer
stopped after 27, selecting epoch 20 (NLL 1.2494). The minimum improvement threshold
is applied when selecting checkpoints, so a smaller later change below 0.001 does
not replace the incumbent. This is not a multi-seed architecture comparison.

Per-pilot top-1 accuracy, in percent:

| Pilot | GRU 1 s | GRU 2 s | GRU 5 s | Transformer 1 s | Transformer 2 s | Transformer 5 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 001 | 45.89 | 45.20 | 45.24 | 49.48 | 47.64 | 47.77 |
| 002 | 69.29 | 68.56 | 63.33 | 69.10 | 68.25 | 63.79 |
| 003 | 53.62 | 54.08 | 50.97 | 62.43 | 61.77 | 57.96 |
| 004 | 59.01 | 58.87 | 58.68 | 66.78 | 66.32 | 66.90 |
| 005 | 62.72 | 62.62 | 56.64 | 67.14 | 66.33 | 61.30 |
| 006 | 51.33 | 50.65 | 48.18 | 57.98 | 57.24 | 55.17 |
| 007 | 54.32 | 52.85 | 51.28 | 55.75 | 56.40 | 53.65 |
| 008 | 37.56 | 37.41 | 36.36 | 40.59 | 39.60 | 38.77 |
| 009 | 59.81 | 59.95 | 57.46 | 62.70 | 62.51 | 59.03 |
| 010 | 48.79 | 48.27 | 46.24 | 56.46 | 56.43 | 53.66 |

When the future label differs from the selection at the origin, transformer
top-1 accuracy is 38.69%, 39.89% and 40.18%, versus 33.35%, 35.03% and 36.56%
for the GRU. Those subsets contain 2,264, 4,254 and 8,659 windows respectively;
they differ across horizons, so their scores are not a matched-window comparison.
Full per-class, confusion and unchanged-subset results are in the report.

Both selected checkpoints reproduced every saved validation probability exactly
in fresh GPU processes, which exited successfully. The GRU training process
itself returned exit code 1 after saving its final summary and validation files,
without a Python traceback. This runtime event is recorded in
`gru/runtime_note.json`; the independent verification is in `verification.json`.
All notebook code cells also executed successfully in a headless check.

## Task and information boundary

Selections are logged before execution. At origin 6 seconds, the three labels
are the recorded choices at 7, 8 and 11 seconds. The origin's observation is
available before its contemporaneous selection is executed. Labels are joined by
exact 10 Hz tick after checking `time_s == step * 0.1`; there is no nearest-time
join, interval label, or forward fill. Each missing future selection is masked
independently, including an absent decision at the terminal frame. Origins with
no available horizons are excluded.

Inputs are 51 frames spanning five seconds, with 51 recorded observations plus
three backward acceleration features. All 56 raw frames required by the 0.5-second
acceleration lookback must be live and follow reset. Origins are every second,
beginning at six seconds. Alive flags select valid histories but are not model
inputs. Neither future tracks nor target decisions affect feature values.

Pilot identity, checkpoint identity, final outcomes, target skills and continuous
action parameters are excluded from encoder inputs. The true selection at the
origin is saved only to stratify evaluation into changed/unchanged future labels.
This diagnostic uses privileged information and is not a deployable persistence
baseline or a model input. "Same" compares the two endpoint selections; a label
may change and return between them.

The target vocabulary retains all 33 catalogue entries, in alphabetical order.
Some names share executed guidance: `lag_pursuit`, `turn_cold`, `extend`,
`recommit` and `pursue_target` can emit identical zero-change commands. Left
beaming and left cranking both set a -90-degree heading change. The descriptions
therefore do not establish physically distinct manoeuvres. See
[`skill_manager.py`](../bvr_sim_source/bvr_sim/agents/skill_manager.py).
No existing records are relabelled or merged.

## Splits, alignment audit and class support

The scenario assignment must match the existing motion experiment's saved
`scenario_splits.json` exactly: seed 1729, stratified by requested range, with all
ten versions of each scenario kept together. Scaling and optional class weights
are fitted on training windows only.

| Partition | Scenario groups | Episodes | Origins | Valid +1 / +2 / +5 s labels |
| --- | ---: | ---: | ---: | --- |
| Training | 210 | 2,100 | 146,690 | 146,690 / 145,163 / 140,516 |
| Validation | 45 | 450 | 30,802 | 30,802 / 30,466 / 29,458 |
| Reserved test | 45 | 450 | Not prepared | Not evaluated |

Preparation skips test row groups entirely. Checksums cover complete source
shards, but test rows do not contribute to label audits, scaling or performance.
The smaller origin count than motion forecasting comes from requiring actual
future decision records: a terminal physical frame need not have a decision.

The 217,141 training/validation decisions contain 30 labels, with exact one-second
cadence and no duplicate timestamps. After history and future-label alignment,
29 labels have training support and 26 have validation support at each horizon.
`accelerate_to_speed`, `commit`, `dive_defense` and `drag_missile` have no aligned
development labels. At +1 second, `climb_to_altitude` has only three training
examples and one validation example; `lock_target` has eight training examples
and no validation examples. These classes cannot support reliable estimates.

`decision_support.csv` reports raw development decision counts.
`label_support.csv` reports the actual supervised-label counts by split, horizon
and pilot, including zero counts. Early selections removed by the history
requirement must not be mistaken for training examples.

## Models, objective and reporting

Both available encoders reuse the motion architecture: two GRU layers or two
transformer blocks of width 128, with four attention heads per transformer block.
A categorical projection provides three independent 33-way output distributions.
No cross-horizon consistency constraint is imposed. Dropout is 0.1.

The initial objective is unweighted cross-entropy, averaged over available labels
within each horizon and then across represented horizons. An optional
`sqrt_inverse` weighting mode uses training support only, normalizes by the
training-frequency weighted mean, and caps weights at 5; it is disabled in the
initial configuration. Unsupported classes remain in the output vocabulary.

Training uses AdamW, learning rate 0.0003, weight decay 0.01, gradient clipping 1,
batch size 1,024, BF16 CUDA autocasting and seed 42. The budget is 30 epochs with
early stopping after seven non-improving epochs. Checkpoints are selected by
mean **unweighted validation negative log-likelihood** across horizons, with
minimum improvement 0.001. Validation never supplies class weights.

The frequency baseline emits one training-fitted distribution per horizon for
every validation window. Its top-1 accuracies are 17.84%, 17.99% and 18.46%; top-3
accuracies are 50.03%, 50.41% and 51.48%. It uses no target identity or private
current skill. Ties are resolved in alphabetical catalogue order.

Reports include top-1/top-3 accuracy, per-class precision/recall/F1 and counts,
confusion matrices, per-pilot results and changed/unchanged subsets at each
horizon. `macro_f1_catalogue` averages all 33 labels, assigning zero to undefined
F1; `macro_f1_supported` averages classes with truth support in the evaluated
subset. Both are reported so absent classes and changing denominators remain
visible. Summary metrics weight valid windows equally, not episodes equally.
NLL clips probabilities at 1e-12 for numerical stability.

Overlapping windows are correlated. This split measures new initial scenarios
for the known pilot population, not generalisation to unseen pilots. Probability
outputs have not been assessed for calibration. The observations contain
idealised simulator tracks, not noisy or intermittent radar measurements.

## Commands and artifacts

Use [notebook 09](../notebooks/09_skill_forecasting.ipynb) to inspect saved results,
counts, training curves and confusion matrices. Its default Run All does not
train or evaluate the test set. The portable report is
`artifacts/skill_forecasting/initial_v1/report.html`.

```powershell
.venv\Scripts\python.exe -X utf8 scripts/train_skill_forecaster.py prepare
.venv\Scripts\python.exe -X utf8 scripts/train_skill_forecaster.py baseline
.venv\Scripts\python.exe -X utf8 scripts/train_skill_forecaster.py train --architecture gru
.venv\Scripts\python.exe -X utf8 scripts/train_skill_forecaster.py train --architecture transformer
.venv\Scripts\python.exe -X utf8 scripts/train_skill_forecaster.py verify --architecture gru
.venv\Scripts\python.exe -X utf8 scripts/train_skill_forecaster.py verify --architecture transformer
.venv\Scripts\python.exe -X utf8 scripts/train_skill_forecaster.py report
.venv\Scripts\python.exe -X utf8 scripts/train_skill_forecaster.py status
```

The completed prepared dataset is reused; do not run `prepare` over it. Use
`--data` for another preparation and `--output` for another experiment.
`--config` selects a different configuration. Preparation and new training
refuse nonempty destinations. `--resume` requires the original configuration,
data hashes and source; frozen source entry points are retained in each run's
`source/scripts/` directory. Pass explicit original data/output/config paths when
running a source snapshot. Hardware differences may affect numerical results.

Prepared artifacts retain the class mapping, split assignments, source checksums,
training-fitted scaling, exact preparation source and alignment audit. Each run
contains `best.pt`, `last.pt`, `run.json`, `history.csv`, `progress.json`,
`summary.json`, validation probabilities and metrics, and source snapshots.
Verification reloads the best checkpoint in a fresh process and compares every
validation probability exactly. Generated artifacts stay under ignored
`artifacts/skill_forecasting/`; nothing is published automatically.

```python
from bvr_behavior_prediction.skill_forecasting.inference import SkillPredictor

predictor = SkillPredictor("artifacts/skill_forecasting/initial_v1/gru/best.pt")
# At least 56 consecutive 10 Hz B01 frames, oldest first, original 51-feature scale.
forecast = predictor.predict(observed_history)
print(forecast["horizons_s"])      # [1, 2, 5]
print(forecast["probabilities"])   # shape (3, 33), columns match forecast["classes"]
print(forecast["top3_skills"])     # shape (3, 3)
```

Tests cover exact time alignment, missing/terminal decisions, future-input and
private-label independence, the acceleration lookback, invalid timestamps,
masked gradients, class support and frequency priors, metric denominators,
unread test partitions, training-only scaling, inference parity, checkpoint
reload and resume guards. The motion-forecasting tests are retained.
