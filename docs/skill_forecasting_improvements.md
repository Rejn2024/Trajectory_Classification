# Improving catalogue-selection forecasts

This experiment preserves the original results in
`artifacts/skill_forecasting/initial_v1/` and the original prepared data in
`artifacts/skill_forecasting/data_v1/`. New runs are under
`artifacts/skill_forecasting/improvement_v2/` (first internal screen) and
`artifacts/skill_forecasting/improvement_v3/` (targeted follow-up and final comparison).
A checksum inventory of the original
artifacts is checked after the experiment.

The task remains prediction of the exact selected catalogue entries at +1, +2
and +5 seconds. The 33-class vocabulary, forecast origins, missing-label masks,
pilot population and outer scenario partitions are unchanged. Catalogue names
are not assumed to denote distinct physical manoeuvres.

## Validation result

The frozen ensemble improves top-1 accuracy on the original validation scenarios
at every horizon. The comparison uses identical origins and missing-label masks:

| Horizon | Valid origins | Original transformer | Improved ensemble | Gain | Paired 95% interval for gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| +1 s | 30,802 | 58.97% | 65.62% | +6.65 points | +5.60 to +7.84 points |
| +2 s | 30,466 | 58.38% | 65.17% | +6.79 points | +5.69 to +8.04 points |
| +5 s | 29,458 | 55.90% | 63.48% | +7.58 points | +6.39 to +8.95 points |

Intervals resample all 45 validation scenario groups, keeping their ten pilot
episodes and overlapping windows together. They are descriptive development
intervals, not a correction for all model selection.

This improvement also appears beyond overall accuracy:

| Metric | +1 s, old → new | +2 s, old → new | +5 s, old → new |
| --- | ---: | ---: | ---: |
| Top-3 accuracy | 88.40% → 90.10% | 88.22% → 89.82% | 86.51% → 89.25% |
| Macro-F1, all 33 entries | 0.3010 → 0.4120 | 0.2953 → 0.4115 | 0.2739 → 0.3665 |
| Negative log-likelihood (lower is better) | 1.2216 → 1.1155 | 1.2361 → 1.1324 | 1.2903 → 1.1643 |
| Accuracy when the future selection differs from the current one | 38.69% → 47.66% | 39.89% → 50.49% | 40.18% → 55.33% |

All ten pilots improve in top-1 accuracy at all three horizons. This does not mean
every class or subset improves: for example, +5-second unchanged-selection
top-3 accuracy decreases from 91.12% to 90.81%, and its NLL increases from 1.0575
to 1.0813. The evidence concerns new scenarios for the same ten trained pilots;
unseen pilots have not been tested. The reserved test set remains untouched.

Explore the comparison, per-pilot and per-class results in
[notebook 10](../notebooks/10_skill_forecasting_improvements.ipynb) or the
[standalone report](../artifacts/skill_forecasting/improvement_v3/validation/report.html).

## Frozen final model

The four-variant uniform mixture won the internal comparison. It is refitted
with seeds 42, 43 and 44 for each component, giving **12 models** whose class
probabilities are averaged equally:

| Component | History | Width | Residual blocks | Dropout | Label smoothing | Fixed epochs | Fixed temperature |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Temporal MLP | 20 s | 256 | 3 | 0.20 | 0.02 | 46 | 1.4654 |
| Wider temporal MLP | 20 s | 512 | 3 | 0.30 | 0.03 | 32 | 1.4014 |
| More smoothed temporal MLP | 20 s | 256 | 3 | 0.30 | 0.08 | 40 | 1.1691 |
| Longer-history temporal MLP | 60 s | 256 | 3 | 0.20 | 0.02 | 40 | 1.4238 |

Every component uses the same 91 observable/derived features per token. Validity
indicators distinguish episode-start padding. The ensemble is more expensive
than the original single transformer and can use substantially more past data;
this is an explicit change in the input-history budget, not a like-for-like
architecture-only comparison.

The twelve models contain 28,977,060 parameters and occupy about 116 MB of
checkpoint files. On this machine with four CPU threads, 30 warm single-origin
calls had a median of 20.3 ms and a 95th percentile of 22.0 ms, including feature
construction for all members. This is a local timing check, not a deployment
latency guarantee or a comparison with the original transformer's runtime.

Before the final refit, its mean top-1 accuracy on the two internal selection
partitions was 59.09% and 62.87%, versus 51.87% and 51.38% for the control. Across
the two partitions, mean NLL improved from 1.3599 to 1.2581 and catalogue macro-F1
from 0.2192 to 0.3285. The six-model wide/long alternative had almost identical
internal accuracy (60.96% averaged across splits versus 60.98%); the registered
highest-accuracy rule selected the twelve-model version. The small difference
between those two mixtures is not evidence of a decisive architecture advantage.

## Why these changes

In the initial transformer, training loss continued falling while validation
loss deteriorated. More training alone was not a supported way to improve
generalisation. Many errors also confused labels with overlapping control
implementations. We therefore test regularization, explicit observable geometry,
a longer causal history, and a uniform ensemble of independently initialized
models. These are candidate comparisons, not a complete factorial ablation.

Geometry is reconstructed only from B01 observations and their backward
acceleration estimates: target velocity, speed, course, line-of-sight direction,
closure, relative aspect, turn rate, relative energy and observed missile-slot
occupancy. A closest-approach projection uses current relative position and
velocity; it does not read a future trajectory. Some features are deliberately
redundant transformations that make nonlinear relationships easier to learn.

Long-history candidates use the latest five seconds densely and earlier samples
at one-second intervals, up to 20 seconds in the past. Transformer candidates
use 51 recent tokens and 15 older tokens. Temporal MLPs use 11 recent tokens
spaced 0.5 seconds apart and the same 15 older tokens. Each token's acceleration
needs an additional 0.5 seconds of raw history. No additional early origins are
discarded: unavailable older tokens are explicitly masked within the episode.

The MLP processes a chronological vector of observations with residual dense
blocks. The modified transformer combines its final token, a masked mean over
the history, and the current observed features. The auxiliary candidate also
predicts the current selected label with a 0.25-weight supervised loss. That
label is an auxiliary training target only; neither it nor a future label is
passed into the network. Other candidates have no auxiliary target.

## Selection without repeated outer-validation tuning

The protocol and source snapshots are saved before screening any new candidate.
All comparisons used to choose the new architecture occur inside the original
210 training scenario groups. Each internal split is stratified by starting
range and keeps all ten pilot versions together:

| Purpose | Scenario groups | Episodes | Use |
| --- | ---: | ---: | --- |
| Fit | 147 | 1,470 | Gradients and feature normalization |
| Stop | 21 | 210 | Select epoch by NLL or mean accuracy, as registered; fit temperature |
| Select | 42 | 420 | Compare candidate architectures after checkpoint selection |

Internal split seeds are 271828 and 314159. The first split screens nine fixed
candidates: the original encoder, stronger regularization, geometry transformer,
geometry MLP, 20-second transformer, its auxiliary variant, 20-second MLP, wider
20-second transformer and wider 20-second MLP. The latter two increase dropout
and weight decay as well as width. Exact settings are in `protocol.json` and
`candidates()` in the implementation.

The two best eligible candidates and the control are repeated on the second
internal split. Eligibility requires selection NLL no worse than control +0.02
and catalogue macro-F1 no worse than control -0.01. Among eligible candidates,
the highest mean top-1 accuracy across horizons and the two internal splits wins.
This limits reliance on one favourable split, though it does not eliminate
selection uncertainty.

The final candidate, median selected epoch count, seeds 42/43/44 and uniform
probability averaging are written to `frozen_selection.json` before fitting on
all 210 original training groups. These final runs use a fixed epoch count;
the outer validation set does not select epochs, ensemble weights or seeds.
Normalization is refitted on original-training histories only.

The first screen confirmed that the 20-second MLP improved mean internal selection
accuracy from 51.63% to 54.75% across the two internal partitions. It also showed
that top-1 accuracy frequently improved after the log-loss optimum. No new model
was evaluated on the original validation set during that screen.

The targeted follow-up retains the control and NLL-selected MLP, and tests four
accuracy-selected variants: 20-second MLP, a wider version, stronger label
smoothing, and a 60-second MLP. These settings are registered in
`configs/training/skill_forecasting_accuracy_search.json`. The 60-second model
uses 11 recent and 55 older tokens, with the same episode-start masking. The
same two internal partitions, NLL/F1 eligibility guards and three-seed ensemble
rule apply. Prior internal results motivated this follow-up; it is not an
independent replication of the first search.

For accuracy-selected candidates, a single temperature is fitted to stop-group
predictions after the epoch is selected. It adjusts confidence without changing
an individual model's class ranking. The selection groups assess its resulting
NLL independently. Final fits use the median of the two internal temperatures;
outer validation never fits this parameter. Accuracy-based stopping targets the
requested metric, while the NLL and macro-F1 guards reject gains accompanied by
substantial deterioration on the internal selection groups.

Individual accuracy-selected models improved top-1 but failed the NLL guard on
the internal selection groups. A narrow/wide probability average improved both
metrics, motivating the recorded `ensemble_protocol.json` extension. It compares
four fixed uniform mixtures (narrow/wide, those two plus stronger smoothing,
wide/60-second, and all four) against the control and the NLL-selected MLP. No
mixture weights are fitted. The top two eligible mixtures are checked on the
second internal partition before choosing the final mixture. Each selected
component is then refitted with three fixed seeds; the final ensemble gives
equal weight to every component/seed. The initial search, targeted follow-up and
mixture extension all remain recorded, including unsuccessful candidates.

Only this final frozen mixture reaches the original 45-group validation set for
a comparison.
It was already used during the original experiment and is therefore still a
development validation set. The reserved 45-group test set is neither prepared
nor evaluated. No observed validation gain establishes a guarantee of zero
overfitting or generalisation to new pilots.

Training uses AdamW, initial learning rate 0.0003, a fixed 60-epoch cosine schedule
towards 0.00003, gradient clipping at 1, batch size 1,024, and BF16 on CUDA. Inner
training stops after ten epochs without a 0.001 improvement in the registered
stopping metric (lower NLL or higher mean top-1 accuracy). Regularized candidates
use label smoothing from 0.02 to 0.08 without rare-class weighting. Missing future
labels have no loss or gradient.

## Comparison and checks

The new and old predictions are compared on exactly the same validation origins.
Reports include all three horizons, top-1/top-3 accuracy, catalogue/supported
macro-F1, per-class support and precision/recall, per-pilot scores, confusion
matrices and changed/unchanged subsets. No label is merged to increase accuracy.

Paired uncertainty intervals resample the 45 complete scenario groups 5,000
times, preserving all pilots and overlapping windows within a sampled group.
They describe uncertainty in this development comparison, not a final test or
a correction for all historical model selection. Independently reloaded
checkpoints must reproduce every saved validation probability exactly.

Tests cover inner/outer group separation, training-only normalization, strictly
past-only history indexing, padding invariance, offline/streaming feature parity,
rotation-invariant geometry, missing-label gradients, auxiliary-label separation,
ensemble reloads and group-level uncertainty calculations.

The 32 original and improvement tests pass. Independent checkpoint reloads
reproduce every member and ensemble probability exactly. Checksums verify that
the original results and prepared-data manifest are unchanged. The full history
audit checks all 177,492 prepared origins under five history configurations and
finds zero future-frame or cross-episode reads. Every final model's saved
normalization uses exactly the 210 original training groups. Audit details are
saved in `verification.json`, `full_window_audit.json` and
`normalization_audit.json` under `improvement_v3`.

All seven code cells in notebook 10 execute successfully, including the
observation-only predictor. Streaming and prepared features agree exactly at
6-, 20- and 65-second origins for both history lengths on a real validation
episode. The execution log, figures and timing/parity checks are saved under
`improvement_v3/notebook_check/`.

## Running and inspecting the experiment

For the completed follow-up, inspect notebook 10 or run:

```powershell
.venv\Scripts\python.exe -X utf8 scripts/improve_skill_forecaster.py status --output artifacts/skill_forecasting/improvement_v3
```

The follow-up workflow, using a fresh `--output` directory to preserve existing
results, is:

```powershell
.venv\Scripts\python.exe -X utf8 scripts/improve_skill_forecaster.py initialize --output artifacts/skill_forecasting/another_search --design configs/training/skill_forecasting_accuracy_search.json
.venv\Scripts\python.exe -X utf8 scripts/improve_skill_forecaster.py screen --output artifacts/skill_forecasting/another_search
.venv\Scripts\python.exe -X utf8 scripts/ensemble_skill_forecaster.py register --output artifacts/skill_forecasting/another_search
.venv\Scripts\python.exe -X utf8 scripts/ensemble_skill_forecaster.py confirm --output artifacts/skill_forecasting/another_search
.venv\Scripts\python.exe -X utf8 scripts/ensemble_skill_forecaster.py refit --output artifacts/skill_forecasting/another_search
.venv\Scripts\python.exe -X utf8 scripts/ensemble_skill_forecaster.py evaluate --output artifacts/skill_forecasting/another_search
```

The preservation check expects `preserved_initial_hashes.json`, the inventory of
the original run. This experiment's inventory is retained in both experiment
directories. Verification for this completed experiment is:

```powershell
.venv\Scripts\python.exe -X utf8 scripts/ensemble_skill_forecaster.py verify
```

Completed individual fits are reused. Incomplete fits are protected from
overwrite; they must be preserved before restarting in a separate destination.
Source/configuration hashes prevent silently changing an experiment after it is
registered. Use `--output` for a separate experiment. The initial `improvement_v2`
screen retains its own older source snapshot; reproduce its fits with that
snapshot and explicit data/output paths. The original
experiment remains accessible through notebook 09 and its original report.

`ImprovedPredictor` accepts a list of checkpoint paths and returns the uniform
mean of their probability distributions. Supply consecutive live B01
observations in the original 51-feature scale, excluding the reset frame. At
least 56 frames are required; 206 or 606 frames provide the full 20- or 60-second
history and acceleration lookback. Shorter histories mask only the unavailable older tokens.
Pilot identity, private decisions, final outcomes and future observations are
not inference arguments.

Keep the full available history up to that maximum during deployment. Short
histories naturally occur during episode startup; deliberately truncating a late
encounter to only five seconds changes the available information and has not
been separately validated.
