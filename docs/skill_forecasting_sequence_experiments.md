# Sequence architecture experiments (v4)

This experiment attempts to improve on the **v3 twelve-member MLP ensemble**.
The original Transformer remains a historical comparison, not the acceptance
baseline. Earlier checkpoints, predictions, scores and presentation remain intact.

## Registered comparison

The dataset, 33-class ordering, ten frozen RL pilots, tactical B01 viewpoint,
scenario split, forecast origins, target masks and exact +1/+2/+5-second targets
are unchanged. Inputs are the same 91 observation-derived features and the same
26-token (20-second) or 66-token (60-second) sparse histories. True current skill,
pilot identity, future observations and final flight outcomes are not inputs.

The primary ranking metric is mean top-1 accuracy across the three horizons,
then averaged equally across the two existing internal partitions. The practical
target is at least **one percentage point** above v3; a smaller positive result
must be described as such. Each promoted recipe must improve mean accuracy in
both partitions, with mean NLL no more than 0.02 worse, catalogue macro-F1 no more
than 0.01 worse, and no individual horizon accuracy more than 0.003 worse than
the corresponding current-ensemble reference. These are eligibility tolerances,
not permission to hide a regression in reporting.

## Bounded candidates

| Candidate family | Histories | Change being tested |
|---|---|---|
| Multiscale residual MLP | 20, 60 seconds | Separate recent, older and current-state branches before a residual trunk |
| Temporal convolutional network | 20, 60 seconds | Shared causal temporal filters, dilations, masked pooling and current-state bypass |
| GRU | 20, 60 seconds | Recurrent state over valid observations, masked pooling and current-state bypass |
| Wider long-history MLP | 60 seconds | Existing architecture with width 512 instead of 256 |
| Deeper MLP | 20 seconds | Existing width-512 architecture with five rather than three residual blocks |

Exact settings are in
[`skill_forecasting_sequence_search.json`](../configs/training/skill_forecasting_sequence_search.json).
These are predefined model recipes with different widths and regularization,
not an experiment that isolates architecture while holding parameter count fixed.
No extra candidates are added after observing this screen. TCN and GRU receive
deterministic relative-time and gap coordinates to identify the transition from
older one-second sampling to recent half-second sampling; these convey the
already-known history layout, not another sensor stream. The multiscale MLP
separates those regions explicitly. All three retain current-state information.

The TCN uses left padding and per-token normalization, and removes invalid
activations after every block. The GRU compacts valid observations before packing
them, preventing padded history from updating its state. Their design draws on
the sequence-model families in [Bai et al. (2018)](https://arxiv.org/abs/1803.01271)
and [Cho et al. (2014)](https://aclanthology.org/D14-1179/); neither paper predicts
which architecture will work best for this pilot dataset.

## Selection and preservation

1. Screen all eight candidates with seed 42 on internal partition 0, against the
   explicitly labelled four-member seed-42 v3 reference.
2. Shortlist at most two architectures using a predefined probability blend:
   `(1-w) * baseline + w * candidate`, with `w` in `{0.25, 0.5, 0.75, 1}`.
3. Complete a twelve-member internal baseline: all four incumbent variants and
   seeds 42, 43 and 44, separately in each partition. Reuse only historical
   internal seed-42 outputs, never incumbent final weights trained on these
   selection groups. Fit shortlisted candidates with the same three seeds in
   both partitions. Confirm the predefined blends, including a uniform mixture
   of both shortlisted architectures. Weight zero retains the incumbent.
4. Fit normalization on fit groups only. Choose epochs and temperature on that
   partition's stop groups. Rank recipes on its selection groups. Internal
   scores retain their own stop-fitted temperatures; they do not use temperatures
   pooled across partitions.
5. Freeze the selected recipe, weight, seeds and median epoch/temperature values
   across its six internal runs. Refit only new components on all original
   training groups. Existing v3 final models and probabilities stay fixed.
6. Evaluate one frozen recipe on the existing outer validation origins, with a
   paired bootstrap over entire shared scenario groups. Do not tune after this
   comparison. If no recipe passes internal confirmation, retain v3 and skip a
   new outer-validation evaluation. The reserved test is never prepared or scored.

A blended ensemble may have 15 or 18 members versus the incumbent's 12. This is
an explicitly reported increase in model budget. Standalone candidate scores
and blend scores separate architectural behavior from the effect of combining
more models; they do not establish an architecture-only causal improvement.

All previous dataset and experiment files are recorded in a preservation hash
manifest. Frozen source, partition, baseline and checkpoint hashes are checked
before reuse. Unsuccessful runs are retained alongside successful runs.

The two internal partitions have already guided earlier development and share
some scenario groups. Their agreement is a useful stability check, not two
independent tests. The outer validation has also been inspected previously.
Results remain development evidence for new scenarios involving the same ten
trained pilots, not an untouched-test or unseen-pilot generalization claim.

## Reproduction

Run these commands from the repository root with the prepared `data_v1` dataset
and preserved v3 artifacts available. Use a fresh `--output` directory for a new
experiment. Initialization freezes source and data hashes; changing code later
requires a new experiment or execution from the saved source snapshot.

```powershell
.venv\Scripts\python.exe scripts/sequence_skill_forecaster.py initialize
.venv\Scripts\python.exe scripts/sequence_skill_forecaster.py screen
.venv\Scripts\python.exe scripts/sequence_skill_forecaster.py confirm
.venv\Scripts\python.exe scripts/sequence_skill_forecaster.py refit
.venv\Scripts\python.exe scripts/sequence_skill_forecaster.py evaluate
```

The last two commands are no-ops when confirmation retains the incumbent.
Completed fits can be reused after integrity checks; incomplete run directories
are preserved rather than overwritten. Outer evaluation refuses a second start.
All model artifacts remain under `artifacts/skill_forecasting/improvement_v4/`.
The lightweight published report contains metrics and provenance, not the large
local checkpoints or observation arrays.

The CUDA GRU implementation uses stacked single-layer recurrent modules and
explicit dropout between packed activations. This preserves inter-layer dropout
while avoiding a cuDNN shutdown failure reproduced with stock multi-layer GRU
dropout on this Windows environment. Synthetic CUDA BF16 forward/backward checks
and process completion passed before dataset training began.

## Untested follow-up identified during the code review

The existing features provide B01-relative missile positions and velocities, but
do not explicitly calculate each missile's geometry relative to the observed
A01 aircraft. That relationship can be reconstructed from the same current
observation, without adding information. For missile slot `j`, beginning at
`k = 19 + 7*j`, the relative position in metres is
`[50000, 50000, 10000] * (obs[k:k+3] - obs[9:12])`; the relative velocity in
metres per second is `1200 * obs[k+3:k+6] - 600 * obs[12:15]`.

The unequal velocity scales come from
[`CompactObsSpace`](../bvr_sim_source/bvr_sim/src_py/observation_space.py).
Only present slots (`obs[k+6] > 0`) should contribute. Unmasked zero padding would
create fictitious missiles, and surviving slots can change index, so their
identities must not be assumed stable across time. Range, closing speed and
constant-velocity closest approach are plausible derived features for a future
separately registered study. **They were not added to v4, and no performance gain
has been measured for them.**

## Results

The internal search froze a 50/50 probability blend of the unchanged v3 ensemble
and three 60-second TCNs (seeds 42, 43 and 44). Each new TCN has width 128, six
residual convolution blocks with dilations 1/2/4/8/16/32, dropout 0.2 and 917,657
parameters. The input has 66 timestamps; known relative-time/gap coordinates and
validity accompany the 91 observation features. The readout concatenates the last
hidden state, masked mean hidden state and current observation. Three 33-class
heads predict the unchanged horizons.

The selected new models use 42 fixed refit epochs and temperature
1.4599881045960732, both determined from the six internal fits. Their averaged
probability receives half the final weight, with the existing v3 probability
receiving the other half. This is **not** a uniform average of all 15 models.
The total parameter count is 31,730,031, compared with v3's 28,977,060; parameter
count alone is not an inference-latency measurement.

Internal mean accuracy improved from 59.4632% to 60.5334% on partition 0 and
62.9631% to 63.6433% on partition 1. NLL and catalogue macro-F1 improved on both.
The mean gain, 0.8752 percentage points, passed the promotion guards but did not
reach the predeclared one-point practical target. Larger MLPs, the GRUs and the
two-architecture mixture did not win the registered comparison.

The single frozen outer-validation comparison improved every primary horizon:

| Horizon | Retained v3 | Frozen v4 | Gain | Paired scenario 95% interval |
|---|---:|---:|---:|---:|
| +1 second | 65.62% | 67.09% | +1.48 pp | [+0.92, +2.11] pp |
| +2 seconds | 65.17% | 66.62% | +1.45 pp | [+0.89, +2.09] pp |
| +5 seconds | 63.48% | 64.86% | +1.38 pp | [+0.77, +2.11] pp |

Mean accuracy rose from **64.76% to 66.19%**, a **1.44-point gain** with a joint
scenario-bootstrap interval of **[+0.90, +2.08] points**. The point estimate meets
the one-point practical target. The interval uses the same 5,000 resamples of all
45 shared scenario groups across horizons, averaging the three gains within each
resample. It is conditional on these frozen models and this development set;
it does not adjust for the preceding model search. It does not establish unseen
pilot performance or replace the still-reserved test.

Mean NLL improved from 1.1374 to 1.0929 and catalogue macro-F1 from 0.3967 to
0.4024. Brier score improved at all three horizons. **ECE worsened**: 2.94/2.93/3.42
points became 4.16/3.63/4.36 points at +1/+2/+5 seconds. There was no recalibration
or further model tuning after this validation comparison.

Accuracy point estimates improved in 25 of 30 pilot/horizon combinations and
for nine of ten pilots at +5 seconds. Pilot 008 regressed at all three horizons
by 1.43/1.35/0.73 points; its paired intervals include zero. Some skill F1 scores
also fell, including `decelerate_to_speed` at +1/+2 seconds. The full pilot and
skill tables preserve these regressions and their support counts.

There were **37 new fits**: 34 internal development/reference fits and three final
refits, plus eight reused historical internal runs. Summed logged fit time was
57.09 minutes (49.50 internal, 7.59 final); setup/loading are excluded and some
confirmation fits shared the GPU, so this is neither wall time nor an isolated
throughput benchmark. All 490 pre-existing dataset/experiment files passed the
preservation audit. The saved new ensemble exactly reproduces its registered
weighted combination of the preserved baseline and three new prediction arrays.

The [five-page PDF and complete portable evidence](experiments/skill_forecasting_v4/README.md)
include candidate evolution, pilot and skill comparisons, temporal breakdowns,
training histories, exact protocol bytes and the frozen source archive. The
earlier 41-slide catalogue presentation remains available unchanged.
