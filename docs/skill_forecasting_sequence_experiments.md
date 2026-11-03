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

## Results

Pending the registered experiment. The retained v3 validation accuracies are
65.6159%, 65.1710% and 63.4802% at +1, +2 and +5 seconds respectively. Results,
timings, per-pilot and per-skill comparisons will be added after selection and
any frozen validation evaluation are complete.
