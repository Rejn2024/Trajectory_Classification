# Skill catalogue forecasting: study guide and experiment review

Review date: **3 November 2026**. This guide accompanies the **41-slide PDF**.
Slides 1–26 retain their numbering and results; slide 19 clarifies the internal
ensemble-size comparison. Slides 27–41 add architecture, training cost, experiment
evolution and descriptive pilot/skill/time analyses.

This review inspected source code, completed training metadata and fixed saved
validation predictions. It fitted no models, changed no checkpoints or
calibration, and performed no reserved-test evaluation. Full temporal decisions
were decoded from **450 validation row groups only**, with zero test groups read.

## A useful reading sequence

1. Slides 2, 26–29: define the forecasting task and follow one input through the network.
2. Slides 23–25, 30–31: learn what the losses mean, what training cost, and why the experiments changed.
3. Slides 5–17, 32–35: inspect headline metrics, each pilot, each skill and calibration.
4. Slides 36–39: separate forecast horizon, elapsed encounter time, label persistence and physical behaviour.
5. Slides 40–41: decide how to report the claim and what to test next.

## 1. What is trained, and what is predicted?

There are two learning problems. PPO trains ten independent pilot policies to
select actions using different reward profiles. The pilot actor–critic is part
of the aircraft controller. A complete 150-epoch × 75-flight × ten-pilot stage
uses **112,500 training flights**, excluding its validation/evaluation flights
and other experiment stages. Pilots change during that training.

After selecting and freezing their checkpoints, a separate recorder generated
**300 new starting scenarios × ten pilots = 3,000 encounters**. B01 is the scripted
tactical observer; A01 is the trained target pilot. The forecaster is a separate
supervised network learning from B01's observations to predict the exact selected
catalogue entry at +1, +2 and +5 seconds. It has no PPO critic and does not update
the pilots. A high forecast score describes predictability, not combat competence.

The outer split is 210/45/45 scenario groups, or 2,100/450/450 encounters. All ten
pilot versions of a scenario stay together. The 146,690 training and 30,802
validation origins are overlapping windows, not independent encounters. The
reserved test targets remain unprepared. Every pilot appears in training and
validation: the claim is **new scenarios for known pilots**, not unseen policies.

## 2. From observations to an MLP prediction

The original GRU/transformer use 54 features per timestamp: 51 recorded observation
values and three backward target-acceleration estimates. The improved models add
37 causal geometry features, for 91 values. These include aircraft position and
velocity, relative geometry, missile information, energy proxies, speed/course,
closure/aspect, turn-rate estimates and constant-velocity closest approach.
Closest approach is computed from current position/velocity, never a future track.

The 20-second model samples 26 timestamps; the 60-second model samples 66. Older
observations are one second apart; the latest five seconds are half a second apart.
The acceleration estimate needs a further half-second of raw lookback. A flag per
timestamp identifies missing early history. Features are normalised using fitting
data only; invalid timestamps are zeroed and flagged.

For N examples, the organised input is `[N, timestamps, 91]`. Appending flags and
flattening gives `[N, 2392]` or `[N, 6072]`: an ordinary 2D feature table. The column
position identifies the lag, so `speed(t-2)` and `speed(t)` are distinct columns.
Future skill labels belong in y, not X. Pilot ID, checkpoint ID, private skill
choices, private continuous action parameters and final outcomes are excluded.

For the standard model, the exact computation is:

```text
2,392 input values
  → Linear(2392, 256) → GELU
  → three residual blocks
  → LayerNorm(256) → Linear(256, 99)
  → reshape to 3 horizons × 33 logits
  → divide logits by the fixed temperature
  → softmax separately for each horizon
```

Each residual block computes:

```text
output = x + [LayerNorm → Linear(256,512) → GELU → Dropout
              → Linear(512,256) → Dropout](x)
```

Linear layers learn weighted combinations. GELU supplies nonlinearity. LayerNorm
controls activation scale. Dropout disables sampled activations during training,
and the residual addition carries the original representation around the block.
The wider variant uses width 512 and expands to 1,024 inside each block.
**Three residual blocks means eight linear layers in the whole network**, including
input and output projections. There is no recurrent state or attention in these MLPs.

Each member emits all three horizons; there is no one-member-per-pilot arrangement
and no explicit constraint enforcing consistency between the three future labels.
The twelve members' calibrated probabilities are averaged uniformly: each member
has weight 1/12, each four-variant family 1/4. Nine members use 20 seconds and three
use 60 seconds. Seeds 42/43/44 change initialization and training order.


| Variant | History s | Input columns | Width | Blocks | Parameters/member | Epochs | Temperature |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20 s / standard | 20 | 2392 | 256 | 3 | 1428835 | 46 | 1.4654 |
| 20 s / wider | 20 | 2392 | 512 | 3 | 4430435 | 32 | 1.4014 |
| 20 s / smoothing | 20 | 2392 | 256 | 3 | 1428835 | 40 | 1.1691 |
| 60 s / standard | 60 | 6072 | 256 | 3 | 2370915 | 40 | 1.4238 |


The ensemble has **28,977,060 parameters**, versus 439,651 for the original
transformer and 234,723 for the original GRU. Input budget and compute both changed;
the result does not isolate an architecture-only improvement. The existing local
four-thread CPU timing was 20.3 ms median / 22.0 ms p95 over 30 warm single-origin
calls, including features. This is not a deployment guarantee or a matched runtime
comparison with the original networks.

## 3. Training, loss and timing

Training uses AdamW, mini-batches of 1,024 origins, learning rate 0.0003,
gradient-norm clipping at 1.0 and CUDA BF16 autocasting. Improved runs use a cosine
schedule toward 0.00003 over 60 epochs, stopping at their saved epoch budgets. Label
smoothing is 0.02/0.03/0.08/0.02; dropout 0.20/0.30/0.30/0.20 and weight decay
0.03/0.05/0.05/0.03 across the four variants. All are mechanisms intended to control
fitting, not guarantees against overfitting.

With a hard correct label, cross-entropy / NLL is `-ln(p_correct)`: p=0.9 gives
0.105, p=0.5 gives 0.693, and p=0.1 gives 2.303. Zero is perfect; the values are
not percentages. Training loss is recorded during weight updates with dropout;
validation NLL is measured using fixed end-of-epoch weights and no dropout.
Improved training also uses smoothed targets, while NLL uses hard labels, so the
absolute train/validation gap is not a pure generalisation-gap measurement.

The loss averages valid examples within a horizon, then the represented horizons.
Missing future decisions are masked, never carried forward. Accuracy cares about
the highest-probability label; NLL cares about probability assigned to truth.
Accuracy can rise while NLL worsens—for example, if remaining mistakes receive
increasingly confident predictions. The accuracy/NLL divergence was observed
internally and motivated calibrated probability mixtures; the precise error-level
cause was not decomposed in this review. Positive scalar temperature scaling preserves a single
model's class order, but calibrating members before averaging can change the
ensemble's final order. [Guo et al., 2017](https://proceedings.mlr.press/v70/guo17a.html)

Original checkpoints were selected at GRU epoch 15 and transformer epoch 20,
using validation NLL and a minimum-improvement threshold. Improved candidates
used internal stop groups; final refits used fixed epoch budgets. Per-pilot
forecast losses were not logged each epoch, so no such curves were reconstructed.

The following totals count each completed fit once, excluding copied validation
summaries. They are sums of recorded fit-stage elapsed timers:


| stage | runs | epochs_sum | elapsed_seconds_sum | minutes_sum |
| --- | --- | --- | --- | --- |
| initial_v1 | 2 | 49 | 89.737 | 1.496 |
| improvement_v2 | 11 | 217 | 352.326 | 5.872 |
| improvement_v3 | 11 | 466 | 533.309 | 8.888 |
| final_refits | 12 | 474 | 925.0 | 15.417 |


Total: **31.67 minutes across 36 runs**.



Final refits account for 15.42 minutes. This is not complete project wall time.
Improved timers exclude loading, feature construction and model/optimizer setup;
they include the training loop, internal stopping evaluation, checkpoint writes
and temperature fitting, and stop before selection-set prediction. Original
timers additionally include final validation/report work. These boundaries differ.
Simulation, dataset preparation, broader evaluation, audits and presentation work
add separate costs. RTX 4090/CUDA 13.0 is recorded for the original/v3 experiments;
v2 has no separate environment file. See training_run_inventory.csv for all runs.

## 4. How the experiments evolved

1. **Initial experiment:** frequency baseline, GRU and transformer; 5 s / 54 features.
   The transformer led the initial validation comparisons, but its training loss
   continued falling after validation NLL ceased improving.
2. **v2 internal search:** nine first-split candidates explored regularisation,
   geometry, auxiliary current-label supervision, longer history and MLPs. Two
   further second-split runs provided confirmation. Unsuccessful candidates remain
   in the record; the 20-second MLP was promising.
3. **v3 targeted search:** accuracy-selected checkpoints, temperature scaling,
   wider/smoothed networks and 60-second history. Eleven fits across the two internal
   partitions. Individual accuracy gains could breach the NLL guard.
4. **Recorded mixture extension:** fixed uniform mixtures were compared internally.
   Averaging improved accuracy and probability quality. Four variants averaged
   60.9817% internal accuracy versus 60.9598% for the wide/long pair; NLL favoured
   four variants (1.2581 versus 1.2807), while macro-F1 slightly favoured the pair.
   Internal confirmation used one seed per variant; the final recipe uses three.
5. **Freeze and refit:** variants, epoch counts, temperatures and seeds were fixed,
   then twelve models fitted on all 210 training groups. Outer validation was used
   for the preserved original-versus-improved comparison. The reserved test was not used.

Internal groups comprise 147 fit / 21 stop / 42 select, wholly within the original
training partition. Two split seeds reduce reliance on one favourable partition,
but the search was adaptive. Internal and outer-validation scores must not be
drawn as a single comparable improvement curve. The evidence supports the combined
procedure; it does not isolate how much each change contributed.

The outer validation set already informed the original models. This review adds
diagnostics to that same development set; it does not turn it into an untouched
test. Repeated model selection can overfit the selection criterion itself.
[Cawley & Talbot, 2010](https://www.jmlr.org/beta/papers/v11/cawley10a.html)

## 5. Pilot-by-pilot interpretation

All 30 pilot/horizon top-1 point estimates improve over the original transformer.
That does not establish a significant improvement for every pilot. The table
below uses +5-second origins; accuracy/share/change/ECE values are percentages or
percentage points, and gain intervals are paired, pointwise 95% intervals.


| pilot_id | count | new_top1 | gain | gain_lower | gain_upper | dominant_skill | dominant_share | endpoint_change_rate | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pilot_001 | 2845 | 54.06 | 6.292 | 0.622 | 12.594 | lag_pursuit | 50.193 | 18.067 | 8.743 |
| pilot_002 | 3057 | 75.597 | 11.809 | 8.128 | 15.389 | extend | 31.632 | 40.301 | 9.262 |
| pilot_003 | 2878 | 63.899 | 5.942 | 1.752 | 10.807 | recommit | 35.719 | 25.747 | 10.867 |
| pilot_004 | 2897 | 67.794 | 0.897 | -4.644 | 6.524 | turn_cold | 44.667 | 17.915 | 7.523 |
| pilot_005 | 3049 | 74.155 | 12.857 | 9.071 | 16.542 | crank_target_right | 22.204 | 44.736 | 6.823 |
| pilot_006 | 2987 | 62.337 | 7.164 | 2.202 | 12.472 | lag_pursuit | 29.628 | 33.88 | 1.996 |
| pilot_007 | 3001 | 54.415 | 0.766 | -2.714 | 4.261 | extend | 39.72 | 18.427 | 4.357 |
| pilot_008 | 2871 | 46.082 | 7.315 | 1.3 | 13.238 | turn_cold | 29.781 | 23.755 | 8.464 |
| pilot_009 | 2990 | 69.03 | 10.0 | 4.901 | 14.798 | lag_pursuit | 26.656 | 38.395 | 10.914 |
| pilot_010 | 2883 | 66.077 | 12.418 | 7.514 | 17.797 | turn_cold | 59.07 | 31.009 | 4.186 |


Intervals resample whole scenario groups 5,000 times, keeping windows together;
they are not corrected for 30 comparisons or historical model selection. Pilot
identity was not provided as a feature. Different dominant labels, switch rates,
episode lengths and reward profiles influence the interpretation of difficulty.
The table describes association, not a causal attribution of accuracy to any one
factor. A predictable pilot is not necessarily a better or worse combat pilot.

Per-pilot ECE ranges from about 2.00 to 10.91 points despite pooled ECE of 3.42.
Opposing overconfidence and underconfidence can cancel when pooled. Mean confidence
matching accuracy is not enough to establish calibration across confidence bins.
Conditional calibration is a reporting gap to investigate internally before
considering any new calibration method.

## 6. Per-skill results: terminology and support

- Precision = TP / (TP + FP): when we predict this skill, how often are we right?
- Recall = TP / (TP + FN): when this skill actually occurs, how often do we recognise it?
- F1 = 2TP / (2TP + FP + FN): balances precision and recall.
- Top-1 accuracy: fraction of all eligible origins with the correct selected label.
- Top-3 accuracy: fraction with truth among three highest-scored labels.
- Catalogue macro-F1 averages all 33 entries, using zero for undefined stored class scores.
- Supported macro-F1 averages only labels with truth support; the set can change by pilot/horizon.

Per-skill 'accuracy' is ambiguous. If the denominator is true occurrences of that
skill, the metric is recall. One-versus-rest accuracy includes true negatives and
can be high for an almost-never-predicted rare class. Keep precision, recall, F1,
truth support and predicted count together. Seven labels have no validation target
support, so performance on them is not measured. Eleven labels account for 96.42%
of +5-second target origins; the remaining fifteen supported labels are sparse.

pilot_skill_metrics.csv includes every pilot, all 33 labels, all horizons and both
original-transformer/ensemble results. skill_support_context.csv adds distinct
encounter, scenario and pilot counts per label. Many adjacent origins can come
from one sustained label run; raw support is not the number of independent cases.
Use the pilot × skill heatmap to locate misses, then inspect precision and
predicted counts to distinguish under-prediction from broad over-prediction.

## 7. Three meanings of time

**Observation history** is how far backwards X extends. **Forecast horizon** is
how far ahead the target lies. **Elapsed encounter time** is when the forecast is
issued. Increasing one does not imply increasing either of the others.

On the same 29,458 origins, ensemble top-1 is 65.60 / 65.14 / 63.48% at +1/+2/+5 s,
versus 58.87 / 58.29 / 55.90% for the original transformer. Thus longer-horizon
accuracy decreases even when the eligible starting observations are held fixed.
Supported class sets are 25/25/26, so use fixed-33 macro-F1 for equal-vocabulary
comparisons. Different future targets still have different label distributions.

On each horizon's full eligible set, endpoint-change fractions are 7.35 / 13.96 /
29.39%. These are different cohorts. Same endpoint labels can conceal an intervening
switch and return. Repeating the current true label is a privileged diagnostic,
not an observation-only forecasting baseline, because that label is unavailable.

Elapsed-time results for the ensemble at +5 seconds are:


| origin_time_bin | count | episodes | scenario_groups | supported_skills | top1_accuracy | changed_endpoint_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| 6-20 s | 6411 | 450 | 45 | 24 | 57.386 | 40.009 |
| 21-60 s | 11531 | 389 | 39 | 23 | 64.392 | 32.703 |
| 61+ s | 11516 | 225 | 24 | 17 | 65.96 | 20.172 |


Later cases have more history, fewer changes and a different surviving population.
Only 225 encounters and 24 scenario groups remain in the late bin. Supported labels
fall from 24 to 23 to 17. The higher late top-1 cannot isolate the benefit of more
history. Late top-3 actually falls slightly versus the transformer: 90.48% to 90.06%.
The complete table includes both models, all horizons, NLL and F1.

The full validation decision streams contain 38,181 choices and 37,731 adjacent
one-second pairs. 91.77% repeat the same label, with 3,104 switches. There are 3,554
runs, including 450 episode-ending runs whose durations are incomplete. A run means
consecutive identical selections, not a physical manoeuvre. Completed runs have
median duration 5 s and p90 16 s; excluding censored endings makes these conditional
summaries, not unbiased estimates of every duration. Gaps would also censor runs;
none occurred here. Full streams contain 27 labels because early/terminal choices
outside forecasting eligibility are included. Per-pilot temporal summaries:


| pilot_id | decisions | one_second_repeat_fraction | switches | completed_runs | right_censored_runs | completed_run_median_s | completed_run_p90_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| population | 38181 | 91.773 | 3104 | 3104 | 450 | 5.0 | 16.0 |
| pilot_001 | 3679 | 94.056 | 216 | 216 | 45 | 4.5 | 21.0 |
| pilot_002 | 3889 | 89.204 | 415 | 415 | 45 | 5.0 | 13.0 |
| pilot_003 | 3735 | 92.195 | 288 | 288 | 45 | 5.0 | 21.0 |
| pilot_004 | 3808 | 94.898 | 192 | 192 | 45 | 6.0 | 20.9 |
| pilot_005 | 3890 | 88.739 | 433 | 433 | 45 | 6.0 | 14.0 |
| pilot_006 | 3804 | 91.088 | 335 | 335 | 45 | 6.0 | 17.0 |
| pilot_007 | 3840 | 94.256 | 218 | 218 | 45 | 5.0 | 20.0 |
| pilot_008 | 3768 | 92.479 | 280 | 280 | 45 | 5.0 | 20.0 |
| pilot_009 | 3953 | 90.02 | 390 | 390 | 45 | 6.0 | 13.0 |
| pilot_010 | 3815 | 91.061 | 337 | 337 | 45 | 5.0 | 14.0 |


## 8. Catalogue semantics and observability

The execution code allows several labels to share guidance components. For example,
lag_pursuit, extend, turn_cold, recommit and pursue_target can emit the same zero-change
commands; beam_target_left and crank_target_left both emit a −90° heading change;
launch, secondary_shot and short_range_attack all set shoot. This does not prove
identical trajectories or identical semantics in every context. It does show why
exact symbolic label prediction is stricter than coarse physical-motion prediction.

No labels were merged. A supplementary behaviour-family score would need a justified
mapping fixed independently of which grouping raises validation scores. The present
evidence does not establish an irreducible accuracy ceiling.

## 9. Report the estimand, not just a percentage

The primary skill score weights eligible forecast origins equally. Equal-encounter
accuracy averages within each eligible encounter first; equal-pilot accuracy averages
within each pilot first. Long encounters contribute more under the primary definition.
The three accuracy columns are percentages; horizon is in seconds and the remaining
numeric columns are counts:


| model | horizon_s | count | pooled_windows | equal_episodes | equal_pilots | episodes | pilots |
| --- | --- | --- | --- | --- | --- | --- | --- |
| transformer | 1 | 30802 | 58.967 | 56.618 | 58.841 | 450 | 10 |
| ensemble | 1 | 30802 | 65.616 | 63.804 | 65.461 | 450 | 10 |
| transformer | 2 | 30466 | 58.38 | 55.718 | 58.249 | 450 | 10 |
| ensemble | 2 | 30466 | 65.171 | 63.009 | 65.016 | 450 | 10 |
| transformer | 5 | 29458 | 55.9 | 52.978 | 55.799 | 450 | 10 |
| ensemble | 5 | 29458 | 63.48 | 60.854 | 63.345 | 450 | 10 |


A practical report should contain: task/observation boundary; groups/episodes/origins;
primary top-1/top-3/F1 plus NLL/Brier/calibration; paired uncertainty; all-pilot and
all-skill breakdowns; temporal strata and negative results; selection chronology,
seeds and cost. State units, averaging and denominators in each table. Keep test
results separate from development history when the final assessment is performed.

Dependent observations require evaluation blocks suited to the intended use. Here,
scenario blocking targets new encounters for known pilots; unseen pilots require a
different blocked design. [Roberts et al., 2017](https://nsojournals.onlinelibrary.wiley.com/doi/10.1111/ecog.02881)
Cluster resampling preserves within-group structure under assumptions about the
clusters; it does not make windows independent or undo selection bias.
[Field & Welsh, 2007](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-9868.2007.00593.x)

## 10. Proposed experiments, with measurable questions

1. **Data-size learning curves.** Use nested fitting-group subsets with fixed internal
   stop/select groups and repeated seeds. Determine whether gains plateau before
   committing to broader scenario collection. Preserve natural-prevalence evaluation.
2. **Matched ablations.** Compare 54/91 features and 5/20/60-second histories while
   controlling model, training protocol and compute. Separate architecture changes
   from feature/history changes rather than attributing the combined gain to one.
3. **Checkpoint objective and calibration.** Compare NLL- and accuracy-selected
   checkpoints, with/without calibration, using internal groups. Report accuracy,
   macro-F1 and probability quality together; reject materially harmful trade-offs.
4. **Ensemble size.** Test the six- versus twelve-member refit recipes with paired
   scores, training and inference cost. The measured 0.02-point internal difference
   was between two and four variants using one seed each; it was not a final
   six-versus-twelve evaluation and does not establish that twelve are necessary.
5. **Rare skills and changes.** Collect broader relevant training cases and test
   cautious reweighting/current-skill-estimation plus transition models. Current
   true skill must remain a target or diagnostic, never an inference feature.
   Measure common-skill regressions, NLL and calibration alongside rare-skill gains.

Freeze acceptance criteria before comparisons, retain failed runs, and keep the
reserved test for the final frozen protocol. Any future use of the present validation
diagnostics to choose models remains development, not independent confirmation.
Unseen-pilot and sensor-noise evaluations would support broader claims and should
have their own explicit design.

Segmental metrics can reveal errors hidden by pointwise accuracy in action-sequence
tasks. [Lea et al., 2017](https://arxiv.org/abs/1611.05267) motivates a supplementary
future analysis, not a replacement or reinterpretation of the existing per-label
F1. Define future-time alignment, segment matching, gaps and censored endings first.

## Reproducibility and files

- [Full PDF](skills_catalogue.pdf), [speaker notes](speaker_notes.md), [existing comparison](comparison.md).
- [Per-pilot review](data/pilot_review.csv), [pilot/skill metrics](data/pilot_skill_metrics.csv), [label support context](data/skill_support_context.csv).
- [Weighting comparison](data/accuracy_weightings.csv), [common origins](data/temporal_common_origins.csv), [elapsed-time strata](data/temporal_elapsed_phases.csv).
- [Temporal dynamics](data/temporal_pilot_dynamics.csv), [run records](data/temporal_validation_runs.csv), [transition counts](data/temporal_transition_counts.csv).
- [Architecture and timing provenance](data/architecture_review.json), [all fit timers](data/training_run_inventory.csv), [temporal audit](data/temporal_evidence.json), [review contract](data/review_evidence.json).

The default builder reads this portable bundle only. `--collect-review` regenerates
the review from fixed local metadata/predictions and validation-only decision rows.
It does not train, run model inference, alter selected checkpoints or evaluate test
targets. Bundle and source SHA-256 checksums make the evidence auditable.
