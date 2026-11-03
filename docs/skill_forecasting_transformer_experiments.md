# Transformer challenge for skill catalogue forecasting (v5)

This study tests whether Transformers can match or exceed the best previously
tested forecaster. The reference is the **v4 fifteen-model ensemble**: half the
probability from twelve MLPs, half from three 60-second TCNs. Its existing
validation accuracies are 67.09%, 66.62% and 64.86% at +1/+2/+5 seconds. These are
empirical results, not proof that the architecture or its score is optimal.

The primary result will be a **Transformer-only challenger**. A separately named
secondary result may add Transformers to the incumbent. An incumbent blend cannot
establish that Transformers alone replace the MLP/TCN system.

## Fixed task and comparison

The corpus, ten frozen trained pilots, tactical B01 viewpoint, scenario partitions,
33 labels, forecast origins, missing-target masks and exact +1/+2/+5-second
targets remain unchanged. Inputs retain the same 91 observation-derived features
and 26-token/20-second or 66-token/60-second histories. Current skill labels,
pilot identity and future tracks are not model inputs. No additional simulator
flights or RL-policy training are performed.

The primary metric is the equally weighted mean of the three horizon accuracies.
Internal comparisons then average equally across the two established partitions.
Probability quality and all-33-label macro-F1 remain reported. Matching is assessed
through measured differences and paired scenario intervals; an interval crossing
zero is not proof of equivalence. No noninferiority margin is introduced.

Every internal comparison uses the **full cached fifteen-model reference**:
each MLP has weight 1/24, and each TCN has weight 1/6. All cached internal models
were fitted only on that partition's 147 fit groups. The final incumbent weights,
trained on all 210 training groups, must never predict internal selection groups.

## Eight predeclared recipes

Each architecture is tested with both 20- and 60-second histories. Exact settings
are in [the registered configuration](../configs/training/skill_forecasting_transformer_search.json).

| Family | Attention/readout | Role |
|---|---|---|
| Existing pooled Transformer | Final and masked mean states plus current observation | Matched training/readout control |
| Temporal Transformer | All contextualized history positions retained in a flattened readout | Primary candidate |
| Patch Transformer | Older observations grouped in fives, recent observations in pairs, current token separate | Primary candidate |
| Transformer–MLP hybrid | Attention plus a parallel residual MLP over the full history | Explicitly separate hybrid |

The temporal and patch variants add fixed relative-time/gap coordinates and
validity flags to describe the existing nonuniform sampling. They use three
independently initialized pre-norm attention blocks, width 192 and four heads.
Their learned residual gains start at 0.1. The patch layouts contain nine or
seventeen tokens for the 20- or 60-second history respectively. No patch crosses
the older one-second/recent half-second sampling boundary.

Bidirectional attention is confined to observations already available at the
forecast origin. It does not access future target times. Invalid features are
removed before projection, invalid keys are excluded from attention, and hidden
padding is removed again after each block. Tests cover NaN padding, gradients,
batch independence, patch coverage and streaming checkpoint reload.

"Transformer-only" means every forecasting ensemble member is a Transformer.
Transformers still contain ordinary feed-forward sublayers and classification
heads; the architectural hybrid additionally has an independent full-history
MLP branch. Model counts and parameter budgets are reported rather than assumed
equal across families.

Temporal patching is inspired by
[Nie et al., PatchTST](https://openreview.net/forum?id=Jbdc0vTOcol). Our model jointly
encodes the aircraft features and predicts catalogue classes; it is an adaptation,
not a reproduction of that paper's channel-independent regression experiments.

## Training and frozen selection

All new recipes use AdamW, batch size 1,024, a maximum of 60 epochs and four
warmup epochs from learning rate 0.00003 to 0.0003, followed by a fixed 56-epoch
cosine schedule back to 0.00003. Accuracy stopping has patience 10 and minimum
improvement 0.001. Temperature fitting uses stop groups only. Final refits use
the same fixed schedule prefix; no validation feedback changes their budgets.
The optional warmup leaves the older zero-warmup schedule exactly unchanged.

Earlier Transformer screens used different sampling and NLL stopping. The new
controls also receive sparse history, accuracy stopping and stop-only calibration,
so this study tests the training/readout choices as well as the model family.

1. Screen all eight recipes with seed 42 on partition 0. Rank their standalone
   mean accuracy, keeping single-model quality guards descriptive at this stage.
   Freeze the best two primary-family recipes and the best architectural hybrid.
2. Fit those three recipes with seeds 42/43/44 in both internal partitions.
   Each partition contains 147 fit, 21 stop and 42 selection groups, entirely
   inside the original training partition. Scaling uses fit groups only.
3. Primary recipes are each three-seed Transformer ensemble and the uniform
   six-model pair. Prefer the highest mean accuracy satisfying quality guards in
   both partitions: mean NLL no more than 0.02 worse, all-33 macro-F1 no more than
   0.01 worse and each horizon's accuracy no more than 0.003 worse than v4.
   If none qualify, freeze the highest-accuracy Transformer challenger and label
   its guard failures; evaluating a challenger is not automatic promotion.
4. Secondary recipes blend a confirmed family or the pure Transformer pair with
   the incumbent at new-model weights 0.25/0.5/0.75. They must improve mean accuracy
   and pass the quality guards in both partitions. Freeze the best such blend,
   or omit this secondary result if none qualify.
5. Freeze the primary and optional secondary together. New components use the
   median epoch and temperature across their six internal fits, then three fixed
   refits on all 210 training groups. Refit their union only once.
6. One validation invocation evaluates the already named outputs on identical
   origins. Do not choose another recipe from validation scores or tune afterward.
   Preserve every old result and unsuccessful new fit. The reserved test remains
   neither prepared nor evaluated.

The internal partitions overlap and have already guided development. The outer
validation has also been inspected previously. This is further development
evidence for new scenarios involving the same pilots, not an untouched final test
or an assessment of unseen-pilot generalization.

## Reproduction

Use a new, empty output directory for a new experiment. Existing registered runs
require their saved source snapshots; later live code intentionally fails their
source-hash guards.

```powershell
.venv\Scripts\python.exe scripts/transformer_skill_forecaster.py initialize
.venv\Scripts\python.exe scripts/transformer_skill_forecaster.py screen
.venv\Scripts\python.exe scripts/transformer_skill_forecaster.py confirm
.venv\Scripts\python.exe scripts/transformer_skill_forecaster.py refit
.venv\Scripts\python.exe scripts/transformer_skill_forecaster.py evaluate
```

## Results

Pending the registered search. Transformer-only and incumbent-blend results will
be kept distinct, including any regressions, extra model cost and uncertainty.
