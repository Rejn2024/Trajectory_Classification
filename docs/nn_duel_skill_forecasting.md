# Own-skill-conditioned NN-duel forecasting

Run these notebooks in order on `forecasting_ed_261104`:

1. `notebooks/12_prepare_nn_duel_skill_forecasting.ipynb` discovers the latest complete
   packaged release from notebook 11, selectively downloads its Git LFS data, surveys
   label support, and writes the prepared dataset.
2. `notebooks/13_nn_duel_skill_forecasting.ipynb` reads that dataset and fits three
   independently seeded Transformer networks using the 09b training/evaluation pipeline.
   Adjust `TRAINING_SEEDS` to change the number of networks.

The current release is `shared/corpora/nn-duels-20k-v1/eee3474c5493`: 10,000 physical
flights, 20,000 perspectives, and 42,196,060 recorded frames. Discovery ranks tracked
manifest files by their last Git commit timestamp; local unpublished packages use
manifest modification time. Partial packages are excluded. Set
`BVR_NN_FORECAST_SOURCE` to select a particular packaged directory, or
`BVR_NN_ROOT` to locate the checkout from a different kernel directory.

## What enters the model

For A01 observing B01, the input is A01's 51 observations, the same three backward
acceleration features as 09b, and 33 indicators for A01's current selected catalogue
skill: **87 features per frame**. B01's future skill remains the target. B01 observing
A01 uses exactly the same relative-role schema. Both directions train a shared model.

The own A01 selection is stored in **B01/target_decisions**, because those files label
the target of the named perspective. Preparation verifies actual aircraft IDs and
flight/timestamp alignment. Selections apply for their recorded one-second decision
interval; a missing own decision interval raises an error. The collector selects one
catalogue skill per aircraft at a time, so the catalogue-wide indicator vector is
one-hot on usable history frames. It does not provide concurrent-skill annotations.

The model receives 51 frames at 10 Hz, spanning five seconds. Five extra past frames
support the backward acceleration calculation. Origins start at 6 seconds and advance
by one second. Opponent labels at +1/+2/+5 seconds require exact decisions and use
independent missing masks. A terminal frame at a new decision boundary has no new
selection; it never creates a usable forecast beyond the recorded decisions. Inputs
exclude opponent decisions, pilot/checkpoint identities, future observations and outcomes.

## Preparation, splits and provenance

Both perspectives of a flight stay together in the same `scenario_group`. The split
targets 40% train, 30% test and 30% validation. Unlike 09b's discrete initial ranges,
NN-duel ranges are continuous: preparation uses up to ten range quantile strata and
bounded seeded pair swaps to balance per-skill/horizon episode presence and window
support while preserving each stratum's group quotas. Counts and absences are saved
in `skill_support_before_split.csv` and `skill_support_by_split.csv`.

Normalization uses each frame occurring in a training history once. Skill indicators
remain binary with mean 0 and scale 1. The prepared directory contains a memory-mapped
`features.npy`, `anchors.npy`, `targets.npy`, sample metadata, split assignments,
scaling, support tables and a SHA-256 manifest. Source, catalogue and feature-helper
hashes record lineage. A completion receipt, `artifacts/nn_duel_skill_forecasting/latest_dataset.json`,
points notebook 13 to the exact verified dataset. `BVR_NN_FORECAST_DATA` overrides
the prepared path. A partial output must be removed or a fresh path chosen before
retrying preparation; a complete, matching output can be reused.

The raw release is about 10.3 GB, and the 87-channel frame array is about 13.7 GiB.
Allow at least 30 GB of free disk space, with additional room for model checkpoints,
epoch reports and per-seed predictions. Preparation reads episode row groups and
writes frames without expanding all overlapping windows. The new trainer defaults
to prefetched batches on CUDA and direct memory-mapped batches on CPU; its default
does not attempt to cache the full frame array on a 16 GiB GPU.

## Training and interpretation

The architecture, optimizer, loss, batch sizes and early stopping follow 09b: two
Transformer layers, width 192, six heads, dropout 0.1, AdamW, independent missing-label
masks, and an equal mean over available horizon losses. The input projection derives
its width from the prepared manifest. Each seed trains for up to 100 epochs; with
early stopping enabled, the first 50 epochs always run and patience is ten afterward.
CUDA uses BF16 when available; CPU uses FP32.

As in 09b, strict improvements in **test NLL** select checkpoints. Test results are
therefore selection-biased. Validation is evaluated only after every configured seed
finishes fitting and selection. All seeds are reported; the first configured seed
is used for detailed plots, without choosing a winner from validation scores.

Runs live under `artifacts/nn_duel_skill_forecasting/runs/`. Checkpoints save the
ordered feature names, input dimension, classes, training-only scaling, architecture,
dataset digest and task contracts. `EXISTING_RUNS` supports resume or report-only
execution with the same data, configuration and source. Saved metrics include
per-horizon and per-class NLL, accuracy, F1, support, confusion matrices, probabilities,
learning curves, reliability plots and a training-frequency baseline. MLflow is enabled
by default in a separate `nn-duel-own-skill-forecasting` experiment; set
`BVR_NN_FORECAST_MLFLOW=0` to disable it or `MLFLOW_TRACKING_URI` for a remote store.

Catalogue selections remain labels rather than proof of distinct executed physical
maneuvers. A comparison with the older 09b corpus changes both data and inputs; an
own-skill ablation on these same flights and splits is needed to isolate the value
of the appended skill features.
