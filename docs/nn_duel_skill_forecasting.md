# Own-skill-conditioned NN-duel forecasting

## Notebook 28: HGT + TCN

After notebook 26 has prepared the v2 skill/intent overlay, run
`28_skill_intent_target_relative_missile_hgt_tcn_nn_duel_skill_forecasting.ipynb`.
This is notebook 27's single-seed recipe with a **Temporal Convolutional Network
(TCN)** replacing only its temporal Transformer and learned position embeddings.
The HGT-v2 spatial encoder and hierarchical skill/auxiliary intent heads are shared.
Four width-96 residual blocks each contain two causal kernel-3 convolutions;
dilations 1/2/4/8 give a receptive field of 61 frames, covering the 51-frame input.
Per-time channel LayerNorm preserves causality. The final temporal position feeds
the existing output heads. Spatial graph attention remains present.

The notebook uses the same dataset receipt or `BVR_NN_INTENT_DATA` override as 27.
Run artifacts live separately under `artifacts/nn_duel_skill_intent_tcn_forecasting/`,
with `comparisons/latest_hgt_hierarchical_intent_tcn_run.json`. Optional tracking
uses its own experiment; `BVR_NN_INTENT_TCN_MLFLOW=0` disables it. The family/side
weight caps are 100/20, matching notebook 27's actual reviewed configuration.
Unweighted exact-skill test NLL continues to select checkpoints; weighted F1 is
reported at each horizon. No performance improvement is assumed.

Start a fresh TCN run. Resume requires a TCN `EXISTING_RUN` with matching source,
configuration and data. Notebook-27 Transformer checkpoints have incompatible
temporal weights. The model factory saves/reloads kernel and dilation settings,
and final evaluation retains the same skill, family, conditional-direction,
credit, intent and baseline reports. Bounded HGT frame caching and streaming
epoch metrics are supported. Full flight-data training and GPU performance
must be measured on the intended training machine.

## Preparing the NN-duel forecasting dataset

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

## Non-overlapping training workflow

Run `notebooks/14_prepare_nonoverlap_nn_duel_skill_forecasting.ipynb` and then
`notebooks/15_nonoverlap_nn_duel_skill_forecasting.ipynb`. Notebook 14 prepares a
separate dataset directly from the same notebook-11 source; notebook 12 is not a
prerequisite. Notebook 15 uses **width 96, six heads, two layers and dropout 0.25**,
with notebook 13's L1 regularisation, no 50-epoch minimum, and overtraining shutdown.

The non-overlap policy covers the full raw input footprint: 51 model frames plus
five backward-acceleration lookback frames, or the inclusive interval `[origin-55,
origin]`. Fixed candidate origins 60, 120, 180, ... are six seconds apart, leaving
four unused raw ticks between adjacent retained intervals. Slots alternate A01/B01;
a stable SHA-256 hash of the split seed and flight ID determines the starting view.
Thus even the two views of the same physical interval cannot both enter training.
Missing or ineligible slots stay empty. Each view's origins are normally twelve
seconds apart. Selection uses no skill labels beyond the existing eligibility mask.
This removes shared input intervals, but adjacent forecast labels may lie in later
histories and flights still contain correlated behaviour; it does not establish
statistical independence.

The same seeded 40/30/30 scenario assignments as notebook 12 are computed from all
eligible candidates, then only training is thinned. Test/validation retain the
original one-second stride and identical windows/labels. Scaling is refitted from
retained training histories only; skill indicators remain unscaled. Both aircraft
perspectives of every flight remain in the same partition. Rare skills may lose
training support: `skill_support_before_split.csv` describes all candidates, while
`skill_support_by_split.csv` describes retained examples. `window_sampling_audit.csv`
reports eligible/retained/dropped counts per flight/view; `nonoverlap_audit.json`
records verified raw-history disjointness. The loader rechecks this guarantee and
the view schedule when reading the dataset.

Outputs, the latest-dataset receipt and run directories live under
`artifacts/nn_duel_nonoverlap_forecasting/`. Override the prepared path with
`BVR_NN_NONOVERLAP_DATA`; source/download/root overrides are the same as notebook 12.
The new `nn-duel-own-skill-nonoverlap-horizon-v1` contract rejects dense training
caches. MLflow uses a separate `nn-duel-nonoverlap-skill-forecasting` experiment and
local database; disable it with `BVR_NN_NONOVERLAP_MLFLOW=0`. Fresh runs are required
for the new architecture/data. Resume requires unchanged source, data and settings.
The frame memory map remains about 13.7 GiB; window thinning reduces training work
and index size, not the raw frame-array storage. Existing prepared arrays consume
additional disk space.

## Class-weighted training

Notebook 15 uses unweighted masked cross-entropy averaged equally over forecast
horizons. Common skills still contribute more examples within each horizon. L1,
dropout, non-overlap sampling and balanced split proportions do not compensate for
unequal class contributions to that loss.

After notebook 14, run
`notebooks/16_balanced_nonoverlap_nn_duel_skill_forecasting.ipynb` to fit the same
width-96, two-layer, six-head model with dropout 0.25 and training-only class weights.
It reuses the exact same prepared dataset, receipt, three seeds and stopping rules.
Each horizon gets its own counts from the retained training targets, excluding `-1`.
For a supported class, `r = min(5, sqrt(max_count / class_count))`; divide these raw
weights by their training-frequency weighted mean so the mean example weight is one.
Thus a 16:1 count ratio gives rare examples four times the weight of common examples,
and the rare/common weight ratio cannot exceed `max_class_weight_ratio=5.0`.
The setting `class_weighting="sqrt_inverse"` enables this in notebook 16; shared
trainer defaults leave it disabled for the earlier notebooks.

The per-horizon data loss averages `weight[target] * cross_entropy` over valid labels,
then averages represented horizons equally. It divides by the valid-label count,
not by the minibatch sum of weights, so single-class minibatches retain their class
weight. Training-frequency normalisation keeps the overall weighting scale comparable
with L1. No training windows are duplicated or resampled. Classes absent from training
receive zero positive-target weight and retain their output logits; weighting cannot
supply their missing positive examples. The notebook displays all zero-support pairs.

Counts and weights are written to `class_weights.json` and `class_weights.csv` in
each run, recorded in `run.json` and checkpoints, and attached to MLflow. Resume
requires unchanged settings, data and source. History/MLflow record the evaluation-mode
weighted training CE as `train_data_loss` and its sum with L1 as `train_objective`.
`train_nll`, `test_nll`, checkpoint selection, early stopping, the overtraining gap,
final metrics and reliability plots all remain **unweighted**. Compare rare-class
recall/F1 and supported macro-F1 alongside overall NLL, accuracy and calibration;
weighting may trade some common-class accuracy or calibration for rare-class recall.

The new runs and default MLflow database live in
`artifacts/nn_duel_balanced_forecasting/`, using the
`nn-duel-balanced-skill-forecasting` experiment. The dataset still comes from
`artifacts/nn_duel_nonoverlap_forecasting/latest_dataset.json` or
`BVR_NN_NONOVERLAP_DATA`. Disable tracking with `BVR_NN_BALANCED_MLFLOW=0`.
Notebook 15 need not run first. Start fresh runs for the new objective; use the
selection partition called "test" for comparisons, and keep final validation held
out until every seed completes.

## Enemy-missile features

Run `notebooks/17_prepare_enemy_missile_nn_duel_skill_forecasting.ipynb`, then
`notebooks/18_enemy_missile_nn_duel_skill_forecasting.ipynb`. Notebook 17 prepares
a new dataset directly from notebook 11's complete two-view recordings; notebooks
14–16 need not run first. The recordings already contain each aircraft's friendly
missiles. This supplies four enemy slots for the other observer without rerunning
flights or changing the NN pilots, simulator, original corpora or earlier datasets.

Each slot has seven channels, analogous to the original friendly slots:
relative North/West/Up position (3), relative North/West/Up velocity (3), and range
(1). The new layout for the 33-skill catalogue is:

| Channels (zero-based) | Meaning |
| --- | --- |
| 0–53 | Existing observation/energy and causal acceleration values |
| 54–81 | Four enemy-missile slots, seven consecutive channels each |
| 82–114 | Current own-skill indicators, one per catalogue entry |

For A observing B, read B's friendly slots (`obs_19`–`obs_46`) at exactly the same
flight, step and timestamp. Let `r_MB` and `v_MB` be a missile's position and velocity
relative to B, and `r_BA`, `v_BA` the B-minus-A aircraft quantities already present
in A's observation. Then `r_MA = r_MB + r_BA` and `v_MA = v_MB + v_BA`.
The code translates world North/West/Up vectors; it does not rotate body-frame axes.
Raw position components use `[50000, 50000, 10000]` metre scales. Aircraft relative
velocities use 600 m/s, whereas missile-relative velocities use 1200 m/s, so the
normalised missile velocity adds half the aircraft relative-velocity channels.
Range is recomputed from physical position and divided by 50000 m, including the
correct vertical scale. B observing A uses the same calculation with roles reversed.

The original opposing friendly-slot ordering is retained; this is not a nearest-four
range selection. If the source has more than four live friendly missiles, the same
four source slots are represented. All-seven-zero source slots remain zero before
standardisation, preventing phantom missiles at the opposing aircraft position.
The compact interface's all-zero absence convention is preserved, with no added
presence bit. These are idealised contemporaneous simulator quantities; live use
requires equivalent observed enemy-missile tracks. Opponent decisions, future missile
states, pilot identity and outcomes are not enemy-feature inputs.

The dataset uses the same fixed scenario splits, non-overlapping training histories
and target masks as notebook 14 for the same source/seed. Both input streams are
aligned before transformation. Scaling uses retained training histories for all
82 numeric channels; the final 33 skill indicators remain binary. Missing slots
are zero in raw space; their normalised values follow the fitted means/scales.
The loader verifies the new feature order and the
`nn-duel-own-skill-enemy-missile-nonoverlap-horizon-v1` contract, and rejects older
87-channel caches. `enemy_missile_audit.csv` reports raw-frame slot occupancy by
flight/view and is logged to MLflow alongside the dataset manifest and other audits.

Notebook 18 retains width 96, two layers, six heads, dropout 0.25, capped training-only
class weights, L1 and both stopping controls. Its projection accepts 115 channels;
output horizons and the catalogue vocabulary remain unchanged. Checkpoints store
ordered feature names, scaling and input width; inference must reproduce that schema.
`reconstruct_enemy_missiles(observer_raw, opposite_raw)` reproduces the 28 new
channels when both raw views are available. The new numeric channels precede own
skills, so old checkpoints cannot be used with the new dataset. Unweighted NLL
continues to drive checkpoint selection, stopping and final reporting.

Data, receipt, runs and default MLflow storage live under
`artifacts/nn_duel_enemy_missile_forecasting/`, in the
`nn-duel-enemy-missile-skill-forecasting` experiment. Override the data path with
`BVR_NN_ENEMY_MISSILE_DATA`, or disable tracking with `BVR_NN_ENEMY_MISSILE_MLFLOW=0`.
Source/root/download overrides are shared with notebook 14. Start fresh model runs.
The separate 115-channel frame array is about 18.1 GiB for the archived release;
allow at least 35 GB for source downloads and preparation, plus existing arrays,
model checkpoints and reports. Streaming remains the default rather than caching
the full array on a 16 GiB GPU.

## BF16 graph mask/reduction overflow in notebook 20

`RuntimeError: value cannot be converted to type c10::BFloat16 without overflow`
at `_TypedMessageBlock`'s `masked_fill` is a graph reduction dtype mismatch. Under
autocast, node residual states can be FP32 while the message MLP returns BF16.
The old code used the FP32 minimum as the BF16 mask value, which is out of range;
mixed source/destination dtypes can also fail the scatter operations.

Graph message aggregation now explicitly uses FP32 messages, sum/max/count
buffers and residual updates. Linear layers can still use BF16 autocast. The
mask minimum is taken from the message tensor's actual reduction dtype. No graph
edges, target masks, loss weights or prepared features are changed.

After pulling the fix, restart the Jupyter kernel and rerun notebook 20's setup
and training cells with `RESUME=False` and `EXISTING_RUNS={}` for new timestamped
runs. Keep notebook 17's existing prepared data. This changes the graph source;
do not bypass the source guard to resume an old graph run under changed code.

## Nonfinite evaluation in notebooks 18–20

A first-epoch `ValueError: Nonfinite evaluation loss` means the selection/reporting
pass produced a nonfinite loss after fitting. The traceback alone does not establish
whether inputs, parameters or GPU attention/precision kernels caused it. The shared
trainer now checks evaluation stages and records persistent failures, including
sample identities, in the run's `nonfinite_evaluation.json`.

Notebooks 18–20 explicitly use `evaluation_precision="fp32"` and
`evaluation_attention="math"`, while training can retain `precision="bf16"` on CUDA.
The evaluation policy separates selection from BF16 and fused inference-attention
kernels; it is a conservative starting policy rather than a confirmed diagnosis of
the reported GPU failure. Cross-entropy, checkpoint NLL and final metrics remain
unweighted and exclude L1. Use the same evaluation policy for all three architecture
experiments; a comparison with earlier mixed-precision results changes the protocol.

After pulling this fix, restart the Jupyter kernel so imported modules are refreshed.
Rerun notebook setup and training with `RESUME=False` and `EXISTING_RUNS={}` to create
fresh timestamped runs. A failure before epoch one completed can leave no usable
epoch checkpoint. Resume still requires unchanged source, data and configuration;
do not bypass those guards or delete a previous run to reuse its path. Notebook 17's
existing verified dataset and receipt remain compatible, so no data preparation or
flight collection is needed for this training/evaluation change.

Complete historical final predictions can still be plotted using `TRAIN_MODEL=False`,
`RESUME=False` and `REEVALUATE_SAVED=False`, with every configured seed supplied in
`EXISTING_RUNS`. Only this saved-output path can ignore the newly introduced evaluation
settings; it does not resume fitting or silently re-evaluate an old model under the
new policy. Loss, dataset, catalogue and feature-order checks remain active. If a
fresh run still fails, retain its numerical diagnostic file together with the printed
runtime/device settings and report the failing stage/sample; do not replace nonfinite
values or omit failed examples to obtain a score.

## Windows PyArrow import failure

`ImportError: DLL load failed while importing lib: An Application Control policy
has blocked this file` means Windows rejected a native component while importing
PyArrow. The traceback alone does not identify whether the blocked file is
`pyarrow.lib` or a dependent DLL. This happens before any flight data is read.
Both notebooks check the Parquet import first and report the active kernel's
Python executable, environment prefix, installed PyArrow version and original error.

Reproduce the import in the same environment from Anaconda Prompt:

```bat
conda run -n trajectory-nn-clean python -c "import sys; print(sys.executable); import pyarrow.parquet; import pyarrow; print(pyarrow.__version__)"
```

Immediately afterward, open Event Viewer > Applications and Services Logs > Microsoft
> Windows > CodeIntegrity > Operational and inspect the corresponding block event.
The machine's policy administrator can use that event's file path and signing details
to supply a policy-approved PyArrow build or authorize the affected component.
Installing another copy of the same blocked binary does not resolve the policy.
If the approved replacement is distributed through Conda, install it into
`trajectory-nn-clean` using the administrator's approved channel and build; the
traceback alone does not establish which package manager supplied the current copy.

After repair, rerun the import command above, restart the Jupyter kernel, and rerun
the notebook. The early check improves the diagnosis; it cannot change Windows policy.
Changing model settings, selecting CPU, or changing the flight-data directory does
not resolve a native-library policy block. A missing-package or ordinary missing-DLL
error receives separate instructions, rather than being labeled a policy rejection.

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

The architecture, optimizer, masked data loss and batch sizes follow 09b: two
Transformer layers, width 192, six heads, dropout 0.1, AdamW, independent missing-label
masks, and an equal mean over available horizon losses. The input projection derives
its width from the prepared manifest. Notebook 13 adds L1 regularisation with
`l1_lambda=1e-6`: the training loss includes this coefficient times the sum of
absolute trainable matrix/tensor weights, including position embeddings but excluding
biases and LayerNorm scales. AdamW weight decay remains 0.01. Evaluation NLL and
checkpoint selection exclude L1; history records `train_l1_penalty`, the evaluation-mode
`train_objective`, and the test-minus-train `generalization_gap` separately.

Each seed trains for up to 100 epochs. Ordinary early stopping has no warmup
(`early_stopping_min_epochs=0`): ten consecutive epochs without a new all-time
minimum test NLL stop training, as early as epoch 11. Independently, the overtraining
shutdown stops after three consecutive epochs with test NLL more than 5% above
its all-time best, a test-minus-train gap greater than 0.1 NLL, and train NLL below
its value at the best-test checkpoint. A recovery resets the overtraining counter.
These configurable thresholds are starting values to assess on the selection partition.
`best.pt` always retains the lowest-test-NLL checkpoint. History, resume checkpoints
and `training_status.json` record the counter and stop reason (`overtraining` or
`patience`). Disable both stopping switches to use the full epoch budget.

Start fresh run directories after this change. The existing provenance guard rejects
resume with changed source or configuration; old checkpoints cannot be resumed under
the new objective. Shared-trainer defaults keep L1 and overtraining shutdown disabled
for other notebooks; notebook 13 explicitly enables them.
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
