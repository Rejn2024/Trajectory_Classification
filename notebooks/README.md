**Varied pilot population:** [30_train_varied_transformer_pilots.ipynb](30_train_varied_transformer_pilots.ipynb)
trains ten pilots as in notebook 06, with recorded random Transformer widths/layers
and PPO loss coefficients. Its 91-feature frames include the four nearest incoming
hostile missiles. See the [guide](../docs/varied_transformer_pilots.md) for the
observation layout and checkpoint compatibility.

**Continue and collect:** [31_continue_varied_pilots_and_generate_corpus.ipynb](31_continue_varied_pilots_and_generate_corpus.ipynb)
loads those ten saved pilots, adds 200 simple-opponent epochs and 300 tactical-opponent
epochs per pilot, then collects 10,000 NN-versus-NN flights with all 91 policy features.
Run All performs preflight, training, tactical assessments, collection and verification;
the same job directory resumes committed epochs and flights. See the
[weekend campaign guide](../docs/varied_pilot_weekend.md).

**Current workflow:** train catalogue-v4 pilots with notebooks 04–06, then use
[29_trajectory_state_forecasting.ipynb](29_trajectory_state_forecasting.ipynb) for
future speed, heading and altitude regression. See the
[guide](../docs/trajectory_state_forecasting.md). Earlier forecasting notebooks
retain their historical targets and results.

# Notebooks

For joint skill and offensive/defensive/context-dependent intent forecasting, run
[26: prepare intent supervision](26_prepare_skill_intent_target_relative_missile_nn_duel_skill_forecasting.ipynb),
then [27: train one HGT network](27_skill_intent_target_relative_missile_hgt_nn_duel_skill_forecasting.ipynb).
Notebook 26 requires a verified notebook-24 v2 dataset and writes a small,
hash-pinned label overlay without copying its large feature map. Keep the base
dataset directory available for training/resume. The frozen taxonomy assigns all
33 skills to 9 offensive, 11 defensive and 13 context-dependent entries; unknown
skills fail preparation. Original future-skill masks, inputs, splits, scaling and
non-overlapping training windows are preserved. Intent labels are supervision only.
The v2 overlay also freezes the 29-family/four-mirror-pair hierarchy and writes
family/direction targets and support tables. Direction is masked for singletons
and missing skill labels. Re-run notebook 26 into a fresh small overlay to replace
an earlier intent-only overlay; the notebook-24 feature map can be reused.

Notebook 27 fits **one network once, with seed 42**, using notebook 25's HGT-v2
and temporal encoder. Its skill readout predicts 29 families and left/right
conditioned on each of the four mirrored families. Singleton families have no
direction prediction. Joint probabilities reconstruct the original 33 classes.
The skill objective awards base family credit plus an additional handedness bonus:
for mirrored labels, `C = (a*P(family) + b*P(family,side))/(a+b)`; for singletons,
`C=P(family)`. Base, unweighted scoring uses `-log(C)`. The training objective
balances its family and handedness components separately as described below,
averaging over all valid labels/horizons. Default `a=b=1` gives half credit for correct family/wrong
side and full credit for both correct. The unweighted side penalty is bounded by
`log(2)` (scaled by the fitted handedness weight during training); wrong-family side predictions do not earn bonus credit. Reward weights
are configurable as `family_reward` and `handedness_reward`.
Notebook 27 balances family frequencies using `family_sqrt_inverse` and fits
separate L/R weights **within each mirrored family at each horizon**, only on
retained training labels. Default `handedness_weighting="sqrt_inverse"` uses capped
inverse-square-root support (ratio cap 5), normalized to conditional mean one.
Use `"inverse"` for a stronger correction or `"none"` to disable side balancing.
The shared family weight multiplies `family CE + w_hand * side_penalty`; L/R
imbalance is not counted again in family weights. Zero-support sides are audited,
never invented. Both weight states are persisted and checked on resume, with
`handedness_weights.json/csv` for review. Base reward/NLL metrics stay unweighted.
Direction supervision selects the true family, never a hard family prediction.
The joint 33-skill NLL remains the checkpoint selection metric; it is separate
from the modified partial-credit training loss. Reward metrics are saved in
`final_hierarchical_credit_metrics.csv`. Existing notebook-26 v2 data can be reused,
but start fresh notebook-27 runs when changing the objective.
The auxiliary intent loss remains `0.25 * intent CE`, with L1 as before.
Unweighted joint skill test NLL still selects checkpoints. Reports compare intent-head
predictions with grouped skill probabilities and training-frequency baselines.
Validation is evaluated after selection. A catalogue grouping does not establish
the latent intent of an ambiguous manoeuvre, and one seed does not measure
variation across fits.

Data, runs and tracking use `artifacts/nn_duel_skill_intent_forecasting/`, with the
singular receipt `comparisons/latest_hgt_hierarchical_intent_run.json`. Set `BVR_NN_INTENT_DATA`
to override the intent dataset, or `BVR_NN_INTENT_MLFLOW=0` to disable tracking.
Set the single `EXISTING_RUN` for resume/report-only use. `best.pt` retains all
heads and frozen intent/family/direction ordering. `return_hierarchy=True` exposes
family, conditional-direction and intent logits plus joint skill log probabilities.
The default forward/MLflow export returns joint original-skill log probabilities;
softmax remains compatible with the previous interface. Optional `decode_hierarchy`
selects the family first and then its side; primary metrics use joint leaf argmax.
Separate direction reports condition on the true paired family and mask singletons.
Start fresh hierarchical runs; flat-head checkpoints cannot resume as this architecture.

Notebook 27 enables bounded, pass-local frame reuse during evaluation: overlapping
histories share their HGT/frame readouts, with at most 65,536 cached unique rows
and 8,192 rows per graph inference batch. Caches are rebuilt for each frozen
evaluation pass and preserve original row/observer identity and temporal positions.
Epoch metrics accumulate on the device, with small totals copied after each split;
full prediction arrays remain available from final evaluation. Fail-fast input
checks are retained; output-stage flags share one status transfer per batch, with
detailed failure diagnostics.
FP32/math evaluation, the loss and per-epoch checkpoint selection are preserved.
Per-class tables are displayed every five epochs and at training completion,
including early stopping; every epoch's CSV reports are still saved. These
execution settings require fresh runs under the source/config resume checks.

For missile positions measured from the opposing aircraft, run
[24: prepare target-relative missile positions](24_prepare_target_relative_missile_nn_duel_skill_forecasting.ipynb),
then [25: train the HGT](25_target_relative_missile_graph_nn_duel_skill_forecasting.ipynb).
These are counterparts to notebooks 22 and 23. Friendly missile positions are
relative to the opposing aircraft; hostile missile positions are relative to the
observer. Friendly range is recomputed from the new position. Friendly missile velocity is
relative to the opposing aircraft; hostile missile velocity is relative to the observer. Raw-zero empty slots and slot order are
preserved. The graph translates friendly positions into the common observer frame
for edge geometry and also restores friendly world velocities from their target-relative
inputs, keeping target-relative position and velocity in the missile node inputs.
Both accelerations, the 115-channel layout, splits, labels, training protocol and
three seeds remain as in 22/23. Checkpoints save `target_relative_missiles=True`,
`target_relative_missile_velocities=True` and `observer_acceleration=True` for factory
reload. The v2 data contract requires fresh preparation and training; earlier
position-only prepared datasets are rejected.

The new contract rejects older caches and uses the separate artifact root
`artifacts/nn_duel_target_relative_missile_forecasting/`. Set
`BVR_NN_TARGET_RELATIVE_MISSILE_DATA` to override its prepared dataset or
`BVR_NN_TARGET_RELATIVE_MISSILE_MLFLOW=0` to disable optional tracking.
Run notebook 24 before 25; neither notebook 22 nor new simulator collection is required.

Notebook 25 uses HGT model version 2: two layers (width 64, four graph heads),
with separate attention/aggregation for aircraft, ownership and targeting
families. Learned masked gates combine the three family summaries. Eighteen edge
features include relative position/velocity, distance/closing speed, bounded
closest-approach time/distance, bearing alignment, relative speed, and both
endpoints' kinetic, gravitational potential and total specific energy (J/kg).
A trainable per-head threat prior combines a physical distance/CPA/alignment score
with a learned residual from all edge features. It biases missile–target edges
in both directions; empty slots are excluded throughout.

AdamW warms up over 5% of planned optimizer updates from 10% of the peak `3e-4`,
then cosine-decays to 5% of peak. Both stopping rules wait until after epoch 20.
Scheduler state and optimizer-update count are checkpointed for exact resume.
The temporal Transformer remains width 96 with six temporal heads. Existing v2
prepared data can be reused; start fresh HGT-v2 training runs. Older HGT-v1
checkpoints retain their original factory reload path. Completed HGT receipts
are `comparisons/latest_hgt_runs.json`.

For observer acceleration in place of absolute observer North/West/Up position, run
[22: prepare both-aircraft accelerations](22_prepare_observer_acceleration_enemy_missile_nn_duel_skill_forecasting.ipynb),
then [23: train the typed graph](23_observer_acceleration_enemy_missile_graph_nn_duel_skill_forecasting.ipynb).
These are counterparts to notebooks 17 and 20. Observer acceleration uses the same
causal backward 0.5-second difference and 100 m/s² scale as opponent acceleration.
The three position channels are removed; the other 48 original observations,
both accelerations, four reconstructed enemy slots and own-skill indicators give
115 features for the 33-skill catalogue. Splits, labels and non-overlapping training
histories are preserved. The graph explicitly consumes observer acceleration in
its observer node, with the correct shifted aircraft/friendly-missile channels.
The distinct data contract prevents reusing notebook-17 caches with the same width.

The new notebooks use `artifacts/nn_duel_observer_acceleration_enemy_missile_forecasting/`
for data, receipts, runs and MLflow, preserving the original experiment outputs.
Set `BVR_NN_OBSERVER_ACCELERATION_DATA` to override the prepared-data directory or
`BVR_NN_OBSERVER_ACCELERATION_MLFLOW=0` to disable optional tracking. Notebook 23
retains notebook 20's model, loss, three seeds and stopping/evaluation settings.
Its checkpoints save `observer_acceleration=True` for correct factory reload.
Run notebook 22 before 23; notebook 17 is not a prerequisite.

For four additional enemy-missile slots, run
[17_prepare_enemy_missile_nn_duel_skill_forecasting.ipynb](17_prepare_enemy_missile_nn_duel_skill_forecasting.ipynb),
then train the three architectures using the same prepared dataset:

- [18: ordered Transformer](18_enemy_missile_nn_duel_skill_forecasting.ipynb) uses a dense frame projection.
- [19: unordered missile sets](19_enemy_missile_set_nn_duel_skill_forecasting.ipynb) pools friendly/enemy missile embeddings without graph messages.
- [20: typed graph](20_enemy_missile_graph_nn_duel_skill_forecasting.ipynb) passes geometry-conditioned messages between aircraft and missile nodes.
- [21: saved-run comparison](21_compare_enemy_missile_nn_duel_skill_forecasting.ipynb) verifies matched datasets, protocols, seeds and prediction targets before comparing all three families. It does not train models or load the large prepared feature array.

Notebook 17 reconstructs four enemy slots from the opposite view's same-time friendly
slots, changing the reference position/velocity and recomputing range. Each slot has
seven values, giving 115 features per frame. It preserves the non-overlap policy,
splits and targets. Notebooks 18–20 retain notebook 16's class weighting, temporal
width 96, six heads, two layers, dropout 0.25, L1 and stopping controls. Each trains
seeds 42/43/44 in separate architecture-named run directories. The set and graph
encoders respect raw-zero absence and do not assign temporal identity to missile
slot numbers. All training notebooks persist diagnostic plots, calibration and
per-class tables, sample identities and a manifest for **every** seed, alongside
the final predictions. Their completed-family receipts live at
`artifacts/nn_duel_enemy_missile_forecasting/comparisons/latest_<architecture>_runs.json`.
Notebook 21 reads those receipts or explicit frozen run lists and writes a new
timestamped comparison directory with CSV/PNG/JSON outputs and scenario-group
bootstrap intervals. Checkpoint selection uses "test"; validation remains final
descriptive evaluation, never a rule for choosing the best family or seed.

For report-only runs of 18–20, set `TRAIN_MODEL=False`, provide `EXISTING_RUNS` for
all configured seeds and leave `REEVALUATE_SAVED=False` to reuse existing final
predictions. Set `BVR_NN_ENEMY_MISSILE_MLFLOW=0` to disable optional tracking.
Keep the completed receipts or explicit paths to reproduce a historical comparison;
each successful new family execution updates its latest receipt. Existing saved
training source/configuration guards still apply to resume.
No new simulator collection is needed; live inference needs equivalent enemy tracks.
See the [feature details](../docs/nn_duel_skill_forecasting.md#enemy-missile-features).

Notebooks 18–20 use FP32 evaluation with math attention, separately from optional
CUDA BF16 training. After pulling an evaluation fix, restart the kernel and start
fresh runs; notebook 17's prepared dataset can be reused. Persistent numerical
failures save `nonfinite_evaluation.json` with stage/sample details. See
[nonfinite-evaluation troubleshooting](../docs/nn_duel_skill_forecasting.md#nonfinite-evaluation-in-notebooks-1820).

For class-imbalance compensation, run notebook 14 and then
[16_balanced_nonoverlap_nn_duel_skill_forecasting.ipynb](16_balanced_nonoverlap_nn_duel_skill_forecasting.ipynb).
Notebook 16 reuses the same non-overlapping dataset and width-96/dropout-0.25 model
as notebook 15, with per-horizon capped inverse-square-root class weights fitted
only on retained training labels. It saves weight/support audits and separate runs;
checkpoint selection and evaluation NLL remain unweighted for comparison with 15.
See the [weighting details](../docs/nn_duel_skill_forecasting.md#class-weighted-training).

For non-overlapping training histories and a smaller model, run
[14_prepare_nonoverlap_nn_duel_skill_forecasting.ipynb](14_prepare_nonoverlap_nn_duel_skill_forecasting.ipynb),
then [15_nonoverlap_nn_duel_skill_forecasting.ipynb](15_nonoverlap_nn_duel_skill_forecasting.ipynb).
Notebook 14 keeps disjoint 56-frame raw training histories across both views of
each physical flight, alternating views on six-second slots. It preserves notebook
12's scenario splits and dense test/validation windows. Notebook 15 uses width 96,
six heads, two layers and dropout 0.25, plus notebook 13's L1 and stopping controls.
These notebooks use separate prepared data, runs and MLflow tracking; notebook 12
is not a prerequisite. See the [sampling details](../docs/nn_duel_skill_forecasting.md#non-overlapping-training-workflow).

For forecasts conditioned on the observing aircraft's own current skill, run
[12_prepare_nn_duel_skill_forecasting.ipynb](12_prepare_nn_duel_skill_forecasting.ipynb),
then [13_nn_duel_skill_forecasting.ipynb](13_nn_duel_skill_forecasting.ipynb).
Notebook 12 reads notebook 11's latest complete NN-duel release and prepares both
perspectives with 54 observation/acceleration features plus 33 own-skill indicators.
For A observing B, A's current skill enters the inputs and B's future skill remains
the target at +1/+2/+5 seconds. Whole flights stay together in 40/30/30 grouped splits.
Notebook 13 trains new independently seeded networks with the 09b architecture,
test-based checkpoint selection and final held-out validation. Both notebooks
provide timed cells, provenance, support audits and reuse/resume controls.
See the [workflow guide](../docs/nn_duel_skill_forecasting.md) for data alignment,
resource requirements and configuration.

For additional manoeuvre forecasting data, open
[11_nn_vs_nn_training_data.ipynb](11_nn_vs_nn_training_data.ipynb). **Run All collects
20,000 new NN-versus-NN pairings (40,000 aircraft perspectives) and pushes incremental Git LFS backups** on
`retraining_environment_261111`. It uses the frozen post-tactical notebook-07 population,
new actual reset geometries, a 300-second/first-defeat limit, both observation
perspectives, shared GPU actor batching with a CPU/GPU throughput benchmark,
an end-to-end runtime estimate, resumable episode checkpoints and
exact-horizon skill-support audits. Collection and publication switches are in the
first cell. The existing 09b corpus and preparation cache remain separate.
Every Run All creates a new series ID, seed, work directory and output directory;
previous flights, benchmarks, verification receipts and logs are not reused.
Series IDs use 12 hexadecimal characters and benchmark directories use `b/`
to keep Windows snapshot paths short. Setup checks all archive destinations
before extraction; a work root that is still too long reports how to shorten it.
Defaults are `artifacts/nn_duels_20k_v1/<series-id>/` and
`shared/corpora/nn-duels-20k-v1/<series-id>/`. `BVR_NN_WORK` and `BVR_NN_OUTPUT`
override the parent directories; each still gets a fresh series subdirectory.
Rerun only the collection cell to continue the series active in the same kernel.
Earlier series remain intact. An explicit `BVR_NN_SEED` overrides random seed selection;
`BVR_NN_FLIGHTS` permits smaller validation runs.
See the [GPU batching guide](../docs/nn_duel_gpu_batching.md) for expected speedup,
hardware assumptions and validation limits.
Human-readable progress appears about every five minutes across rollout and backup
stages; detailed command logs stay under the working directory's `notebook_logs/`.

For the skill-forecasting improvement study, open
[10_skill_forecasting_improvements.ipynb](10_skill_forecasting_improvements.ipynb).
**Run All reads saved results and demonstrates inference.** It compares the
frozen ensemble with the preserved initial models, including scenario-group
uncertainty intervals. See the [improvement guide](../docs/skill_forecasting_improvements.md).

For future catalogue selections, open
[09_skill_forecasting.ipynb](09_skill_forecasting.ipynb). **Run All reads saved
development results and demonstrates observation-only inference.** It shows
exact +1/+2/+5-second labels, class support, frequency baselines, per-pilot and
changed/unchanged metrics, and confusion matrices. Training controls default to
off. See the [skill-forecasting guide](../docs/skill_forecasting.md).

For position and velocity forecasting, open
[08_motion_forecasting.ipynb](08_motion_forecasting.ipynb). **Run All reads saved
validation results and demonstrates checkpoint inference; it does not retrain or
evaluate the reserved test set.** It compares transformer and GRU forecasts at
1, 2 and 5 seconds against constant-velocity and constant-acceleration baselines.
See the [forecasting guide](../docs/motion_forecasting.md) for training commands,
grouped partitions, input definitions and the initial results.

For current pilot experiments, use the [experiment index](../docs/pilot_experiments.md).
`05_compare_population_runs.ipynb` is a read-only report: select a named experiment
in its first code cell and **Run All** to refresh progress, tables and plots. Saved
historical reports live in `results/`. Training notebooks start new runs.

[09b: paired A01/B01 skill forecasting](09b_paired_skill_forecasting.ipynb) combines
the synchronized observation streams from both published corpora. It uses the same
five-second history and +1/+2/+5-second skill targets as notebook 09, with six
attention heads, scenario-grouped 40/30/30 train/test/validation splits and up to 100 epochs.
Optional early stopping is enabled in the notebook: after the first 50 epochs,
10 consecutive epochs without a new minimum test loss conclude training (earliest
stop: epoch 60). Set `EARLY_STOPPING=False` for the full 100 epochs.
Run All downloads the needed LFS files and trains; each epoch displays train/test
per-class loss and F1. Every improving test-loss checkpoint is retained, and the
selected checkpoint receives final inline diagnostics on all three partitions.
Validation is held out until that final report; test is used for model selection.
The notebook includes setup, resumption instructions and performance recommendations.
Every executable cell reports CPU and elapsed wall time using `%%time`, with CUDA
synchronization in cells that submit GPU work. Diagnostic figures use the
noninteractive Agg backend, display the saved PNG bytes inline, and report each
layout/render/write/display stage. See [diagnostic recovery](../docs/paired_skill_diagnostics.md)
to regenerate the report from an existing run without retraining.
A resource audit describes current batching,
and the throughput recommendations include conditional speedup and memory estimates.
CUDA defaults now use BF16 autocast, training/evaluation batches of 512/1024 and
a normalized cache of unique frames on the GPU. CPU execution falls back to FP32
and direct memory-mapped batches. Set `DATA_BACKEND="prefetch"` for a reusable,
spawn-safe CPU worker pool with pinned buffers and overlapping CUDA transfers.
The training-only batch benchmark compares 512 and 1024 without changing the fitted
model or RNG state. `COMPILE_MODEL` and `BENCHMARK_COMPILATION` opt into compilation
and an eager/compiled comparison; saved weights remain portable. Earlier run folders
are preserved; acceleration settings use a fresh run directory.
MLflow tracking is enabled by default, with a dedicated SQLite store at
`artifacts/paired_skill_forecasting/mlflow.db`. The setup cell prints the local UI
command; set `MLFLOW_TRACKING_URI` to use a tracking server. Each training run gets
a unique folder/name and logs configuration, data/source lineage, runtime/backend,
epoch/horizon/class metrics, final metrics, selected model and diagnostic figures.
`EXISTING_RUN` plus `RESUME=True` reuses the original MLflow run ID. The final
notebook section compares runs on the same prepared dataset and plots their loss
histories. Models are saved in the MLflow PyTorch flavor with their input schema;
the raw corpora and large prepared/prediction arrays are not uploaded.

Start with `01_generate_smoke_dataset.ipynb`. It is intentionally executable without
BVR Sim: a small deterministic dynamics stub exercises the repository's real scenario
sampler, policies, episode runner, feature generation, manifest, and Parquet writer.
This makes it possible to validate the data contract before connecting an external
simulator.

The generated files default to `artifacts/datasets/bvr_f16_1v1_smoke_demo/` and are
safe to delete. Set `BVR_NOTEBOOK_OUTPUT` to write elsewhere. When BVR Sim is installed
at the pinned revision, replace only the `env_factory` shown near the end of the
notebook; keep the validation and persistence cells unchanged.

For a simulator-backed pilot, use `02_generate_jsbsim_skill_dataset.ipynb`. It runs
100 seeded 1-v-1 flights through JSBSim, selects labelled behaviours through BVR
Sim's `SkillManager` and a reproducible stochastic schedule manager, records canonical
features at 10 Hz, and emits one Tacview ACMI replay per flight. Use
`BVR_DATASET_FLIGHTS` for a shorter validation run before the full build. Flights run in
spawn-compatible loky worker processes on Windows and POSIX; set `BVR_DATASET_WORKERS=1`
to run serially while debugging. Install the `video` extra before running this notebook.

Train a skill classifier with `03_train_jsbsim_skill_classifier.ipynb`. It reads the
canonical Parquet shards from notebook 02, splits complete flights between
train/test/validation (approximately 60%/20%/20%) while stratifying on every skill
demonstrated in a flight, then constructs five-second windows and fits normalization
on the training split only. It trains a Transformer classifier, uses test loss for
checkpoint selection, reserves validation for the final report, and records parameters,
metrics, and artifacts with MLflow. Set `BVR_TRAIN_DATASET` to load a dataset from a
non-default location. By default, MLflow stores tracking data in
`artifacts/mlflow.db` using its SQLite backend; set `BVR_MLFLOW_TRACKING_URI` to use a
different database or tracking server.

If Parquet loading fails with `ArrowKeyError: No type extension with name
arrow.py_extension_type found`, update the environment with `pip install -e '.[ml]'`,
restart the notebook kernel, and run all cells again. The project requires
PyArrow 14.0.1 or newer because 14.0.1 includes PyArrow's own legacy-extension
hotfix; restarting clears any partially imported pandas modules from the old session.

Train the hybrid-action reinforcement-learning pilot with
`04_train_rl_pilot.ipynb`. The notebook runs the production `PPOTrainer` against the
repository-pinned Python BVR Sim environment and its JSBSim F-16 flight-dynamics model,
then plots return and PPO loss and records deterministic final-model ACMI replays for every
fixed evaluation scenario. A JSON manifest maps each Tacview file to its seed, scenario,
return, and terminal information. Its
transformer receives two seconds of 10 Hz history, including separate kinetic and potential
energy approximations for both aircraft. After every epoch, the plotted pilot return is a
deterministic evaluation over the same seeded engagement set-ups; the training batch likewise
reuses its initially sampled set-ups and simulator seeds. The best checkpoint is selected by
the fixed-set evaluation mean and reloaded before the final recorded flight. Install the
`ml` and `video` extras, and use `BVR_PILOT_EPOCHS`, `BVR_PILOT_EPISODE_SECONDS`,
`BVR_PILOT_SCENARIOS`, `BVR_PILOT_EVALUATION_SCENARIOS`, `BVR_PILOT_WORKERS`,
`BVR_PILOT_DEVICE`, and `BVR_PILOT_OUTPUT` to configure the experiment.

Choose a population workflow:

- [05: original combat rewards](05_train_multiple_pilots.ipynb) trains 10 pilots for
  **150 epochs each** by default. It restores notebook 04's event-based formula with
  coefficients sampled from explicit random ranges and fresh training scenarios each
  epoch. All pilots share separate validation and test cases. No pilot is automatically
  assigned the original weights. The default learning rate is now `0.00003`.
- [06: hybrid rewards](06_train_multiple_pilots.ipynb) preserves the previous notebook
  05 experiment: relative evade/pursue/eliminate preferences and capped shared combat
  guidance, with defaults of 10 pilots and 80 epochs each.

Set `BVR_PILOT_COUNT` to any positive count and `BVR_PILOT_EPOCHS` to override the epoch
budget. Each pilot has its own reward, model, checkpoint, and MLflow run. Both notebooks
retain persistent processes, the 80% worker budget, CUDA graphs, and matched seeds
(`BVR_PILOT_SEED_STRIDE` selects different seeds). Their default outputs are separated
under `artifacts/rl_pilot_notebook/combat/` and `artifacts/rl_pilot_notebook/hybrid/`.
Both compare checkpoints within the population and record ACMI replays. Their benchmark
formulas differ, so cross-population comparisons require a common evaluation reward.
Notebook 04 is unchanged. Existing batches are preserved. See the
[multi-pilot guide](../docs/multi_pilot_training.md) for arbitrary custom reward
formulas, a short smoke run, and loading checkpoints after restarting the kernel.

## Hierarchical skill forecasting with a TCN

[28: HGT + TCN skill/intent forecasting](28_skill_intent_target_relative_missile_hgt_tcn_nn_duel_skill_forecasting.ipynb)
is the temporal-convolution counterpart to notebook 27. Run notebook 26 first,
then Run All in 28. It reuses the prepared v2 dataset, HGT-v2 graph encoder,
family/direction/intent heads and training/reporting policy, replacing temporal
attention with four causal residual convolution blocks covering all 51 frames.
Its checkpoints and reports have a separate output directory. See the
[forecasting guide](../docs/nn_duel_skill_forecasting.md#notebook-28-hgt--tcn)
for settings and reuse.
