# Notebooks

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
`forecasting_ed_261104`. It uses the frozen post-tactical notebook-07 population,
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
