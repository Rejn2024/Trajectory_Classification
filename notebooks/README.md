# Dataset-generation notebooks

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

### RL pilot team names in Tacview

The training API consistently calls the **NN-controlled aircraft blue** and its opponent
red. The pinned Python BVR Sim backend currently has the opposite simulator-coalition
assignment because its built-in `simple` opponent controls the simulator's blue side.
Consequently, in evaluation ACMI files as currently generated:

| Tacview appearance | Aircraft ID | Training role |
| --- | --- | --- |
| Red | `A01` | NN-controlled aircraft (called blue by the training API) |
| Blue | `B01` | `simple` baseline opponent (called red by the training API) |

This is a backend colour inversion, not two NN policies. New integrations should retain
the public convention that the NN aircraft is blue and, where the simulator permits it,
also assign that aircraft to the blue ACMI coalition. The evaluation manifest's terminal
information includes the aircraft IDs and current ACMI colours so recorded files remain
unambiguous until that simulator-side migration is made.
