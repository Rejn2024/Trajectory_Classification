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
`BVR_DATASET_FLIGHTS` for a shorter validation run before the full build.

Train a skill classifier with `03_train_jsbsim_skill_classifier.ipynb`. It reads the
canonical Parquet shards from notebook 02, splits complete flights between
train/validation/test before constructing five-second windows, fits normalization on
the training split only, and trains/evaluates a GRU tactical-skill classifier. Set
`BVR_TRAIN_DATASET` to load a dataset from a non-default location.

If Parquet loading fails with `ArrowKeyError: No type extension with name
arrow.py_extension_type found`, update the environment with `pip install -e '.[ml]'`,
restart the notebook kernel, and run all cells again. The project requires
PyArrow 14.0.1 or newer because 14.0.1 includes PyArrow's own legacy-extension
hotfix; restarting clears any partially imported pandas modules from the old session.
