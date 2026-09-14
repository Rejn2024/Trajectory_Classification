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

## Windows PyTorch DLL recovery

On Windows, `ImportError: DLL load failed while importing _C` means that Python found
PyTorch, but Windows could not load one of its compiled dependencies. This is an
environment installation problem rather than a notebook or dataset error. First make
sure the notebook is using the intended environment: run `import sys; print(sys.executable)`
in a new cell and confirm that it points into `trajectory-classification`.

Repair the environment from an **Anaconda Prompt**, not from the running notebook:

```powershell
conda activate trajectory-classification
python -m pip uninstall -y torch torchvision torchaudio
python -m pip cache purge
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[ml,dev]"
python -m pip install ipykernel
python -c "import torch; print(torch.__version__); print(torch.rand(1))"
```

The explicit CPU wheel is the simplest supported option for this classifier and avoids
CUDA toolkit/driver mismatches. To use an NVIDIA GPU, replace that install command with
the command produced by the PyTorch installation selector for the installed driver.
Do not mix Conda and pip PyTorch packages in one environment. If the clean reinstall
still reports a missing DLL, install or repair the current **Microsoft Visual C++
Redistributable (x64)**, reboot, and rerun the one-line import check.

After the command-line check succeeds, register/select the environment if necessary:

```powershell
python -m ipykernel install --user --name trajectory-classification --display-name "Python (trajectory-classification)"
```

Restart the Jupyter kernel (not merely the cell) before running the training notebook
again. A fresh Python 3.11 or 3.12 environment is preferable if the environment contains
packages copied from another machine or Python installation.
