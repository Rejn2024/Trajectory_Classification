# Notebook 05 combat population: 2026-10-06

Ten independently trained PPO pilots from the completed run
`20261006T140858_362365Z`: 150 epochs each, 75 fixed training scenarios and
50 separate fixed validation scenarios, against BVR Sim's simple baseline opponent
using JSBSim aircraft dynamics. This is the original combat reward population from
[historical notebook 05](../../../notebooks/results/20261006_notebook05.ipynb), with independently
varied relative event weights. Notebook 06's hybrid rewards were not used here.

Each pilot directory contains its selected `best_model.pt`, the original `config.json`
and `manifest.json`, and all 150 rows of `training_metrics.jsonl`. The checkpoints and
epoch logs use Git LFS. Small JSON metadata and this README use ordinary Git.

## Download and load

From the repository root, after checking out the `multi-pilot-training` branch:

```powershell
git lfs install --local
git lfs pull --include="shared/models/notebook05-combat-20261006T140858_362365Z/**" --exclude=""
python -m pip install -e ".[ml]"
```

```python
from bvr_behavior_prediction.rl import load_trained_pilot

pilot = load_trained_pilot(
    "shared/models/notebook05-combat-20261006T140858_362365Z/pilot_005",
    device="cpu",  # CUDA is optional for inference.
)
# Supply the normalized 20 x 51 history produced by BluePilotEnvironment.
action = pilot.act(observation_history, deterministic=True)
```

The loader checks the checkpoint SHA-256 and skill/parameter ordering. These are
weights-only checkpoints for inference/evaluation; they contain no optimizer state
for exact training resumption. Running flights also needs the simulator dependencies
described in the [training guide](../../../docs/multi_pilot_training.md).

## Validation results

| Pilot | Selected epoch | Common return | Survival | Elimination |
| --- | ---: | ---: | ---: | ---: |
| pilot_001 | 75 | 133.51 | 42% | 66% |
| pilot_002 | 150 | 78.38 | 40% | 48% |
| pilot_003 | 90 | 138.67 | 58% | 54% |
| pilot_004 | 25 | 126.31 | 46% | 58% |
| pilot_005 | 140 | 169.46 | 46% | 70% |
| pilot_006 | 45 | 101.89 | 30% | 62% |
| pilot_007 | 135 | 159.42 | 54% | 66% |
| pilot_008 | 85 | 143.37 | 56% | 56% |
| pilot_009 | 135 | 163.26 | 42% | 74% |
| pilot_010 | 105 | 115.56 | 40% | 60% |

`pilot_005` has the highest common validation return in this population. This ranking
uses the same 50 validation scenarios that selected checkpoints; it is not an unseen
test or evidence that overfitting is absent. Pilot-specific selection returns use
different reward weights and should not be compared directly. Evaluate all pilots on
new seeds and geometries before drawing generalization or behavioural-style conclusions.
See [summary.json](summary.json) and each pilot's original manifest for full metrics.

## Provenance and reproduction

The [publication manifest](manifest.json) records hashes, seeds, the reviewed source
snapshot `37cab41c9952efe7a5187c735f46662149285c2b`, simulator provenance, and the LFS inventory.
Training used an uncommitted working tree. The source was committed afterwards, and
the run did not record a Git revision or source hashes; the publication snapshot is
therefore not independently verified as an exact training-time checkout.

The original per-pilot configs/manifests are preserved. Their absolute output and
MLflow paths refer to the training machine; the loader does not use those paths.
The top-level `reward` is the actual objective, overriding the legacy base values
under `training.reward`. For another run, use a new output directory. At publication,
notebook 05 defaulted to 50 fixed training scenarios; this run used `BVR_PILOT_SCENARIOS=75`.
The active notebook now uses a revised recipe with random reward ranges, refreshed
training scenarios and a separate final test. This population predates those changes.
All pilots used seed 7, 22 process workers, CUDA graph updates, 50 PPO passes per epoch,
and 64 transitions per minibatch. Full network and training settings are in each config.

## LFS footprint

The 10 checkpoints total **4,451,730 bytes (4.25 MiB)**; epoch logs add
**1,499,521 bytes**, for **5,951,251 bytes (5.68 MiB)** across 20 LFS
objects. The largest file is 445,173 bytes, below GitHub's 2 GB Free/Pro per-file limit.
The repository owner's remaining quota was not independently available for inspection.
This immutable batch excludes intermediate checkpoints, simulator logs, MLflow databases,
and replay/video corpora. Keep future releases in new versioned directories; replacing
or deleting files does not reclaim historical LFS storage automatically.

The selective pull above downloads only this population and avoids the separate
128 MiB transfer-test payload. Fresh clones normally download all current LFS files;
set `GIT_LFS_SKIP_SMUDGE=1` for cloning if selective download is desired. Every complete
download of this population consumes approximately 5.68 MiB of LFS bandwidth.
See the [sharing guide](../../README.md),
[GitHub per-file limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage),
and [LFS billing](https://docs.github.com/en/billing/concepts/product-billing/git-lfs).
