# RL and forecasting data handover

Verified on **4 November 2026**, for integration of `forecasting` into `main`.
All paths below are relative to the repository root. The existing local checkout
is `C:\Users\jttur\VSCodeProjects\Trajectory_Classification`.

## Start here

| What you need | Main location | Available from GitHub? |
| --- | --- | --- |
| Frozen tactical-adapted RL pilots used to create forecasting data | `artifacts/rl_pilot_experiments/20261010T200223_142625Z_tactical/training/` | Full run is local; the ten selected weights and frozen simulator are also in the corpus reproduction ZIP below |
| Raw forecasting encounters: tactical B01 observes trained A01 | [shared/corpora/trained-pilot-episodes-v1/](../shared/corpora/trained-pilot-episodes-v1/README.md) | Yes, Git LFS for large files |
| Opposite-view encounters: trained A01 observes tactical B01 | [shared/corpora/tactical-episodes-v1/](../shared/corpora/tactical-episodes-v1/README.md) | Yes, Git LFS; a different observation perspective |
| Prepared motion forecasting arrays and shared scenario split | `artifacts/motion_forecasting/data_v1/` | Local and ignored; rebuild from the raw corpus |
| Prepared skill forecasting arrays, labels and scaling | `artifacts/skill_forecasting/data_v1/` | Local and ignored; rebuild using the motion scenario split |
| Forecaster weights, histories and full search outputs | `artifacts/motion_forecasting/initial_v1/` and `artifacts/skill_forecasting/` | Local and ignored; transfer separately or retrain |
| Latest 22-slide presentation and study guide | [Skills forecasting revision](presentations/skills_forecasting_revision/README.md) | Yes, including the actual PDF |

**A merge or clone does not transfer `artifacts/`.** Code, configurations and
portable reports are ordinary Git files. Published large artifacts under
`shared/` use Git LFS. The prepared datasets and full local training directories
are deliberately ignored. Keep an independent copy of the local directories
listed here if retaining existing trained models and exact experiment histories.

## 1. RL training: where the pilots and training records live

RL training generates simulator rollouts online, updates the PPO policy and
discards the in-memory rollouts. For the tactical-adapted population there were
**10 pilots × 150 epochs × 75 training flights = 112,500 training flights**.
There is no saved dataset containing every one of those flight trajectories.
The later **3,000-flight corpus** is a separate collection using frozen policies.

The population registry is [configs/pilot_experiments.json](../configs/pilot_experiments.json).
The main preserved lineage is:

| Stage | Local population directory |
| --- | --- |
| Fresh-scenario combat baseline | `artifacts/rl_pilot_notebook/combat/20261007T212241_526315Z/` |
| Lower learning rate | `artifacts/rl_pilot_experiments/20261008T130810_285510Z_lr3e-05/training/` |
| Selected-skill probability correction; parent of adaptation | `artifacts/rl_pilot_experiments/20261009T153646_051001Z_selected_skill_parameters_recovery/training/` |
| **Tactical adaptation; forecasting teachers** | **`artifacts/rl_pilot_experiments/20261010T200223_142625Z_tactical/training/`** |

In the tactical job, each `training/pilot_001` through `pilot_010` contains:

- `best_model.pt`, `initial_model.pt`, `initialization.json`, `config.json` and `manifest.json`.
- `training_metrics.jsonl`: 150 epoch records with training/validation returns,
  losses and timings. `mean_return` is the validation return; check
  `evaluation_performed` before treating it as a new evaluation. There are
  31 recorded validation evaluations per pilot.
- `demonstrations/epoch_0030` through `epoch_0150`: five saved demonstration
  trajectories per pilot, including `trajectory.npz`, ACMI and summaries;
  **50 saved trajectories across the population**.
- `test_episodes.json`: historical evaluation records. These cases have already
  informed development and should not be called an untouched final test.

The job's parent directory also retains `experiment.json`, `status.json`, frozen
`source/`, `baseline_assessment/` and `adapted_assessment/`.

The [published October 6 population](../shared/models/notebook05-combat-20261006T140858_362365Z/README.md)
contains older weights and training logs. **It is not the tactical-adapted
population used for forecasting.** The latter's ten selected weights are
available in the trained-pilot corpus's `reproduction.zip`.

### Generating RL training runs

| Entry point | Purpose and output |
| --- | --- |
| [notebooks/05_train_multiple_pilots.ipynb](../notebooks/05_train_multiple_pilots.ipynb) | Train a new combat population; current defaults are ten pilots, 150 epochs and 75 fresh flights per epoch. Writes a new timestamped directory under `artifacts/rl_pilot_notebook/combat/`. |
| [scripts/run_pilot_experiment.py](../scripts/run_pilot_experiment.py) | Prepare and run matched learning-rate, selected-skill and other controlled interventions. See the [experiment workflow](multi_pilot_training.md#matched-ppo-experiments). |
| [scripts/run_tactical_adaptation.py](../scripts/run_tactical_adaptation.py) | Prepare a frozen job and adapt a completed selected-skill population against the tactical opponent. |
| [notebooks/06_train_multiple_pilots.ipynb](../notebooks/06_train_multiple_pilots.ipynb) | Separate hybrid-reward population; not the forecasting teachers' lineage. |

For a new tactical adaptation from the existing local parent population:

```powershell
$baselineBatch = 'artifacts/rl_pilot_experiments/20261009T153646_051001Z_selected_skill_parameters_recovery/training'
$adaptationPlan = .\.venv\Scripts\python.exe scripts/run_tactical_adaptation.py --prepare $baselineBatch --seed-offset 4000000 | ConvertFrom-Json
$adaptationScript = Join-Path $adaptationPlan.source 'scripts/run_tactical_adaptation.py'
.\.venv\Scripts\python.exe -u $adaptationScript --run-job $adaptationPlan.job
```

Preparation creates a new timestamped job; run its **frozen script**. The runner
requires CUDA. Its parent must be a complete training batch with `pilots.json`
and valid checkpoints/configurations. The raw-corpus reproduction ZIP is not
such a batch: it lacks full training history and `pilots.json`. Adaptation loads
the selected policy/critic weights and intentionally starts a fresh Adam optimizer;
the reproduction ZIP is not an optimizer-state training-resumption archive.

Defaults are 150 epochs and 300 assessment scenarios. The historical default
`--seed-offset 1000000` now fails the prior-confirmation overlap check in this
local checkout. The example uses **4000000**, checked against the current local
RL plans, registry and raw-corpus seed ranges at handover. Recheck for overlap
before using it for a later independent assessment; executing this example
makes those ranges part of a new experiment.
Current source is not automatically identical to a historical frozen snapshot;
the older October 6 run also used a different scenario/reward recipe.

[Notebook 07](../notebooks/07_tactical_adaptation.ipynb) and
[the population comparison notebook](../notebooks/05_compare_population_runs.ipynb)
read saved reports; **Run All does not launch training**. Further details:
[pilot experiment index](pilot_experiments.md) and
[multi-pilot training guide](multi_pilot_training.md).

## 2. Raw flight datasets: download, inspect and regenerate

The forecasting source is **`shared/corpora/trained-pilot-episodes-v1/`**:
scripted tactical **B01 observes trained A01**. It contains **300 shared scenarios
× ten frozen pilots = 3,000 encounters**, with 2,555,165 observation frames.
Scenario seeds are `3000000–3000299`; starting ranges are 18, 30 and 45 km,
100 scenarios each. Observations/tracks are recorded at 10 Hz, selected skills
normally at 1 Hz, with at most 120 seconds per encounter.

| File or directory | Contents and forecasting use |
| --- | --- |
| `observations/pilot_NNN.parquet` | The observer's 51 raw features, with episode, step and time keys |
| `tracks/pilot_NNN.parquet` | Aircraft physical tracks, velocity, alive status and launch information; the loader controls which causal fields enter features |
| `target_decisions/pilot_NNN.parquet` | Privileged A01 catalogue selections and parameters, used to construct supervised targets, **never model inputs** |
| `episodes.parquet` | Pilot/scenario/seed identifiers, checkpoint provenance, initial states and outcomes; metadata is not automatically an input feature |
| `config.json`, `schema.json`, `metrics.json`, `manifest.json` | Collection settings, field definitions, counts and file checksums |
| `reproduction.zip` | Frozen simulator/policy source and resources, ten adapted pilot checkpoints/configurations/manifests, collection scripts and `experiment.json` |

`shared/corpora/tactical-episodes-v1/` records the **opposite perspective**:
trained A01 observes scripted B01. Its `decisions/` records the learned
observer's own selections, unlike forecasting's `target_decisions/`.
Both corpora reuse matched scenario cases. If combined, both views and all
pilots for a scenario must remain in the same partition.

The published manifests describe 36 files each: **657,433,191 bytes** for the
trained-pilot corpus and **637,801,324 bytes** for the opposite view.
Both are materialized locally and both full checksum verifications passed at
handover. Collector working directories also remain locally at
`artifacts/trained_pilot_episode_data_v1/` and
`artifacts/tactical_episode_data_v1/`.

### Environment and selective download

Run commands from the repository root. Use Python 3.10 or newer and a project
environment with the ML/simulator dependencies. For a new environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[ml,video,dev]"
git lfs install --local
git lfs pull --include="shared/corpora/trained-pilot-episodes-v1/**" --exclude=""
# Optional opposite perspective:
git lfs pull --include="shared/corpora/tactical-episodes-v1/**" --exclude=""
```

The repository's dependency ranges are not an exact historical environment lock.
Consult the saved manifests/source for provenance; hardware and library changes
can affect numerical and bitwise reproduction. A normal Git checkout can contain
small LFS pointer files until `git lfs pull` materializes the data.

Verify the downloaded packages without training or collecting new flights:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/generate_tactical_corpus.py verify --perspective tactical
.\.venv\Scripts\python.exe -X utf8 scripts/generate_tactical_corpus.py verify --perspective learned
```

### Generate a new raw forecasting corpus

The entry point is [scripts/generate_tactical_corpus.py](../scripts/generate_tactical_corpus.py),
with [scripts/tactical_corpus_schema.py](../scripts/tactical_corpus_schema.py).
Its `--perspective` names the **observer**. Set **`tactical`** for forecasting;
the CLI default is `learned`, the opposite view.

To use the published historical collector and frozen pilots without the full
local RL job, extract the trained-pilot reproduction ZIP and use new destinations:

```powershell
Expand-Archive -LiteralPath shared/corpora/trained-pilot-episodes-v1/reproduction.zip -DestinationPath artifacts/trained-pilot-source-copy
$collector = 'artifacts/trained-pilot-source-copy/scripts/generate_tactical_corpus.py'
$corpusArgs = @(
    '--perspective', 'tactical',
    '--job', 'artifacts/trained-pilot-source-copy',
    '--work', 'artifacts/trained-pilot-collection-copy',
    '--output', 'artifacts/trained-pilot-package-copy',
    '--episodes-per-pilot', '300',
    '--seed', '3000000',
    '--workers', '8',
    '--device', 'cuda'
)
.\.venv\Scripts\python.exe -X utf8 $collector collect @corpusArgs
.\.venv\Scripts\python.exe -X utf8 $collector status @corpusArgs
.\.venv\Scripts\python.exe -X utf8 $collector package @corpusArgs
.\.venv\Scripts\python.exe -X utf8 $collector verify @corpusArgs
```

These commands perform frozen-policy inference, not PPO training. Choose unused
paths; compatible collection work can resume, while packaging protects an
existing publication. Output stays under ignored `artifacts/` until deliberately
published. `--device cpu` is supported for collection. `--limit` is only a smoke
test and does not produce the full corpus.

To use the existing local teacher job instead, invoke the repository collector
with `--job artifacts/rl_pilot_experiments/20261010T200223_142625Z_tactical`.
For opposite-view collection, set `--perspective learned` and distinct new work
and output paths. The trained-pilot ZIP includes both collection scripts; the
opposite-view ZIP includes source/weights but omits those scripts.

## 3. Prepared forecasting datasets and their generators

Both tasks use the trained-pilot raw corpus above. The authoritative existing
split is **`artifacts/motion_forecasting/data_v1/scenario_splits.json`**:
seed **1729**, **210 training / 45 validation / 45 reserved test scenarios**.
All ten pilots for a scenario stay together. Skill preparation explicitly
references this split; never independently random-split overlapping windows.

| Dataset | Local path | Prediction origins: train / validation / test |
| --- | --- | --- |
| Motion | `artifacts/motion_forecasting/data_v1/` | 147,718 / 31,029 / 32,457 |
| Skills | `artifacts/skill_forecasting/data_v1/` | 146,690 / 30,802 / 0 |

Motion test arrays exist but have not been evaluated. Skill preparation skips
reserved test row groups entirely. The different origin totals reflect task
target availability. Skill validation has **30,802 / 30,466 / 29,458** valid
labels at +1 / +2 / +5 seconds; comparisons must use the same masks.

| Prepared file | Motion shape | Skill shape |
| --- | --- | --- |
| `features.npy` (float32) | `(2555165, 54)` | `(2168263, 54)` |
| `anchors.npy` (int64) | `(211204,)` | `(177492,)` |
| `residuals.npy`, `constant_acceleration.npy` (float32) | Each `(211204, 3, 6)` | — |
| `valid.npy` (bool) | `(211204, 3)` | — |
| `targets.npy` (int64) | — | `(177492, 3)` |
| `current.npy` (int64; diagnostic label only) | — | `(177492,)` |

Both directories include `samples.parquet`, `scaling.json`, `data_manifest.json`,
`scenario_splits.json` and `split_assignments.csv`. Skills additionally has
`classes.json`, `alignment_audit.json`, `decision_support.csv`,
`label_support.csv` and `preparation_source/`.

The stored feature matrix has the 51 observer fields plus three causal
acceleration values. Improved skill loaders derive **37 additional causal
features** for 91 total. Histories are assembled from the frame matrix and
anchors; the base data is not stored as a separate 3D tensor per model.
All skill improvements still use **`data_v1`**, including v4 and v5. There is no
replacement `data_v4` or `data_v5`. Targets represent 33 catalogue choices;
unavailable future labels use `-1`. True current skill and pilot identity are
diagnostic metadata, not model inputs.

### Rebuild prepared data in fresh directories

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/train_motion_forecaster.py prepare --corpus shared/corpora/trained-pilot-episodes-v1 --data artifacts/motion_forecasting/data_handover
.\.venv\Scripts\python.exe -X utf8 scripts/train_skill_forecaster.py prepare --corpus shared/corpora/trained-pilot-episodes-v1 --reference-splits artifacts/motion_forecasting/data_handover/scenario_splits.json --data artifacts/skill_forecasting/data_handover
```

Use unused output directories. These preparation commands do not train models
or evaluate the reserved test. They read the published raw corpus; if using a
newly collected package, change both `--corpus` arguments explicitly.
To rebuild only skills against the original split, use
`--reference-splits artifacts/motion_forecasting/data_v1/scenario_splits.json`.

Entry points and default configurations:

- [scripts/train_motion_forecaster.py](../scripts/train_motion_forecaster.py),
  [motion_forecasting.json](../configs/training/motion_forecasting.json),
  [motion guide](motion_forecasting.md), [Notebook 08](../notebooks/08_motion_forecasting.ipynb).
- [scripts/train_skill_forecaster.py](../scripts/train_skill_forecaster.py),
  [skill_forecasting.json](../configs/training/skill_forecasting.json),
  [skill guide](skill_forecasting.md), [Notebook 09](../notebooks/09_skill_forecasting.ipynb).

## 4. Forecasting experiment outputs and continuation

| Stage | Preserved local results | Training/search entry point |
| --- | --- | --- |
| Initial motion GRU and Transformer | `artifacts/motion_forecasting/initial_v1/{gru,transformer}/` | [train_motion_forecaster.py](../scripts/train_motion_forecaster.py) |
| Initial skill frequency, GRU and Transformer | `artifacts/skill_forecasting/initial_v1/` | [train_skill_forecaster.py](../scripts/train_skill_forecaster.py) |
| v2 geometry/history study | `artifacts/skill_forecasting/improvement_v2/` | [improve_skill_forecaster.py](../scripts/improve_skill_forecaster.py) |
| v3 twelve-MLP ensemble | `artifacts/skill_forecasting/improvement_v3/` | [improve_skill_forecaster.py](../scripts/improve_skill_forecaster.py) for initialization/screening, then [ensemble_skill_forecaster.py](../scripts/ensemble_skill_forecaster.py) |
| v4 fifteen-member MLP/TCN reference | `artifacts/skill_forecasting/improvement_v4/` | [sequence_skill_forecaster.py](../scripts/sequence_skill_forecaster.py) |
| v5 six-Transformer primary and 75% Transformer / 25% v4 secondary blend | `artifacts/skill_forecasting/improvement_v5_transformers/` | [transformer_skill_forecaster.py](../scripts/transformer_skill_forecaster.py) |

The v3–v5 directories retain `inner_0/`, `inner_1/`, `refit/`, `validation/`,
`source/`, `protocol.json` and `frozen_selection.json`; v5 additionally records
`selection_receipt.json`. These contain candidate/refit weights, histories,
saved predictions, fitted preprocessing and selection evidence.

For continuation, preserve **whole run directories plus their prepared data**,
not only `best.pt`. v4/v5 require preserved earlier baseline artifacts and
cached predictions; downloading the raw corpus alone is insufficient to resume
those exact searches. Frozen protocols contain absolute paths and source/data
hash checks. Preserve them unchanged; use the matching layout and frozen source
for an existing run, or register a new experiment in a new output directory
when changing paths, inputs or code.

The committed [v4 source snapshot](experiments/skill_forecasting_v4/source_snapshot.zip)
and [v5 source snapshot](experiments/skill_forecasting_v5_transformers/source_snapshot.zip)
contain registered code and configuration, **not datasets or model weights**.
Portable reports remain in [v4](experiments/skill_forecasting_v4/README.md),
[v5](experiments/skill_forecasting_v5_transformers/README.md) and the
[presentation folders](presentations/skills_forecasting_revision/README.md).
Read the [improvement guide](skill_forecasting_improvements.md),
[sequence experiment guide](skill_forecasting_sequence_experiments.md) and
[Transformer experiment guide](skill_forecasting_transformer_experiments.md)
before starting a new search.

The current comparison remains **development validation**, with fixed scenario
groups, origin eligibility and horizon masks. Mean accuracy is 66.19% for v4,
65.88% for the pure Transformer ensemble and 66.26% for the secondary blend;
neither Transformer comparison establishes a significant accuracy gain.
These scores measure unseen scenarios for the same ten trained pilots; they do
not establish generalization to unseen pilots.
Preserve original results and the reserved test boundary. Features must use
only observations available at the prediction time; privileged target labels,
future rows, pilot/checkpoint identity and final outcomes must not enter inputs.
Fit normalization and select models using the appropriate training partitions.

## Handover verification

Locations, array headers, split manifests, checkpoint lineage, CLI arguments and
archive contents were inspected. Both raw-corpus checksum verifications passed
(36 files and 3,000 episodes each). The generation and training commands above
were checked against their implementations but **were not executed during this
handover**. No new training, data collection or reserved-test evaluation was
performed.
