# Pilot experiment index

Open [the comparison notebook](../notebooks/05_compare_population_runs.ipynb), select
the **Trajectory pilots (.venv)** kernel, and **Run All**. It only reads saved results.
The first code cell selects a named experiment from [the registry](../configs/pilot_experiments.json).
The default is `selected_skill_parameters`; choose `lower_learning_rate` to revisit
the completed comparison. Save the notebook after refreshing to keep its outputs.

## Current experiment: selected-skill parameters

Started 9 October 2026, using the completed lower-learning-rate population as baseline.
Ten pilots use 150 epochs each, at learning rate **0.00003**. The recovery run retains
pilots 001-005 unchanged and trains 006-010 from their original seeds.
Only the action probability and expected parameter entropy calculations change.
Network dimensions, skills, parameter bounds, rewards, seeds and simulator stay fixed.
The original 100 test cases provide a development comparison. Both populations also
receive 100 predeclared fresh confirmation cases (seed 620000).

A worker disconnected during pilot 006, collecting epoch 144. The first five pilots
were complete. The [failed attempt](../artifacts/rl_pilot_experiments/20261009T092543_792046Z_selected_skill_parameters/) remains intact, including its
143 completed epochs for pilot 006. Its weights-only checkpoint cannot resume the
optimiser, so pilot 006 starts again. The recovery adds two bounded batch retries
with restored random states; rewards, learning settings and evaluation cases are unchanged.

- [Experiment directory](../artifacts/rl_pilot_experiments/20261009T153646_051001Z_selected_skill_parameters_recovery/)
- [Settings and provenance](../artifacts/rl_pilot_experiments/20261009T153646_051001Z_selected_skill_parameters_recovery/experiment.json)
- [Live progress log](../artifacts/rl_pilot_experiments/20261009T153646_051001Z_selected_skill_parameters_recovery/stderr.log)
- [Job status](../artifacts/rl_pilot_experiments/20261009T153646_051001Z_selected_skill_parameters_recovery/status.json)
- [Population status](../artifacts/rl_pilot_experiments/20261009T153646_051001Z_selected_skill_parameters_recovery/training/pilots.json)
- [New models and epoch metrics](../artifacts/rl_pilot_experiments/20261009T153646_051001Z_selected_skill_parameters_recovery/training/)
- [Baseline models](../artifacts/rl_pilot_experiments/20261008T130810_285510Z_lr3e-05/training/)

The runner writes `comparison.html` and `comparison.json` in the experiment directory
after training, then adds fresh confirmation results. Each pilot directory contains
its selected model, configuration, manifest, epoch metrics and test outcomes.
Do not launch the same job again: training runs independently of the report notebook.

## Preserved populations

| Population | Models and results | Assessment |
| --- | --- | --- |
| 5 October: original weight sweep | [80-epoch run](../artifacts/rl_pilot_notebook/20261005T231808_624505Z/) | Validation only; earlier reward implementation |
| 6 October: varied relative weights | [150-epoch run](../artifacts/rl_pilot_notebook/combat/20261006T140858_362365Z/), [published models](../shared/models/notebook05-combat-20261006T140858_362365Z/) | Validation only; published historical population |
| 7 October: random reward ranges and fresh training batches | [150-epoch run](../artifacts/rl_pilot_notebook/combat/20261007T212241_526315Z/) | Original test mean clean wins: **28.2%** |
| 8 October: lower learning rate | [150-epoch run](../artifacts/rl_pilot_experiments/20261008T130810_285510Z_lr3e-05/training/), [comparison report](../artifacts/rl_pilot_experiments/20261008T130810_285510Z_lr3e-05/comparison.html) | Original test clean wins **36.8%**; fresh confirmation **54.4% versus 44.5%** for its baseline |
| 9 October: selected-skill probabilities | [Current run](../artifacts/rl_pilot_experiments/20261009T153646_051001Z_selected_skill_parameters_recovery/training/) | Results pending; use live status above |

Compare rates only within matching evaluation suites. The fresh confirmation cases
for the learning-rate experiment differ from its original test cases. Scenario
confidence intervals do not measure variability between training seeds.

Historical notebook outputs are in [notebooks/results](../notebooks/results/).
The completed learning-rate snapshot is `20261009_learning_rate_comparison.ipynb`.
Shared models and MP4 demonstrators remain under [shared](../shared/README.md).

## Repository layout

| Location | Purpose |
| --- | --- |
| `notebooks/04_train_rl_pilot.ipynb` | Original single-pilot training |
| `notebooks/05_train_multiple_pilots.ipynb` | Population training; now defaults to learning rate 0.00003 |
| `notebooks/05_compare_population_runs.ipynb` | Read-only comparison and progress report |
| `notebooks/06_train_multiple_pilots.ipynb` | Separate hybrid reward experiment |
| `scripts/run_pilot_experiment.py` | Prepare a frozen source snapshot and run one controlled intervention |
| `configs/pilot_experiments.json` | Names and relative paths for the report notebook |
| `artifacts/rl_pilot_notebook/` | Preserved notebook populations |
| `artifacts/rl_pilot_experiments/` | Full controlled experiments |
| `artifacts/archive/20261009/` | Temporary checks, helper scripts, verification and interrupted runs |

The cleanup archived files rather than deleting results. See the
[move manifest](../artifacts/archive/20261009/cleanup_manifest.json) for original and
archived paths. Completed run directories remain at their recorded paths; all old
models, reports and source snapshots remain available. Archived diagnostic metadata
may still contain its historical absolute paths. The `artifacts` directories are
local and ignored by Git; they are not automatically downloaded with the repository.

For launch commands and objective details, see the
[matched experiment workflow](multi_pilot_training.md#matched-ppo-experiments) and
[selected-skill objective](multi_pilot_training.md#selected-skill-parameter-objective).
