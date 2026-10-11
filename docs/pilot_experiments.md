# Pilot experiment index

The [offline PDF/HTML deck and technical report](presentations/README.md)
provide portable access to the experiment results.
These include the completed results and need no notebook runtime or model downloads.
Open [the executed tactical report](../notebooks/07_tactical_adaptation.ipynb) for
the full saved tables. Its **Run All** only reads existing results on the training PC.

For earlier experiments, open [the comparison notebook](../notebooks/05_compare_population_runs.ipynb), select
the **Trajectory pilots (.venv)** kernel, and **Run All**. It only reads saved results.
The first code cell selects a named experiment from [the registry](../configs/pilot_experiments.json).
The default is `selected_skill_parameters`; choose `skill_conditioned_parameters`,
`clean_win_bonus` or `lower_learning_rate` to revisit other comparisons.
Save the notebook after refreshing to keep its outputs.

## Completed experiment: tactical adaptation

Completed **11 October 2026 at 09:16 BST**, after about 12 hours 14 minutes including
all assessments and fine-tuning. All ten pilots completed **150 additional epochs**;
all epoch metrics are finite, with **zero simulator-worker retries**. The original
51 baseline files and all frozen source files passed SHA-256 checks. Each initial
checkpoint matches its own parent; each selected checkpoint matches its manifest.

| Matched tactical suite | Clean wins before → after | Survival before → after | Mutual destruction before → after |
| --- | --- | --- | --- |
| 300-case assessment, seed 1120000 | 31.73% → 38.67% | 76.80% → 87.23% | 4.00% → 3.93% |
| Separate 100-case confirmation, seed 1220000 | **32.7% → 38.6%** | **75.1% → 83.8%** | 9.1% → 10.0% |

Confirmation clean wins improved **5.9 percentage points**, paired scenario bootstrap
95% interval **+1.2 to +11.0 points**. Nine pilots improved and one tied. Pilot 005
has the highest observed confirmation clean-win rate (**44%**, survival **86%**);
pilot 007 is close at 43%. Pilot 010 leads the different 300-case assessment. These
rankings do not establish a statistically superior universal champion.

Each suite shares cases across pilots. Its 1,000 or 3,000 flights are not that many
independent scenarios. Bootstrap intervals resample scenarios jointly across pilots
and do not measure training-seed uncertainty. Neither suite selected checkpoints.
Retention against the simple opponent after adaptation has **not** been evaluated.

Launched **10 October 2026 at 21:02 BST** from the ten completed selected-skill pilots.
First, each frozen pilot faces the same **300 tactical scenarios** (seed 1120000).
Only after all ten assessments finish does the job fine-tune independent copies for
**150 additional epochs per pilot**, with **75 fresh scenarios per epoch**, **22 CPU
simulator workers**, CUDA and the existing **0.00003 learning rate**.

Each copy loads its own policy and critic weights. Adam starts fresh because the
parent checkpoints contain weights only. Reward profiles, network architecture,
skills, aircraft physics and PPO settings are preserved. The opponent changes from
`simple` to `tactical`; all simulated opponent labels reflect that selection.
The initial checkpoint remains eligible for selection, using each pilot's own
tactical validation reward on **50 separate cases** (seed 1100000). Fine-tuning
training seeds start at 1000007 and refresh each epoch.

After training, the selected models repeat the 300-case tactical assessment.
A separate **100-case confirmation** (seed 1220000) compares the parent and adapted
populations against tactical. Neither assessment nor confirmation determines
checkpoint selection. Results on the simple opponent belong to the earlier reports
and must not be directly compared with tactical outcomes as a measure of improvement.

- [Tactical report notebook](../notebooks/07_tactical_adaptation.ipynb)
- [HTML outcome table](../artifacts/rl_pilot_experiments/20261010T200223_142625Z_tactical/tactical_report.html)
- [Frozen plan](../artifacts/rl_pilot_experiments/20261010T200223_142625Z_tactical/experiment.json)
- [Stage status](../artifacts/rl_pilot_experiments/20261010T200223_142625Z_tactical/status.json)
- [Evaluation progress](../artifacts/rl_pilot_experiments/20261010T200223_142625Z_tactical/stdout.log)
- [Training progress](../artifacts/rl_pilot_experiments/20261010T200223_142625Z_tactical/stderr.log)
- [Tactical starting results](../artifacts/rl_pilot_experiments/20261010T200223_142625Z_tactical/baseline_assessment/)
- [Adapted models and epoch metrics](../artifacts/rl_pilot_experiments/20261010T200223_142625Z_tactical/training/)

The preparation/runner is `scripts/run_tactical_adaptation.py`. Its frozen copy
finished this one population and comparison. Notebook 07
only reads saved results. The original population and all earlier attempts remain
intact. This is checkpoint fine-tuning against a scripted opponent, not self-play.

## Stopped experiment: terminal clean-win bonus

Stopped **10 October 2026 at 21:02 BST** to redirect compute to tactical adaptation.
Pilots 001-006 completed, and pilot 007 completed 38 epochs; all saved files remain.
Across those six matching pilots, development clean wins were **49.8% versus 51.5%**
for the baseline, with survival **63.8% versus 68.3%**. Confirmation did not run, so
this is an incomplete experiment, not a final population comparison.

Launched **10 October 2026 at 12:46 BST**, using the successful **selected-skill population with
a shared parameter predictor** as baseline. The reward intervention adds **250
points once at episode end for a confirmed elimination while surviving**. Original
event coefficients and death penalties remain unchanged. Mutual destruction,
survival without elimination, and an opponent crash earn no bonus.

Recovered **10 October 2026 at 20:48 BST** after Windows denied replacement of
`training/pilots.json` at 19:13 BST, just before pilot 007 started. Pilots 001-006
had completed all training and evaluation; the recovery copied their files unchanged
and restarted at pilot 007. Manifest replacement now retries brief Windows
access/sharing failures for up to about six seconds, preserving the previous complete
file if access remains denied. Rewards, seeds, models, PPO settings and evaluation
suites are unchanged; the [original attempt](../artifacts/rl_pilot_experiments/20261010T114552_188643Z_clean_win_bonus/)
remains intact. The repair and population recovery checks passed (45 tests).

The run was configured for ten fresh pilots, each with **150 epochs and 75 scenarios per epoch**,
at learning rate **0.00003**, using **22 simulator workers and CUDA**. Skills,
simulator physics, default opponent, architecture and PPO settings remain matched.
The existing validation selection procedure uses each pilot's own reward including
the bonus. Common benchmark returns exclude the bonus for both populations.

The planned confirmation would have given both populations **300 new matched scenarios
(seed 920000)**. Earlier confirmation suites, including the skill-conditioned
experiment's cases, are reserved and cannot be reused. Adoption requires at least
a **3-point clean-win gain**, a positive paired 95% confidence interval, at most a
**2-point survival drop** and at most a **2-point mutual-destruction increase**.
The latter margins use point estimates; training-seed variation remains unmeasured.

- [Frozen plan and criteria](../artifacts/rl_pilot_experiments/20261010T194825_532536Z_clean_win_bonus_recovery/experiment.json)
- [Job status](../artifacts/rl_pilot_experiments/20261010T194825_532536Z_clean_win_bonus_recovery/status.json)
- [Epoch progress log](../artifacts/rl_pilot_experiments/20261010T194825_532536Z_clean_win_bonus_recovery/stderr.log)
- [New population and models](../artifacts/rl_pilot_experiments/20261010T194825_532536Z_clean_win_bonus_recovery/training/)
- [Preserved baseline](../artifacts/rl_pilot_experiments/20261009T153646_051001Z_selected_skill_parameters_recovery/training/)
- [Exact reward changes](../artifacts/rl_pilot_experiments/20261010T194825_532536Z_clean_win_bonus_recovery/reward_differences.json)

Select `clean_win_bonus` in the comparison notebook to read this stopped job. **Run All** only refreshes
saved progress and results; it does not start training. Own returns and critic
losses use different reward targets across these populations, so judge improvement
by the common outcome tables. This job is stopped; refreshing the report does not
restart it.

## Completed experiment: skill-conditioned parameters

Completed **10 October 2026 at 12:37 BST**, before the request to stop it. All ten
pilots and both confirmation populations finished; its models and results are retained.
Fresh confirmation clean wins fell **57.97% to 50.70%** (**-7.27 points**, paired
95% interval **-9.23 to -5.53**). Survival rose **70.93% to 71.93%** and mutual
destruction rose **20.13% to 20.87%**. The experiment failed its clean-win criteria;
the selected-skill population remains the baseline for the next trial.

The [completed notebook snapshot](../notebooks/results/20261010_skill_conditioned_comparison.ipynb)
contains all tables and seven embedded figures. Its job's `plots/` directory also
contains PNG/PDF exports. These results use seed 820000, so its 57.97% baseline rate
must not be compared directly with the 56.0% measured on the earlier 620000 suite.

Launched **10 October 2026 at 01:02 BST** against the completed selected-skill
population. Each skill now has its own continuous parameter predictor and standard
deviations. The original predictor is copied to every skill at initialization,
preserving the initial action distribution, backbone, skill selector and critic.
Training can then specialise parameters independently for each skill.

This was one matched training run per reward profile: **ten pilots, 150 epochs each,
75 fresh scenarios per epoch, learning rate 0.00003, 22 simulator workers and CUDA**.
Rewards, PPO objective and coefficients, checkpoint selection by own validation
reward, simulator and simple opponent remain matched. Models train from scratch;
the baseline checkpoints remain available and unchanged.

After the original development comparison, both populations faced **300 fresh matched
confirmation scenarios (seed 820000)**, giving 3,000 flights per population.
The predeclared adoption criteria are a population mean clean-win gain of at least
**3 percentage points**, a paired 95% scenario-bootstrap interval entirely above
zero, no more than a **2-point survival drop**, and no more than a **2-point mutual
destruction increase**. The latter margins use point estimates. The report records
every check and recommends retaining the baseline if any fails. One training seed
per profile still leaves training-seed uncertainty unmeasured.

The job ended after this comparison without launching follow-up training.

- [Frozen plan and criteria](../artifacts/rl_pilot_experiments/20261010T000147_332865Z_skill_conditioned_parameters/experiment.json)
- [Job status](../artifacts/rl_pilot_experiments/20261010T000147_332865Z_skill_conditioned_parameters/status.json)
- [Epoch progress log](../artifacts/rl_pilot_experiments/20261010T000147_332865Z_skill_conditioned_parameters/stderr.log)
- [Population status and model directories](../artifacts/rl_pilot_experiments/20261010T000147_332865Z_skill_conditioned_parameters/training/)
- [Preserved baseline](../artifacts/rl_pilot_experiments/20261009T153646_051001Z_selected_skill_parameters_recovery/training/)

Select `EXPERIMENT = "skill_conditioned_parameters"` in the comparison notebook to
revisit this completed job. `comparison.json` and `comparison.html` contain the
full decision and paired evaluations. No restart is required.

## Completed experiment: selected-skill parameters

Completed **9 October 2026 at 22:12 BST**, using the completed lower-learning-rate
population as baseline. All ten pilots finished 150 epochs at learning rate **0.00003**.
The recovery retained pilots 001-005 unchanged and trained 006-010 from their original seeds.
Only the action probability and expected parameter entropy calculations changed.
Network dimensions, skills, parameter bounds, rewards, seeds and simulator stay fixed.
The original 100 test cases provide a development comparison. Both populations also
received 100 predeclared fresh confirmation cases (seed 620000).

Fresh confirmation mean clean wins increased **42.4% to 56.0%**: **+13.6 percentage
points**, with a paired scenario bootstrap interval of **+9.8 to +17.4**. Eight pilots
improved, pilot 006 tied, and pilot 010 regressed by six points. Survival was nearly
unchanged (**70.7% to 70.9%**); mutual destruction decreased **21.8% to 20.7%**.
The improvement mainly reflects more eliminations while surviving. These intervals
measure scenario uncertainty; repeat training seeds before claiming a robust improvement.

The [completed notebook snapshot](../notebooks/results/20261010_selected_skill_comparison.ipynb)
contains all final tables and seven embedded figures. Standalone **PNG and PDF**
exports are in the local [plots directory](../artifacts/rl_pilot_experiments/20261009T153646_051001Z_selected_skill_parameters_recovery/plots/), including
[population outcomes](../artifacts/rl_pilot_experiments/20261009T153646_051001Z_selected_skill_parameters_recovery/plots/population_outcomes.png)
and [paired pilot improvements](../artifacts/rl_pilot_experiments/20261009T153646_051001Z_selected_skill_parameters_recovery/plots/paired_clean_win_changes.png).

A worker disconnected during pilot 006, collecting epoch 144. The first five pilots
were complete. The [failed attempt](../artifacts/rl_pilot_experiments/20261009T092543_792046Z_selected_skill_parameters/) remains intact, including its
143 completed epochs for pilot 006. Its weights-only checkpoint cannot resume the
optimiser, so pilot 006 restarted. The recovery added two bounded batch retries
with restored random states; no further worker failures were recorded. Rewards,
learning settings and evaluation cases were unchanged.

- [Experiment directory](../artifacts/rl_pilot_experiments/20261009T153646_051001Z_selected_skill_parameters_recovery/)
- [Settings and provenance](../artifacts/rl_pilot_experiments/20261009T153646_051001Z_selected_skill_parameters_recovery/experiment.json)
- [Training log](../artifacts/rl_pilot_experiments/20261009T153646_051001Z_selected_skill_parameters_recovery/stderr.log)
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
| 9 October: selected-skill probabilities | [Completed run](../artifacts/rl_pilot_experiments/20261009T153646_051001Z_selected_skill_parameters_recovery/training/), [saved report](../notebooks/results/20261010_selected_skill_comparison.ipynb) | Original test clean wins **50.0% versus 36.8%**; fresh confirmation **56.0% versus 42.4%** |
| 10 October: skill-conditioned predictors | [Completed run](../artifacts/rl_pilot_experiments/20261010T000147_332865Z_skill_conditioned_parameters/training/), [saved report](../notebooks/results/20261010_skill_conditioned_comparison.ipynb) | Original test clean wins **42.5% versus 50.0%**; fresh confirmation **50.70% versus 57.97%** |

Compare rates only within matching evaluation suites. The fresh confirmation cases
for the learning-rate experiment differ from both its original test cases and the
selected-skill confirmation suite. Its earlier 54.4% and current 42.4% baseline results
come from different cases, not a changed baseline model. Scenario
confidence intervals do not measure variability between training seeds.

Historical notebook outputs are in [notebooks/results](../notebooks/results/).
The completed learning-rate snapshot is `20261009_learning_rate_comparison.ipynb`;
the selected-skill snapshot is `20261010_selected_skill_comparison.ipynb`.
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
