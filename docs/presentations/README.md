# Population PPO and tactical adaptation

**Report formats:** [PDF slide deck](presentation.pdf) · [offline browser deck](presentation.html).
Download either file from the `multi-pilot-training` branch and open it locally.
The PDF provides a standalone slide deck. For HTML, use GitHub's **Download raw file** button;
GitHub shows its source rather than hosting the slideshow. Both files are standalone:
no Jupyter, Python, network connection, GPU, model weights or Git LFS are required.
In HTML, use arrow keys; **N** toggles supplementary notes and **F** enters fullscreen.

The deck contains **13 slides**. Slides 1–10 cover the methodology and principal results;
slides 11–13 cover observed behaviour, limitations and supporting evidence. The report
is generated from existing files without additional training, simulations or evaluation flights.

## Executive summary

The single-pilot PPO workflow was extended to ten independent pilots. Each has its
own reward coefficients and checkpoint, while sharing the same aircraft, skills
and scenario schedule. The model already includes an actor that chooses actions
and a critic that estimates future reward. Each pilot is trained in a separate
learning run using individual engagements.

Common combat outcomes provide a consistent basis for comparison. A clean win means
eliminating the opponent by missile while surviving. Each pilot’s own reward still
guides its learning, but it is not used to rank pilots against one another.

Against the simple scripted opponent, reducing the learning rate improved clean-win
performance. Correcting the PPO probability calculation to include only parameters
used by the selected skill produced a further improvement. A larger set of parameter heads underperformed;
a terminal clean-win reward bonus trial was stopped without evidence of improvement.
Those results were preserved, and the stronger shared-head population was retained.

Copies of those ten pilots received a further 150 training epochs against the scripted
tactical opponent. On separate matched confirmation scenarios, average clean wins
increased from **32.7% to 38.6%**, and survival from **75.1% to 83.8%**. Mutual
destruction increased slightly, indicating a tradeoff despite the improvements in
clean wins and survival. All ten runs completed. The original models remain unchanged.

## Evaluation results

### Tactical adaptation results

The completed run took approximately **12 hours 14 minutes**, including before/after
assessments and confirmation as well as gradient updates.

| Suite, same cases before and after | Clean wins | Survival | Mutual destruction | Clean-win change, paired 95% interval |
| --- | --- | --- | --- | --- |
| 300-case assessment / pilot, seed 1120000 | 31.73% → 38.67% | 76.80% → 87.23% | 4.00% → 3.93% | +6.93 pp [+3.87, +9.90] |
| Separate 100-case confirmation / pilot, seed 1220000 | **32.7% → 38.6%** | **75.1% → 83.8%** | **9.1% → 10.0%** | **+5.9 pp [+1.2, +11.0]** |

These are equal-weight means over ten pilots. On confirmation, losses without
elimination fell from **15.8% to 6.2%**. The common outcome score rose from **12.35
to 27.4**. Nine pilots improved their clean-win rate; one tied.

| Pilot | Confirmation clean wins before → after | Survival after | Selected additional epoch |
| --- | --- | --- | --- |
| 001 | 34% → 34% | 81% | 90 |
| 002 | 32% → 38% | 84% | 55 |
| 003 | 32% → 37% | 85% | 135 |
| 004 | 33% → 39% | 84% | 75 |
| 005 | **31% → 44%** | **86%** | 150 |
| 006 | 30% → 39% | 83% | 115 |
| 007 | 33% → 43% | 85% | 115 |
| 008 | 37% → 38% | 83% | 145 |
| 009 | 34% → 35% | 85% | 85 |
| 010 | 31% → 39% | 82% | 125 |

Pilot 005 has the highest **observed** confirmation clean-win rate, followed closely
by pilot 007. Pilot 010 leads the distinct 300-case assessment (43.33%). These rankings
are subject to uncertainty and do not establish universal superiority. The checkpoint
is selected by its own reward on validation, not its confirmation clean-win result.

### Shared development benchmark against the simple opponent

The principal comparison uses **the same 100 scenarios per pilot**, seeds
**211307–211406**, for every population below. All four runs retain the same ten reward
profiles, training seed **7**, 150-epoch scenario schedule and 50-case validation suite
starting at **111257**. Rates are equal-weight means over the ten pilots.

| Population | Clean-win rate | Interpretation |
| --- | --- | --- |
| Original, LR 0.0003 | **28.2%** | Starting population |
| Reduced LR 0.00003 | **36.8%** | +8.6 percentage points; retained |
| Reduced LR + selected-skill probabilities | **50.0%** | +13.2 percentage points; retained |
| Separate parameter heads per skill | **42.5%** | −7.5 percentage points versus selected-skill baseline; not retained |

The terminal +250 clean-win bonus trial completed only six pilots. Their matched
development comparison was **51.5% → 49.83%**; this subset must not be compared with
the ten-pilot population means as if the composition were unchanged.

These cases were originally stored as `test_episodes.json`. Once their results informed
experimental choices, they became a **development benchmark**. They are separate from
both training flights and checkpoint-selection validation, and are not an untouched
final test. Simple and tactical rates also represent different opponent conditions.
The original weight-sweep and relative-weight populations had validation-only results
and different reward setups.

### Recorded training and validation results

The learning-curve slide uses saved epoch records, without running new flights. Training
return is recorded on 75 fresh scenarios per epoch; validation return is recorded on
the fixed 50-case suite at epoch 1, every fifth epoch and the final epoch. Own validation
reward selects each pilot's checkpoint. Curves compare matching reward profiles across
runs; own reward is unsuitable for ranking pilots with different coefficients.

[learning_curves.csv](data/learning_curves.csv) provides all 6,000 recorded pilot-epochs
from the four complete populations, including training return, validation return when
measured, best validation return and actor/critic losses.
[development_summary.csv](data/development_summary.csv) provides selected epochs,
checkpoint hashes, recorded validation benchmark returns, survival and elimination,
and shared development outcomes for all 40 models. The historical validation records
do not contain joint clean-win counts; separate survival and elimination rates are
insufficient to reconstruct them.

### Historical confirmation evaluations

These completed evaluations are preserved as supplementary evidence, not used as the
main cross-experiment progress table. Each row paired its baseline and candidate on
the same cases, but the suites differ between rows.

| Intervention | Cases / pilot and seed | Clean wins before → after |
| --- | --- | --- |
| Reduced learning rate | 100, 420000 | 44.5% → 54.4% |
| Selected-skill probabilities | 100, 620000 | 42.4% → 56.0% |
| Separate parameter heads per skill | 300, 820000 | 57.97% → 50.70% |

These suites, including the tactical confirmation reported above, have already been
inspected. They cannot become untouched final tests again. Ongoing development should
use recorded training/validation outputs and fixed development cases. **Final testing
is reserved until model, reward and training choices are frozen**, using unused cases
shared across all final candidates. The completed experiment plans and results remain
historical records; this reporting change does not rerun or modify those experiments.

## Training and evaluation methodology

1. **Population initialization.** A reward profile is sampled for each pilot and
   remains fixed throughout a matched experiment. Each pilot has a separate model.
2. **Experience collection.** One scenario is one seeded 1-v-1 duel, with specified
   starting range, bearing, headings, speeds and altitudes. Each epoch has 75 fresh
   scenarios, each lasting at most 120 seconds. The seed schedule is shared across
   pilots. Their different actions lead to different flight trajectories.
3. **Observation and action selection.** Approximately every second, the policy
   receives 20 frames of 51 observation features (two seconds at 10 Hz), chooses one of 33 skills, and supplies
   the relevant numeric parameters. The skill and JSBSim controllers fly the aircraft.
4. **Reward calculation.** Locks, launches, evaded missiles, kills, crashes and being shot
   down contribute to the pilot's own reward. The reward function is not the neural
   network loss: it supplies the learning targets used to construct that loss.
5. **PPO optimisation.** Advantages estimate whether an action performed better than the
   critic expected. PPO makes better-than-expected actions more likely, with a clipped
   change to keep updates bounded. A minibatch is **256 decision transitions**, not
   256 scenarios. Up to ten shuffled update passes follow each collection, with a KL
   limit that can stop them early. CPU processes run simulation; CUDA runs neural work.
6. **Checkpoint selection.** Evaluation uses 50 fixed, separate validation cases at epoch 1,
   every fifth epoch and the end. The checkpoint with the highest own validation reward
   is selected. For tactical fine-tuning, the unmodified initial checkpoint is also eligible. All 150 epochs
   run, even if the selected checkpoint comes from an earlier epoch.
7. **Comparative evaluation.** Simple-opponent development uses the same fixed cases
   across experiments, alongside recorded training and validation outputs. The completed
   tactical experiment used a 300-case before/after assessment and a separate 100-case confirmation.
   Neither selects checkpoints. Fine-tuning starts from each pilot's actor **and critic**
   weights with a fresh Adam optimizer; original checkpoints are preserved.

The tactical opponent is a **scripted controller** that responds to incoming missiles,
supports its own missiles and attacks. It replaces the simple pursuit-and-fire policy
while keeping the aircraft and physics. This is **not self-play**. Scenario templates
may retain a generic `red_policy` label; the backend's explicit `opponent_type`, recorded
evaluation metadata and frozen plan determine the actual JSBSim opponent.

## Reward design, optimisation objectives and behaviour

The current population uses **NumPy seeded independent uniform sampling**, not Halton
or Hilbert sampling. Seed **42** reproduces the coefficients, subject to the same
generator implementation and sampling order. An arbitrary population size is supported.
The exact ten profiles are in [reward_profiles.csv](data/reward_profiles.csv).

| Reward coefficient | Sampling range |
| --- | --- |
| Crashed | −450 to −300 |
| Shot down | −450 to −300, independent of crash coefficient |
| Ground clearance | Fixed +0.02 per eligible decision |
| Acquire target lock | 0 to +2 |
| Fire with lock | 0 to +2 |
| Avoid incoming missile | 0 to +15 |
| Opponent destroyed | Fixed +250 |
| Fire without lock | −12 to −4 |

Ranges encode plausible priorities; they were not established as optimal by a search.
Fixed kill rewards and heavier death penalties discourage mutual destruction in the
terminal event subtotal. Other accumulated rewards still influence behaviour. Random
profiles cover tradeoffs without hard-coded style names, but may produce similar
policies or unhelpful combinations. They do not guarantee distinct aggression levels.
The separate terminal +250 clean-win bonus **is not part of tactical training**.

There are three different quantities in the reports:

- **Own reward:** each pilot's different coefficients, used for learning and validation
  selection. Comparing these raw returns between pilots is not a fair skill ranking.
- **Common benchmark reward:** one event formula applied to everyone. Reward-component
  tables show average **earned points per flight** (event counts × coefficients), not
  the original coefficients themselves.
- **Common outcome score:** +100 for a surviving missile elimination, 0 for survival
  without elimination, −50 for mutual destruction, −100 for own loss without opponent
  destruction. Clean-win, survival and mutual-destruction rates remain the clearer
  primary measures. Mutual destruction includes both aircraft dying even without a
  confirmed missile elimination.

The transformer has width 64, four attention heads and two layers. A skill head emits
33 probabilities, a shared Gaussian head predicts 16 numeric slots, and the value head
predicts future discounted reward. Only slots used by the chosen skill enter its PPO
action log probability. Adding per-skill heads was a separate architecture trial;
the retained population uses the shared head.

For a transition, the temporal-difference error is
`delta_t = scaled_reward_t + gamma * next_value_t - value_t` (with the terminal mask).
GAE sums these errors with decaying weights `(gamma * lambda)^k` to form advantage.
Here `gamma=0.99`, `lambda=0.95`; batch advantages are normalized. Rewards are scaled
by **0.01 for learning**, while reported returns retain original reward units.

Policy loss is the negative mean of PPO's clipped probability-ratio × advantage
objective. Positive advantage encourages the recorded action; negative advantage
discourages it. Value loss fits return targets; entropy encourages exploration.
The total is `policy_loss + 0.5 * value_loss - 0.01 * entropy`. Small or negative total
loss is not a clean-win score and is not evidence for or against leakage on its own.

The behaviour heatmap shows decision fractions, not episode percentages. For example,
001 often uses lag pursuit, 010 often turns cold, and 005 mixes beaming, cranking and
pursuit. This is observable variety, not proof that reward weights alone caused it.

## Generalisation, overfitting and evaluation limitations

There is no observed direct seed overlap among tactical training, validation,
assessment and confirmation. Checkpoints were chosen without the report suites.
Improvements on the separate confirmation suite support generalisation to those
cases. They do **not** prove absence of overfitting to a scripted opponent, simulator
or scenario generator. One training seed was used per reward profile. Bootstrap
intervals resample the **100 or 300 shared scenarios jointly across pilots**, using
2,000 draws and seed 8128; ten pilots do not make ten times as many independent cases.
These intervals omit training-seed variation. Retention against simple after tactical
adaptation has not been tested. No real-world performance claim follows from this.

Potential future work includes measuring simple-opponent
retention, evaluating broader geometries and opponent policies, and repeating training
seeds. Population opponents/self-play are another design option, but were not used
here. These investigations are outside the scope of the reported experiments.

## Notebook and module reference

| File | Role |
| --- | --- |
| [04](../../notebooks/04_train_rl_pilot.ipynb) | Original single-pilot actor–critic PPO workflow; already has a critic |
| [05 training](../../notebooks/05_train_multiple_pilots.ipynb) | Independent population, original combat-event reward family with varied coefficients |
| [05 comparison](../../notebooks/05_compare_population_runs.ipynb) | Read-only report; defaults to the completed selected-skill comparison |
| [06](../../notebooks/06_train_multiple_pilots.ipynb) | Preserved hybrid evade/pursue/eliminate reward alternative |
| [07](../../notebooks/07_tactical_adaptation.ipynb) | Read-only completed tactical report; embedded outputs are published |
| [multi_pilot.py](../../bvr_behavior_prediction/rl/multi_pilot.py) | Population orchestration, persistence, initialization and checkpoint selection |
| [pilot.py](../../bvr_behavior_prediction/rl/pilot.py) | Transformer actor/critic and selected-skill action probabilities |
| [trainer.py](../../bvr_behavior_prediction/rl/trainer.py) | Rollouts, advantages, PPO updates, batching and diagnostics |
| [reward.py](../../bvr_behavior_prediction/rl/reward.py) / [reward_population.py](../../bvr_behavior_prediction/rl/reward_population.py) | Event reward definitions and reproducible coefficient sampling |
| [evaluation.py](../../bvr_behavior_prediction/rl/evaluation.py) | Shared outcome and behaviour evaluation |
| [bvr_jsbsim_backend.py](../../bvr_behavior_prediction/rl/bvr_jsbsim_backend.py) | Aircraft simulation adapter and explicit opponent choice |
| [run_pilot_experiment.py](../../scripts/run_pilot_experiment.py) | Frozen, matched experiments and paired comparisons |
| [run_tactical_adaptation.py](../../scripts/run_tactical_adaptation.py) | Before/after tactical adaptation workflow |

Notebook 04 and 05 share the PPO/simulator foundations and original event-reward
family; they are not identical experiments. Population reward coefficients, training
budgets and later experiment settings differ. Notebook 07 reports the explicitly
recorded fine-tuning settings rather than changing 04 or 05's training cells.

## Evidence and reproducibility

- [results.json](data/results.json): settings, both tactical suites, all ten reward profiles,
  selected epochs, checkpoint hashes, learning curves, the shared development comparison
  and historical confirmation results.
- [learning_curves.csv](data/learning_curves.csv): 6,000 recorded epochs from four simple-opponent
  populations; missing validation values indicate epochs without validation.
- [development_summary.csv](data/development_summary.csv): 40 model summaries with
  checkpoint-selection validation statistics and shared development outcomes.
- [paired_outcomes.csv](data/paired_outcomes.csv): **4,000 paired rows** = (300 + 100)
  cases × 10 pilots. Each row records before/after outcomes for the same seed and pilot.
- [pilot_outcomes.csv](data/pilot_outcomes.csv): per-pilot before/after rates for both
  suites, provided in CSV format (rates are fractions, e.g. 0.44 means 44%).
- [source_hashes.json](data/source_hashes.json): SHA-256 of input reports used for the export.
- [checksums.json](checksums.json): checksums for the portable package files.
- [Exporter](../../scripts/build_pilot_presentation.py): `--collect` audits existing
  local results, verifies pairing, recomputes confidence intervals and writes this
  package. Without `--collect`, it renders from committed data. Neither mode imports
  training code, loads weights or runs simulation.

Local full artifacts remain under the two directories recorded in `data/results.json`:
the selected-skill recovery population and the tactical adaptation job.
These are ignored local experiment directories, not downloads included with this
presentation. The portable package intentionally contains results rather than model
weights. Historical MP4s in `shared/` depict earlier populations, not these tactical
pilots. Existing model/video LFS objects are not required to present this deck.

Publication checks: 114 relevant unit tests passed, using small fixtures rather than
real simulator runs. Three filesystem tests initially exceeded Windows path length
under the long audit directory; those three passed when rerun in a shorter workspace
directory. The read-only selected-skill, stopped-bonus and tactical reports executed
without cell errors, with their figures embedded. The export independently reproduced
both tactical bootstrap intervals and verified 150 finite-metric epochs per pilot,
all initial/selected checkpoint hashes, parent/source preservation and seed separation.
