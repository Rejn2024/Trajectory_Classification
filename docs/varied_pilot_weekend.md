# Continue notebook 30 and collect a 91-feature NN duel corpus

[Notebook 31](../notebooks/31_continue_varied_pilots_and_generate_corpus.ipynb)
runs the full sequence through
[`run_varied_pilot_weekend.py`](../scripts/run_varied_pilot_weekend.py):

1. Validate all ten completed notebook-30 pilots, checkpoint hashes, catalogue,
   observation contract, dependencies and requested device.
2. Train every pilot for **200 additional epochs against simple opponents**.
3. Assess every selected pilot on **300 shared tactical cases**.
4. Train every pilot for **300 additional epochs against tactical opponents**.
5. Repeat the matched tactical assessment using the adapted selected policies.
6. Freeze the exact adapted checkpoints, source and simulator assets; choose an
   inference backend; collect **10,000 NN-versus-NN flights** and fully verify them.

Both training stages and their validation flights now run for **up to 300 in-game
seconds**, retaining earlier simulator termination on defeat. Validation runs
**after every epoch** against the corresponding opponent, using the fixed stage
validation scenarios. Improved selected weights are saved after each improvement.

Install the repository's `ml,video` extras in the selected notebook kernel before
Run All. The notebook sets `BVR_WEEKEND_SOURCE` to
`<checkout>/artifacts/rl_pilot_simplified/varied_transformers/20261112T232045_249258Z`,
the batch named in notebook 30's completed training output. It resolves this path
relative to the checkout on Windows and Linux and respects an existing environment
override. The later `20261113T195250_365660Z` timestamp appears in setup output,
not the completed training result. Point `SOURCE` or `BVR_WEEKEND_SOURCE` elsewhere
if the batch was copied. It must contain `pilots.json` and all ten pilot directories.
These trained artifacts are Git-ignored and must exist on the execution machine;
the saved notebook results do not include the weight files. No old 51-input
checkpoints or freshly initialized pilots are substituted for missing weights.

The default campaign directory is
`artifacts/rl_pilot_simplified/varied_weekend_v2/`. This separates the new duration
and validation schedule from earlier campaigns. Use a fresh JOB/OUTPUT for this
version; 120-second validation scores must not select 300-second checkpoints.
The default corpus directory is
`shared/corpora/nn-duels-varied-v1/<job-directory-name>/`. Use `BVR_WEEKEND_JOB` and
`BVR_WEEKEND_OUTPUT` to choose other locations. Notebook 31 does not commit or push
data. Back up these directories using your normal storage workflow.

## Preserved training recipe and selection

Each pilot's saved width, layer/head counts, selected-skill parameter contract,
learning rate, PPO settings, value coefficient, entropy coefficient, training
reward scale and complete hybrid reward definition are loaded from its files.
The existing `PPOTrainer._update` implements the unchanged notebook-30 objective:
`policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy`.
The roster is never resampled. The notebook displays the effective per-pilot
architecture, coefficients and source scenario budgets before training.

The continuation explicitly sets `episode_duration_s=300` and
`evaluation_interval=1`; `BVR_WEEKEND_EPISODE_SECONDS` supports short installation
checks. The simulator's own maximum step count receives the same duration, so
raising the wrapper limit cannot leave an unnoticed 120-second native cutoff.
Source reward parameters remain unchanged, including the reward recipe's saved
threat-exposure normalization time (120 seconds for the recorded source batch).

Each stage starts from the previous stage's validation-selected policy **and
critic**, using fresh Adam because notebook 30 saved inference weights. This
matches notebook 07's stage-boundary initialization. The initial checkpoint is
eligible for selection on the new stage's own validation scenarios. The selected
checkpoint may therefore remain the initial policy if later updates do worse.

Training seeds are offset from the original source seeds by one million for
simple continuation and two million for tactical adaptation. Validation seeds
are 400,000,000 and 500,000,000; the separate matched assessment starts at
600,000,000. Backend benchmarks use 710,000,000; the corpus defaults to 800,000,000.
Seed ranges are checked for overlap with the source training, validation and test
ranges and between data roles. Pilots within a stage intentionally share cases.
The saved choice of fixed or resampled training scenarios is preserved.

The tactical assessment does not select checkpoints. It compares survival,
confirmed elimination, clean wins and mutual destruction on matched cases.
After inspection it is a development comparison. Raw training rewards and PPO
losses have different meanings between pilots with different objectives.

## Unattended operation and restart

Keep the machine awake. Run All uses a child process, records full logs under
`JOB/logs/`, and displays tqdm-style notebook bars for the current pilot, overall
training epochs and final NN collection. The bars refresh every two seconds and
show elapsed time, rate, current phase, loss/return and ETA forecasts. They count
committed epochs and resume at saved counts. `JOB/training_progress.json` supplies
the live phase and measurements; `status.json` identifies the overall stage.

Pilot forecasts use the most recent five completed epoch timings. Overall
training forecasts use observed averages for each stage; an unmeasured tactical
stage initially uses the simple-stage average. The first epoch reports
"estimating" until measurements exist, and an expired forecast is labelled
"re-estimating". These are approximate training-computation forecasts, excluding
setup, assessments and final collection; architecture differences, checkpoint I/O
and variable flight lengths can change actual wall time. Collection has its own
ETA based on measured flights/hour. Widget support comes from the `ipywidgets`
video extra, with tqdm's standard text fallback where needed.

CPU simulator processes remain warm across
epochs; policy inference and PPO use CUDA when available. The worker default is
80% of logical CPUs, capped at 32, with configurable Torch CPU threads. These are
concurrency limits, not reserved CPU capacity.

Every epoch atomically commits `resume.pt`: current model, best model, Adam,
gradient scaler, Python/NumPy/Torch/CUDA RNGs, training-scenario schedule position,
validation scores and metrics. An interruption restores this state and repairs
derived metrics/selected-weight files. It does not count a partial epoch as
complete. The new runner records JSONL metrics locally rather than starting
MLflow runs. `best_model.pt` remains separate from the current training state and
is rewritten on validation improvement, plus recovery if a crash left it stale.
`best_epoch` and `best_model_improved` identify selection events in the JSONL
metrics; `evaluation_performed` is one for every epoch. Initialization still
preserves `initial_model.pt` and evaluates it as a selection candidate.

Score-only validation uses the exact deterministic actor choices but skips
critic values, Gaussian log probabilities and stored training trajectories.
Training histories are released after PPO, before validation. CUDA training
rollout outputs use one combined device-to-host transfer instead of four per
decision. The existing batched policy, fused CUDA Adam, mixed precision and CUDA
graph update settings remain available. Epoch/scenario counts, PPO update passes,
minibatch size and loss/reward coefficients are not reduced by these changes.
`rollout_seconds`, `update_seconds`, `evaluation_seconds` and the recorded training
collection profile expose the remaining costs.

Completed pilots and assessments are reused. Each completed flight writes
checksummed files before its completion marker. Transport failures retry the
original batch; collection retains successful flights and retries pending cases
at most twice per invocation. Persistent errors stop with `failure.json` and a
traceback. Advisory locks are released by the OS when a process exits, including
after a crash. A still-running process retains its lock.

Restart the kernel and Run All using the **same JOB and settings**. The source
batch is pinned after the first preflight; later runs do not silently select a
newer batch. The campaign checks source/assets, parent checkpoints, configuration
and software versions. Restore the recorded environment to resume it, or choose
new JOB and OUTPUT directories for a changed experiment. The final archive also
rejects code changes made during a running campaign.

On a CUDA machine, `auto` benchmarks local CPU inference and central CUDA
inference on separate flights. The central inference owner groups pilots by
architecture, including attention-head counts even when weight shapes match,
keeping their weights independent and restoring original request
order. CPU-reference parity is checked for every group before collection. The
measured faster backend is saved. Without CUDA, `auto` uses local CPU inference;
`central_cpu` exercises the grouped service for installation checks. Simulation
processes stay warm for the entire collection; raw per-flight files are durable
before the final packaging pass.

## Final data contract

Both aircraft are controlled by the ten **tactical-selected** policies. The 100
ordered pairs, including self-pairs, receive 100 distinct physical geometries
each at the default 10,000-flight budget. Notebook 11's stratified geometry
sampler and reset checks are reused. Flights stop at 300 seconds or first defeat,
including a crash. All outcomes are retained. Requested speed control remains
enabled, as on this branch's training backend.

For each physical flight there are two indexed perspectives. Each side has
`observations`, `tracks` and `target_decisions` Parquet shards, with one flight per
row group and 100 flights per shard by default. `episodes.parquet` retains the
pilot identities, selected checkpoint hashes, geometry group, duration and
survival outcomes. Keep both perspectives in the same future dataset split.

| Columns | Actual network features |
| --- | --- |
| `obs_00`–`obs_46` | Compact own/enemy aircraft state and four friendly missile slots |
| `obs_47`–`obs_86` | Four nearest live hostile missiles targeting this observer, ten values each |
| `obs_87`–`obs_90` | Own kinetic/potential energy, target kinetic/potential energy |

`schema.json` names all 91 features and records their divisors/formulas. Threat
slots call the **same extraction function as training**: presence, relative NWU
position/velocity, range, signed closing speed and seeker state; sorted by range
then UID, with zero padding. Inputs are simulator truth, including missiles whose
seekers are inactive. They are not radar/MWS-gated measurements.

Observations and tracks are sampled at 10 Hz, including reset and terminal
frames. Decisions occur at 1 Hz before the next physics step. A decision consumes
the preceding 20 chronological frames including the current frame, repeating
reset on the left as necessary. No additional terminal decision is generated.
`target_decisions` describes the **other aircraft**, whose input observations are
in the opposite perspective. Its `policy_history_sha256` records the exact
20-by-91 float32 input used for inference. Final verification reconstructs and
checks every such history, in addition to checksums, geometry, cadence and counts.

`tactical_reproduction.zip` includes exact code, simulator assets, configurations,
all ten adapted checkpoints and campaign/source provenance. There is no fitted
preprocessing, split or forecast-window construction. Existing loaders/models
that hard-code 51 columns need to be configured for the expanded feature set.

## Short execution check

The notebook's final cell documents environment overrides for one epoch per
stage, three one-second training/validation scenarios, three tactical assessment
cases, and three three-second NN flights. Choose fresh smoke JOB/OUTPUT paths.
This still exercises all ten pilots. Clear the overrides and use fresh output
paths for production. A short check establishes execution, not learned flight
quality. The full 5,000 additional epochs and 10,000 flights may take longer than
a weekend depending on hardware; the saved epoch timings show actual throughput.

## Validation

The focused backend, PPO, process-worker, collection, inference, runtime,
speed-control, continuation and progress suite passed **151 tests**, with
**12 CUDA checks skipped** on the CPU environment.
Recovery tests inject a failure after an epoch commit and reproduce the
uninterrupted policy, selected weights and Adam state exactly; a killed-process
test also verifies automatic lock recovery.

Selection tests cover both opponents, validation at every epoch, saving only
strict improvements and recovery after a committed epoch. Score-only validation
matches complete validation actions and returns, including skill-conditioned
parameter heads. Progress tests cover resumed counts, phase changes, ETA aging,
unknown/expired estimates and collection progress.

A paired CPU benchmark used three 30-second JSBSim validation scenarios and
three repetitions per path. Parent Python allocation peaks fell from about
0.83 MB to 0.15 MB (82%) with identical returns. Median wall times were both about
2.9 seconds; this check does not establish an overall CPU speedup. CUDA transfer
latency improvements require measurement on CUDA hardware. Separate real-JSBSim
level-flight duration checks, with combat suppressed to avoid early defeats,
reached exactly 300 seconds, 300 decisions and 3,000 simulator steps through both
backend opponent configurations.

Every notebook cell also ran against real JSBSim with all ten 4/8-head pilots,
two additional epochs per stage, three one-second training/validation cases and
three three-second NN flights. All 40 epochs recorded fresh validation. The
executed tqdm widgets reached 40/40 training epochs and 3/3 collection flights;
the six perspectives passed full 91-feature/history/geometry verification.
Running every cell again preserved all 38 selected-checkpoint/raw-flight Parquet
hashes and retained exactly two epochs per pilot per stage. Notebook schema,
cell compilation, Ruff and `git diff --check` passed.

Full production training/collection and execution on CUDA hardware were not run.
