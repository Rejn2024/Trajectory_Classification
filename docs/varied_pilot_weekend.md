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

Notebook setup now defaults to `BVR_WEEKEND_MODE=new`: each new kernel allocates
unique timestamped `varied_weekend_v3_...` directories under
`artifacts/rl_pilot_simplified/` and `shared/corpora/nn-duels-varied-v1/`.
It ignores old `BVR_WEEKEND_JOB`/`BVR_WEEKEND_OUTPUT` environment pins in this mode,
so a stale campaign manifest cannot block a whole new run. Old jobs and corpora
remain intact. New training starts from the notebook-30 source checkpoints.
Rerunning setup in the same kernel retains the newly allocated paths; restarting
the kernel starts another fresh campaign. The paths are printed before preflight.

To select explicit paths or resume after a kernel restart, set
`BVR_WEEKEND_MODE=resume`, `BVR_WEEKEND_JOB`, and `BVR_WEEKEND_OUTPUT` if customized,
before setup. All pinned source/settings checks still apply to an existing job;
no manifest is deleted or weakened to make an incompatible resumption succeed.
Notebook 31 does not commit or push data. Back up campaign/output directories
using your normal storage workflow.

## Paired training schedule

Notebook 31 and the campaign CLI now default to `paired` scheduling. Within
**each** simple and tactical stage, pilots run in the fixed order **1&2, 3&4,
5&6, 7&8, 9&10**. Both members must finish before the next pair starts. The
existing stage boundaries remain: all simple training, the first matched tactical
assessment, all tactical training, the second assessment, then corpus collection.

Each pilot runs in a separate interpreter with its own policy, critic, optimizer,
RNG state, CUDA context, cached graphs and simulator pool. A start rendezvous lets
both interpreters initialize before releasing the pair. Each pilot remains
on-policy: its next training batch follows its own completed PPO update. Processes
can overlap simulation, validation and GPU updates; this does not guarantee GPU
kernels in different contexts will execute simultaneously on every platform.

`--workers` / `BVR_WEEKEND_WORKERS` is a **total** simulator budget for the active
pair. Fourteen means **seven per pilot**; twelve means six each. Odd totals give
the first pilot one extra worker. Paired mode requires at least two workers.
Notebook defaults cap this budget at fourteen while reserving two logical CPUs
when possible: a 16-logical-CPU machine gets 14, an 8-CPU machine gets 6. Notebook
Torch defaults to two threads in the coordinator, giving one Torch CPU thread to
each pilot process. `--pilot-schedule serial` or
`BVR_WEEKEND_PILOT_SCHEDULE=serial` selects the sequential reference behavior.

Children write only their own pilot checkpoints, metrics, execution records,
phase records and failures. Only the coordinator writes population manifests and
global progress. The pair bar reports both phases and combined committed epochs;
ETA uses observed wall-time throughput rather than adding overlapping epoch times.
Per-pilot stdout/stderr logs live under `JOB/logs/`. `pair_events.jsonl` records
ordering, PIDs, worker allocation and pair wall time. Pair failure interrupts the
peer and closes its nested simulator processes. Completed pilots and committed
epochs are retained for a restart with the same paired settings.

Use a **fresh campaign** for this scheduling change. The older performance-repair
migration described below does not authorize converting serial jobs to this new
schedule or changing worker allocations. New-kernel Run All already selects fresh
directories. Seven workers per pilot is a reasonable trial on a 16-logical-CPU
host, not a guarantee of higher throughput or sufficient RAM/VRAM; measure 12 versus
14 total workers on the target machine.

### All-up execution test

`tests/test_paired_campaign.py` exercised the complete campaign on Linux CPU with
real JSBSim and a generated ten-pilot **test population**, not the unavailable
production weights. It used two epochs per pilot per stage, three three-second
training/validation cases per epoch, three matched assessment cases per pilot,
and three three-second NN corpus flights. Two total simulator workers were used
to fit the test environment. The test verified:

- All ten pairs (five per stage) ran in order, with actual execution intervals
  overlapping within each pair and no overlap between successive pairs.
- All **40 epochs** validated, both assessment passes completed, and the final
  **three flights/six perspectives** passed the corpus verification pipeline.
- Rerunning the completed campaign preserved selected-model and Parquet hashes
  and did not add training epochs.

The initial all-up test plus budget/progress checks passed in **203 seconds**.
The final build's all-up, budget, progress and peer-failure tests passed
**4 tests in 249 seconds**; separate notebook/progress checks passed 17 tests and
the broader affected regression run passed 61 tests (three CUDA checks skipped).
A separate serial reference for pilots 1 and 2, using the same one-worker-per-pilot
allocation, produced **bit-identical current/selected policy tensors, Adam state,
Torch RNG state and selection scores in both simple and tactical stages**.
Additional checks cover peer cancellation and aggregate progress counts. These
tests establish execution/restart correctness; they do not establish training
quality or an epoch/campaign speedup on Windows/CUDA. The 14-worker production
configuration has not been load-tested on that hardware.

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
training epochs and final NN collection. Widgets refresh every two seconds and
show elapsed time, rate, current phase, loss/return and ETA forecasts. They count
committed epochs and resume at saved counts. `JOB/training_progress.json` supplies
the live phase and measurements; `status.json` identifies the overall stage.

On Windows, an atomic JSON replacement can briefly deny a progress read. The
notebook skips unreadable or incomplete progress snapshots and retries on the
next poll, including its final refresh. This keeps a display-file lock from
terminating the campaign or hiding the child process's actual error and log.
Required checkpoint and campaign-manifest validation retains its strict reads.

Pilot forecasts use the most recent five completed epoch timings. Overall
training forecasts use observed averages for each stage; an unmeasured tactical
stage initially uses the simple-stage average. The first epoch reports
"estimating" until measurements exist, and an expired forecast is labelled
"re-estimating". These are approximate training-computation forecasts, excluding
setup, assessments and final collection; architecture differences, checkpoint I/O
and variable flight lengths can change actual wall time. Collection has its own
ETA based on measured flights/hour. Widget support comes from the `ipywidgets`
video extra, with a plain text event fallback where needed.

The fallback now prints plain progress events instead of terminal animation.
Unchanged two-second polls emit nothing; a phase change updates the pilot line,
and the overall line updates only when its committed epoch count or stage changes.
Widget displays still refresh in place every two seconds. Rendering is controlled
in one place, so a count update cannot first draw a second line with an old postfix.

CPU simulator processes remain warm across
epochs; policy inference and PPO use CUDA when available. The CLI worker default is
80% of logical CPUs, capped at 32; notebook paired defaults are described above. These are
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

### Investigating the notebook-31 slowdown

At branch revision `bc01bf4`, the saved campaign cell contained **20,018 overall
bar redraws for only 394 distinct completed-epoch counts**, in 3.16 million text
characters. The output reports `IProgress not found`; `tqdm.auto` therefore used
terminal animation. `CampaignProgress.refresh()` forcibly redrew both bars every
two seconds even when nothing changed. These were display updates, not repeated
training epochs. Cursor-up/carriage-return operations also explain stale loss
and ETA fragments appended to the overall line. The event fallback removes this
growing notebook/browser/serialization workload. The old retained campaign output
has been cleared from the edited notebook; the original evidence remains in Git.

The slowdown is not monotonic: differences between saved completed-epoch elapsed
times put pilot 001 at about **250 seconds/epoch during epochs 11–100**, then
**43 seconds/epoch during epochs 116–200**. Pilot 002 averages about **58
seconds/epoch during epochs 11–100**. Polling gives approximately two-second
resolution and the first interval includes initialization. The corrupted bar
postfixes cannot reliably attribute this time to individual training phases.

Two simulator resource issues are repaired:

- A `NoneType` type annotation imported `torch._C` into every simulator worker,
  despite policy inference taking place in the parent. Using `str | None` removes
  that dependency. A real fresh-process simulator test checks that stepping a
  flight does not load Torch.
- `BVR3DEnv.close()` was empty. Aircraft/enemy/sensor reference cycles retained
  their native JSBSim engines until cyclic GC. `JSBSimFDM.close()` now releases
  both references to its native engine; environment close and reset call it.
  Reset also clears completed missiles from the previous flight. With cyclic GC
  disabled, repeated close/reset tests verify immediate native-handle release
  and unchanged seeded initial observations.

An isolated Linux CPU probe creating 100 two-aircraft environments reduced final
process RSS from about **316 MB to 64 MB** after both repairs. With GC disabled,
20 closed environments previously retained 40 FDMs with live native engines;
the repaired path retains zero native handles even while Python cycles remain.
RSS includes loaded libraries and allocator caching. This establishes resource
cleanup and lower per-process memory here, not a measured speedup of the original
Windows/CUDA training campaign.

An epoch is also a variable amount of computation. The source recipe uses **60
PPO passes and minibatches of 64**, processing every collected decision. If all
60 flights survive for 300 one-second decisions, PPO performs
`60 * ceil(18,000 / 64) = 16,920` minibatches in one epoch. Longer survival can
increase simulation and update time substantially without a throughput regression.
The existing saved `rollout_transitions` and `update_minibatches` distinguish that
case; it cannot be decided from the bar output alone. No production checkpoints
or raw JSONL metrics from the Windows machine are included in this checkout.

The notebook now has a read-only timing table, grouped into ten-epoch median
windows. Compare decision/minibatch counts against milliseconds per decision and
per minibatch. New rows also record validation decision/game-time counts,
validation policy/simulator timings, PPO KL and CUDA allocated/reserved/peak
memory. KL diagnostics do not change the optimizer updates. For a **new JOB and
OUTPUT**, optional `BVR_WEEKEND_TARGET_KL=0.02` / `--target-kl 0.02` enables the
trainer's existing KL early stopping (threshold `1.5 * target`) instead of always
using all inherited passes. It changes the training recipe and is not enabled
automatically for the existing campaign. Validate its flight quality separately.

Completed pilots and assessments are reused. Each completed flight writes
checksummed files before its completion marker. Transport failures retry the
original batch; collection retains successful flights and retries pending cases
at most twice per invocation. Persistent errors stop with `failure.json` and a
traceback. Advisory locks are released by the OS when a process exits, including
after a crash. A still-running process retains its lock.

For resumption after a kernel restart, select `BVR_WEEKEND_MODE=resume` and the
**same JOB, OUTPUT and settings** before Run All. The source
batch is pinned after the first preflight; later runs do not silently select a
newer batch. The campaign checks source/assets, parent checkpoints, configuration
and software versions. Restore the recorded environment to resume it, or choose
new JOB and OUTPUT directories for a changed experiment. The final archive also
rejects code changes made during a running campaign.

For the progress-file permission repair of paired jobs created at `7b3916f`,
restart the kernel and set `BVR_WEEKEND_MODE=resume` and `BVR_WEEKEND_JOB` to the
saved job before setup. Retain the same OUTPUT, source and experiment settings.
Resume setup already enables `--accept-performance-fix` unless
`BVR_WEEKEND_ACCEPT_PERFORMANCE_FIX=0` is set. This repair accepts the known
LF/CRLF hashes of only the runner and progress module from that revision,
records the upgrade in `JOB/performance_fix.json`, and preserves saved training
state. Fresh mode starts a new campaign instead.

**Historical serial repair only:** the following source-upgrade procedure applied
before paired scheduling was introduced; paired mode needs a fresh campaign.
To resume a campaign pinned to `bc01bf4` or `26b3446` with that earlier repair revision, stop its old process
and restart the kernel before Run All. Notebook setup enables this known
operational repair by default (`ACCEPT_PERFORMANCE_FIX=True`); set
`BVR_WEEKEND_ACCEPT_PERFORMANCE_FIX=0` to retain strict source rejection. The CLI equivalent is
`--accept-performance-fix`. Preflight takes the campaign lock and checks the
known LF/CRLF hashes of the ten operational repair files. All other source/assets,
parent checkpoints, software and experiment settings must still match exactly.
The original plan and an appended upgrade history are retained in `JOB/performance_fix.json`
and included in the final reproduction archive. Committed model, optimizer,
scaler, RNG and epoch state are preserved. The flag may remain set after migration;
repeated preflight is idempotent. Changing KL or other budgets still needs a new
experiment directory.

### Epoch acceleration

CUDA action selection now combines distribution validity with the existing
action-output transfer, avoiding five Python tensor-boolean checks per stochastic
rollout batch. Invalid probabilities/scales receive safe sampling inputs on the
device, and the original validity flag rejects the batch on the CPU before actions
reach a simulator. This prevents a CUDA sampling assertion from poisoning the
device. Valid probabilities and samples retain their values and RNG consumption.
Score-only validation uses the same combined validation/transfer approach.

One full-minibatch forward/backward CUDA graph is cached per trainer. Later epochs
reuse it while tensor shapes/dtypes/devices/strides, parameter and buffer storage,
training mode, attention heads, action parameterization, loss coefficients and AMP
scale storage match. A changed scale **value** at the same address is read by the
graph; replacement scale storage triggers recapture. Small eager batches preserve
the cache, and eager remainder gradients are replaced by graph-owned buffers before
replay. Invalidated graphs and trainer shutdown explicitly release captured
storage. `cuda_graph_cache_hit` and `cuda_graph_capture_seconds` expose reuse.

Process workers use a parent-owned shared-memory array for observation histories.
Only actions and step metadata cross pipes after reset/attachment. The parent
copies independent snapshots before the next command so training histories cannot
be overwritten by later simulator steps. Mappings are reused across matching
batches and resized after worker acknowledgments when layouts change. The owner
closes/unlinks mappings after normal shutdown and transport failures. Default
`auto` falls back to pipes if allocation is unavailable or observations are not
supported numeric histories; explicit `shared_memory` requires that transport.
`training_simulator_shared_memory` and its evaluation counterpart record usage.

`PilotTrainingConfig.simulator_transport`, CLI `--simulator-transport`, and notebook
`BVR_WEEKEND_SIMULATOR_TRANSPORT` accept `auto`, `shared_memory`, or `pipe`. Omit
the notebook override to retain compatibility with an existing pinned job. A
transport-only benchmark can be run without training or production checkpoints:

```bash
python scripts/benchmark_pilot_transport.py --workers 2 --scenarios 60 --decisions 100 --repeats 5
```

On this Linux CPU environment, alternating five repetitions measured median
transport times of **0.1015 seconds with pipes** and **0.0503 seconds with shared
memory** for 100 batches of sixty 20-by-91 float32 histories, including independent
parent snapshots. This is about **2.02x** for synthetic communication. It excludes
worker startup, physics, policy inference and PPO and is not an epoch speedup.

Epoch timing now includes both durable restart and selected-weight writes.
`compute_seconds` and `checkpoint_seconds` split those costs; `epoch_seconds`
includes both. `checkpoint_timing_recorded=1` identifies a complete measurement.
A crash immediately after committing `resume.pt` can leave a compute-only row
with flag zero, since a checkpoint cannot record its own write duration. Completed
JSONL timings are recovered on resume only when their epoch and non-timing fields
match the committed checkpoint. Uncommitted/mismatched rows are discarded. The
notebook table shows complete timing counts and excludes unknown checkpoint
durations from checkpoint medians.

For a **new JOB and OUTPUT**, the notebook now accepts
`BVR_WEEKEND_MINIBATCH_SIZE=256` / CLI `--minibatch-size 256`, alongside
`BVR_WEEKEND_TARGET_KL=0.02` / `--target-kl 0.02`. Both default to inheritance.
At the 18,000-transition limit, increasing minibatches from 64 to 256 reduces the
60-pass maximum from 16,920 to 4,260 optimizer steps; it changes gradient averaging
and optimization, so this is an explicit experiment rather than an automatic
resumption change. KL stopping may reduce passes further. Keep these settings
unchanged to resume that experiment and compare flight quality on matched cases.

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

### Performance repair verification

The repair/backend/PPO/process/progress suite passed **94 tests**, with **11 CUDA
checks skipped**. In particular, 1,000 unchanged progress polls redraw each text
view once, phase changes do not redraw the overall view, widget polling remains
live, and text events contain no terminal cursor-control codes. LF/CRLF campaign
upgrade tests preserve saved state and provenance, reject changes to budgets,
KL, checkpoints and unrelated source, and refuse a locked campaign.

A two-epoch real-JSBSim/process-worker/PPO execution recorded all new phase/workload
and KL fields, validated every epoch, and preserved model tensors and epoch counts
when rerun from its completed restart checkpoint. Notebook cells compile; Ruff
on the changed training/progress/test files and `git diff --check` pass.

Related pilot/notebook checks passed another **106 tests**. Three existing notebook
checks fail on both this checkout and an unmodified `bc01bf4` worktree: the training
notebook CPU-default and batch-size assertions, and notebook 30's assertion that
executed cells have no saved execution count. Those notebooks were not changed.
The original Windows/CUDA campaign and its raw metrics/checkpoints remain
unavailable here; its end-to-end speedup has not been measured.

### Acceleration verification

The main relevant suite passed **172 tests**, with **20 CUDA checks skipped**.
After the final provenance and unsupported-observation refinements, the focused
recovery/action/transport suite passed **62 tests**, with **12 CUDA checks
skipped**, including four additional buffer/fallback cases. Coverage includes
exact pipe/shared policy, Adam and RNG parity; independent observation snapshots;
layout reuse/resizing; failure cleanup; allocation fallback; invalid layout
rejection; seeded deferred-validation parity; and graph-cache invalidation.
Cached CUDA replay, AMP scale changes and invalid-input device recovery have
CUDA-conditional tests, but were not executed on this CPU environment.

A real JSBSim/process-worker run completed two additional epochs with three
ten-second training and validation flights per epoch, recording shared-memory
usage, KL, compute and checkpoint timings. Restarting the completed run preserved
model tensors and epoch counts. Tests with delayed checkpoint writes verify that
epoch timing includes those writes; crash/restart tests retain unknown timings
honestly and recover complete timings without accepting uncommitted metrics.
Notebook cells compile, changed-file Ruff and `git diff --check` pass. Production
Windows/CUDA training and its epoch speedup remain unmeasured.
