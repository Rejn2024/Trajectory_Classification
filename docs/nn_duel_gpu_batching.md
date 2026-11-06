# Shared GPU rollout inference and estimated runtime

The updated `notebooks/11_nn_vs_nn_training_data.ipynb` serves independent CPU/JSBSim
worlds from one GPU actor process. The new default work/output paths are
`artifacts/nn_duels_batched_v1` and `shared/corpora/nn-duels-batched-v1`. The original
validation sample is preserved. The full 10,000-flight campaign has not been
launched in the cloud; it is intended for the user's compute.

## Expected benefit

A small three-flight CPU profile, with each flight reaching 300 game seconds,
spent 4.16 seconds in NN decisions out of 33.40 summed episode wall seconds:
approximately **12.5%**. Physics, sensors and weapons dominated. Eliminating the
entire inference cost would give about **1.14x** throughput on this profile.
GPU offload may therefore provide a modest improvement, or lose to IPC/batching
overhead. This is a noisy CPU profile, not a CUDA performance measurement.

The default `BACKEND="auto"` benchmarks local CPU inference and, when CUDA is
available, shared GPU inference at two worker counts, then selects the fastest
successful whole-flight configuration. At least thirty full-duration-limited
flights per candidate cover all scenario families; use 100+ for a steadier estimate.
Every candidate uses the same cases and seeds. Benchmark data stays separate from
the training corpus. Failed GPU candidates appear in the table; auto mode may
retain CPU inference. An existing campaign keeps its recorded backend.

Set `BACKEND="central_cuda"` to force CUDA. `central_cpu` exercises the same shared
owner path for tests and is not recommended for production. Central batching uses
deterministic policies; stochastic collection remains available with `local_cpu`.

## Implementation

Workers send both aircraft histories, each 20-by-51, once per game second over CPU
pipes. One owner gathers ready requests within a configurable 2 ms batching window.
`torch.func.stack_module_state` and `vmap` evaluate the ten independent actor weights
in one stacked forward. Padding is ignored when returning ordered requests. The
policies are never averaged, and histories never cross engagements.

Only the actor encoder, skill logits and parameter means are evaluated; the unused
critic/log-probability calculations are omitted. Workers never initialize CUDA or
load CUDA models. The owner caches the weights and uses FP32 with TF32 disabled.
Guidance, simulator cadence, sensors/weapons, first-defeat termination and both
recording perspectives retain their original contracts.

At each owner startup, eighty histories across all ten actors are compared against
CPU reference logits/means with absolute and relative tolerances of `2e-5`; selected
skills must agree. Failure aborts that candidate. Small arithmetic/batch-shape
differences can accumulate in flight trajectories, so GPU flights are not promised
to be bit-identical across hardware or worker counts. Source/software, backend,
precision and batch settings are recorded in the immutable campaign configuration.
Dispatch sizes, padding and inference time are saved in `inference_metrics.json`.

## Estimated complete duration

For the **10 physical core / 16 thread Intel CPU and RTX 5080 reported in 09b**,
budget roughly **4–8 hours end to end**, assuming about **20 Mbps sustained upload**.
This is a provisional planning range. The old cloud serial benchmark took 80.75
seconds for ten flights, mean duration 222.56 game seconds. An idealized extrapolation
to eight effective workers gives about 2.8 hours for acquisition before storage
and network overhead. GPU offload may reduce that modestly; JSBSim physics stays
on CPU. CPU speed, thermal limits, flight durations and bandwidth can move the
actual time outside this range.

The notebook replaces this extrapolation with measured whole-flight throughput
and measured packaging/verification time. It separately budgets repeated prefix
checksum reads, per-snapshot Git overhead and payload upload. Defaults of 400 MB/s
for reads, ten seconds per Git snapshot and 20 Mbps upload are explicit, editable
assumptions. The sensitivity range varies compute/disk costs and is not a confidence
interval. Live collection cycles update the ETA using actual collection,
verification and upload time.

Expect about 10–15 GiB of compressed payload and reserve 40–55 GiB free for work,
publication and LFS objects. Unchanged semantic checks use receipts keyed by actual
payload hashes, configuration and episode index rows. Every payload byte is still
checksummed; publication verifies each snapshot once, and final verification uses
no receipt.

## Validation

- Twenty-four collector, actor and schema tests passed. The CUDA-specific test
  was skipped because the cloud environment has no GPU.
- Tests cover independent policies, uneven padding, request order, decoded
  actions, exact future masks, verification receipts and estimate arithmetic.
- The real ten-pilot stacked actor passed its eighty-history CPU parity check;
  maximum absolute error was about `1.43e-6`.
- All notebook cells executed through the shared owner path on CPU: one- and
  two-worker benchmark candidates, four short engagements, runtime estimate,
  packaging, verification and the 09b-compatible feature/target example. These
  short probes validate orchestration, not 300-second-flight throughput.
- CUDA numeric parity and speed remain to be checked on the user's GPU. The
  notebook performs both checks before choosing that backend.

No extra package beyond the existing ML/simulator dependencies is required. Use
a CUDA-enabled PyTorch build supporting the GPU. Changing source, software,
backend or batch settings requires a fresh campaign; the historical validation
data remains verifiable with its archived collector.

## Human-readable notebook progress

`PROGRESS_INTERVAL_SECONDS` defaults to **300 seconds**, exposed in the first cell
and optionally overridden by `BVR_NN_PROGRESS_SECONDS`. A single reporting clock
spans collection chunks, packaging, verification and LFS uploads. Long quiet stages
still receive keepalives. Benchmark probes are labelled separately from main flights.

Each update shows physical flights saved/target and percentage, aircraft perspectives,
elapsed collection time, observed end-to-end rate and ETA, current stage, CPU/GPU
backend, configured worker count, failed flights in the latest batch, and last
verified/pushed backup with its age. An unavailable ETA or a command still awaiting
its first completed-flight record is stated explicitly. Start, final completion
and failures appear immediately; intermediate backup boundaries do not flood output.
Completion is reported only after the final full verification passes.

Raw child stdout/stderr goes directly to uniquely named files under
`WORK/notebook_logs/`, including Git/LFS diagnostics. Human-readable updates also
have a persistent transcript there. A command failure prints its log location and
a short tail immediately. Notebook interruption terminates the command's own process
group, including its workers, without targeting unrelated processes.

Reporting runs outside the flight processes. A single blocked waiter thread in the
notebook process detects command completion; the notebook otherwise sleeps until
the next report. It reads the existing small `progress.json` only on heartbeats and
collection-command completion. It adds no worker messages, trajectory scans,
checksum passes, GPU polling or CUDA synchronization. Missing or temporarily unreadable
counters retain the last known count. Optional transcript-write failures do not stop
the collection. The flight collector and its saved source hashes are unchanged by
this reporting update, so existing campaign resumption is preserved.

Progress validation: eleven reporting tests cover cadence, quiet stages, benchmark
isolation, missing counters, raw-output logging, immediate failures/completion and
interruption cleanup. The combined collector/actor/schema/reporting suite passed
35 tests with one CUDA-only skip. All notebook cells also executed with a one-second
test interval using the existing four-flight sample, exercising final verification
and confirming raw command JSON stays out of notebook output. No new flights were
generated for that notebook check, and the full campaign remains unstarted here.

## Fix for frozen simulator import failures (2026-11-06)

The Windows failure `Wrong simulator source imported` occurs during worker/owner
initialization. Local CPU collection then reports `BrokenProcessPool`; shared CUDA
collection reports `Central inference startup failed`. The final notebook error
`Every rollout backend failed` is the consequence of those startup failures.

Prepending the snapshot to `sys.path` cannot override packages already cached in
`sys.modules` or an editable-install finder earlier in `sys.meta_path`. The collector
now activates `nn_duel_source.py` inside each isolated worker and inference owner.
Its finder resolves both `bvr_sim` and `bvr_behavior_prediction`, including their
submodules, exclusively from the frozen snapshot. Cached copies of those packages
are cleared first. Missing snapshot modules raise instead of falling through to an
installed package; unrelated dependencies retain normal import resolution. The
existing policy/simulator origin checks remain in place and report actual paths.
The helper is included in the collector hashes and archived reproduction sources.

The notebook reuses successful benchmark measurements and retries failed candidates.
Failure rows and the final exception retain command log paths and output excerpts.
Setup directories now include the collector source hash, as benchmark directories
already did, so a failed pre-fix setup cannot block the repaired collector.

After updating, restart the notebook kernel and Run All. The reported failure
occurred before preparation of the main campaign, so the default paths can be
reused. If `WORK/config.json` already exists for an older main campaign, select fresh
`BVR_NN_WORK` and `BVR_NN_OUTPUT` paths first: the source hash change deliberately
prevents combining collector versions in one immutable campaign. Keep prior
collections and published manifests intact.

Validation: the original initializer fails three installed-package collision cases;
the repaired initializer passes all four combinations of cached packages and
priority editable-install finders. The collector/actor/schema/progress/import and
benchmark retry suites pass **41 tests**, with one CUDA-only skip. Real four-flight,
eight-game-second runs pass with two simulator workers for both `local_cpu` and
`central_cpu`; the shared owner passes its 80-history frozen-actor parity check.
All notebook code cells also pass with four short flights, one/two-worker shared
CPU benchmarks, final full verification and publication disabled. This check
retains a deliberately stale pre-fix setup and uses the new source-keyed setup.
Windows and CUDA execution still require validation on the user's machine.
