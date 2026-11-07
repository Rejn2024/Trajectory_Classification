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

## Windows OpenMP startup failure (2026-11-06)

`OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already
initialized` is a native-library abort. The subsequent `BrokenProcessPool` and
`OSError: handle is closed` messages are consequences of dead workers. Reducing
worker counts does not repair two OpenMP runtimes inside each process. A common
cause is a Conda MKL/Intel OpenMP installation combined with a pip PyTorch wheel
that supplies another Intel OpenMP DLL; the log alone does not identify which
packages supplied the two copies.

The collector initializes Torch's CPU runtime before importing NumPy, Pandas,
Arrow or the simulator on Windows, including when Windows spawn imports the
collector in a new worker/owner. The inference module and notebook also load
Torch before NumPy. The notebook first probes the native dependency mix in a
separate process, before importing scientific libraries into the kernel or
launching a pool. It exercises Torch, NumPy and SciPy BLAS matrix operations and
OpenCV, and imports the simulator's runtime dependencies. Logs contain the kernel
executable, candidate Intel OpenMP DLL paths, import stages, and (on success)
loaded native thread-pool paths. A native abort is retained as the original
error in a `WORK/notebook_logs/native_runtime_*.log` and stops setup immediately.
If the Windows kernel/IDE already imported NumPy, the disposable probe also
initializes NumPy before Torch, matching the existing kernel instead of assuming
that restarting a cell can change its native-library load order. This can require
a clean environment even when a fresh Torch-first worker probe succeeds.
The standalone check is:

```powershell
python scripts/nn_duel_runtime.py check
```

Restart the kernel after updating. Import order fixes the startup collision when
both libraries can reuse the first runtime; it cannot repair incompatible DLLs
in an installed environment. If the probe still aborts, create a clean environment
instead of deleting/renaming DLLs or setting `KMP_DUPLICATE_LIB_OK=TRUE`. That flag
suppresses the runtime's protection and can leave crashes or incorrect results.
For the reported RTX 5080, this is an example using CUDA 12.8 PyTorch wheels:

```powershell
conda create -n trajectory-nn-clean python=3.12 pip -y
conda activate trajectory-nn-clean
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[ml,video]"
python scripts/nn_duel_runtime.py check
python -m ipykernel install --user --name trajectory-nn-clean --display-name "Trajectory NN clean"
```

Run these from the repository root. Install the numerical/ML packages with pip in
this new environment so NumPy/SciPy use their wheel-provided libraries rather than
adding Conda MKL alongside PyTorch. Select **Trajectory NN clean** as the PyCharm
notebook kernel and restart it. Check the printed `sys.executable` path. Use a
current PyTorch CUDA wheel that supports the installed NVIDIA driver and GPU if
choosing a different CUDA version. The old environment and data are retained.

The runtime bootstrap is included in collector source hashes. Source-keyed setup
and benchmark directories are regenerated automatically; an already initialized
main campaign requires fresh work/output directories after source/software changes.

Validation of this change: 51 focused tests passed with one CUDA-only skip. Tests
exercise Windows bootstrap ordering, fail-fast notebook ordering, a child that
terminates natively with an OpenMP error, and propagation of worker thread limits
without adding a duplicate-runtime override. The real native probe, a six-worker
CPU collection of six short flights, and every notebook cell (shared CPU actor
benchmarks, four eight-second flights, packaging and full verification) passed in
the Linux environment. The Windows DLL collision and CUDA execution could not be
validated on this Linux host; use the probe in the selected Windows kernel's
Python environment before starting the full campaign.

## Switching kernels after a failed setup

`Resume configuration/source/software differs; use a new work directory` under
`WORK/setup/<id>` can occur after installing the clean kernel: the old setup key
included collector sources but omitted software versions and preparation settings.
The notebook now derives that key from collector hashes, all recorded package and
Python versions, flight count/seed/duration, deterministic policy and batch settings,
and parent archive/manifest hashes. An identical configuration still reuses its setup;
a changed environment or preparation request gets a separate setup directory.
Previous setup configurations remain intact.

After updating, rerun from the first notebook cell in the clean kernel. Old failed
setup folders can remain in place. The collector's immutable configuration checks
and source hashes are unchanged by this notebook-only fix. If the mismatch instead
refers to the main `WORK/config.json`, select fresh `BVR_NN_WORK` and `BVR_NN_OUTPUT`
paths before rerunning the first cell; keep the old campaign rather than editing its
configuration to combine different software versions.

Validation: the regression reproduces the reported `prepare` exception using a
changed Torch version, then successfully prepares a separate setup while retaining
the old setup and main campaign configurations. Version and preparation-setting
cases fail with the old notebook key and pass with the new key. The focused suite
passes 65 tests with one CUDA-only skip; all notebook cells also pass with a stale
legacy setup retained, four short shared-CPU flights and final full verification.

## Git unavailable only during publication

A publication traceback ending in `FileNotFoundError: [WinError 2]` at
`git("branch", "--show-current", ...)` means the child cannot launch Git. That
branch check runs after snapshot verification, so this failure has already passed
local payload verification and has not started its Git upload.

Previously, the download helper could discover `C:\Program Files\Git\cmd\git.exe`
while the notebook left `GIT_EXECUTABLE=None`. Publication then launched a child
without that directory in PATH. The notebook now retains the discovered executable
for subsequent commands and checks Git/LFS, the publication branch and the staging
area before collection. Discovery also runs with `DOWNLOAD_LFS=False` when
publication is enabled. A local-only run with both flags false still needs no Git.

To retry the already packaged backup in the existing kernel:

```python
GIT_EXECUTABLE = r"C:\Program Files\Git\cmd\git.exe"
run("publish", backend=BACKEND)
```

Use the actual Git installation path if different. Once publication succeeds,
rerun the collection cell: completed episodes are verified and skipped. Keep the
same WORK and OUTPUT. To use automatic discovery after updating the checkout,
restart the clean kernel and rerun setup. The collector scripts and source hashes
are unchanged by this fix, so existing campaign configurations remain resumable.

Validation: 77 focused tests pass with one CUDA-only skip. Tests cover Windows Git
paths containing spaces, publication when downloads are disabled, local-only runs
without Git, early branch/staging checks, and an actual child that launches Git
after discovery despite the original PATH omitting Git. All collector source files
are byte-for-byte unchanged from the campaign-compatible base revision.

## Git LFS lock verification blocks a saved backup (2026-11-07)

The supplied `20261106T231411Z_add517_0041_publish.log` records local commit
`e1bbbf1` with 1,400 physical flights (2,800 aircraft perspectives). Both
`git lfs fsck` checks pass. The subsequent `git push` fails with
`Fatal error: Unable to verify locks`, after reporting that origin does not
support the Git LFS locking API. The Python `CalledProcessError` propagates this
Git failure. The log does not establish whether that API is unsupported or
inaccessible with the current credentials. The CRLF-to-LF notices are warnings;
they did not cause the push failure. The 1,400-flight backup is committed locally;
its upload is not confirmed by this log.

This corpus writes immutable shards and does not use Git LFS file locks. Notebook
publication setup now sets
`lfs.https://github.com/Rejn2024/Trajectory_Classification.git/info/lfs.locksverify=false`
in this checkout's local Git configuration, after the branch/staging checks and
before collection. Global Git settings and other LFS endpoints retain their lock
policy. Checksums, semantic verification, LFS integrity checks, the pre-push LFS
upload hook, the explicit LFS push and remote-commit confirmation remain active.
Authentication, quota, network and non-fast-forward failures still propagate.

To recover the stopped run without collecting another batch, run this once in a
terminal at the repository root:

```powershell
git config --local lfs.https://github.com/Rejn2024/Trajectory_Classification.git/info/lfs.locksverify false
git pull --no-rebase origin forecasting_ed_261104
```

The pull merges the published code fix with the local, unpushed corpus backup;
both commits are preserved. Run it while collection is stopped. Then run this in
the existing notebook kernel:

```python
run("publish", backend=BACKEND)
```

After it succeeds, rerun the collection cell. Publication re-verifies the saved
snapshot, reuses an existing commit when there are no new package changes, pushes
through the normal LFS hook and checks that the remote branch matches local HEAD.
Keep the same WORK and OUTPUT and the same kernel/software/campaign settings;
completed flights are verified and skipped. All five collector source files and
the archived corpora are unchanged by this notebook-only repair, preserving the
existing campaign's source hashes. A fresh run applies the setting during setup.

Apply the notebook-only change without replacing or normalizing the collector
files in the stopped checkout. The published campaign records CRLF byte hashes
for `tactical_corpus_schema.py`, `nn_duel_source.py` and `nn_duel_runtime.py`;
their LF-normalized Git versions have different hashes. This pre-existing
cross-platform difference is separate from the lock failure, and the repair
leaves the original Windows collector bytes in place.

Validation: 76 focused Git/setup/progress/benchmark/collector/import/runtime/
actor/schema tests pass, with one CUDA-only skip. A real-Git regression first
fails against the old notebook, then confirms that the repaired setup overrides
an inherited endpoint lock setting and that the notebook's publication child
sees `false`. Global settings, other endpoints and the pre-push hook are
preserved; local-only setup changes no Git configuration. The notebook schema
validates and all six code cells compile. All five collector source files are
byte-for-byte unchanged from the branch base revision.

The stopped Windows checkout and its unpushed commit are not available in the
cloud environment. Recovery must run there, where the new LFS payloads exist.
