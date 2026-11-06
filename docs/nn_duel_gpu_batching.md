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
