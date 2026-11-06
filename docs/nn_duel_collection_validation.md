# NN versus NN acquisition: handoff and validation

Open `notebooks/11_nn_vs_nn_training_data.ipynb` on `forecasting_ed_261104`.
The default target is **10,000 new physical flights / 20,000 perspectives**.
The user requested that the full collection run on their own compute; that
campaign has **not** been started in the cloud environment.

Install `python -m pip install -e ".[ml,video]"` in the notebook kernel's environment,
install Git LFS, check out the branch, and run the notebook. `COLLECT` and `PUBLISH`
default to true. Publication commits only the corpus directory and pushes the
current branch after each 100-flight snapshot. The first cell exposes all controls.
Use a fresh work/output directory if changing source, software, seed, policy mode
or duration. Interruptions with unchanged configuration resume verified episodes.

## What was validated

- All notebook cells executed with ten real JSBSim engagements, including the
  packaging, verification, 09b-compatible feature/label example and support audit.
  Seven reached 300 seconds; three ended at defeat at 14.6, 24.7 and 39.7 seconds.
- A separate execution exercised the notebook's complete worker benchmark path.
  Its two ten-flight runs produced identical data checksums across worker counts.
- The validation sample is stored separately in
  `shared/corpora/nn-duels-validation-v1`: **ten flights, twenty perspectives**.
  Its `complete=true` flag refers only to its ten-flight configuration. It is not
  the future 10,000-flight release. Its manifest, checkpoints and collector copies
  preserve the exact validation provenance.
- LFS uploaded eleven objects (approximately 11 MB of new payload); the remote
  branch commit was verified after pushing. The validation package also references
  the already published tactical reproduction archive. The production output path
  `shared/corpora/nn-duels-v1` remains unused.
- Ruff and whitespace checks passed. Eighteen collector/schema tests passed,
  covering geometry uniqueness and pair balance, exact future masks, mirrored
  roles, immutable shards, checksums, first-defeat termination, resumability,
  exclusive writers, notebook validity and Windows reader-lock retries.
- Twenty-seven existing 09b tests passed; three optional GPU/MLflow tests were
  skipped. One existing notebook test was excluded after confirming that the
  branch's original notebook at `82cb16c` lacks the `paired-03` cell ID the test
  expects. The original 09b notebook and forecasting implementation were unchanged.
  The prefetch test requires local socket access in the sandbox and passed with
  that access enabled.

## Small CPU benchmark

Measured on the cloud environment's three detected CPUs with CPU FP32 inference,
not on the user's training machine. Both runs used the same ten scenarios, mean
duration 222.56 simulated seconds, and about 1.14 MB of compressed data per flight.

| Worker processes | Wall seconds | Flights/hour |
| --- | ---: | ---: |
| 1 | 80.75 | 445.82 |
| 2 | 68.17 | 528.07 |

Two workers reduced elapsed time by about 15.6% in this small sample. These rates
are not estimates for the user's hardware. The notebook benchmarks that machine
before selecting its worker count. No 10,000-flight job was launched here.

The implementation uses persistent CPU simulator processes with one native
inference thread per process, cached models, inference mode, self-pair batching,
omitted unused rewards/expert labels, and incremental compressed storage. It does
not implement centralized GPU rollout inference or GPU physics. The notebook
explains that a batched GPU inference server would need separate implementation
and measurement for these small networks. GPU availability alone is not claimed
as an acceleration result. Future forecasting-model training can use the GPU.

The final file-writing adjustment opens temporary files with write access for
Windows fsync and pins JSON, README and copied collector source to LF, so Git's
Windows line-ending conversion cannot invalidate publication checksums. This was
covered by the collector tests; native Windows and CUDA execution remain to be
checked on the user's machine.

## Corpus and skill-gap interpretation

Each perspective contains its actual observer's 51-feature input at 10 Hz, physical
tracks and the opposing NN's privileged 1 Hz selections. The helper produces 09b's
54 causal features and exact +1/+2/+5-second target contract, without changing the
original 09b loader's fixed corpus. Future preparation must index the new shards
explicitly. Keep both perspectives of a flight together under `scenario_group`.

The notebook discusses zero/rare skills observed in the saved 09b survey, proposes
energy, aspect and missile-exposure scenarios, and measures eligible support by
pilot and scenario family. These proposals do not guarantee selection of missing
skills by frozen pilots. The preserved simulator forces full throttle; speed
catalogue labels may therefore share executed controls. Stochastic inference or
curriculum adaptation would be separately versioned studies.
