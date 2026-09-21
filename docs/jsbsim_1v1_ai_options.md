# Options for obtaining or training a JSBSim 1-v-1 policy

## Executive recommendation

Use the **BVR Sim PPO path already vendored in this repository**, train a local
checkpoint against the supplied F-16 1-v-1 convergence scenario, and add a separate
rollout-to-canonical-dataset adapter. This is the lowest-risk route because it already
uses JSBSim, the same F-16 family, a Gymnasium interface, native discrete tactical
actions, a scripted opponent, deterministic evaluation, and both Python and faster C++
backends.

Do **not** assume that an arbitrary downloaded policy is plug-compatible. A useful
checkpoint is inseparable from its observation ordering and normalization, action
encoding, aircraft/config revision, timestep, termination logic, and reward. No frozen
`.pt`, `.pth`, or `.ckpt` policy is present in the pinned source tree. BVR Sim's local
trainer does, however, write a portable checkpoint containing the policy weights,
normalizer, action dimensions, configuration hash, and version metadata.

The important dataset caveat is that a low-level RL action is not a tactical-skill
label. Notebook 03 expects observable relative-geometry features plus a supervised
`tactical_label`. For scientifically defensible classifier data, either:

1. train a **hierarchical policy whose high-level action is one of the existing skill
   labels** (recommended), so the selected skill is genuine controller ground truth; or
2. retain low-level PPO actions as privileged provenance and use separately named,
   kinematics-derived labels (turn/climb/acceleration), never relabelling inferred
   behaviour as `commanded_skill` or `target_skill`.

## What is available in this checkout

| Requirement | BVR Sim support | Consequence |
|---|---|---|
| JSBSim F-16 1-v-1 | Supplied convergence configs select `F16` and `fdm_type: jsbsim` | No simulator translation is required. |
| Local RL | `bvr_sim_rl` trains PPO through [skrl](https://skrl.readthedocs.io/) | Start with the Python backend; use C++ for the real run. |
| Opponent | The convergence config uses `tactical_random` blue control | Suitable for curriculum stage 1 and reproducible evaluation. |
| Action space | `MultiDiscrete([15, 15, 9, 2])`: heading, altitude, speed, fire | Save all four bins as privileged data. |
| Frozen evaluation | `python -m bvr_sim_rl.evaluate` checks the config SHA-256 | Prevents silently evaluating a checkpoint under a different scenario. |
| Replay | The adapter can emit ACMI for the first vector environment | Visually inspect representative successes and failures. |
| Classifier-ready output | Not yet implemented for RL rollouts | Add an exporter; do not feed flattened RL observations directly to notebook 03. |

Upstream resources:

* [BVR Sim repository](https://github.com/lizi-Margin/bvr_sim) — the authoritative
  upstream for the code pinned here.
* [JSBSim repository](https://github.com/JSBSim-Team/jsbsim) and
  [reference manual](https://jsbsim-team.github.io/jsbsim-reference-manual/) — dynamics
  and property definitions.
* [Gymnasium custom-environment guide](https://gymnasium.farama.org/introduction/create_custom_env/)
  — useful when extending the wrapper without breaking reset/termination semantics.
* [skrl documentation](https://skrl.readthedocs.io/) — the PPO implementation used by
  the bundled entry point.

## Proposal A — train the bundled PPO policy locally (fastest viable route)

Run from `bvr_sim_source/` so paths resolve as documented.

### 1. Create an isolated environment and smoke-test the Python path

```bash
cd bvr_sim_source
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e '.[rl]'
python -m bvr_sim_rl \
  --backend python \
  --config scripts/experiments/ppo/f16_1v1_convergence_python.jsonc \
  --timesteps 128 --num-envs 1 --rollouts 16 --seed 1111 \
  --logdir runs/smoke
```

The smoke run validates installation only; it is not evidence of learning.

### 2. Build and validate the accelerated backend

```bash
bash bvr_sim/build_linux.sh            # Windows: bvr_sim\build_windows.bat
python scripts/tests/test_cpp.py
```

If the native build is unavailable, the Python backend remains a correct functional
fallback, but expect substantially less sample throughput.

### 3. Train three seeds, not one anecdotal run

```bash
for seed in 1111 2222 3333; do
  python -m bvr_sim_rl \
    --backend cpp \
    --config scripts/experiments/ppo/f16_1v1_convergence_cpp.jsonc \
    --timesteps 100000 --num-envs 8 --rollouts 256 --seed "$seed" \
    --logdir "runs/f16_1v1/seed_${seed}"
done
tensorboard --logdir runs/f16_1v1
```

Here `timesteps` is the number of vector steps; with eight environments, 100,000
steps collects approximately 800,000 transitions. Treat this as a starting budget,
not a convergence guarantee. Watch `Info / return_per_step`, outcome rates, and episode
length rather than raw return alone.

### 4. Evaluate frozen policies against an untouched seed range

```bash
for seed in 1111 2222 3333; do
  python -m bvr_sim_rl.evaluate \
    --backend cpp \
    --config scripts/experiments/ppo/f16_1v1_convergence_cpp.jsonc \
    --checkpoint "runs/f16_1v1/seed_${seed}/ppo_cpp_seed${seed}.pt" \
    --episodes 500 --seed 30000 \
    --output "runs/f16_1v1/seed_${seed}/eval.json"
done
```

Compare against the tactical baseline with the same episode seeds:

```bash
python -m bvr_sim_rl.evaluate \
  --method tactical --backend cpp \
  --config scripts/experiments/ppo/f16_1v1_convergence_cpp.jsonc \
  --episodes 500 --seed 30000 --output runs/f16_1v1/tactical_eval.json
```

Promote a checkpoint only if its configuration hash matches, it beats declared
baselines on held-out seeds, it does not exploit termination or unsafe-envelope bugs,
and ACMI inspection confirms credible flight. Report the evaluator's Wilson interval,
not only point win rate.

## Proposal B — make the RL policy generate notebook-03-compatible data

### Preferred design: hierarchical skill policy

Keep the existing bounded skill controllers from notebook 02 as the low-level layer.
At a slower decision interval (initially 2–5 seconds), let PPO select from the six
catalogue entries:

```text
MAINTAIN, PURSUE, BEAM, CRANK_LEFT, CRANK_RIGHT, EXTEND
```

This provides an exact `tactical_label` and `commanded_skill` for every 10 Hz sample,
while JSBSim still determines the trajectory. Initially train against the scripted
opponent, then use a frozen-policy league (current policy versus sampled older
checkpoints) to improve diversity without the instability of unrestricted simultaneous
self-play.

### Export contract

The rollout exporter should reuse the canonical schema and writers rather than invent a
second format. For every episode it should:

1. sample and persist the scenario seed and initial geometry;
2. run the frozen policy in inference mode and log state at exactly 10 Hz;
3. compute the same target-centric `MODEL_FEATURE_COLUMNS` as notebook 02;
4. write `episode_id`, `time_s`, and `tactical_label`, plus all required trajectory
   columns;
5. record high-level skill and low-level action bins only in privileged columns;
6. write `label_map.json`, `episodes.parquet`, sharded `trajectories/*.parquet`, a
   manifest containing simulator/config/checkpoint hashes, and sampled ACMI replays;
7. run the existing schema, cadence, finiteness, leakage, and episode-boundary checks.

Notebook 03 can then consume the output unchanged through `BVR_TRAIN_DATASET`. Keep
entire engagements in one split, and keep every rollout from the same policy checkpoint
and seed family in the same partition to avoid policy-fingerprint leakage.

### If low-level PPO is retained

Exporting low-level PPO flight is still useful for out-of-distribution testing and
kinematic classification. Use names such as `derived_turn_label` and include the
labelling rule/version in the manifest. Do not fabricate the six tactical labels from
single frames. If tactical segmentation is required, annotate temporal segments and
measure inter-rater agreement or validate a documented rule against held-out human
labels.

## Proposal C — download an external model (screening route, not first choice)

Public projects are valuable implementations and baselines, but a release advertised
as “1-v-1 JSBSim” is not automatically compatible with this repository.

| Resource | Likely value | Main integration risk |
|---|---|---|
| [Light Aircraft Game (LAG)](https://github.com/liuqh16/LAG) | JSBSim air-combat environment and 1-v-1 research code; useful for reward/curriculum comparisons | Checkpoint availability and licenses must be verified at the chosen revision; observation/action semantics differ. |
| [Gym-JSBSim](https://github.com/Gor-Ren/gym-jsbsim) | A compact example of wrapping JSBSim as a Gym environment | Flight-control tasks, not a drop-in 1-v-1 combat policy. |
| [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) and [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) | Mature local PPO tooling and checkpoint/evaluation patterns | No reason to expect a zoo model trained for this custom observation/action space. |
| [HARL](https://github.com/PKU-MARL/HARL) | HAPPO/HATRPO/MAPPO-family training for later multi-agent experiments; an adapter is already present | More moving parts than the one-controlled-aircraft PPO path; not needed for the first dataset. |
| [marlbenchmark/off-policy](https://github.com/marlbenchmark/off-policy) | Alternative MARL algorithms; a compatibility adapter is present | Framework compatibility does not imply a compatible pretrained policy. |

Before downloading any checkpoint, require all of the following:

* an explicit model license and redistribution permission;
* immutable release/tag and SHA-256 for every artifact;
* observation names, ordering, units, history stacking, and normalizer state;
* exact action meaning and bounds;
* JSBSim aircraft/config revision, integration and decision timestep;
* opponent distribution, reward, termination rules, and seed protocol;
* deterministic inference code and credible held-out evaluation.

Reject a model if any of the first five items cannot be reconstructed. If it passes,
write a narrow adapter and evaluate it before collecting classifier data; do not reshape
or truncate tensors until dimensions happen to match.

## Curriculum and experiment design

1. **Installation gate:** one deterministic Python episode, one C++ episode, identical
   declared action meanings, and readable ACMI output.
2. **Learning gate:** three-seed PPO beats random and is compared with the tactical
   baseline on 500 fixed, unseen episodes.
3. **Robustness gate:** evaluate a grid of range, bearing, altitude, speed, and opponent
   seeds beyond the training distribution; report each stratum.
4. **Hierarchy gate:** train high-level skill selection with minimum dwell time and an
   action-change penalty; verify all classes have adequate occupancy.
5. **Dataset pilot:** export 100 episodes, run every schema/leakage check, and train
   notebook 03 as an end-to-end compatibility test.
6. **Scale-up:** only after the pilot, generate the large corpus. Mix policy seeds,
   checkpoint ages, opponents, and geometries, retaining provenance for each episode.
7. **Classifier evaluation:** reserve entire policies/opponents and geometry strata for
   validation. Compare scripted-only, RL-only, and mixed training sets on the same
   untouched evaluation corpus.

## Suggested deliverables

* `rl_skill_env.py`: high-level categorical skill wrapper with dwell-time handling.
* `collect_rl_skill_dataset.py`: frozen-policy rollout and canonical Parquet/ACMI writer.
* `configs/rl_dataset.yaml`: seeds, checkpoint hashes, scenario distributions, sample
  cadence, and shard size.
* `dataset_manifest.json`: code/config/model hashes, versions, label provenance, units,
  counts, failures, and quality-control results.
* A short model card for each policy: intended use, training distribution, baseline
  results with uncertainty, known failure modes, and license.

## Practical decision

Start with Proposal A plus a 100-episode export pilot. If the primary objective is the
six-class skill classifier, implement Proposal B before scaling training: hierarchical
skill selection gives the classifier meaningful ground truth. Investigate LAG or other
downloaded checkpoints only as comparison systems unless their complete interface and
provenance contract can be reproduced. This order minimizes integration risk and avoids
spending classifier compute on labels that the RL controller never actually generated.
