# BVR Sim Air-Combat Manoeuvre Recognition and Next-Step Forecasting
## Revised implementation plan — F-16 first, aircraft type explicit throughout

**Status:** revised after review of the current `lizi-Margin/bvr_sim` repository and the BVR Sim paper  
**Repository baseline reviewed:** `lizi-Margin/bvr_sim`, `master`, commit `db4f95657c081d120442022a339bed62d340c93b` (2 September 2026)  
**Primary initial platform:** F-16 vs F-16  
**Implementation language:** Python  
**Training environment:** local laptop  
**Primary simulator:** BVR Sim only

---

# 1. Executive summary

The project should use **BVR Sim as a synthetic-data generator**, not initially as an RL-training problem.

The first objective is to generate controlled, labelled 1-v-1 F-16 trajectories, then train causal sequence models to answer three progressively harder questions:

1. **What is the target aircraft doing now?**
2. **Is the target about to change behaviour?**
3. **What action or tactical behaviour is likely next?**

The architecture should make aircraft type a first-class variable from the start, even though the first several experiments should remain **F-16 vs F-16**.

The proposed progression is:

```text
BVR Sim
    ↓
aircraft/scenario registry
    ↓
scripted and built-in behaviour generators
    ↓
privileged simulator logging
    ↓
canonical target-centric trajectory dataset
    ↓
feature/window generation
    ↓
simple baselines
    ↓
causal GRU / TCN sequence models
    ↓
current behaviour recognition
    ↓
transition prediction
    ↓
next BVR Sim action prediction
    ↓
multi-aircraft generalisation
```

The most important design changes relative to the earlier plan are:

- treat BVR Sim's **native action** correctly as a four-branch discrete action:
  `MultiDiscrete([15, 15, 9, 2])`;
- separate **kinematic behaviour** from **tactical intent** rather than forcing both into one class taxonomy;
- maintain a strict split between **observable features** and **privileged simulator labels** to avoid target leakage;
- evaluate next-step forecasting against strong trivial baselines such as **repeat the current action** or **remain in the current manoeuvre**;
- model manoeuvre transitions explicitly rather than relying only on fixed-horizon class prediction;
- record aircraft identity, aircraft-model version, controller version, simulator commit and backend from the first dataset.

---

# 2. Verified BVR Sim properties relevant to this project

The current BVR Sim repository is a good fit for the project because it already provides:

- Gymnasium-style interfaces;
- interchangeable Python and accelerated C++ backends;
- JSBSim-based aircraft dynamics;
- configurable aircraft, sensors, weapons and opponents;
- structured entity observations;
- scripted opponents;
- Tacview/ACMI replay;
- built-in RL adapters;
- a shared high-level tactical control abstraction above aircraft-specific controllers.

At the reviewed repository revision, the README lists these supported aircraft families:

```text
F-15
F-16
F/A-18
F-22
F-4N Phantom II
AJ 37 Viggen
JA 37 Viggen
```

The published BVR Sim paper documents the earlier core set of F-15, F-16, F/A-18 and F-22. The repository has therefore already expanded beyond the paper.

For this project, that is useful because cross-aircraft generalisation can eventually be tested without redesigning the data model.

The repository's supplied F-16 1-v-1 convergence configuration uses, among other settings:

```text
dt = 0.4 s
max_steps = 400
red unit_spec = F16
blue unit_spec = F16
fdm_type = jsbsim
obs_type = entity
blue opponent = tactical_random
initial separation = 25 nm
```

The same configuration demonstrates that aircraft type is already naturally represented by the BVR Sim configuration field:

```json
"unit_spec": "F16"
```

This field should be wrapped by the project's own aircraft registry rather than referenced directly throughout the ML code.

---

# 3. Important correction: BVR Sim's native action space

The current source and PPO quick-start documentation define the simulator action space as:

```text
MultiDiscrete([15, 15, 9, 2])
```

representing:

```text
delta_heading
delta_altitude
delta_speed
shoot
```

Therefore, the most direct definition of a combatant's **next step** is not initially a continuous regression problem.

It is a **multi-head categorical forecasting problem**:

\[
P(a_{t+h}^{\psi}),
\quad
P(a_{t+h}^{h}),
\quad
P(a_{t+h}^{V}),
\quad
P(a_{t+h}^{fire})
\]

where the respective numbers of categories are:

\[
15,\;15,\;9,\;2.
\]

This yields four output heads:

```text
heading-action head     → 15 classes
altitude-action head    → 15 classes
speed-action head       → 9 classes
fire-action head        → 2 classes
```

Physical command deltas can still be reconstructed or added as regression targets later, but the native discrete action should be the primary target because it is exactly what BVR Sim executes.

The action-space implementation contains normalisation logic and nominal physical scaling constants, including a maximum heading delta of 45 degrees and numeric altitude/speed limits of 80. Their precise physical interpretation should be confirmed against the active controller path before those values are used as regression labels.

---

# 4. Core research questions

The project should answer a sequence of increasingly difficult questions.

## 4.1 Current kinematic behaviour

Given a causal observation history:

\[
X_{t-k:t},
\]

classify the target's current motion state.

Examples:

```text
TURN_LEFT
TURN_RIGHT
STRAIGHT

CLIMB
DESCEND
LEVEL

ACCELERATE
DECELERATE
STEADY_SPEED
```

These should not necessarily be collapsed into one mutually exclusive class.

---

## 4.2 Current tactical behaviour

Given the same history, classify relational BVR behaviour such as:

```text
PURSUE
BEAM
CRANK_LEFT
CRANK_RIGHT
EXTEND
RECOMMIT
DEFEND
MAINTAIN
```

This is distinct from the kinematic behaviour.

For example, an aircraft can simultaneously be:

```text
TURN_LEFT
CLIMB
ACCELERATE
CRANK_LEFT
```

A single flat label taxonomy cannot represent this cleanly.

---

## 4.3 Behaviour-transition forecasting

Predict:

\[
P(\text{behaviour changes within } H \text{ seconds}
\mid X_{t-k:t})
\]

for horizons such as:

```text
0.8 s
2 s
4 s
8 s
```

This addresses a key weakness of fixed-horizon next-class prediction: most samples may simply remain in the same manoeuvre.

---

## 4.4 Next tactical class conditional on transition

When a transition is likely, predict:

\[
P(M_{next} \mid
\text{transition}, X_{t-k:t})
\]

For example:

```text
CRANK → DEFEND
PRESS → CRANK
EXTEND → RECOMMIT
```

This is closer to genuine tactical forecasting than simply asking whether the label at `t+4 s` matches the current label.

---

## 4.5 Native BVR Sim action prediction

Predict the four next action branches:

\[
a_{t+h}
=
(a^\psi, a^h, a^V, a^{fire})
\]

using four categorical heads.

This provides an objective, simulator-native definition of "next step".

---

# 5. Strict separation between observations and privileged labels

This is one of the most important implementation rules.

The simulator can expose information that would never be available to an external observer, including:

```text
opponent policy name
skill name
issued action
desired control command
internal controller state
future action
future manoeuvre label
```

These are valuable **ground-truth labels**, but they must never accidentally enter the model input.

Define two explicit schemas.

## 5.1 Observable schema

Contains only information that the model is allowed to use:

```text
time
target position / relative position
target velocity / relative velocity
range
range rate
bearing
line-of-sight angles
relative altitude
relative heading
target speed
target vertical speed
turn rate
possibly target attitude
possibly sensor-state indicators
aircraft identity, only in aircraft-aware experiments
```

## 5.2 Privileged schema

Used only for labelling/evaluation:

```text
target policy
target skill
target native action
target controller command
weapon release
future action
future skill
scenario seed
hidden simulator state
```

The data-loading code should enforce this separation programmatically.

A useful safeguard is an explicit allow-list:

```python
MODEL_FEATURE_COLUMNS = [...]
PRIVILEGED_COLUMNS = [...]
```

and a unit test asserting:

```python
set(MODEL_FEATURE_COLUMNS).isdisjoint(PRIVILEGED_COLUMNS)
```

---

# 6. Aircraft type as an explicit domain variable

Aircraft type should exist at four levels.

## 6.1 Simulator configuration

```python
@dataclass
class AircraftConfig:
    aircraft_type: str
    bvr_unit_spec: str
    fdm_type: str
    controller_id: str | None
    model_version: str | None
    reference_speed_ms: float | None
    reference_altitude_m: float | None
    max_g: float | None
```

Initial entry:

```python
F16 = AircraftConfig(
    aircraft_type="F16",
    bvr_unit_spec="F16",
    fdm_type="jsbsim",
    controller_id="default",
    model_version="repo-pinned"
)
```

---

## 6.2 Aircraft registry

```python
AIRCRAFT_REGISTRY = {
    "F16": F16,
}
```

Later:

```python
AIRCRAFT_REGISTRY = {
    "F16": ...,
    "F15": ...,
    "FA18": ...,
    "F22": ...,
    "F4N": ...,
    "AJ37": ...,
    "JA37": ...,
    "TYPHOON": ...,
}
```

Typhoon should remain a future registry entry unless and until a validated BVR Sim-compatible model is added.

---

## 6.3 Dataset metadata

Every episode should store:

```text
observer_aircraft_type
target_aircraft_type

observer_aircraft_model_version
target_aircraft_model_version

observer_controller_version
target_controller_version
```

---

## 6.4 Optional ML feature

Aircraft identity should **not** be supplied to the first model.

Later compare:

### Aircraft-agnostic

\[
f(X)
\]

### Aircraft-aware

\[
f(X, e_{own}, e_{target})
\]

where `e` is an aircraft embedding.

This turns aircraft type into an explicit experimental factor rather than an uncontrolled confounder.

---

# 7. Use a target-centric representation

The model should not learn absolute latitude, longitude or global heading unnecessarily.

For each observation, transform the engagement into a target/observer-relative frame.

For an observer aircraft at position \(\mathbf p_o\) and heading \(\psi_o\), transform target position:

\[
\mathbf r_t
=
R(-\psi_o)
(\mathbf p_t-\mathbf p_o).
\]

A useful convention is:

```text
+x = observer forward
+y = observer right
+z = up
```

Also transform velocity:

\[
\mathbf v_{rel}
=
R(-\psi_o)
(\mathbf v_t-\mathbf v_o).
\]

Derived features should include:

```text
range
range rate
relative bearing
relative heading
relative altitude
relative speed
LOS azimuth
LOS elevation
LOS angular rate
target aspect
angle-off
target turn rate
target climb rate
target acceleration
```

This should make the model substantially more invariant to arbitrary map orientation.

---

# 8. Use BVR Sim's entity observation as a reference representation

BVR Sim's current preferred observation design is:

```text
obs_type = "entity"
```

The repository describes this as a fixed-width entity table where aircraft, missiles and other objects share a common schema.

Relevant entity features include:

```text
normalised relative position
normalised relative velocity
range
radial velocity
altitude / altitude difference
entity type
missile-specific fields
sin/cos azimuth
sin/cos elevation
```

For the initial 1-v-1 project, there is no need to feed the raw entity table directly into a Set Transformer or GNN.

Instead:

1. log the simulator entity observation;
2. convert it into a canonical 1-v-1 relational feature vector;
3. train a simple causal sequence model.

However, retaining entity-table data or enough raw state to reconstruct it will make later expansion to multi-aircraft scenarios easier.

---

# 9. Revised manoeuvre labelling strategy

The earlier flat eight-class taxonomy should be replaced by a hierarchical or multi-head label structure.

## 9.1 Kinematic lateral head

```text
STRAIGHT
TURN_LEFT
TURN_RIGHT
```

## 9.2 Kinematic vertical head

```text
LEVEL
CLIMB
DESCEND
```

## 9.3 Kinematic energy/speed head

```text
STEADY_SPEED
ACCELERATE
DECELERATE
```

## 9.4 Tactical relational head

Initial tactical classes:

```text
MAINTAIN
PURSUE
BEAM
CRANK_LEFT
CRANK_RIGHT
EXTEND
```

Later:

```text
RECOMMIT
DEFEND
MISSILE_EVASION
SUPPORT
```

This structure acknowledges that tactical intent and physical execution are not the same variable.

---

# 10. Label source hierarchy

Use labels in decreasing order of confidence.

## 10.1 Level A — controller/skill ground truth

If the aircraft is executing a known scripted behaviour:

```text
skill_name = crank_left
```

that is the cleanest tactical-intent label.

BVR Sim currently contains a `SkillManager` with built-in examples including:

```text
crank_maneuver
missile_evasion
disengage
maintain_position
```

These can be used as references and potentially as data-generation components.

However, project-specific deterministic data-generation policies should still be implemented so label semantics remain controlled and versioned.

---

## 10.2 Level B — native action labels

Log the exact executed BVR Sim action:

```text
delta_heading bin
delta_altitude bin
delta_speed bin
shoot
```

This is unambiguous and should be retained for every timestep.

---

## 10.3 Level C — derived kinematic labels

Kinematic labels can be derived from actual motion using thresholds and hysteresis.

For example:

```text
TURN_LEFT     if smoothed yaw rate < -threshold
TURN_RIGHT    if smoothed yaw rate > +threshold
STRAIGHT      otherwise
```

Similarly for:

```text
climb rate
speed derivative
```

These labels describe what the aircraft actually did, not merely what was commanded.

---

# 11. Handle manoeuvre transitions explicitly

Windows around policy switches are intrinsically ambiguous.

Suppose:

```text
PURSUE → CRANK_LEFT
```

at time \(t_s\).

The aircraft will not instantaneously exhibit the new kinematics because the inner-loop controller and airframe have inertia.

Do not blindly assign every timestep after `t_s` the new observed-motion label.

Retain at least:

```text
commanded_skill
executed_kinematic_state
time_since_skill_change
time_to_next_skill_change
transition_flag
```

For early experiments, one option is to exclude a short ambiguity interval around transitions when training the pure current-manoeuvre classifier.

Later, those transition windows become the central data for the forecasting model.

---

# 12. Scripted data-generation policies

The first dataset should come from deliberately controlled behaviours.

Suggested project policies:

```text
MaintainPolicy
TurnPolicy
ClimbPolicy
DescendPolicy
AcceleratePolicy
DeceleratePolicy
PursuitPolicy
BeamPolicy
CrankPolicy
ExtendPolicy
```

All should inherit from a common interface:

```python
class TacticalPolicy:
    label: str

    def reset(self, rng):
        ...

    def act(self, own_state, opponent_state, time_s):
        ...
```

The output should ultimately map into BVR Sim's native action representation.

---

# 13. Randomise behaviours, not just scenarios

Every behavioural class should vary internally.

For example, a crank should vary:

```text
direction
offset angle
duration
speed command
altitude command
reaction delay
switch timing
```

Scenario randomisation should include:

```text
initial range
initial altitude
altitude difference
initial speed
speed difference
heading difference
aspect
lateral offset
initial vertical rate
```

The goal is to learn the invariant structure of a behaviour rather than a narrow trajectory template.

---

# 14. Avoid unrealistic scenario sampling

Uniform randomisation over all possible geometries can produce many implausible or uninformative episodes.

Use staged sampling.

## Stage A — controlled geometry

Narrow scenario ranges, primarily for software validation.

## Stage B — realistic broad geometry

Expand range, altitude, aspect and speed distributions.

## Stage C — edge cases

Deliberately include difficult geometries:

```text
low closure
near-beam
large altitude differences
very high/low energy
rapid policy transitions
```

Record a scenario-bin identifier so performance can be stratified later.

---

# 15. Episode perspective and data multiplication

Each 1-v-1 engagement can potentially yield two recognition examples:

```text
Red observing Blue
Blue observing Red
```

but only if both sides have valid labels.

Store explicit roles:

```text
observer_id
target_id
observer_team
target_team
```

Do not use "red" and "blue" as semantic equivalents of "observer" and "target" in the ML pipeline.

This avoids an unnecessary team-colour bias and makes the same code reusable when roles are reversed.

---

# 16. Dataset schema

Use one canonical row per target-perspective timestep.

Suggested fields:

```text
episode_id
perspective_id
step
time_s

observer_id
target_id

observer_aircraft_type
target_aircraft_type

# Observer state
observer_x
observer_y
observer_z
observer_vx
observer_vy
observer_vz
observer_heading
observer_pitch
observer_roll
observer_speed

# Target state
target_x
target_y
target_z
target_vx
target_vy
target_vz
target_heading
target_pitch
target_roll
target_speed

# Relative features
rel_x_body
rel_y_body
rel_z_body
rel_vx_body
rel_vy_body
rel_vz_body

range
range_rate
relative_bearing
relative_heading
relative_altitude
relative_speed
target_aspect
angle_off
los_azimuth
los_elevation
los_rate

# Derived target dynamics
target_turn_rate
target_climb_rate
target_acceleration

# Observable sensor flags when enabled
track_valid
track_age
sensor_mode

# Privileged labels
target_skill
target_skill_id
time_since_skill_change
time_to_next_skill_change
transition_within_2s
transition_within_4s

target_action_heading_bin
target_action_altitude_bin
target_action_speed_bin
target_action_fire

target_next_skill
```

---

# 17. Episode metadata schema

Use a separate `episodes.parquet` table:

```text
episode_id
seed

bvr_sim_commit
bvr_sim_backend
bvr_sim_config_hash
jsbsim_version

observer_aircraft_type
target_aircraft_type
observer_model_version
target_model_version

initial_range_nm
initial_altitude_difference_m
initial_heading_difference_deg
initial_speed_difference_ms
initial_aspect_bin

observer_policy
target_policy

weapons_enabled
sensor_mode

episode_duration_s
termination_reason
```

This is essential for reproducibility and group-based splitting.

---

# 18. Store simulator provenance

Every dataset build should have a manifest such as:

```yaml
dataset_id: bvr_f16_1v1_v001

simulator:
  repository: lizi-Margin/bvr_sim
  commit: db4f95657c081d120442022a339bed62d340c93b
  backend: cpp
  dt: 0.4

aircraft:
  observer: F16
  target: F16

fdm:
  type: jsbsim

observation:
  type: entity

weapons:
  enabled: false

generation:
  seed_base: 100000
```

Also store:

```text
Python version
package lockfile
OS
dataset creation date
config hash
feature schema version
label schema version
```

---

# 19. Backend strategy

Use both BVR Sim backends, but for different purposes.

## Python backend

Use for:

```text
debugging
inspection
unit/integration development
small scenario checks
feature verification
```

## C++ backend

Use for:

```text
production dataset generation
large Monte Carlo sweeps
later RL experiments
```

The current repository reports approximately 260 C++ steps/s for 1-v-1 at the 0.4 s decision interval in its benchmark, but actual laptop throughput should be measured locally rather than assumed.

Do not silently mix Python- and C++-generated data in the same dataset version.

If both are used, record backend and test whether they produce materially different distributions.

---

# 20. Data storage

Recommended formats:

```text
Parquet / PyArrow
NumPy
Pandas
```

Suggested layout:

```text
datasets/
└── bvr_f16_1v1_v001/
    ├── manifest.yaml
    ├── aircraft.parquet
    ├── episodes.parquet
    ├── trajectories/
    │   ├── shard_000.parquet
    │   ├── shard_001.parquet
    │   └── ...
    └── label_map.json
```

Use moderate-sized shards rather than one file per episode.

A reasonable initial target is:

```text
100–1000 episodes per shard
```

depending on episode duration and row width.

---

# 21. Dataset size strategy

Do not generate 40,000 episodes before proving that the labels and feature pipeline work.

Use three scales.

## Smoke dataset

```text
~20–50 episodes per behaviour
```

Purpose:

```text
integration testing
visual inspection
schema checks
feature sanity checks
```

## Pilot dataset

```text
~200–500 episodes per behaviour
```

Purpose:

```text
baseline model training
class separability
leakage checks
initial learning curves
```

## Main dataset

Scale only after plotting performance versus training set size.

Potentially:

```text
1,000–5,000+ episodes per behaviour
```

if learning curves justify it.

Simulation throughput is likely not the limiting factor; bad labels and leakage are bigger risks.

---

# 22. Window generation

With the supplied 0.4 s decision interval:

\[
f = 2.5\,Hz.
\]

Candidate history lengths:

| History | Samples |
|---|---:|
| 2 s | 5 |
| 4 s | 10 |
| 8 s | 20 |
| 10 s | 25 |
| 20 s | 50 |

Start with:

```text
10 s = 25 timesteps
```

but treat history length as an experimental variable.

Do not permanently materialise every overlapping training window unless needed.

Prefer an indexed PyTorch `Dataset` that reads trajectory shards and constructs windows dynamically.

---

# 23. Split strategy

Never random-split overlapping windows.

Use episode-level grouping.

Initial split:

```text
70% train episodes
15% validation episodes
15% test episodes
```

Then add harder splits.

## Geometry OOD

Hold out combinations of:

```text
range
aspect
altitude difference
speed difference
```

## Parameter OOD

Example:

```text
train crank offset: 25–45 degrees
test crank offset: 45–60 degrees
```

## Policy OOD

Train on some controller parameterisations and test on others.

## Aircraft OOD

Later:

```text
train: F16
test: F15
```

or:

```text
train: F16 + F15 + FA18
test: F22
```

---

# 24. Leakage and sanity checks

Before trusting any neural result, run the following tests.

## 24.1 Shuffled-label test

Training on shuffled labels should collapse to chance-level performance.

## 24.2 Privileged-feature test

Assert that:

```text
skill name
policy name
future action
current opponent action
```

are absent from the operational feature matrix.

## 24.3 Team-colour test

Swap Red/Blue roles and verify that performance does not depend materially on colour encoding.

## 24.4 Heading-invariance test

Rotate the entire scenario and verify that target-centric features remain nearly unchanged.

## 24.5 Episode-overlap test

Assert that no `episode_id` appears in more than one split.

## 24.6 Scenario duplicate test

Detect near-identical seeds/configurations crossing train/test boundaries.

---

# 25. Baseline models

The first benchmark suite should deliberately include simple models.

## 25.1 Rule-based current-manoeuvre recogniser

Use thresholds on:

```text
turn rate
climb rate
acceleration
relative bearing
range rate
aspect
```

This produces an interpretable floor.

---

## 25.2 Persistence forecast

For next manoeuvre:

```text
predict future manoeuvre = current manoeuvre
```

This is a critical baseline.

A fixed-horizon classifier that does not beat this baseline is not useful.

---

## 25.3 Repeat-current-action forecast

For native next action:

```text
a_hat(t+h) = a(t)
```

or, if current opponent action is intentionally unavailable to the inference problem:

```text
a_hat(t+h) = inferred modal recent action
```

The exact baseline should reflect the intended observation assumptions.

---

## 25.4 Constant-velocity trajectory forecast

For any future-geometry prediction:

\[
\hat p_{t+h}
=
p_t + h v_t.
\]

Any learned trajectory predictor should outperform this.

---

## 25.5 Tabular ML

Use:

```text
RandomForestClassifier
HistGradientBoostingClassifier
```

on window summary statistics.

Useful statistics include:

```text
mean / std / min / max turn rate
net heading change
mean climb rate
altitude change
speed change
mean range rate
range-rate trend
aspect trend
LOS-rate statistics
```

---

# 26. First neural sequence model: causal GRU

A small GRU remains the recommended first neural architecture.

Example:

```python
class GRUEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

    def forward(self, x):
        _, h = self.gru(x)
        return h[-1]
```

Why GRU first:

```text
causal
small
easy to debug
fast on a laptop
well suited to short sequences
strong baseline for temporal dynamics
```

Avoid a bidirectional LSTM for the online recognition model because it uses future context inside the input window.

A Bi-LSTM can still be used as an **offline upper-bound experiment**, but it should not be confused with deployable causal recognition.

---

# 27. Second sequence architecture: TCN

After the GRU, add a Temporal Convolutional Network as a competing causal sequence baseline.

Advantages:

```text
parallel training
stable gradients
controllable receptive field
no recurrent hidden-state dependence
```

There is no need to introduce a Transformer until GRU and TCN results justify more complexity.

For 25–50 timestep sequences and tens of input features, a Transformer may provide little benefit.

---

# 28. Current-behaviour model structure

Use one shared sequence encoder with separate heads.

```text
trajectory window
       ↓
   GRU / TCN
       ↓
 latent state
   ├──────── lateral kinematic head
   ├──────── vertical kinematic head
   ├──────── speed/energy head
   └──────── tactical behaviour head
```

This gives:

\[
L =
\lambda_L L_{lat}
+
\lambda_V L_{vert}
+
\lambda_E L_{energy}
+
\lambda_T L_{tactical}.
\]

Initially use equal weights unless class imbalance or gradient domination is observed.

---

# 29. Next-action forecasting architecture

Reuse the sequence encoder:

```text
trajectory window
       ↓
    encoder
       ↓
 latent state
   ├──────── heading branch: 15 classes
   ├──────── altitude branch: 15 classes
   ├──────── speed branch: 9 classes
   └──────── fire branch: 2 classes
```

For forecast horizon \(h\):

\[
L_{action}
=
CE(\hat a^\psi, a^\psi_{t+h})
+
CE(\hat a^h, a^h_{t+h})
+
CE(\hat a^V, a^V_{t+h})
+
\lambda_f CE(\hat a^{fire}, a^{fire}_{t+h}).
\]

Because weapon release may be rare, the fire head will probably require:

```text
class weighting
balanced sampling
or separate event-based evaluation
```

once weapons are enabled.

---

# 30. Transition forecasting architecture

Add a transition head:

\[
P(T_H=1 \mid X)
\]

where:

```text
T_H = 1
```

means a tactical-state transition occurs within horizon \(H\).

For example:

```text
transition_2s
transition_4s
transition_8s
```

These can be independent binary heads.

Then add:

```text
next_tactical_class
```

conditioned either explicitly or conceptually on a future transition.

This is preferable to relying exclusively on:

```text
class_at_t_plus_4s
```

because the latter is heavily biased toward class persistence.

---

# 31. Multi-task model after baselines are stable

Only after separate models work should they be unified:

```text
                        ┌─ lateral state
                        ├─ vertical state
                        ├─ energy state
trajectory → encoder ───┼─ tactical state
                        ├─ transition probability
                        ├─ next tactical state
                        ├─ heading action
                        ├─ altitude action
                        ├─ speed action
                        └─ fire action
```

This may encourage the encoder to learn a useful latent representation of tactical state.

Do not start here; debugging a ten-head model before individual tasks are understood would make errors hard to isolate.

---

# 32. Evaluation metrics — current recognition

Report:

```text
macro F1
per-class precision
per-class recall
confusion matrix
balanced accuracy
Brier score
negative log likelihood
```

For temporally segmented behaviour, also consider:

```text
segment-level F1
transition detection delay
false transition rate
```

Detection delay:

\[
\Delta t_{detect}
=
t_{detected} - t_{actual}.
\]

This is particularly useful because a classifier that becomes correct only near the end of a manoeuvre may have high frame accuracy but low operational value.

---

# 33. Evaluation metrics — transition forecasting

For each horizon:

```text
AUROC
PR-AUC
Brier score
NLL
precision at useful recall levels
```

PR-AUC is important if transitions are relatively rare.

Also plot:

\[
P(\text{transition in }H)
\]

calibration curves.

---

# 34. Evaluation metrics — native action forecasting

For each branch:

```text
top-1 accuracy
top-k accuracy where meaningful
macro F1
cross-entropy / NLL
Brier score
```

Also measure joint action accuracy:

\[
P(
\hat a^\psi=a^\psi
\land
\hat a^h=a^h
\land
\hat a^V=a^V
\land
\hat a^{fire}=a^{fire}
).
\]

Because exact joint accuracy can be harsh, report both joint and branch-level results.

If action bins are ordinal, also report mean absolute bin error.

For example:

\[
MAE_{bin,\psi}
=
E[
|\hat i_\psi-i_\psi|
].
\]

This captures whether an incorrect action was nevertheless close to the correct command.

---

# 35. Probability calibration

Forecast probabilities are likely to be more valuable than hard classes.

After model selection, evaluate calibration using:

```text
reliability diagrams
Brier score
expected calibration error
negative log likelihood
```

Use temperature scaling as an initial calibration method if needed.

Do calibration on the validation set only and report final metrics on untouched test data.

---

# 36. Aircraft-normalised features

Once multiple platforms are introduced, consider both physical and capability-normalised variables.

Examples:

\[
V_{norm} =
\frac{V}{V_{ref}(A)}
\]

\[
h_{norm} =
\frac{h}{h_{ref}(A)}
\]

\[
n_{norm} =
\frac{n}{n_{max}(A)}.
\]

Do not discard the physical values.

Retain both:

```text
speed_ms
speed_norm

altitude_m
altitude_norm
```

and let experiments determine which representation transfers best.

---

# 37. Aircraft-aware versus aircraft-agnostic experiments

Once at least two aircraft types are available in the dataset, compare:

## Model A — aircraft agnostic

Uses only trajectory/geometry features.

## Model B — categorical aircraft identity

Adds one-hot aircraft type.

## Model C — learned aircraft embedding

```python
nn.Embedding(num_aircraft_types, embedding_dim)
```

## Model D — aircraft-normalised physics

Uses capability-normalised features but not aircraft identity.

These experiments answer a useful scientific question:

> Is the network recognising the manoeuvre, or exploiting the characteristic dynamics of a particular airframe?

---

# 38. Cross-aircraft generalisation programme

Suggested progression:

## Experiment 1

```text
train: F16
test: F16
```

Establish the basic task.

## Experiment 2

```text
train: F16
test: F15
```

Zero-shot domain transfer.

## Experiment 3

```text
train: F16 + F15 + FA18
test: F22
```

Held-out-aircraft transfer.

## Experiment 4

Fine-tune the held-out-aircraft model with:

```text
50
100
250
500
1000
```

new episodes and plot performance against adaptation data.

## Experiment 5

If a Typhoon model is eventually integrated:

```text
train: existing BVR Sim aircraft
test: Typhoon
```

Treat Typhoon as a strict unseen-domain test before fine-tuning.

---

# 39. Weapon-disabled phase first

The initial experiment should disable weapons.

Reasons:

```text
simpler causal structure
cleaner manoeuvre labels
fewer episode terminations
less class imbalance
easier debugging
```

The objective of Phase 1 is not to learn combat effectiveness.

It is to validate:

```text
simulation
logging
labels
coordinate transforms
sequence modelling
forecasting
```

---

# 40. Weapon-enabled phase later

Once current behaviour and action forecasting work, enable missile engagements.

Add tactical states such as:

```text
PRESS
LAUNCH
CRANK
SUPPORT
DEFEND
MISSILE_EVASION
EXTEND
RECOMMIT
```

BVR Sim already supports:

```text
missile loadouts
missile launch
engagement outcomes
missile-specific entity observations
```

At this stage the model can forecast:

```text
probability of transition to DEFEND
probability of fire action
likely post-launch manoeuvre
```

but these should remain separate tasks with explicit metrics.

---

# 41. Perfect-state versus sensor-limited datasets

Use two clearly differentiated data regimes.

## Regime A — oracle / simulator truth

Purpose:

> Determine whether behaviour is theoretically recoverable from the kinematics.

## Regime B — observable/sensor-limited

Add:

```text
measurement noise
track dropout
track latency
irregular update intervals
limited detection
partial observability
```

Then compare performance loss:

\[
\Delta M =
M_{truth} - M_{sensor}.
\]

This distinguishes model limitations from sensing limitations.

---

# 42. Software stack

Recommended initial dependencies:

```text
Python 3.10+ or 3.11
BVR Sim
JSBSim
Gymnasium

NumPy
SciPy
Pandas
PyArrow

scikit-learn

PyTorch
TensorBoard

MLflow

Matplotlib
Plotly

PyYAML
pytest
```

Optional:

```text
Pydantic
```

for validated configuration models.

Do not initially add:

```text
PyTorch Lightning
Ray
Dask
Spark
Hydra
Weights & Biases
```

unless the project develops a concrete need for them.

---

# 43. MLflow usage

MLflow becomes useful once experiments branch.

Log:

```text
dataset_id
dataset schema version
label schema version
BVR Sim commit
backend

train aircraft
test aircraft
aircraft-feature mode

feature set
history length
forecast horizon

model architecture
hidden size
layers
dropout
learning rate
batch size

macro F1
Brier score
NLL
transition PR-AUC
action branch accuracy
```

Store artifacts:

```text
confusion matrices
calibration plots
forecast-horizon plots
learning curves
scenario-stratified plots
model checkpoints
configuration files
```

---

# 44. Repository structure

```text
bvr_behavior_prediction/
│
├── simulator/
│   ├── bvr_adapter.py
│   ├── aircraft_config.py
│   ├── aircraft_registry.py
│   ├── scenario_config.py
│   └── backend_validation.py
│
├── policies/
│   ├── base.py
│   ├── kinematic.py
│   ├── pursuit.py
│   ├── beam.py
│   ├── crank.py
│   ├── extend.py
│   └── transitions.py
│
├── generation/
│   ├── scenario_sampler.py
│   ├── episode_runner.py
│   ├── privileged_logger.py
│   ├── dataset_builder.py
│   └── manifest.py
│
├── data/
│   ├── schema.py
│   ├── observable_columns.py
│   ├── privileged_columns.py
│   ├── transforms.py
│   ├── relational_features.py
│   ├── labels.py
│   ├── windows.py
│   └── torch_dataset.py
│
├── models/
│   ├── rule_baseline.py
│   ├── sklearn_baseline.py
│   ├── gru.py
│   ├── tcn.py
│   ├── multitask.py
│   └── aircraft_embedding.py
│
├── training/
│   ├── train_baseline.py
│   ├── train_current_state.py
│   ├── train_transition.py
│   ├── train_next_action.py
│   └── losses.py
│
├── evaluation/
│   ├── classification.py
│   ├── transitions.py
│   ├── action_forecasting.py
│   ├── calibration.py
│   ├── generalisation.py
│   ├── leakage_checks.py
│   └── plots.py
│
├── configs/
│   ├── aircraft/
│   │   └── f16.yaml
│   ├── scenarios/
│   ├── policies/
│   └── training/
│
├── tests/
│   ├── test_action_mapping.py
│   ├── test_coordinate_transform.py
│   ├── test_schema.py
│   ├── test_no_label_leakage.py
│   ├── test_episode_split.py
│   └── test_bvr_smoke.py
│
└── notebooks/
    └── exploratory_analysis.ipynb
```

---

# 45. Implementation milestones

## Milestone 0 — simulator validation

Goal:

```text
install BVR Sim
run F16 vs F16
validate Python backend
build and validate C++ backend
generate ACMI replay
confirm entity observations
confirm exact native actions
```

Acceptance criterion:

> One deterministic F-16 1-v-1 episode can be reproduced and all relevant state/action fields are understood.

---

## Milestone 1 — canonical logging

Implement:

```text
AircraftConfig
AircraftRegistry
ScenarioConfig
BVRSimAdapter
episode metadata
observable/privileged schema split
Parquet logger
```

Acceptance criterion:

> A generated trajectory can be loaded independently of BVR Sim and reconstructed into target-relative plots.

---

## Milestone 2 — primitive kinematic dataset

Use:

```text
F16 vs F16
weapons off
perfect observations
```

Generate controlled:

```text
straight
left/right turn
climb/level/descent
accelerate/steady/decelerate
```

These are multi-head labels, not one flat class.

Acceptance criterion:

> Rule-based and tree-based baselines recover these states reliably on unseen episodes.

---

## Milestone 3 — tactical relational behaviours

Add:

```text
pursue
beam
crank
extend
```

Randomise geometry and policy parameters.

Acceptance criterion:

> Tactical labels remain separable across broad held-out initial conditions.

---

## Milestone 4 — causal GRU recogniser

Input:

```text
~10 s history
target-relative trajectory features
```

Outputs:

```text
lateral state
vertical state
speed state
tactical state
```

Compare against:

```text
rule baseline
Random Forest
gradient boosting
```

Acceptance criterion:

> The sequence model improves meaningfully over non-temporal baselines, especially near ambiguous geometry.

---

## Milestone 5 — transition prediction

Construct explicit transition examples and targets:

```text
transition within 2 s
transition within 4 s
transition within 8 s
next tactical state
```

Compare with manoeuvre-persistence baselines.

Acceptance criterion:

> Forecast performance remains above calibrated persistence/event-rate baselines on held-out scenarios.

---

## Milestone 6 — native next-action prediction

Train four categorical output heads:

```text
15 heading bins
15 altitude bins
9 speed bins
2 fire states
```

Initially:

```text
fire = always disabled
```

so only the first three heads are active.

Acceptance criterion:

> The model beats repeat/most-common-action baselines across multiple horizons.

---

## Milestone 7 — weapon-enabled BVR behaviour

Enable missiles and add:

```text
fire head
missile entity observations
defensive behaviour
post-launch behaviour
```

Acceptance criterion:

> Weapon-related forecasts are evaluated with class-imbalance-aware event metrics, not raw accuracy alone.

---

## Milestone 8 — sensor degradation

Introduce:

```text
noise
dropout
latency
partial tracks
```

Acceptance criterion:

> Performance degradation is quantified relative to the oracle-state model.

---

## Milestone 9 — multi-aircraft data

Add available aircraft one at a time.

Start:

```text
F16 ↔ F15
```

then expand.

Acceptance criterion:

> Dataset schema and model code require no structural change when aircraft type changes.

---

## Milestone 10 — unseen-aircraft generalisation

Hold an aircraft out entirely during training.

Compare:

```text
aircraft agnostic
aircraft embedding
normalised physics
few-shot fine-tuning
```

This is the main proof that aircraft type was correctly treated as a domain variable.

---

# 46. First concrete experiment

The first experiment should be smaller and cleaner than the earlier plan.

## Simulator

```text
BVR Sim
C++ backend for main generation
Python backend for debugging
dt = 0.4 s
```

## Aircraft

```text
observer = F16
target = F16
```

## Weapons

```text
disabled
```

## Observation regime

```text
perfect simulator state
```

## Tactical behaviours

```text
MAINTAIN
PURSUE
BEAM
CRANK_LEFT
CRANK_RIGHT
EXTEND
```

## Kinematic heads

```text
lateral:
  STRAIGHT
  TURN_LEFT
  TURN_RIGHT

vertical:
  LEVEL
  CLIMB
  DESCEND

speed:
  STEADY
  ACCELERATE
  DECELERATE
```

## Pilot dataset size

```text
~200–500 episodes per tactical behaviour
```

before scaling.

## History

```text
10 s / 25 samples
```

## Models

```text
rules
Random Forest / HistGradientBoosting
GRU-128 × 2
TCN
```

## Metrics

```text
macro F1
per-class recall
confusion matrices
Brier score
NLL
detection latency
```

---

# 47. Second concrete experiment: forecasting

Use the same dataset structure.

## Transition targets

```text
transition within 2 s
transition within 4 s
transition within 8 s
```

## Native action targets

```text
heading action at +0.8 / +2 / +4 s
altitude action at +0.8 / +2 / +4 s
speed action at +0.8 / +2 / +4 s
```

Compare against:

```text
current-state persistence
most-common action
repeat inferred action
```

Plot:

\[
F1(h),\quad
NLL(h),\quad
Brier(h),\quad
ActionAccuracy(h).
\]

This horizon curve is more informative than one headline forecasting score.

---

# 48. Key technical risks

## Risk 1 — label leakage

Highest risk.

Mitigation:

```text
privileged schema
feature allow-list
unit tests
shuffled-label tests
```

---

## Risk 2 — trivial temporal persistence

A forecast model may score highly simply because manoeuvres last for many timesteps.

Mitigation:

```text
persistence baselines
transition-conditioned evaluation
event-based metrics
```

---

## Risk 3 — flat labels hide simultaneous behaviours

Mitigation:

```text
separate kinematic and tactical heads
```

---

## Risk 4 — overfitting to scripted policy signatures

The network may identify the exact controller rather than the underlying behaviour.

Mitigation:

```text
parameter randomisation
multiple implementations of the same behaviour
held-out policy parameters
eventual RL-generated trajectories
```

---

## Risk 5 — aircraft-specific dynamics become shortcuts

Mitigation:

```text
aircraft-agnostic features
target-relative coordinates
held-out-aircraft tests
aircraft-normalised variables
```

---

## Risk 6 — simulator/backend drift

Mitigation:

```text
pin BVR Sim commit
record backend
record model/controller versions
version every dataset
```

---

## Risk 7 — synthetic-to-real gap

This project initially measures inference inside BVR Sim, not necessarily real-world air-combat prediction.

Mitigation:

```text
state this scope explicitly
introduce sensor degradation
test domain shifts
avoid interpreting simulator accuracy as real-world accuracy
```

---

# 49. Testing requirements

The project should contain automated tests for at least:

```text
BVR Sim smoke episode
native action conversion
angle wrapping
coordinate rotation
range/range-rate calculation
aspect calculation
window indexing
transition labels
no privileged columns in model input
episode split isolation
deterministic seed behaviour
Parquet schema
aircraft registry lookup
```

The coordinate and angle tests are especially important because errors around:

```text
±π
0/360 degrees
bearing convention
body-frame sign convention
```

can silently corrupt the dataset.

---

# 50. Recommended model-development order

Use this order:

```text
1. Rule-based recogniser
2. Random Forest / HistGradientBoosting
3. GRU
4. TCN
5. Multi-task GRU/TCN
6. Only then consider Transformer/attention
```

Reasons:

```text
short sequences
small feature vectors
laptop training
need for interpretability
need to expose leakage/triviality early
```

A Transformer should be a hypothesis-driven addition, not a default architecture.

---

# 51. Future use of entity-set models

If the project later expands beyond 1-v-1, BVR Sim's entity representation makes set-based models attractive.

Candidates:

```text
Deep Sets
Set Transformer
relation network
graph neural network
graph attention network
```

For 1-v-1, these are unnecessary.

For 2-v-2 or larger scenarios, they become useful because the model must process a variable number of:

```text
allied aircraft
enemy aircraft
missiles
other entities
```

without depending on arbitrary entity ordering.

---

# 52. Reference papers and resources

## BVR Sim

Haocheng Sun and Mulai Tan, **"BVR Sim: An Open and High-Throughput Environment for Heterogeneous Air-Combat Reinforcement Learning"**, arXiv:2608.25419, 2026.

- https://arxiv.org/abs/2608.25419
- https://github.com/lizi-Margin/bvr_sim

Relevant to:

```text
simulator architecture
heterogeneous aircraft
entity observations
high-level tactical interface
Python/C++ backends
cross-aircraft transfer
```

---

## Hierarchical air-combat RL

Adrian P. Pope et al., **"Hierarchical Reinforcement Learning for Air-to-Air Combat"**, arXiv:2105.00990, 2021.

- https://arxiv.org/abs/2105.00990

Relevant to:

```text
hierarchical control
air-combat policy abstraction
expert knowledge
modular tactical behaviours
```

---

## Tactical intention recognition

Xingyu Wang, Zhen Yang, Haiyin Piao, Shiyuan Chai, Jichuan Huang and Deyun Zhou, **"Intelligent recognition method of target tactical behavior intention in air combat based on deep learning"**, *Engineering Applications of Artificial Intelligence*, 138, 109460, 2024.

- DOI: https://doi.org/10.1016/j.engappai.2024.109460

The paper is directly relevant because it treats air-combat tactical intention as a time-series recognition problem using relative situation information and evaluates convolution + Bi-LSTM + self-attention on 1-v-1 simulation data.

For this project, its Bi-LSTM architecture should be regarded as an offline comparison rather than the default causal online model.

---

## GRU

Kyunghyun Cho et al., **"Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"**, EMNLP 2014.

- https://arxiv.org/abs/1406.1078
- DOI: https://doi.org/10.3115/v1/D14-1179

Relevant as the foundational GRU reference.

---

## Temporal convolutional networks

Shaojie Bai, J. Zico Kolter and Vladlen Koltun, **"An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling"**, arXiv:1803.01271, 2018.

- https://arxiv.org/abs/1803.01271

Relevant to the proposed TCN comparison.

---

## Probability calibration

Chuan Guo, Geoff Pleiss, Yu Sun and Kilian Q. Weinberger, **"On Calibration of Modern Neural Networks"**, ICML 2017.

- https://arxiv.org/abs/1706.04599
- https://proceedings.mlr.press/v70/guo17a.html

Relevant to:

```text
temperature scaling
reliability diagrams
calibrated confidence
```

---

## Deep Sets

Manzil Zaheer et al., **"Deep Sets"**, NeurIPS 2017.

- https://arxiv.org/abs/1703.06114

Relevant to eventual variable-size multi-entity observations.

---

# 53. Final recommended architecture

The project should be thought of as five deliberately separated layers:

```text
AIRCRAFT DOMAIN
F16 / F15 / F22 / future Typhoon / ...
        ↓

BVR SIM SIMULATION
JSBSim dynamics + BVR Sim controller + sensors
        ↓

BEHAVIOUR GENERATOR
scripted skill / tactical policy / later RL policy
        ↓

OBSERVABLE TRAJECTORY
target-centric state + relative geometry
        ↓

ML INFERENCE
current kinematics
current tactical behaviour
transition probability
next tactical behaviour
next native BVR Sim action
```

The central design principle is:

> **The aircraft model, tactical intent, physical execution and ML observation should be separate concepts.**

If those boundaries are maintained, the project can begin with a very controlled F-16 experiment and later expand to multiple aircraft without changing the fundamental data or model architecture.

---

# 54. Recommended immediate implementation sequence

The first coding work should proceed in this order:

1. Pin the BVR Sim repository revision.
2. Install and smoke-test the Python backend.
3. Build and smoke-test the C++ backend.
4. Run the supplied F-16 1-v-1 configuration.
5. Inspect the exact `entity` observation array and its feature ordering.
6. Inspect the exact executed `MultiDiscrete([15,15,9,2])` action mapping.
7. Implement `AircraftConfig` and `AircraftRegistry`.
8. Implement `ScenarioConfig`.
9. Implement a thin `BVRSimAdapter`.
10. Define the observable and privileged schemas.
11. Implement relative-coordinate transforms and unit tests.
12. Implement one simple `MaintainPolicy`.
13. Implement one `TurnPolicy`.
14. Generate a smoke dataset.
15. Plot trajectories and labels.
16. Verify no privileged information enters the feature matrix.
17. Implement the rule-based baseline.
18. Implement the scikit-learn baseline.
19. Generate the pilot tactical dataset.
20. Train the causal GRU.
21. Add transition labels.
22. Add next-action forecasting.
23. Only then scale dataset size or model complexity.

This sequence minimises the chance of producing a large dataset whose labels, coordinate system or forecast target are subtly wrong.
