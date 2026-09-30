# Extending Neural Training to Multi-Aircraft Engagements and Tactical Datalinks

## Purpose

This document proposes an incremental extension of the existing supervised-learning and reinforcement-learning infrastructure so that engagements can contain arbitrary numbers of aircraft on either side. It also describes how to represent a tactical datalink without exposing privileged simulator truth to a deployed policy.

The central design choice is to stop treating an engagement as one fixed-width observer/target vector. Instead, each aircraft receives a variable-size set of locally observable entities and tracks. A permutation-invariant spatial encoder summarizes that set before the existing temporal encoder processes its history.

## Current architecture and constraints

### Supervised models

The supervised pipeline currently builds a fixed-length temporal window for one observer/target pair. The pairwise feature vector contains observer-relative geometry, track validity, and track age. A GRU or TCN reduces the window to one latent vector, and fixed linear heads predict tactical state, transitions, or actions.

This design has useful foundations:

- observer-body relative positions and velocities;
- range, range rate, bearing, aspect, and line-of-sight features;
- episode-bounded temporal windows;
- separate observable and privileged fields;
- interchangeable temporal encoders and task heads.

Its main limitation is that the schema and dataset describe exactly one observer and one target per row. Increasing the input dimension to reserve slots for more aircraft would impose a fixed maximum, make predictions depend on slot ordering, and require a different network shape for different engagement sizes.

### Reinforcement-learning pilot

The RL pilot uses a temporal Transformer over a flat observation history. It outputs a categorical tactical skill, continuous skill parameters, and a scalar value. The rollout collector already batches policy inference across independent simulator instances.

The surrounding environment and JSBSim adapter are nevertheless specialized for one-versus-one combat:

- one controlled aircraft and one opponent have fixed IDs;
- the backend expects a fixed 47-element observation;
- energy features are hard-coded as blue and red values;
- each environment step constructs one blue and one red action;
- scenarios contain one altitude, heading, and speed per side;
- the separate simulator RL wrapper rejects multiple controlled aircraft.

The simulator observation code already iterates over collections of enemies and partners, so the simulator is closer to multi-aircraft support than the current learning boundary. Its output is still flattened to a dimension that varies with the configured team sizes, however, and should be converted to structured entity records before entering a policy.

## Proposed observation representation

For every controlled aircraft and time step, construct:

```text
self_state:       [batch, time, self_features]
entities:         [batch, time, max_entities_in_batch, entity_features]
entity_type:      [batch, time, max_entities_in_batch]
entity_mask:      [batch, time, max_entities_in_batch]
entity_id:        [batch, time, max_entities_in_batch]  # metadata only
```

`max_entities_in_batch` is a batching detail, not a model limit. Each minibatch is padded only to its largest entity set, and the mask prevents padding from influencing attention, pooling, losses, or action selection.

An observation can contain the following token classes:

- the observing aircraft;
- friendly aircraft;
- hostile or unidentified tracks;
- friendly and hostile missiles or missile tracks;
- optional support assets, ground threats, objectives, and defended zones.

Apply the existing observer-centric relational feature calculation independently to every entity. Body-frame geometry provides translation invariance and reduces dependence on global map orientation. Add categorical embeddings for entity class, relation to the observer, aircraft type, sensor source, and track status.

Entity order must not carry tactical meaning. Aircraft IDs are useful for associating tracks over time and applying actions, but raw IDs or spawn positions should not become learned slot semantics.

## Spatial and temporal network architecture

Use a hierarchical encoder:

```text
Per-entity feature projection
        -> spatial entity encoder at each time step
        -> observer-conditioned attention or masked pooling
        -> one engagement latent per observer and time step
        -> existing temporal Transformer, GRU, or TCN
        -> policy, value, classification, and forecasting heads
```

A minimal implementation can use a Set Transformer or cross-attention in which the self-aircraft latent is the query and visible entity latents are keys and values. This creates a fixed-width spatial summary from any number of entities.

For richer coordination, use a graph neural network:

- nodes represent aircraft, tracks, missiles, and support assets;
- edges contain observer-relative geometry;
- edge types distinguish friendly, hostile, sensor detection, datalink report, missile targeting, and command relationships;
- message passing models screening, mutual support, bracketing, target allocation, and missile support.

The encoder must satisfy these properties:

- permutation equivariance while processing individual entities;
- permutation invariance when constructing an engagement summary;
- explicit masks for padding, dead entities, invalid tracks, and unavailable capabilities;
- parameter sharing across aircraft where roles and capabilities permit it;
- source and uncertainty awareness for imperfect tracks.

## Variable-size prediction and action heads

Fixed categorical heads remain appropriate for maneuvers, tactical modes, and other fixed vocabularies. Decisions referring to aircraft must instead operate over the current entity set.

A hybrid multi-aircraft action should contain:

1. a tactical skill or maneuver;
2. a pointer distribution over valid hostile-track tokens;
3. an optional pointer to a friendly aircraft, missile, formation slot, or support asset;
4. continuous skill parameters;
5. an optional communication action.

Target logits can be produced with an attention score between the policy latent and each hostile entity latent. A mask excludes padding, friendlies, dead entities, and unusably stale tracks. This avoids output neurons with meanings such as `target_1` and `target_2`, which would not generalize to different team sizes.

Supervised target-level tasks should similarly emit one prediction per eligible entity. Losses are evaluated only where both the entity mask and label mask are valid. Engagement-level predictions can continue to use the pooled spatial latent.

## Dataset and schema changes

Replace the one-wide-row-per-observer/target representation with normalized logical tables or equivalent nested records.

### Episodes

```text
episode_id, seed, scenario and simulator versions
blue_count, red_count, team compositions
sensor and datalink configuration
episode duration and termination reason
```

### Aircraft states

```text
episode_id, step, time_s, aircraft_id, team_id, aircraft_type
position, velocity, attitude, speed
alive state, fuel, damage, and weapon inventory
privileged tactical labels
```

### Observations and tracks

```text
episode_id, step, observer_id, track_id
truth_entity_id                 # privileged training/evaluation metadata
friendly, hostile, or unknown relation
sensor/datalink source
reported position and velocity
covariance or track quality
track_valid, track_age, last_update_age
classification and confidence
```

### Actions

```text
episode_id, step, actor_id
skill or action
target_track_id
continuous parameters
```

### Datalink messages

```text
episode_id, send_step, delivery_step
sender_id, recipient/team/channel
message_type, referenced_track_id, payload
dropped, corrupted, or expired status
```

The generator should iterate over every observing aircraft, retrieve only the sensor and datalink tracks available to that aircraft, and emit one observation set per observer and time. World state and truth associations must stay in privileged storage. Track identity should persist through loss, prediction, and reacquisition.

The window dataset should return structured tensors and masks rather than one list of flat rows. A custom collator can pad each minibatch to its local maximum entity count. The existing episode-bounded temporal index can remain in use.

## Multi-agent environment API

Change the backend contract to dictionaries keyed by aircraft ID:

```python
observations: dict[str, AgentObservation] = env.reset(seed)

next_observations, rewards, terminated, truncated, info = env.step(
    actions: dict[str, HybridAction]
)
```

The environment should also:

- return per-aircraft rewards and termination flags;
- return a separate engagement-level termination flag;
- remove destroyed aircraft from the active policy mask while allowing teammates to continue;
- expose energy and event information per aircraft rather than as one blue/red pair;
- accept arbitrary team rosters in scenario configuration;
- distinguish controlled, scripted, and learned aircraft independently of team color.

## Multi-agent PPO

Use a shared actor for aircraft with compatible action spaces. Aircraft type, role, doctrine, and capabilities become embeddings or features, while capability masks prevent impossible actions.

The rollout index becomes:

```text
(environment_id, agent_id, time)
```

Each transition stores local structured observations and masks, selected skill and target, continuous parameters, old log probability, reward, agent termination, and engagement termination. Policy inference can batch all active aircraft from all active simulators into one accelerator call.

Begin with a critic that sees the same local information as the actor. Once the multi-agent loop is stable, add centralized training with decentralized execution:

- the actor receives only onboard sensors and delivered datalink information;
- the training-only critic can receive a joint team observation or privileged world state;
- deployment requires only the decentralized actor.

Rewards should combine mission/team outcome, individual survival and weapon employment, and coordination terms. Team rewards should be normalized deliberately so that their magnitude does not grow accidentally with team size.

## Tactical datalink design

The simulator already contains an SA datalink receiver that periodically provides noisy enemy positions and can cue radar scan direction. It is best treated as a prototype AWACS/GCI feed. A tactical link should be made explicit and causal:

```text
sensor measurement
    -> local tracker
    -> message serialization
    -> network delay, capacity, loss, and corruption
    -> recipient inbox
    -> track association and fusion
    -> recipient entity tokens
```

### Track-report contents

A report should include:

- sender and sender timestamp;
- sender-local track ID;
- estimated position and velocity;
- covariance or quality;
- classification and confidence;
- friendly, hostile, or unknown status;
- last sensor update time;
- engagement or weapon-support status.

Intent and command messages can represent engagement assignment, support requests, defended sectors, formation roles, weapon launches, or lost tracks.

### Network behavior

At minimum, model:

- update rate;
- end-to-end latency;
- packet loss;
- finite bandwidth or message count;
- stale-message expiry;
- uncertainty growth with age;
- sender loss or disconnection;
- optional range, line-of-sight, jamming, and corruption;
- duplicate reports and track association.

Each recipient must maintain a separate delayed local view. A datalink update must not instantaneously mutate a globally shared team track.

### Neural track features

Every track token should encode source, report age, time since local confirmation, uncertainty, sender role, link availability, validity, and whether a weapon is being supported. This lets the model distinguish a fresh onboard-radar track from an old, low-quality remote report.

Initially, fuse reports with a deterministic local tracker and present one token per believed entity. Truth IDs may support training labels and metrics but must never be actor inputs.

### Communication-learning stages

Introduce communication progressively:

1. **Fixed broadcast:** automatically send all qualifying tracks.
2. **Rule-based selection:** send only new, changed, high-quality, threatened, or weapon-supported tracks.
3. **Learned communication:** let the policy select whether to transmit, the message type, referenced track, and recipient scope.

For learned communication, include transmission decisions in the PPO log probability and impose explicit bandwidth or message costs. Starting with fixed broadcast isolates datalink transport and fusion bugs from policy-learning failures.

## Training curriculum and evaluation

Use a curriculum while retaining variable entity counts:

1. reproduce current one-versus-one behavior;
2. introduce two-versus-one and one-versus-two;
3. train two-versus-two;
4. randomize team sizes and aircraft mixtures;
5. reserve larger engagements for extrapolation tests;
6. randomize sensor quality, link latency, loss, bandwidth, and jamming.

Useful augmentation includes shuffling entity order, rotating and translating engagement geometry, swapping equivalent aircraft identities, and changing which friendly aircraft is the observer.

Evaluation should report results by team size and include:

- mission outcome and survival;
- target-allocation quality and duplicate shots;
- local and team track coverage;
- missile-support duration;
- communication load and useful-message rate;
- robustness to latency, loss, and jamming;
- no-link, perfect-link, realistic-link, and learned-link ablations;
- decentralized versus centralized critic comparisons.

## Required tests

### Unit tests

- predictions are invariant to entity permutation;
- padding does not affect outputs;
- invalid, dead, friendly, and padded targets cannot be selected;
- message latency, loss, and expiry are causal;
- datalink-disabled observations contain no remote tracks;
- privileged truth identifiers never enter actor features;
- duplicate reports are associated or fused consistently.

### Integration tests

- one-versus-one behavior remains compatible with current baselines;
- two-versus-two reset, step, observation, action, reward, and termination contracts work;
- the engagement continues after one aircraft is destroyed;
- different team sizes coexist in one minibatch;
- target selection remains correct after entity shuffling;
- delivered messages appear only after their scheduled delay.

## Incremental implementation plan

1. Introduce structured `AgentObservation`, entity, and track types.
2. Refactor backend reset and step APIs to use aircraft-ID dictionaries.
3. Add batch-local ragged collation and masks.
4. Add a set-attention encoder before the existing temporal encoder.
5. Add pointer-based target selection and action masks.
6. Generalize scenario rosters, energy reporting, rewards, and termination.
7. Extend PPO rollouts to `(environment, aircraft, time)` and share policy parameters.
8. Integrate fixed datalink delivery, provenance, track age, uncertainty, and local fusion.
9. Add a centralized training critic after decentralized execution is stable.
10. Add rule-based and then learned communication.

This sequence preserves the existing relative-geometry features, temporal encoders, hybrid skill policy, batched rollout collection, and observable/privileged separation while removing fixed one-versus-one assumptions at each system boundary.
