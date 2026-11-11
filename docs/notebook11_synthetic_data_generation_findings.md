# Notebook 11: synthetic-data coverage, skill execution and speed override

Discussion summary, 11 November 2026, on branch `forecasting_ed_261104`.

This note complements the [notebook 27 weighted-F1 discussion](notebook27_weighted_f1_discussion.md).
It reviews [notebook 11](../notebooks/11_nn_vs_nn_training_data.ipynb), its
[collector](../scripts/generate_nn_duel_corpus.py), notebook 27's saved outputs,
and the frozen pilot source used for collection. No new rollouts or training were
run, and no simulator, policy or collection behaviour was changed by this review.

## 1. Selected labels do not necessarily imply distinct physical skills

The catalogue provides names, parameter schemas and textual descriptions of start,
termination and interruption conditions. Those descriptions are not executable
controllers. In the frozen
[`TacticalSkill.execute()`](../bvr_sim_source/bvr_sim/agents/skill_manager.py),
the implemented transformations are essentially:

- Start with zero heading, altitude and speed deltas.
- If `heading_deg`, `altitude_m` or `speed_mps` exists, calculate the corresponding
  error from the aircraft's own state.
- For names ending in `_left` or `_right`, replace the heading delta with a fixed
  signed value of minus or plus pi/2 radians.
- For `launch`, `secondary_shot` and `short_range_attack`, request firing.
- Calculate completion from timeout and the implemented setpoint checks.

There is no separate dive or drag controller. Both `dive_defense` and
`drag_missile` have `threat_id`, `max_g`, `min_altitude_m` and `timeout_s`
parameters, but the executor does not use the first three to generate guidance or
enforce safety limits. In particular, `min_altitude_m` does not match the exact
`altitude_m` key checked by the executor. Parameter type/range validation is not
the same as enforcing a manoeuvre's conditions.

Both skills therefore emit the same numerical guidance:

```python
delta_heading = 0.0
delta_altitude = 0.0
delta_speed = 0.0
shoot = 0
```

Their output dictionaries retain different `skill_name` values, but the downstream
aircraft-control path does not dispatch a specialised controller from those names.
Other equivalences, before normalisation and environment overrides, include:

| Labels | Numerical guidance |
| --- | --- |
| `dive_defense`, `drag_missile`, `last_ditch_break`, `defensive_reversal`, `preemptive_crank` | Zero deltas; no firing request. |
| `pursue_target`, `lead_pursuit`, `lag_pursuit`, `turn_cold`, `extend`, `recommit` | The same zero-delta, no-fire output. |
| `search`, `lock_target`, `commit`, `support_missile`, `abort_support` | The same zero-delta, no-fire output. |
| `beam_target_left`, `crank_target_left`, `notch_left`, `beam_missile_left` | Heading delta -pi/2; zero altitude/speed deltas; no firing request. |
| Corresponding right-sided labels | Heading delta +pi/2; otherwise the same. |
| `launch`, `secondary_shot`, `short_range_attack` | Zero flight-command deltas; firing requested. |

The left/right requests are subsequently clipped and normalised; they do not
establish that the named target-relative geometry is achieved. The kinematic
skills implement setpoint errors, but their names do not enforce direction:
climb and descend, for example, use the same altitude-error calculation.

The collector supplies `execute()` with time and own-aircraft heading, altitude
and speed, without target or missile geometry. It discards the returned completion
flag (`guidance, _ = ...`). The NN selects again every second; a changed name or
parameter set replaces the active skill. The catalogue's full lifecycle is
therefore not enforced in this collection path.

Zero guidance is not zero motion: aircraft dynamics and controller state still
matter. Equal immediate commands also do not imply identical complete flights,
because states and subsequent policy selections can differ.

## 2. What the "forced-throttle setting" actually does

The frozen [environment](../bvr_sim_source/bvr_sim/bvr_env.py) enables:

```python
BYPASS_RL_THROTTLE_CONTROL = True
```

Before passing the action to the aircraft, it applies:

```python
if BYPASS_RL_THROTTLE_CONTROL:
    action_dict["delta_speed"] = 1.0
```

Thus negative, zero and positive skill-generated speed requests all become the
same positive normalised request. The more precise term is **speed-command
override**, rather than a literal fixed engine-throttle position.

In the frozen
[C++ JSBSim adapter](../bvr_sim_source/bvr_sim/src_cxx/simulator/aircraft/fdm/jsbsim_fdm.cxx),
the request is interpreted as:

```cpp
double intent_mach = mach_current + action.delta_speed();
```

The Python JSBSim adapter uses the same relation. With the override, the desired
Mach value is one above the current value at the adapter's command update. This
creates a persistent acceleration demand; the flight controller determines the
actual engine controls. It is not 1 m/s, a fixed target of Mach 1, or proof of a
constant 100% throttle position. Earlier discussion shorthand calling it
"full throttle" should be read with this correction.

An acceleration or deceleration skill's calculated speed error is overwritten.
Observed slowing can still result from climbing, turning, drag or insufficient
thrust. It does not demonstrate that a deceleration request was honoured. The
override also affects how long low-energy initial states persist; the magnitude
of that effect has not been measured here.

Disabling the override would change the environment experienced by the frozen
pilots. It should be evaluated in a separately versioned campaign, with pilot
behaviour reassessed, rather than treated as an equivalent resumption.

## 3. Consequences for rare-skill coverage and forecasting

The stored labels record actual NN selections. They do not independently verify
the physical behaviours described by the catalogue. More `dive_defense` labels,
for example, would not introduce a dedicated dive command in this executor.

Context and history may still make equivalent-command labels predictable. These
findings neither prove that label prediction is impossible nor establish the cause
of notebook 27's F1 scores or rare-label frequencies. They do limit interpreting
those scores as recognition of distinct executed manoeuvres.

Notebook 27's saved retained-training support includes:

| Skill | +1 s labels | +2 s labels | +5 s labels |
| --- | ---: | ---: | ---: |
| `accelerate_to_speed` | 1 | 1 | 0 |
| `climb_to_altitude` | 1 | 3 | 2 |
| `commit` | 0 | 1 | 1 |
| `dive_defense` | 1 | 0 | 0 |
| `drag_missile` | 5 | 4 | 4 |

These are retained labels, not independent flights. They supersede treating the
older notebook 09b zero-support observations as counts for this newer corpus.
Audit all issued decisions, history-eligible decisions, each horizon's labels,
and post-thinning training labels separately. The first forecast origin is at
6 seconds, so the earliest +1/+2/+5 targets are at 7/8/11 seconds. Initial-only
selections can therefore be absent from forecast support even when issued.

Count independent flights, selection runs and transitions alongside windows.
Preserve genuine preceding history when collecting later events; do not fabricate
history or extend labels beyond an episode's final action.

## 4. Recommended changes to the sampling methodology

Notebook 11 already has ten start families and stratified continuous samples,
with initial altitude approximately 1.5-11.5 km, true airspeed 200-380 m/s and
horizontal separation 6-65 km. The next step is joint validation and coverage,
rather than simply widening these marginal bounds.

- **Aircraft envelope:** sample feasible altitude-speed combinations conditional
  on aircraft configuration, mass, fuel and atmosphere. Check Mach, dynamic
  pressure, lift and load-factor margins, thrust feasibility, ground clearance
  and initialisation consistency. A maximum speed and service ceiling do not
  define a rectangular feasible region. Simulator acceptance alone does not
  validate real-world limits. Reject/resample invalid proposals and record why;
  clipping creates artificial boundary concentrations.
- **Energy:** cover feasible low/middle/high states and relative energy differences.
  Specific energy height `h + V_TAS**2/(2*g)` is useful, but retain altitude and
  speed separately. Randomise the fixed aircraft roles in `low_energy_climb` and
  `high_low`, and balance condition coverage across pilot pairs.
- **Aspect and range:** measure both aircraft's line-of-sight-relative headings,
  closure, horizontal separation and slant range from actual states. Balance
  handedness and roles. Family names and global marginal stratification do not
  guarantee joint or within-family coverage. Excluding nearly antiparallel starts
  is not necessary for physical realism; novelty can come from other dimensions.
- **Missile exposure:** audit actual versus observed exposure, multiplicity,
  duration and lifecycle stage in valid simulated histories. A named defensive
  start family does not guarantee exposure. Any continuation must preserve
  consistent aircraft, missile, sensor, controller and policy-history state.
- **Collection design:** retain a broad component alongside undercovered feasible
  conditions, log sampling rules and retain all outcomes. Inspect frozen-policy
  probabilities when labels remain absent: deterministic argmax can leave a
  catalogue entry unselected despite nonzero probability. Stochastic selection
  changes the behaviour distribution and needs separate provenance. Forced
  selections do not demonstrate autonomous choice by the frozen pilots.

First decide whether the study targets frozen-policy label prediction or distinct
physical behaviour. The latter requires separately versioned executor work and
behaviour validation before broader collection can establish physical-skill
coverage.

Keep development evaluation examples fixed and reserve fresh scenario groups for
independent assessment. Both perspectives, continuations and related variants
belong in the same partition. Report weighted F1 alongside macro-F1, per-skill
precision/recall, independent support and physical-behaviour checks. The common
classes dominate the existing weighted F1; rare-label coverage is a separate
success criterion. Without an empirical operational distribution, describe the
result as physically plausible, coverage-oriented synthetic data.

## Evidence and verification

The original review used commit `14e91b11ee1d7e7fbcffcf5dd0c7671421d99aa4`;
the branch was refreshed to `49cc56296b48f8b630c7d592973a8864ab163897` before
adding this summary. The frozen archive was
[`trained-pilot-episodes-v1/reproduction.zip`](../shared/corpora/trained-pilot-episodes-v1/reproduction.zip),
SHA-256 `ca592273338d0cd9c45e543de24fda752045de090f6f6dbd23d319c5e44993f1`,
matching its publication manifest. Its skill manager, environment and C++ JSBSim
adapter matched the reviewed checkout after newline normalisation. Direct calls
to the frozen skill executor confirmed the command equivalences above. This was
a source/executor audit, not an end-to-end flight experiment or an evaluation of
new training data.
