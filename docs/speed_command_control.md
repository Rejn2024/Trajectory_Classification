# Requested speed control and legacy override

For a branch-level record, see the
[retraining environment modification summary](retraining_environment_261111.md).

The Python environment now honours requested normalised speed deltas by default.
Previously it unconditionally replaced them with `+1.0`. Heading and altitude
requests follow the existing control path. This change addresses general flight
control plumbing; it does not implement the missing combat-specific skill
controllers described in the [executor audit](notebook11_synthetic_data_generation_findings.md).

## Live Python environment

```python
config["bypass_rl_throttle_control"] = False  # Default: preserve speed requests.
```

Set this boolean to `True` only to reproduce the legacy forced-positive-speed
behaviour. It is captured per environment when constructed, so two instances can
use different modes. Non-boolean values are rejected to prevent a string such as
`"false"` being interpreted as enabled. Dictionary actions are copied before
applying an override, preserving the caller's requested action for reuse/auditing.

The flag controls the `BVR3DEnv` Python action path; it is not a new option for the
separate `BVR3DEnvCpp` environment. The Python environment can still drive the
native JSBSim flight-dynamics adapter, as notebook 11 does.

With the override disabled, a negative speed request remains negative at the
aircraft interface, zero remains zero, and a positive request remains positive.
The existing normalisation and downstream flight controller still determine how
those requests translate into engine controls. This does not directly set an
engine-throttle percentage or guarantee acceleration in every flight condition.

## Notebook 11 and its frozen simulator

Notebook 11 imports a verified archived simulator, so editing the live checkout
alone cannot alter its speed handling. New campaigns now record
`native_throttle_override: false` in `config.json`. Each isolated rollout worker
applies that recorded value to the archived simulator's existing
`BYPASS_RL_THROTTLE_CONTROL` switch in memory. The verified archive is not edited.
The value is also passed as environment configuration for simulators supporting
the per-instance option.

The first notebook cell exposes:

```python
FORCE_SPEED_COMMAND = False
```

The collector CLI honours speed requests by default. To explicitly select the
legacy mode for a new campaign, add `--force-speed-command`. The option is recorded
in configuration, setup identity, benchmark identity and the generated corpus
README. Configurations missing the field retain the archived legacy behaviour
when loaded by a worker; newly prepared configurations always contain the field.

Start a fresh series when changing this mode or using the changed collector.
The existing source/configuration checks reject incompatible resumption. The
frozen pilots were trained with the old override, so behaviour with honoured
speed requests needs reassessment. Existing saved notebook outputs describe
historical runs; they are not measurements of the new default.

## Scope and verification

The speed fix makes existing acceleration/deceleration requests available to the
controller. It does not make an unimplemented manoeuvre label produce a distinct
physical behaviour. Skill execution, policy selection and observed motion remain
separate properties to validate.

Regression checks exercise the real Python environment constructor and `step()`
with a recording aircraft double, plus isolated frozen-worker initialisation.
They cover both modes, request signs, input immutability, independent environment
settings, strict boolean parsing, CLI propagation, cache identity and rejection
of changed-mode resumption. They do not establish native JSBSim trajectory,
throttle-response or policy-performance results. No collection or training job
was launched for this change.
