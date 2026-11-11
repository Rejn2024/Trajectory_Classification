# Retraining environment: 11 November 2026

Branch: `retraining_environment_261111`.

Base: `forecasting_ed_261104` at
`d9ba4d8b85c2da4f947161a981bea89fc59ae0a8`.

This branch contains the general speed-control changes following the
[skill-executor audit](notebook11_synthetic_data_generation_findings.md).
It prepares a selectable speed-control mode for future experiments. It does not
contain newly trained pilots or a complete implementation of the tactical-skill
catalogue.

## Modifications

- **Honour speed requests by default.** The Python `BVR3DEnv` action path now
  preserves negative, zero and positive normalised speed requests. The legacy
  forced `+1.0` request is opt-in through the boolean environment setting
  `bypass_rl_throttle_control`.
- **Keep configuration local to each environment.** The validated setting is
  captured at construction, allowing different instances to use different modes.
  Action dictionaries are copied before modification, preserving caller inputs.
- **Apply the choice to notebook 11's archived simulator.** Each isolated worker
  sets the existing frozen simulator switch from the recorded
  `native_throttle_override` configuration. Archive files and checksums remain
  unchanged. The environment configuration also receives the per-instance option.
- **Expose a reproducible collection option.** Notebook 11 defaults to
  `FORCE_SPEED_COMMAND=False`. The collector's `--force-speed-command` flag selects
  legacy mode. Campaign configuration, setup and benchmark identities, and the
  generated corpus README identify the selected mode. Existing checks prevent
  incompatible resumption; a changed mode requires a fresh series.
- **Document the behaviour and add regression checks.** Tests cover request
  propagation, legacy behaviour, input preservation, instance isolation, boolean
  validation, frozen-worker initialisation, CLI/notebook option propagation,
  cache invalidation and configuration mismatch on resumption. Existing notebook
  tests' obsolete 20,000-flight expectations were aligned with its existing
  10,000-flight default; the default flight count itself was not changed.

See [Requested speed control and legacy override](speed_command_control.md) for
configuration details. This is a speed-request change, not a fixed engine-throttle
percentage. It does not guarantee a particular acceleration or rectify the
existing downstream action scaling. The separate `BVR3DEnvCpp` environment is
outside the change; the Python environment can still use the native JSBSim FDM.

## Manoeuvre-controller scope

No tactical-skill executor was changed. The shared numerical outputs identified
in the audit therefore remain shared. The limitation is not that the names are
impossible to implement: this work excludes combat-specific targeting,
weapon-employment and missile-evasion controllers. Ordinary heading, altitude
and airspeed control remain within the implemented general flight-control scope.

## Validation and experimental limits

The targeted regression suite passed 95 tests. It exercises the actual Python
environment constructor and `step()` with an aircraft test double, plus isolated
frozen-worker initialisation and collection configuration/packaging checks.
Documentation links, notebook cell compilation and `git diff --check` were also
checked.

No native flight simulations, new corpus collection or retraining were run.
These checks establish command delivery and configuration consistency, not
physical trajectory, engine response or model-performance improvements. Frozen
pilots were trained under the previous override and need reassessment in the new
mode. Saved notebook outputs remain historical results from earlier runs.

Notebook publication routing was subsequently updated to
`retraining_environment_261111`: notebook 11's branch guard, its instructions and
the collector publisher's default destination agree. All notebook sources were
checked for the previous route. Existing historical outputs and archived corpus
collectors remain records of their original runs. No corpus-publication job was
started by the routing update.
