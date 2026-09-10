# Tactical Skill Manager

`bvr_sim.agents.SkillManager` is a registry of persistent tactical manoeuvres.
It is intended to sit between a slow strategist (rules, a behaviour tree, a
learned selector, or an LLM) and the simulator's aircraft controller.

## Command contract

Every skill consumes a structured observation mapping and returns
`(action, completed)`. Actions use the versioned `tactical_deltas_v1` contract:

- `delta_heading`: radians;
- `delta_altitude`: metres;
- `delta_speed`: metres/second;
- `shoot`: `0.0` or `1.0`.

These are tactical commands, not native `MultiDiscrete` indices. Quantisation
must occur once, at the environment-adapter boundary. The action also includes
`skill_name`, `command_semantics`, and `completed` as logging metadata; remove
those metadata fields if the environment accepts only control fields.

## Built-in vocabulary

- engagement geometry: `pursuit`, `beam`, `crank_maneuver`, `extend`, `recommit`;
- flight path: `turn_to_heading`, `climb`, `descend`, `accelerate`, `decelerate`;
- tactical state: `missile_evasion`, `disengage`, `maintain_position`.

All parameters are validated when the skill is constructed. `SkillManager` is
strict by default and raises `UnknownSkillError` for an unknown name. For an
operational fail-safe, construct it with `strict=False`; the manager then uses
`maintain_position` and records a structured `last_creation_event` explaining
the fallback.

## Persistent execution

Skills contain timers and other state, so do not create one on every simulation
step. `Agent` retains an active instance until it completes or a new selection
interrupts it. `Agent.skill_events` records starts, completions, interruptions,
selector source, reasoning, and simulator time for dataset provenance.

```python
manager = SkillManager()
skill = manager.create_skill(
    "crank_maneuver",
    {"direction": "left", "offset_angle": 30, "duration_s": 25},
)

while True:
    action, completed = skill.execute(observation)
    control = {key: action[key] for key in (
        "delta_heading", "delta_altitude", "delta_speed", "shoot"
    )}
    observation = environment_step(control)
    if completed:
        break
```

Target-relative skills require `target_bearing_rad`. `disengage` uses
`home_bearing_rad` when supplied and otherwise interprets `heading_home` as
degrees. Observations may use `time_s` or the legacy `time` key.
