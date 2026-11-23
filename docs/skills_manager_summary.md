# SkillManager Catalogue Summary

`SkillManager` is the registry and factory for BVR Sim's versioned tactical-skill
vocabulary. It builds its registry from `_catalogue()` and provides methods to list
skills, instantiate them, retrieve an individual contract, or describe the full
catalogue.

The catalogue contains **23 skills** in four categories:

## Kinematic (1)

- `set_trajectory` — sets heading, altitude, and speed together; its emitted
  deltas are relative to the aircraft's current state.

Defined in `bvr_sim_source/bvr_sim/agents/skill_manager.py` in `_catalogue()`.

## Relational (10)

- `pursue_target`
- `lead_pursuit`
- `lag_pursuit`
- `beam_target_left`
- `beam_target_right`
- `crank_target_left`
- `crank_target_right`
- `turn_cold`
- `extend`
- `recommit`

Defined in `bvr_sim_source/bvr_sim/agents/skill_manager.py` in `_catalogue()`.

## Weapon employment (3)

- `search`
- `launch`
- `support_missile`

Defined in `bvr_sim_source/bvr_sim/agents/skill_manager.py` in `_catalogue()`.

## Defensive (9)

- `preemptive_crank`
- `notch_left`
- `notch_right`
- `beam_missile_left`
- `beam_missile_right`
- `drag_missile`
- `dive_defense`
- `last_ditch_break`
- `defensive_reversal`

Defined in `bvr_sim_source/bvr_sim/agents/skill_manager.py` in `_catalogue()`.

## Related implementation locations

- `SkillSpec`, the immutable lifecycle-contract definition.
- `TacticalSkill`, the executable skill implementation.
- `SkillManager`, including `create_skill()`, `list_skills()`, `get_contract()`,
  and `describe_skills()`.
- The unit-test declaration of the complete expected vocabulary:
  `tests/test_skill_manager.py`.

Every skill contract includes its name, category, semantic version, start and
termination conditions, interruption conditions, parameter schema, and a common
guidance-action output contract.
