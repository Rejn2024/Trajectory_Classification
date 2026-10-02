# SkillManager Catalogue Summary

`SkillManager` is the registry and factory for BVR Sim's versioned tactical-skill
vocabulary. It builds its registry from `_catalogue()` and provides methods to list
skills, instantiate them, retrieve an individual contract, or describe the full
catalogue.

The catalogue contains **33 skills** in four categories:

## Kinematic (6)

- `maintain_heading`
- `turn_to_heading`
- `climb_to_altitude`
- `descend_to_altitude`
- `accelerate_to_speed`
- `decelerate_to_speed`

Defined in `bvr_sim_source/bvr_sim/agents/skill_manager.py`, lines 98–110.

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

Defined in `bvr_sim_source/bvr_sim/agents/skill_manager.py`, lines 112–130.

## Weapon employment (8)

- `search`
- `lock_target`
- `commit`
- `launch`
- `support_missile`
- `abort_support`
- `secondary_shot`
- `short_range_attack`

Defined in `bvr_sim_source/bvr_sim/agents/skill_manager.py`, lines 132–146.

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

Defined in `bvr_sim_source/bvr_sim/agents/skill_manager.py`, lines 148–165.

## Related implementation locations

- `SkillSpec`, the immutable lifecycle-contract definition: lines 45–71.
- `TacticalSkill`, the executable skill implementation: lines 168–240.
- `SkillManager`, including `create_skill()`, `list_skills()`, `get_contract()`,
  and `describe_skills()`: lines 243–266.
- The unit-test declaration of the complete expected vocabulary:
  `tests/test_skill_manager.py`, lines 9–18.

Every skill contract includes its name, category, semantic version, start and
termination conditions, interruption conditions, parameter schema, and a common
guidance-action output contract.
