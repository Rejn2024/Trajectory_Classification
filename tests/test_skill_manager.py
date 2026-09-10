import importlib.util
import math
from pathlib import Path
import sys
import unittest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "bvr_sim_source"
    / "bvr_sim"
    / "agents"
    / "skill_manager.py"
)
SPEC = importlib.util.spec_from_file_location("tested_skill_manager", MODULE_PATH)
skill_manager = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = skill_manager
SPEC.loader.exec_module(skill_manager)


def observation(time_s=0.0, heading=0.0, bearing=0.0, speed=250.0):
    return {
        "time_s": time_s,
        "self_status": {
            "position": {"heading_rad": heading, "altitude_m": 8000.0},
            "performance": {"speed_mps": speed},
        },
        "target_bearing_rad": bearing,
    }


class SkillManagerTests(unittest.TestCase):
    def test_registry_contains_expanded_vocabulary_and_schemas(self):
        manager = skill_manager.SkillManager()
        self.assertTrue({
            "crank_maneuver", "missile_evasion", "disengage", "maintain_position",
            "turn_to_heading", "climb", "descend", "accelerate", "decelerate",
            "pursuit", "beam", "extend", "recommit",
        }.issubset(manager.list_skills()))
        self.assertEqual(
            manager.schemas()["crank_maneuver"]["command_semantics"],
            skill_manager.COMMAND_SEMANTICS,
        )

    def test_unknown_skills_are_explicit(self):
        with self.assertRaises(skill_manager.UnknownSkillError):
            skill_manager.SkillManager().create_skill("misspelled")

        manager = skill_manager.SkillManager(strict=False)
        fallback = manager.create_skill("misspelled")
        self.assertEqual(fallback.skill_name, "maintain_position")
        self.assertTrue(manager.last_creation_event.used_fallback)
        self.assertIn("misspelled", manager.last_creation_event.fallback_reason)

    def test_parameters_are_validated(self):
        manager = skill_manager.SkillManager()
        with self.assertRaises(skill_manager.InvalidSkillParameters):
            manager.create_skill("missile_evasion", {"break_direction": "up"})
        with self.assertRaises(skill_manager.InvalidSkillParameters):
            manager.create_skill("climb", {"duration_s": -1})
        with self.assertRaises(skill_manager.InvalidSkillParameters):
            manager.create_skill("disengage", {"unused_parameter": 1})

    def test_crank_is_target_relative_stateful_and_uses_speed_target(self):
        skill = skill_manager.SkillManager().create_skill(
            "crank_maneuver",
            {"direction": "left", "offset_angle": 30, "switch_frequency": 5},
        )
        first, complete = skill.execute(observation(time_s=0, bearing=1.0, speed=240.0))
        switched, _ = skill.execute(observation(time_s=5, bearing=1.0, speed=240.0))
        persisted, _ = skill.execute(observation(time_s=6, bearing=1.0, speed=240.0))

        self.assertFalse(complete)
        self.assertAlmostEqual(first["delta_heading"], 1.0 - math.radians(30))
        self.assertAlmostEqual(switched["delta_heading"], 1.0 + math.radians(30))
        self.assertEqual(switched["delta_heading"], persisted["delta_heading"])
        self.assertEqual(first["delta_speed"], 60.0)
        self.assertEqual(first["command_semantics"], "tactical_deltas_v1")

    def test_timed_completion_and_interrupt_metadata(self):
        skill = skill_manager.SkillManager().create_skill(
            "missile_evasion", {"break_duration": 2, "max_g": 4.5}
        )
        active, complete = skill.execute(observation(time_s=10))
        stopped, complete_after = skill.execute(observation(time_s=12))
        skill.interrupt("test_interrupt")

        self.assertFalse(complete)
        self.assertAlmostEqual(active["delta_heading"], 0.5)
        self.assertTrue(complete_after)
        self.assertEqual(stopped["delta_heading"], 0.0)
        self.assertTrue(skill.interrupted)
        self.assertEqual(skill.interruption_reason, "test_interrupt")

    def test_disengage_uses_home_heading(self):
        skill = skill_manager.SkillManager().create_skill("disengage", {"heading_home": 90})
        command, _ = skill.execute(observation(heading=0.0))
        self.assertAlmostEqual(command["delta_heading"], math.pi / 2)

    def test_observation_must_be_structured(self):
        skill = skill_manager.SkillManager().create_skill("maintain_position")
        with self.assertRaises(TypeError):
            skill.execute("serialized observation")


if __name__ == "__main__":
    unittest.main()
