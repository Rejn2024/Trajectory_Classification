import importlib.util
from pathlib import Path
import sys
import unittest


MODULE = Path(__file__).parents[1] / "bvr_sim_source/bvr_sim/agents/skill_manager.py"
spec = importlib.util.spec_from_file_location("skill_manager", MODULE)
skill_manager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(skill_manager)
SkillManager = skill_manager.SkillManager


EXPECTED = {
    "maintain_heading", "turn_to_heading", "climb_to_altitude", "descend_to_altitude",
    "accelerate_to_speed", "decelerate_to_speed", "pursue_target", "lead_pursuit",
    "lag_pursuit", "beam_target_left", "beam_target_right", "crank_target_left",
    "crank_target_right", "turn_cold", "extend", "recommit", "search", "lock_target",
    "commit", "launch", "support_missile", "abort_support", "secondary_shot",
    "short_range_attack", "preemptive_crank", "notch_left", "notch_right",
    "beam_missile_left", "beam_missile_right", "drag_missile", "dive_defense",
    "last_ditch_break", "defensive_reversal",
}


class SkillManagerTests(unittest.TestCase):
    def setUp(self):
        self.manager = SkillManager()

    def test_catalogue_contains_complete_vocabulary(self):
        self.assertEqual(set(self.manager.list_skills()), EXPECTED)

    def test_every_skill_has_complete_versioned_contract(self):
        for name in EXPECTED:
            contract = self.manager.get_contract(name)
            self.assertEqual(contract["version"], "1.0.0")
            for field in ("start_condition", "output_contract", "termination_condition",
                          "interruption_conditions", "parameter_schema"):
                self.assertTrue(contract[field], (name, field))
            self.assertEqual(contract["output_contract"]["additionalProperties"], False)

    def test_kinematic_execution_and_validation(self):
        skill = self.manager.create_skill("turn_to_heading", {"heading_deg": 90})
        action, completed = skill.execute({"time": 1, "self_status": {"position": {"heading_deg": 0}}})
        self.assertAlmostEqual(action["delta_heading"], 1.57079632679)
        self.assertFalse(completed)
        with self.assertRaises(ValueError):
            self.manager.create_skill("turn_to_heading", {"heading_deg": 500})
        with self.assertRaises(KeyError):
            self.manager.create_skill("not_a_skill")

    def test_weapon_action_obeys_output_contract(self):
        action, completed = self.manager.create_skill("launch").execute({})
        self.assertEqual(set(action), set(self.manager.get_contract("launch")["output_contract"]["required"]))
        self.assertEqual(action["shoot"], 1)
        self.assertTrue(completed)

    def test_agents_package_does_not_eagerly_import_optional_api_clients(self):
        agents_dir = MODULE.parent
        package_name = "bundled_agents_for_test"
        package_spec = importlib.util.spec_from_file_location(
            package_name,
            agents_dir / "__init__.py",
            submodule_search_locations=[str(agents_dir)],
        )
        agents = importlib.util.module_from_spec(package_spec)
        sys.modules[package_name] = agents
        self.addCleanup(sys.modules.pop, package_name, None)

        package_spec.loader.exec_module(agents)

        self.assertIs(agents.SkillManager, agents.SkillManager)
        self.assertNotIn(f"{package_name}.bvr_strategist", sys.modules)


if __name__ == "__main__":
    unittest.main()
