import importlib
import sys
import unittest


from bvr_sim.agents.skill_manager import SkillManager


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
        agents = importlib.import_module("bvr_sim.agents")

        self.assertIs(agents.SkillManager, agents.SkillManager)
        self.assertNotIn("bvr_sim.agents.bvr_strategist", sys.modules)


if __name__ == "__main__":
    unittest.main()
