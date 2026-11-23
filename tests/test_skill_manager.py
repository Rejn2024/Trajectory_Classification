import importlib
import math
import sys
import unittest


from bvr_sim.agents.skill_manager import SkillManager


EXPECTED = {
    "set_trajectory", "pursue_target", "lead_pursuit",
    "lag_pursuit", "beam_target_left", "beam_target_right", "crank_target_left",
    "crank_target_right", "turn_cold", "extend", "recommit", "search", "launch",
    "support_missile", "preemptive_crank", "notch_left", "notch_right",
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

    def test_set_trajectory_emits_relative_heading_altitude_and_speed_errors(self):
        skill = self.manager.create_skill("set_trajectory", {
            "heading_deg": 90, "altitude_m": 9000, "speed_mps": 300,
        })
        obs = {"time": 1, "self_status": {
            "position": {"heading_deg": 30, "altitude_m": 8000},
            "performance": {"speed_mps": 250},
        }}
        action, completed = skill.execute(obs)
        self.assertAlmostEqual(action["delta_heading"], math.radians(60))
        self.assertAlmostEqual(action["delta_altitude"], 1000)
        self.assertAlmostEqual(action["delta_speed"], 50)
        self.assertFalse(completed)
        with self.assertRaises(ValueError):
            self.manager.create_skill("set_trajectory", {"heading_deg": 500})
        with self.assertRaises(KeyError):
            self.manager.create_skill("not_a_skill")

    def test_set_trajectory_completes_only_when_all_setpoints_are_reached(self):
        skill = self.manager.create_skill("set_trajectory", {
            "heading_deg": 90, "altitude_m": 9000, "speed_mps": 300,
        })
        almost = {"self_status": {
            "position": {"heading_deg": 90, "altitude_m": 8000},
            "performance": {"speed_mps": 300},
        }}
        action, completed = skill.execute(almost)
        self.assertFalse(completed)
        self.assertFalse(action["completed"])

        reached = {"self_status": {
            "position": {"heading_deg": 90, "altitude_m": 9000},
            "performance": {"speed_mps": 300},
        }}
        action, completed = skill.execute(reached)
        self.assertTrue(completed)
        self.assertTrue(action["completed"])

    def test_weapon_action_obeys_output_contract(self):
        action, completed = self.manager.create_skill("launch").execute({})
        self.assertEqual(set(action), set(self.manager.get_contract("launch")["output_contract"]["required"]))
        self.assertEqual(action["shoot"], 1)
        self.assertTrue(completed)

    def test_relational_skills_use_target_geometry(self):
        obs = {
            "self_status": {
                "position": {"north": 0.0, "west": 0.0, "up": 5000.0,
                              "heading_deg": 0.0, "altitude_m": 5000.0},
                "performance": {"speed_mps": 250.0, "velocity": [250.0, 0.0, 0.0]},
            },
            "targets": {"primary": {"position": [10000.0, 10000.0, 5000.0],
                                      "velocity": [0.0, 250.0, 0.0]}},
        }

        pursue, _ = self.manager.create_skill("pursue_target").execute(obs)
        self.assertAlmostEqual(pursue["delta_heading"], math.pi / 4, places=6)

        beam, _ = self.manager.create_skill("beam_target_left").execute(obs)
        self.assertAlmostEqual(beam["delta_heading"], -math.pi / 4, places=6)

        lead, _ = self.manager.create_skill("lead_pursuit", {"lead_time_s": 2}).execute(obs)
        self.assertGreater(lead["delta_heading"], math.pi / 4)

    def test_defensive_skills_command_threat_relative_motion(self):
        obs = {
            "self_status": {
                "position": {"north": 0.0, "west": 0.0, "up": 5000.0,
                              "heading_deg": 0.0, "altitude_m": 5000.0},
                "performance": {"speed_mps": 250.0, "velocity": [250.0, 0.0, 0.0]},
            },
            "targets": {
                "highest_priority": {
                    "position": [10000.0, 10000.0, 5000.0],
                    "velocity": [0.0, 250.0, 0.0],
                }
            },
            "missiles": {"highest_priority": {"position": [10000.0, 0.0, 5000.0]}},
        }

        notch, _ = self.manager.create_skill("notch_left").execute(obs)
        # The hostile aircraft is heading west (+90 degrees in NWU), so the
        # left notch reference is north (0 degrees), not the target LOS (+45).
        self.assertAlmostEqual(notch["delta_heading"], 0.0, places=6)
        self.assertLessEqual(abs(notch["delta_heading"]), math.pi / 2)

        notch_right, _ = self.manager.create_skill("notch_right").execute(obs)
        # The exact 180-degree right-side case is represented as -pi by _wrap.
        self.assertLess(notch_right["delta_heading"], 0.0)
        self.assertLessEqual(abs(notch_right["delta_heading"]), math.pi / 2)

        beam, _ = self.manager.create_skill("beam_missile_left").execute(obs)
        self.assertLess(beam["delta_heading"], 0.0)
        self.assertLessEqual(abs(beam["delta_heading"]), math.pi / 2)

        beam_right, _ = self.manager.create_skill("beam_missile_right").execute(obs)
        self.assertGreater(beam_right["delta_heading"], 0.0)
        self.assertLessEqual(abs(beam_right["delta_heading"]), math.pi / 2)

        drag, _ = self.manager.create_skill("drag_missile").execute(obs)
        self.assertGreater(drag["delta_speed"], 0.0)
        self.assertNotEqual(drag["delta_heading"], 0.0)
        self.assertLessEqual(abs(drag["delta_heading"]), math.pi / 2)

        dive, _ = self.manager.create_skill("dive_defense").execute(obs)
        self.assertLess(dive["delta_altitude"], 0.0)

        break_action, _ = self.manager.create_skill(
            "last_ditch_break", {"side": "right", "max_g": 9}
        ).execute(obs)
        self.assertGreater(break_action["delta_heading"], 0.0)
        self.assertLessEqual(abs(break_action["delta_heading"]), math.pi / 2)

    def test_beam_missile_left_and_right_use_nearest_live_missile(self):
        obs = {
            "self_status": {
                "position": {"north": 0.0, "west": 0.0, "up": 5000.0,
                              "heading_deg": -50.0, "altitude_m": 5000.0},
                "performance": {"speed_mps": 250.0, "velocity": [250.0, 0.0, 0.0]},
            },
            "missiles": [
                {"id": "far", "position": [10000.0, 10000.0, 5000.0], "is_alive": True},
                {"id": "near", "position": [1000.0, 0.0, 5000.0], "is_alive": True},
            ],
        }

        left, _ = self.manager.create_skill("beam_missile_left").execute(obs)
        right, _ = self.manager.create_skill("beam_missile_right").execute(obs)
        # The nearest missile is at 0 degrees bearing, so its left/right beams
        # are -/+90 degrees. The 9-g/rate bound limits each to 9 degrees.
        self.assertAlmostEqual(left["delta_heading"], -math.radians(9.0), places=6)
        self.assertAlmostEqual(right["delta_heading"], math.radians(9.0), places=6)

    def test_track_loss_is_safe_and_preserves_side_fallback(self):
        action, _ = self.manager.create_skill("beam_target_left").execute({})
        self.assertEqual(action["delta_heading"], -math.pi / 2)
        for skill_name in self.manager.list_skills():
            action, _ = self.manager.create_skill(skill_name).execute({})
            self.assertTrue(all(math.isfinite(action[key]) for key in (
                "delta_heading", "delta_altitude", "delta_speed")))

    def test_search_sweeps_heading_and_stops_on_detection(self):
        search = self.manager.create_skill("search", {
            "sector_width_deg": 60,
            "sweep_period_s": 20,
            "pattern": "sinusoidal",
            "direction": "right",
        })
        base = {
            "self_status": {
                "position": {"north": 0.0, "west": 0.0, "up": 5000.0,
                              "heading_deg": 0.0, "altitude_m": 5000.0},
                "performance": {"speed_mps": 250.0, "velocity": [250.0, 0.0, 0.0]},
            },
        }
        first, first_done = search.execute({**base, "time": 0.0})
        right, right_done = search.execute({**base, "time": 5.0})
        left, _ = search.execute({**base, "time": 15.0})
        self.assertEqual(first["delta_heading"], 0.0)
        self.assertFalse(first_done)
        self.assertFalse(right_done)
        self.assertAlmostEqual(right["delta_heading"], math.radians(30), places=6)
        self.assertAlmostEqual(left["delta_heading"], -math.radians(30), places=6)

        detected, detected_done = search.execute({
            **base,
            "time": 16.0,
            "targets": {"primary": {"position": [10000.0, 0.0, 5000.0]}},
        })
        self.assertTrue(detected_done)
        self.assertTrue(detected["completed"])
        self.assertEqual(detected["delta_heading"], 0.0)

    def test_support_missile_cranks_to_safe_radar_cone_edge(self):
        support = self.manager.create_skill("support_missile", {
            "support_side": "left",
            "radar_half_fov_deg": 60,
            "radar_margin_deg": 5,
        })
        base = {
            "time": 0.0,
            "self_status": {
                "position": {"north": 0.0, "west": 0.0, "up": 5000.0,
                              "heading_deg": 0.0, "altitude_m": 5000.0},
                "performance": {"speed_mps": 250.0, "velocity": [250.0, 0.0, 0.0]},
            },
            "targets": {"primary": {"position": [10000.0, 0.0, 5000.0]}},
            "support_missiles": [{"id": "M01", "is_alive": True}],
        }
        action, completed = support.execute(base)
        self.assertFalse(completed)
        self.assertLess(action["delta_heading"], 0.0)
        # The 9-g turn-rate bound is applied per control step; it must still
        # command a left crank and never exceed the configured cone edge.
        self.assertLessEqual(abs(action["delta_heading"]), math.radians(55))

        no_missile = dict(base, support_missiles=[])
        stopped, stopped_done = support.execute(no_missile)
        self.assertTrue(stopped_done)
        self.assertTrue(stopped["completed"])

    def test_agents_package_does_not_eagerly_import_optional_api_clients(self):
        agents = importlib.import_module("bvr_sim.agents")

        self.assertIs(agents.SkillManager, agents.SkillManager)
        self.assertNotIn("bvr_sim.agents.bvr_strategist", sys.modules)


if __name__ == "__main__":
    unittest.main()
