import importlib
import math
import sys

import pytest
from bvr_sim.agents.skill_manager import CATALOGUE_VERSION, SkillManager

EXPECTED = ["turn_to_heading", "set_altitude", "set_speed", "launch"]
REMOVED = {
    "maintain_heading", "climb_to_altitude", "descend_to_altitude", "accelerate_to_speed",
    "decelerate_to_speed", "pursue_target", "lead_pursuit", "lag_pursuit", "beam_target_left",
    "beam_target_right", "crank_target_left", "crank_target_right", "turn_cold", "extend",
    "recommit", "search", "lock_target", "commit", "support_missile", "abort_support",
    "secondary_shot", "short_range_attack", "preemptive_crank", "notch_left", "notch_right",
    "beam_missile_left", "beam_missile_right", "drag_missile", "dive_defense",
    "last_ditch_break", "defensive_reversal", "break_left", "break_right",
}
OBSERVATION = {
    "self_status": {
        "position": {"heading_deg": 350, "altitude_m": 7000},
        "performance": {"speed_mps": 250},
    },
}


def test_catalogue_contains_only_distinct_effective_controls():
    manager = SkillManager()
    assert manager.list_skills() == EXPECTED
    assert manager.list_skills("weapon_employment") == ["launch"]
    assert manager.list_skills("kinematic") == EXPECTED[:-1]
    schemas = {
        name: set(manager.get_contract(name)["parameter_schema"]["properties"])
        for name in EXPECTED
    }
    assert schemas == {
        "turn_to_heading": {"delta_heading_deg"}, "set_altitude": {"delta_altitude_m"},
        "set_speed": {"delta_speed_mps"}, "launch": set(),
    }


def test_catalogue_metadata_is_versioned_ordered_and_defensively_copied():
    manager = SkillManager()
    catalogue = manager.get_catalogue()
    assert catalogue["version"] == CATALOGUE_VERSION == "3.0.0"
    assert [contract["name"] for contract in catalogue["skills"]] == EXPECTED
    for contract in catalogue["skills"]:
        assert contract["version"] == "3.0.0"
        for field in ("start_condition", "output_contract", "termination_condition",
                      "interruption_conditions", "parameter_schema"):
            assert contract[field], (contract["name"], field)
        assert contract["output_contract"]["additionalProperties"] is False
    catalogue["skills"][0]["parameter_schema"]["properties"]["delta_heading_deg"]["maximum"] = 1
    catalogue["skills"][0]["output_contract"]["properties"]["shoot"]["enum"].append(2)
    fresh = manager.get_catalogue()
    assert fresh["skills"][0]["parameter_schema"]["properties"]["delta_heading_deg"]["maximum"] == 180
    assert fresh["skills"][0]["output_contract"]["properties"]["shoot"]["enum"] == [0, 1]


@pytest.mark.parametrize("name", sorted(REMOVED | {"not_a_skill"}))
def test_removed_skills_are_not_runtime_aliases(name):
    manager = SkillManager()
    with pytest.raises(KeyError, match="Unknown skill"):
        manager.create_skill(name)
    with pytest.raises(KeyError, match="Unknown skill"):
        manager.get_contract(name)


@pytest.mark.parametrize(
    "name,params,field,expected",
    [
        ("turn_to_heading", {"delta_heading_deg": 20}, "delta_heading", math.radians(20)),
        ("turn_to_heading", {"delta_heading_deg": -20}, "delta_heading", math.radians(-20)),
        ("turn_to_heading", {"delta_heading_deg": 180}, "delta_heading", math.pi),
        ("turn_to_heading", {"delta_heading_deg": -180}, "delta_heading", -math.pi),
        ("set_altitude", {"delta_altitude_m": 80}, "delta_altitude", 80),
        ("set_altitude", {"delta_altitude_m": -80}, "delta_altitude", -80),
        ("set_speed", {"delta_speed_mps": 80}, "delta_speed", 80),
        ("set_speed", {"delta_speed_mps": -80}, "delta_speed", -80),
        ("launch", {}, "shoot", 1),
    ],
)
def test_commands_emit_signed_control_on_only_the_selected_axis(name, params, field, expected):
    manager = SkillManager()
    action, completed = manager.create_skill(name, params).execute(OBSERVATION)
    assert set(action) == set(manager.get_contract(name)["output_contract"]["required"])
    for key in ("delta_heading", "delta_altitude", "delta_speed", "shoot"):
        assert action[key] == pytest.approx(expected if key == field else 0)
    assert completed == action["completed"] == (name == "launch")


@pytest.mark.parametrize(
    "name", EXPECTED[:-1],
)
def test_zero_relative_defaults_request_no_change_at_any_current_state(name):
    action, completed = SkillManager().create_skill(name).execute(OBSERVATION)
    assert completed
    assert all(action[key] == 0 for key in ("delta_heading", "delta_altitude", "delta_speed", "shoot"))


@pytest.mark.parametrize(
    "name,params",
    [("turn_to_heading", {"delta_heading_deg": 30}),
     ("set_altitude", {"delta_altitude_m": -40}), ("set_speed", {"delta_speed_mps": 20})],
)
def test_relative_commands_stay_relative_when_aircraft_state_changes(name, params):
    skill = SkillManager().create_skill(name, params)
    before = skill.execute(OBSERVATION)
    changed_state = {"time": 10, "self_status": {
        "position": {"heading_deg": 30, "altitude_m": 3000},
        "performance": {"speed_mps": 350},
    }}
    assert skill.execute(changed_state) == before
    assert skill.execute({}) == before


@pytest.mark.parametrize(
    "name,params,error",
    [
        ("turn_to_heading", {"delta_heading_deg": 180.1}, ValueError),
        ("turn_to_heading", {"delta_heading_deg": -180.1}, ValueError),
        ("set_altitude", {"delta_altitude_m": -80.1}, ValueError),
        ("set_altitude", {"delta_altitude_m": 80.1}, ValueError),
        ("set_speed", {"delta_speed_mps": 80.1}, ValueError),
        ("set_speed", {"delta_speed_mps": -80.1}, ValueError),
        ("set_speed", {"delta_speed_mps": float("nan")}, ValueError),
        ("set_speed", {"delta_speed_mps": float("inf")}, ValueError),
        ("set_speed", {"delta_speed_mps": True}, TypeError),
        ("set_speed", {"delta_speed_mps": "20"}, TypeError),
        ("turn_to_heading", {"heading_deg": 30}, ValueError),
        ("set_altitude", {"altitude_m": 8000}, ValueError),
        ("set_speed", {"speed_mps": 250}, ValueError),
        ("turn_to_heading", {"tolerance_deg": 3}, ValueError),
        ("set_speed", {"timeout_s": 60}, ValueError),
        ("launch", {"launch_range_m": 40_000}, ValueError),
    ],
)
def test_parameter_validation_rejects_invalid_and_ineffective_parameters(name, params, error):
    with pytest.raises(error):
        SkillManager().create_skill(name, params)


def test_agents_package_does_not_eagerly_import_optional_api_clients():
    agents = importlib.import_module("bvr_sim.agents")
    assert agents.SkillManager is agents.SkillManager
    assert "bvr_sim.agents.bvr_strategist" not in sys.modules
