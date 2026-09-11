import random

from bvr_behavior_prediction.policies.stochastic_skills import DEFAULT_SKILLS, StochasticSkillPolicy
from bvr_behavior_prediction.simulator.scenario_config import ScenarioConfig


class FakeSkill:
    def __init__(self, name, params):
        self.skill_name, self.params = name, params

    def execute(self, obs):
        return {"delta_heading": 0, "delta_altitude": 0, "delta_speed": 0,
                "shoot": self.skill_name == "launch"}, False


class FakeSkillManager:
    def create_skill(self, name, params):
        return FakeSkill(name, params)


def states(**own):
    return {"x": 0, "y": 0, "z": 8000, "speed": 250, **own}, {"x": 30_000, "y": 0, "z": 8000}


def test_parameters_are_bounded_and_seeded():
    first = StochasticSkillPolicy(FakeSkillManager())
    second = StochasticSkillPolicy(FakeSkillManager())
    first.reset(random.Random(17))
    second.reset(random.Random(17))
    assert first.selector_state == second.selector_state
    for name, value in first.selector_state["parameters"].items():
        assert DEFAULT_SKILLS["maintain_position"].parameters[name][0] <= value <= DEFAULT_SKILLS["maintain_position"].parameters[name][1]


def test_skill_persists_until_duration_and_then_uses_geometry_edge():
    policy = StochasticSkillPolicy(FakeSkillManager())
    policy.reset(random.Random(2))
    own, opponent = states()
    end = policy.selector_state["ends_at_s"]
    policy.act(own, opponent, end - 0.001)
    assert policy.label == "maintain_position"
    policy.act(own, opponent, end)
    assert policy.label == "pursue_target"
    assert policy.selector_state["transition_reason"] == "duration_elapsed_geometry"


def test_emergency_interrupts_immediately_and_logs_reason():
    policy = StochasticSkillPolicy(FakeSkillManager())
    policy.reset(random.Random(3))
    own, opponent = states(incoming_missile_active=True)
    action = policy.act(own, opponent, 0.1)
    assert policy.label == "missile_evasion"
    assert policy.transition_log[-1]["transition_reason"] == "incoming_active_missile"
    assert all(0 <= value < size for value, size in zip(action.as_tuple(), (15, 15, 9, 2)))


def test_geometry_can_interrupt_pursuit_before_duration_expires():
    policy = StochasticSkillPolicy(FakeSkillManager())
    policy.reset(random.Random(4))
    own, opponent = states()
    policy._activate("pursue_target", 1.0, "test_setup")
    opponent["x"] = 10_000
    policy.act(own, opponent, 1.1)
    assert policy.label == "launch"
    assert policy.selector_state["transition_reason"] == "target_entered_launch_envelope"


def test_rollouts_default_to_accelerated_cpp_backend():
    assert ScenarioConfig().backend == "cpp"
