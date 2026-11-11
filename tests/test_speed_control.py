"""Exercise the real environment action path without optional simulator libraries.

Only external runtime services are stubbed: the environment class, constructor,
speed-control setup and step implementation are compiled unchanged from source.
An aircraft double records the controls actually received by its step method.
"""

import ast
import copy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


SOURCE = (
    Path(__file__).parents[1] / "bvr_sim_source" / "bvr_sim" / "bvr_env.py"
)


class RecordingAircraft:
    is_alive = True

    def __init__(self):
        self.controls = []

    def step(self, control, **kwargs):
        self.controls.append(copy.deepcopy(control))


class RecordingManager:
    def __init__(self, **kwargs):
        self.actions = []

    def __contains__(self, uid):
        return True

    def step_recorder(self, uid, action):
        self.actions.append((uid, copy.deepcopy(action)))

    def set_last_obs(self, observations):
        pass


class ActionManager:
    action_space = object()

    @staticmethod
    def is_dict_action(action):
        return isinstance(action, dict)

    @staticmethod
    def extract_other_commands(action):
        return action.get("other_commands", {})

    @staticmethod
    def legacy_norm_campus_action(action, **kwargs):
        return action


@pytest.fixture
def environment_type(monkeypatch):
    """Load the complete class, replacing only imported external dependencies."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    names = {
        "BYPASS_RL_CONTROL",
        "BYPASS_RL_THROTTLE_CONTROL",
        "USE_DISTILL_REWARD_ACTION",
        "PRINT_STEP_TIME",
    }
    nodes = [
        ast.ImportFrom(
            module="__future__",
            names=[ast.alias(name="annotations")],
            level=0,
        )
    ]
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in names
            for target in node.targets
        ):
            nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "BVR3DEnv":
            nodes.append(node)
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))

    package_name = "_speed_control_test_runtime"
    package = ModuleType(package_name)
    package.__path__ = []
    action_module = ModuleType(f"{package_name}.action_space")
    action_module.CampusActionSpace = ActionManager
    monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.setitem(sys.modules, action_module.__name__, action_module)
    namespace = {
        "__name__": f"{package_name}.environment",
        "__package__": package_name,
        "gymnasium": SimpleNamespace(Env=object),
        "np": np,
        "spaces": SimpleNamespace(Box=lambda **kwargs: object()),
        "SpawnManager": lambda config: object(),
        "create_observation_space": lambda kind: SimpleNamespace(
            get_obs_dim=lambda red, blue: 6
        ),
        "create_default_reward_manager": lambda config: object(),
        "AircraftRecorderManager": RecordingManager,
    }
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["BVR3DEnv"], namespace


@pytest.fixture
def build_environment(environment_type, tmp_path):
    environment_class, _ = environment_type

    def build(config=None):
        # Run the actual constructor, including speed-control configuration.
        settings = {"red_fighters": {"aircraft": {}}, "blue_fighters": {}}
        settings.update(config or {})
        environment = environment_class(settings, logdir=str(tmp_path))
        aircraft = RecordingAircraft()
        environment.agents = {"aircraft": aircraft}
        # Empty runtime subsystems keep this a control-plumbing test.
        environment._gen_expert_action = lambda: None
        environment._step_ground_units = lambda: None
        environment.episode_done = lambda: (False, {"aircraft": False}, 1, 0)
        environment._get_obs = lambda: {"aircraft": np.zeros(6)}
        environment._get_rewards = lambda info: {"aircraft": 0.0}
        environment._pack_obs = lambda values: values
        environment._pack_rewards = lambda values: values
        environment._pack_dones = lambda values: values
        return environment, aircraft

    return build


def controls(speed):
    return {
        "aircraft": {
            "delta_heading": -0.1,
            "delta_altitude": 0.2,
            "delta_speed": speed,
            "shoot": 0,
        }
    }


@pytest.mark.parametrize("speed", [-0.75, 0.0, 0.75])
def test_requested_speed_reaches_aircraft_by_default(build_environment, speed):
    environment, aircraft = build_environment()
    environment.step(controls(speed))

    assert aircraft.controls == [
        {"delta_heading": -0.1, "delta_altitude": 0.2, "delta_speed": speed}
    ]
    assert environment.recorder_manager.actions[0][1]["delta_speed"] == speed


@pytest.mark.parametrize("speed", [-0.75, 0.0, 0.75])
def test_explicit_legacy_mode_overrides_only_speed(build_environment, speed):
    environment, aircraft = build_environment({"bypass_rl_throttle_control": True})
    environment.step(controls(speed))

    assert aircraft.controls == [
        {"delta_heading": -0.1, "delta_altitude": 0.2, "delta_speed": 1.0}
    ]
    assert environment.recorder_manager.actions[0][1]["delta_speed"] == 1.0


@pytest.mark.parametrize("bypass", [False, True])
def test_step_does_not_mutate_callers_action(build_environment, bypass):
    environment, aircraft = build_environment({"bypass_rl_throttle_control": bypass})
    action = controls(-0.5)
    original = copy.deepcopy(action)
    environment.step(action)

    assert action == original
    assert aircraft.controls[-1]["delta_speed"] == (1.0 if bypass else -0.5)


def test_interleaved_environments_keep_independent_speed_modes(build_environment):
    requested, requested_aircraft = build_environment(
        {"bypass_rl_throttle_control": False}
    )
    legacy, legacy_aircraft = build_environment({"bypass_rl_throttle_control": True})

    for environment in [requested, legacy, requested, legacy]:
        environment.step(controls(-0.5))

    assert [c["delta_speed"] for c in requested_aircraft.controls] == [-0.5, -0.5]
    assert [c["delta_speed"] for c in legacy_aircraft.controls] == [1.0, 1.0]


def test_speed_override_is_disabled_by_default(environment_type, build_environment):
    _, namespace = environment_type
    assert namespace["BYPASS_RL_THROTTLE_CONTROL"] is False
    environment, _ = build_environment()
    assert environment.bypass_rl_throttle_control is False


@pytest.mark.parametrize("invalid", [None, 0, 1, "false", "true", [], {}])
def test_speed_override_requires_a_boolean(build_environment, invalid):
    with pytest.raises(TypeError, match="bypass_rl_throttle_control"):
        build_environment({"bypass_rl_throttle_control": invalid})
