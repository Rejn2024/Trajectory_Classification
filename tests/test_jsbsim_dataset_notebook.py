import ast
import json
from pathlib import Path

import numpy as np


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "02_generate_jsbsim_skill_dataset.ipynb"


def _notebook_source():
    notebook = json.loads(NOTEBOOK.read_text())
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def _target_controls():
    tree = ast.parse(_notebook_source())
    wanted = {"clamp", "angle_error_deg", "target_controls"}
    definitions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    namespace = {"math": __import__("math")}
    exec(compile(ast.Module(definitions, type_ignores=[]), NOTEBOOK.name, "exec"), namespace)
    return namespace["target_controls"]


def test_generation_defaults_to_100000_flights_and_streams_each_flight():
    source = _notebook_source()

    assert 'os.getenv("BVR_DATASET_FLIGHTS", "100000")' in source
    assert "scenarios = [" not in source
    assert "trajectory_rows.extend" not in source
    assert "episode_rows.append" not in source
    assert "writer.write_flight(index, rows, episode_row)" in source
    assert "del rows, episode_row" in source


def test_sample_cadence_is_validated_per_flight_with_tolerance():
    source = _notebook_source()

    assert 'abs(row["time_s"] - step * SAMPLE_DT_S) <= 1e-9' in source


def test_stochastic_manager_uses_runtime_skill_allow_list():
    source = _notebook_source()

    assert 'os.getenv("BVR_DATASET_SKILLS"' in source
    assert "self.available_skills = tuple(available_skills)" in source
    assert "primary = self.available_skills[flight_index % len(self.available_skills)]" in source
    assert "label for label in self.available_skills if label != primary" in source
    assert '"available_skills":list(AVAILABLE_SKILLS)' in source.replace(" ", "")


def test_controller_levels_wings_after_reaching_commanded_bank():
    controls = _target_controls()
    own = {"x": 0.0, "y": 0.0, "heading": 0.0, "roll": np.deg2rad(35), "pitch": 0.0}
    opponent = {"x": 0.0, "y": 1.0}

    aileron, elevator, rudder, throttle = controls("PURSUE", own, opponent)

    assert abs(aileron) < 1e-12
    assert -0.15 <= elevator <= 0.15
    assert rudder == 0.0
    assert 0.0 <= throttle <= 1.0


def test_observer_uses_attitude_feedback_during_integration():
    assert 'observer_fdm.step(target_controls("MAINTAIN", observer, target))' in _notebook_source()
