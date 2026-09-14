import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "02_generate_jsbsim_skill_dataset.ipynb"


def _notebook_source():
    notebook = json.loads(NOTEBOOK.read_text())
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def _validation_assertion():
    notebook = json.loads(NOTEBOOK.read_text())
    validation_cell = next(
        cell
        for cell in notebook["cells"]
        if "intervals = trajectories" in "".join(cell.get("source", []))
    )
    tree = ast.parse("".join(validation_cell["source"]))
    return next(
        statement.test
        for statement in tree.body
        if isinstance(statement, ast.Assert)
        and "intervals.max()" in ast.unparse(statement.test)
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


def test_sample_interval_validation_allows_floating_point_drift():
    intervals = pd.Series([index * 0.1 for index in range(601)]).diff().dropna()
    assert intervals.max() > 0.1

    expression = ast.Expression(_validation_assertion())
    assert eval(compile(expression, NOTEBOOK.name, "eval"), {"np": np}, {
        "intervals": intervals,
        "SAMPLE_DT_S": 0.1,
    })


def test_sample_interval_validation_rejects_intervals_above_tolerance():
    intervals = pd.Series([0.1, 0.100_000_002])
    expression = ast.Expression(_validation_assertion())

    assert not eval(compile(expression, NOTEBOOK.name, "eval"), {"np": np}, {
        "intervals": intervals,
        "SAMPLE_DT_S": 0.1,
    })


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
