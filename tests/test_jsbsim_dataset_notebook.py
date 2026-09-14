import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "02_generate_jsbsim_skill_dataset.ipynb"


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
