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


def test_skill_manager_catalogue_is_shown_directly_after_imports():
    notebook = json.loads(NOTEBOOK.read_text())
    import_cell_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if "from bvr_sim.agents.skill_manager import SkillManager" in "".join(cell.get("source", []))
    )
    catalogue_cell = "".join(notebook["cells"][import_cell_index + 1].get("source", []))

    assert "SkillManager().list_skills()" in catalogue_cell
    assert "Skills available in SkillManager for selection" in catalogue_cell


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


def test_generation_parallelises_flights_with_bounded_ordered_results():
    source = _notebook_source()

    assert 'os.getenv("BVR_DATASET_WORKERS"' in source
    assert 'os.getenv("BVR_DATASET_MAX_IN_FLIGHT"' in source
    assert "ProcessPoolExecutor(max_workers=MAX_WORKERS" in source
    assert "from loky import ProcessPoolExecutor" in source
    assert "executor.submit(run_flight, index, scenario)" in source
    assert "pending = deque()" in source
    assert "index, scenario, future = pending.popleft()" in source
    assert "for index, scenario, result in completed_flights():" in source


def test_single_worker_path_avoids_process_pool():
    source = _notebook_source()

    assert "if MAX_WORKERS == 1:" in source
    assert "yield index, scenario, run_flight(index, scenario)" in source


def test_spawn_based_executor_supports_parallel_generation_without_fork():
    source = _notebook_source()

    assert "multiprocessing as mp" not in source
    assert 'mp.get_context("fork")' not in source
    assert "POSIX fork is unavailable" not in source
    # Only an explicitly requested single worker streams serially.
    assert source.count("yield index, scenario, run_flight(index, scenario)") == 1


def test_skill_instance_is_reused_until_the_scheduled_transition():
    source = _notebook_source()

    assert "if label != active_label:" in source
    assert "active_label, active_skill, contract = stochastic_manager.active" in source
    assert "active_skill.execute" in source


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
