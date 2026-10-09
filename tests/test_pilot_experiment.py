import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "pilot_experiment", Path(__file__).parents[1] / "scripts/run_pilot_experiment.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
paired_comparison = module.paired_comparison


def population(outcomes):
    return {
        pilot: {"episodes": [
            {"seed": 100 + i, "scenario": {"range_m": 18000 + 100 * i},
             "outcome": outcome, "survived": outcome != "mutual_destruction"}
            for i, outcome in enumerate(values)
        ]}
        for pilot, values in outcomes.items()
    }


def test_comparison_preserves_pairing_and_reports_population_mean_not_best_pilot():
    win, draw, mutual = "surviving_elimination", "survived_without_elimination", "mutual_destruction"
    old = population({"a": [win, draw], "b": [mutual, win]})
    new = population({"a": [win, win], "b": [win, draw]})
    result = paired_comparison(old, new)
    assert result["baseline_mean_clean_win_rate"] == 0.5
    assert result["experiment_mean_clean_win_rate"] == 0.75
    assert result["difference_percentage_points"] == 25
    assert result["pilots"][0]["gained_clean_wins"] == 1
    assert result["pilots"][1]["gained_clean_wins"] == 1
    assert result["pilots"][1]["lost_clean_wins"] == 1
    assert paired_comparison(old, new) == result


def test_comparison_rejects_different_cases_or_missing_pilots():
    old = population({"a": ["surviving_elimination"]})
    new = population({"a": ["surviving_elimination"]})
    new["a"]["episodes"][0]["seed"] += 1
    with pytest.raises(ValueError, match="unmatched"):
        paired_comparison(old, new)
    with pytest.raises(ValueError, match="populations differ"):
        paired_comparison(old, {})


def test_prepare_recovery_preserves_original_files_and_study_plan(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    parent = root / "artifacts/rl_pilot_experiments/failed"
    source = parent / "source"
    baseline = root / "baseline"
    previous = parent / "training"
    baseline.mkdir(parents=True)
    previous.mkdir(parents=True)
    complete = {"pilot_id": "first", "status": "completed"}
    pending = {"pilot_id": "second", "status": "failed"}
    for directory, pilots in ((baseline, [complete]), (previous, [complete, pending])):
        module.write(directory / "pilots.json", {"pilots": pilots})
        (directory / "first").mkdir()
        for name in ("config.json", "manifest.json", "best_model.pt", "training_metrics.jsonl", "test_episodes.json"):
            (directory / "first" / name).write_text("fixture")
    repairs = ["bvr_behavior_prediction/rl/simulation_workers.py",
               "bvr_behavior_prediction/rl/trainer.py", "bvr_behavior_prediction/rl/multi_pilot.py",
               "scripts/run_pilot_experiment.py"]
    for name in repairs + ["bvr_behavior_prediction/rl/pilot.py"]:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original")
        updated = root / name
        updated.parent.mkdir(parents=True, exist_ok=True)
        updated.write_text("repair" if name in repairs else "unrelated working-tree edit")
    monkeypatch.setattr(module, "__file__", str(root / "scripts/run_pilot_experiment.py"))
    plan = {"source": str(source), "baseline": str(baseline), "training_output": str(previous),
            "intervention": "selected_skill_parameters", "learning_rate": 3e-5,
            "confirmation": {"seed": 620000, "scenarios": 100},
            "source_hashes": {p.relative_to(source).as_posix(): module.digest(p)
                              for p in source.rglob("*") if p.is_file()},
            "baseline_hashes": module.baseline_inventory(baseline, [complete])}
    module.write(parent / "experiment.json", plan)
    module.write(parent / "status.json", {"status": "training"})
    with pytest.raises(ValueError, match="requires a failed job"):
        module.prepare_recovery(parent)
    module.write(parent / "status.json", {"status": "failed", "error": "lost worker"})
    original = {p: p.read_bytes() for p in parent.rglob("*") if p.is_file()}
    job = module.prepare_recovery(parent)
    recovery = module.read(job / "experiment.json")
    assert recovery["confirmation"] == plan["confirmation"]
    assert recovery["learning_rate"] == plan["learning_rate"]
    assert recovery["recovery"]["reused_pilots"] == ["first"]
    assert recovery["recovery"]["restart_from_scratch"] == ["second"]
    for name in repairs:
        assert (job / "source" / name).read_text() == "repair"
    assert (job / "source/bvr_behavior_prediction/rl/pilot.py").read_text() == "original"
    assert all(path.read_bytes() == value for path, value in original.items())
    for name, value in recovery["source_hashes"].items():
        assert module.digest(job / "source" / name) == value
