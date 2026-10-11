import importlib.util
import copy
from argparse import Namespace
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


@pytest.fixture
def prepared_baseline(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    baseline = root / "baseline_job/training"
    baseline.mkdir(parents=True)
    pilots = []
    for i in range(2):
        pilot_id = f"pilot_{i}"
        directory = baseline / pilot_id
        directory.mkdir()
        (directory / "best_model.pt").write_bytes(b"fixture")
        (directory / "training_metrics.jsonl").write_text('{"epoch_seconds": 1}\n')
        for name in ("manifest.json", "test_episodes.json"):
            module.write(directory / name, {})
        training = {"learning_rate": 3e-5, "selected_skill_parameters": True,
                    "seed": 7, "scenarios_per_epoch": 75, "epochs": 150,
                    "resample_training_scenarios": True, "evaluation_seed": 100000,
                    "evaluation_scenarios_per_epoch": 50, "test_seed": 200000,
                    "test_scenarios_per_pilot": 100, "simulator_workers": 22,
                    "mlflow_tracking_uri": f"sqlite:///job_{i}.db", "output_dir": str(directory)}
        module.write(directory / "config.json", {"training": training, "reward": {"fixture": i}})
        pilots.append({"pilot_id": pilot_id, "status": "completed", "reward": {"fixture": i},
                       "checkpoint": f"{pilot_id}/best_model.pt",
                       "checkpoint_sha256": module.digest(directory / "best_model.pt")})
    module.write(baseline / "pilots.json", {"pilots": pilots})
    module.write(baseline.parent / "experiment.json", {
        "confirmation": {"seed": 620000, "scenarios": 100}, "baseline": str(root / "ancestor"),
    })
    for name in ("bvr_behavior_prediction/placeholder", "bvr_sim_source/bvr_sim/placeholder",
                 "scripts/run_pilot_experiment.py", "notebooks/05_train_multiple_pilots.ipynb"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source fixture")
    monkeypatch.setattr(module, "__file__", str(root / "scripts/run_pilot_experiment.py"))
    args = Namespace(prepare=baseline, intervention="skill_conditioned_parameters", learning_rate=None,
                     confirmation_seed=820000, confirmation_scenarios=300)
    return root, baseline, args


def test_prepare_conditioned_preserves_matched_settings_and_recovered_baseline(prepared_baseline):
    root, baseline, args = prepared_baseline
    original = {p: p.read_bytes() for p in baseline.rglob("*") if p.is_file()}
    module.prepare(args)
    plan = module.read(next((root / "artifacts/rl_pilot_experiments").glob("*/experiment.json")))
    assert plan["learning_rate"] == plan["baseline_learning_rate"] == 3e-5
    assert plan["selected_skill_parameters"] is plan["baseline_selected_skill_parameters"] is True
    assert plan["skill_conditioned_parameters"] is True
    assert plan["baseline_skill_conditioned_parameters"] is False
    assert plan["epochs_per_pilot"] == 150
    assert plan["confirmation"] == {"seed": 820000, "scenarios": 300}
    assert plan["bounded_decision_criteria"]["minimum_clean_win_gain_percentage_points"] == 3
    assert all(path.read_bytes() == value for path, value in original.items())


@pytest.mark.parametrize("invalid", ["learning_rate", "old_confirmation", "training_seed", "different_settings", "already_conditioned"])
def test_prepare_rejects_unmatched_or_contaminated_experiments(prepared_baseline, invalid):
    _, baseline, args = prepared_baseline
    if invalid == "learning_rate":
        args.learning_rate = 1e-4
    elif invalid == "old_confirmation":
        args.confirmation_seed = 620001
    elif invalid == "training_seed":
        args.confirmation_seed = 10
    else:
        path = baseline / ("pilot_1" if invalid == "different_settings" else "pilot_0") / "config.json"
        data = module.read(path)
        data["training"]["epochs" if invalid == "different_settings" else "skill_conditioned_parameters"] = (
            151 if invalid == "different_settings" else True)
        module.write(path, data)
    with pytest.raises(ValueError):
        module.prepare(args)


@pytest.fixture
def decision_fixture():
    plan = {"pilot_count": 2, "confirmation": {"seed": 820000, "scenarios": 300},
            "bounded_decision_criteria": {
                "minimum_clean_win_gain_percentage_points": 3,
                "minimum_ci_lower_bound_percentage_points_exclusive": 0,
                "maximum_survival_drop_percentage_points": 2,
                "maximum_mutual_destruction_increase_percentage_points": 2,
            }}
    comparison = {"status": "completed", "confirmation_suite": plan["confirmation"],
                  "fresh_confirmation": {
                      "difference_percentage_points": 3,
                      "paired_scenario_bootstrap_ci95_percentage_points": [0.1, 6],
                      "pilots": [{"pilot_id": f"p{i}", "scenarios": 300,
                                  "baseline_survival_rate": .7, "experiment_survival_rate": .68,
                                  "baseline_mutual_destruction_rate": .2, "experiment_mutual_destruction_rate": .22}
                                 for i in range(2)],
                  }}
    return comparison, plan


@pytest.mark.parametrize("failure", [None, "small_gain", "uncertain", "survival", "mutual"])
def test_bounded_decision_checks_all_predeclared_gates(decision_fixture, failure):
    comparison, plan = copy.deepcopy(decision_fixture)
    result = comparison["fresh_confirmation"]
    if failure == "small_gain":
        result["difference_percentage_points"] = 2.9
    elif failure == "uncertain":
        result["paired_scenario_bootstrap_ci95_percentage_points"][0] = 0
    elif failure == "survival":
        result["pilots"][0]["experiment_survival_rate"] = .67
    elif failure == "mutual":
        result["pilots"][0]["experiment_mutual_destruction_rate"] = .23
    decision = module.bounded_decision(comparison, plan)
    assert decision["passed"] is (failure is None)
    assert decision["automatic_followup_training"] is False


@pytest.mark.parametrize("failure", ["running", "missing_pilot", "missing_case", "wrong_suite"])
def test_no_bounded_decision_on_incomplete_or_wrong_confirmation(decision_fixture, failure):
    comparison, plan = copy.deepcopy(decision_fixture)
    if failure == "running":
        comparison["status"] = "confirmation in progress"
    elif failure == "missing_pilot":
        comparison["fresh_confirmation"]["pilots"].pop()
    elif failure == "missing_case":
        comparison["fresh_confirmation"]["pilots"][0]["scenarios"] = 299
    else:
        comparison["confirmation_suite"]["seed"] = 10
        plan["confirmation"] = {"seed": 820000, "scenarios": 300}
    with pytest.raises(ValueError):
        module.bounded_decision(comparison, plan)


def test_clean_win_recipe_preserves_weights_and_round_trips():
    from bvr_behavior_prediction.rl.reward import combat_reward_definition

    original = combat_reward_definition("profile").as_dict()
    saved = copy.deepcopy(original)
    changed = module.experiment_reward(original, "clean_win_bonus")
    assert original == saved
    assert changed["version"] == "3"
    assert changed["parameters"] == {**original["parameters"], "clean_win_bonus": 250}
    assert module.definition(changed).as_dict() == changed
    assert module.definition(original).as_dict() == original
    assert module.experiment_reward(original, "selected_skill_parameters") == original
    with pytest.raises(ValueError, match="original CombatReward"):
        module.experiment_reward(changed, "clean_win_bonus")


def test_prepare_clean_win_uses_shared_architecture_and_declares_reward_intervention(prepared_baseline):
    from bvr_behavior_prediction.rl.reward import combat_reward_definition

    root, baseline, args = prepared_baseline
    batch = module.read(baseline / "pilots.json")
    for pilot in batch["pilots"]:
        pilot["reward"] = combat_reward_definition(pilot["pilot_id"]).as_dict()
        path = baseline / pilot["pilot_id"] / "config.json"
        config = module.read(path)
        config["reward"] = pilot["reward"]
        module.write(path, config)
    module.write(baseline / "pilots.json", batch)
    args.intervention = "clean_win_bonus"
    args.confirmation_seed = 920000
    module.prepare(args)
    plan = module.read(next((root / "artifacts/rl_pilot_experiments").glob("*/experiment.json")))
    assert plan["learning_rate"] == 3e-5
    assert plan["selected_skill_parameters"] is True
    assert plan["skill_conditioned_parameters"] is False
    assert plan["reward_intervention"]["clean_win_bonus"] == 250
    assert plan["confirmation"] == {"seed": 920000, "scenarios": 300}
    assert plan["epochs_per_pilot"] == 150


def test_prepare_rejects_previously_used_sibling_confirmation(prepared_baseline):
    root, baseline, args = prepared_baseline
    sibling = root / "sibling"
    sibling.mkdir()
    module.write(sibling / "experiment.json", {"confirmation": {"seed": 820000, "scenarios": 300}})
    (root / "configs").mkdir()
    module.write(root / "configs/pilot_experiments.json", {"experiments": {"sibling": {"job": "sibling"}}})
    with pytest.raises(ValueError, match="previously inspected"):
        module.prepare(args)


def test_bounded_recommendation_names_the_actual_experiment(decision_fixture):
    comparison, plan = decision_fixture
    plan["experiment_label"] = "Clean-win bonus"
    assert "Clean-win bonus" in module.bounded_decision(comparison, plan)["recommendation"]
