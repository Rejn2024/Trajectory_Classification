import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest

from bvr_behavior_prediction.rl.config import PilotTrainingConfig
from bvr_behavior_prediction.rl.reward import combat_reward_definition

spec = importlib.util.spec_from_file_location(
    "tactical_adaptation", Path(__file__).parents[1] / "scripts/run_tactical_adaptation.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def config():
    return PilotTrainingConfig(
        seed=7, epochs=2, scenarios_per_epoch=3, evaluation_seed=100_000,
        evaluation_scenarios_per_epoch=3, test_seed=200_000, test_scenarios_per_pilot=3,
        selected_skill_parameters=True, resample_training_scenarios=True,
    )


def test_tactical_seed_roles_are_disjoint_from_parent_and_previous_experiments():
    settings = config()
    module.check_seed_ranges([settings], 150, {"seed": 1_120_000, "scenarios": 300},
                             1_100_000, 1_220_000, [{"seed": 920_000, "scenarios": 300}])
    for assessment in ({"seed": 7, "scenarios": 300},
                       {"seed": 1_000_007, "scenarios": 300},
                       {"seed": 1_100_000, "scenarios": 300},
                       {"seed": 920_000, "scenarios": 300}):
        with pytest.raises(ValueError, match="overlaps"):
            module.check_seed_ranges([settings], 150, assessment, 1_100_000, 1_220_000,
                                     [{"seed": 920_000, "scenarios": 300}])


def test_prepare_preserves_source_models_and_reward_profiles(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    baseline = root / "baseline"
    pilot_dir = baseline / "first"
    pilot_dir.mkdir(parents=True)
    settings = replace(config(), output_dir=pilot_dir)
    reward = combat_reward_definition().as_dict()
    checkpoint = pilot_dir / "best_model.pt"
    checkpoint.write_bytes(b"fixture checkpoint")
    record = {"pilot_id": "first", "status": "completed", "seed": settings.seed,
              "checkpoint": "first/best_model.pt", "checkpoint_sha256": module.digest(checkpoint),
              "reward": reward}
    module.write(baseline / "pilots.json", {"pilots": [record]})
    module.write(pilot_dir / "config.json", {"training": settings.as_dict(), "reward": reward})
    for name in ("manifest.json", "training_metrics.jsonl", "test_episodes.json"):
        (pilot_dir / name).write_text("fixture", encoding="utf8")
    before = {p: p.read_bytes() for p in baseline.rglob("*") if p.is_file()}
    for directory in ("bvr_behavior_prediction", "bvr_sim_source/bvr_sim", "scripts"):
        (root / directory).mkdir(parents=True)
    for name in ("run_tactical_adaptation.py", "run_pilot_experiment.py"):
        (root / "scripts" / name).write_text("fixture", encoding="utf8")
    monkeypatch.setattr(module, "ROOT", root)
    job = module.prepare(baseline, epochs=150)
    plan = module.read(job / "experiment.json")
    assert plan["epochs_per_pilot"] == 150
    assert plan["opponent_type"] == "tactical"
    assert plan["assessment"] == {"seed": 1_120_000, "scenarios": 300}
    assert plan["baseline_hashes"] == module.baseline_inventory(baseline, [record])
    assert plan["reward_intervention"] is None
    assert not (job / "training").exists()
    assert all(path.read_bytes() == value for path, value in before.items())


def test_report_only_uses_tactical_assessment_results(tmp_path):
    module.write(tmp_path / "experiment.json", {"pilot_count": 1, "epochs_per_pilot": 150,
                                               "assessment": {"scenarios": 300}})
    module.write(tmp_path / "status.json", {"status": "baseline_assessment"})
    module.write(tmp_path / "baseline_assessment/first/evaluation.json", {"summary": {
        "scenarios": 300, "clean_win_rate": .2, "survival_rate": .4,
        "mutual_destruction_rate": .1, "loss_without_elimination_rate": .5, "outcome_score": -35,
    }})
    module.save_progress_report(tmp_path)
    assert "20.0%" in (tmp_path / "tactical_report.html").read_text()
    assert "Before fine-tuning" in (tmp_path / "tactical_results.csv").read_text()
