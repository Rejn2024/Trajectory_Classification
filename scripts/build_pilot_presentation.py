"""Export existing pilot results as an offline presentation. Never runs a simulator.

Use --collect on the training PC to audit and package existing result files.
Without --collect, rebuild the PDF/HTML from the committed data on any PC with
Matplotlib and NumPy. Viewing the exported files needs neither Python nor Git LFS.
Use --collect-forecast to package the saved forecasting validation results.
"""

import argparse
import base64
import csv
import hashlib
import html
import io
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as PlotPath

from presentation_forecasting import add_forecasting_slides, collect_forecasting

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "docs/presentations"
TITLE = "From multi-pilot training to motion forecasting"
OUTCOMES = (
    "surviving_elimination", "survived_without_elimination",
    "mutual_destruction", "loss_without_elimination",
)
SCORES = dict(zip(OUTCOMES, (100, 0, -50, -100)))
BG, FG, MUTED, TEAL, BLUE, RED = "#101e30", "#f2f5f9", "#aebdd0", "#62ddbc", "#8baeff", "#ff958a"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def summarize(episodes):
    counts = Counter(e["outcome"] for e in episodes)
    assert set(counts) <= set(OUTCOMES)
    n = len(episodes)
    result = {k: counts[k] / n for k in OUTCOMES}
    result["elimination"] = sum(bool(e["eliminated"]) for e in episodes) / n
    result["survival"] = sum(bool(e["survived"]) for e in episodes) / n
    assert all((e["outcome"] == OUTCOMES[0]) == (bool(e["eliminated"]) and bool(e["survived"])) for e in episodes)
    result["outcome_score"] = sum(SCORES[e["outcome"]] for e in episodes) / n
    return result


def collect_development(registry, evidence, inventory):
    """Package the fixed simple-opponent benchmark and recorded learning curves."""
    populations = [
        ("original", "Original", registry["lower_learning_rate"]["baseline"]),
        ("lower_learning_rate", "Reduced learning rate", registry["lower_learning_rate"]["training"]),
        ("selected_skill_parameters", "Selected-skill probabilities", registry["selected_skill_parameters"]["training"]),
        ("skill_conditioned_parameters", "Per-skill parameter heads", registry["skill_conditioned_parameters"]["training"]),
    ]
    result = {"opponent": "simple", "role": "shared development benchmark",
              "seed": 211307, "scenarios_per_pilot": 100, "training_seed": 7,
              "validation_seed": 111257, "validation_scenarios_per_pilot": 50,
              "populations": []}
    curve_rows, summary_rows = [], []
    reference_cases, reference_schedule, reference_rewards = None, None, {}
    for key, label, directory in populations:
        population = {"key": key, "label": label, "directory": directory, "pilots": []}
        curves = []
        for i in range(1, 11):
            pid = f"pilot_{i:03d}"
            run = ROOT / directory / pid
            config = evidence(run / "config.json")["training"]
            manifest = evidence(run / "manifest.json")
            evaluation = evidence(run / "test_episodes.json")
            assert config["seed"] == 7 and config["evaluation_seed"] == 111257
            assert config["evaluation_scenarios_per_epoch"] == 50
            assert config["test_seed"] == 211307 and config["test_scenarios_per_pilot"] == 100
            assert sha(run / "best_model.pt") == evaluation["checkpoint_sha256"] == manifest["checkpoint_sha256"]
            if key == "original":
                reference_rewards[pid] = manifest["reward"]
            assert manifest["reward"] == reference_rewards[pid]
            episodes = evaluation["episodes"]
            cases = [(e["seed"], e["scenario"]) for e in episodes]
            assert [e["seed"] for e in episodes] == list(range(211307, 211407))
            if reference_cases is None:
                reference_cases = cases
            assert cases == reference_cases, f"Mismatched development cases: {key}/{pid}"
            path = run / "training_metrics.jsonl"
            inventory[path.relative_to(ROOT).as_posix()] = sha(path)
            metrics = [json.loads(line) for line in path.read_text().splitlines()]
            assert [m["epoch"] for m in metrics] == list(range(1, 151))
            schedule = [(m["training_seed_first"], m["training_seed_last"]) for m in metrics]
            assert schedule == [(7 + epoch * 75, 81 + epoch * 75) for epoch in range(150)]
            if reference_schedule is None:
                reference_schedule = schedule
            assert schedule == reference_schedule
            assert all(m["episodes"] == 75 for m in metrics)
            assert all(math.isfinite(v) for m in metrics for v in m.values() if isinstance(v, float))
            assert [m["epoch"] for m in metrics if m["evaluation_performed"]] == [1, *range(5, 151, 5)]
            selected = max((m for m in metrics if m["evaluation_performed"]), key=lambda m: m["mean_return"])
            assert math.isclose(selected["mean_return"], manifest["selection_return"])
            row = {"population": key, "label": label, "pilot": pid,
                   "selected_epoch": selected["epoch"], "checkpoint_sha256": manifest["checkpoint_sha256"],
                   "final_training_return": metrics[-1]["training_mean_return"],
                   "selected_validation_return": manifest["selection_return"],
                   "validation_benchmark_return": manifest["benchmark_return"],
                   "validation_survival": manifest["survival_rate"],
                   "validation_elimination": manifest["elimination_rate"],
                   **{f"development_{k}": v for k, v in summarize(episodes).items()}}
            population["pilots"].append(row)
            summary_rows.append(row)
            for m in metrics:
                curve_rows.append({"population": key, "pilot": pid, "epoch": m["epoch"],
                                   "training_return": m["training_mean_return"],
                                   "validation_return": m["mean_return"] if m["evaluation_performed"] else None,
                                   "best_validation_return": m["best_return"],
                                   "policy_loss": m["policy_loss"], "value_loss": m["value_loss"]})
            curves.append(metrics)
        population["clean_win_rate"] = float(np.mean([p["development_surviving_elimination"] for p in population["pilots"]]))
        for metric in ("elimination", "survival", "mutual_destruction"):
            population[f"{metric}_rate"] = float(np.mean([p[f"development_{metric}"] for p in population["pilots"]]))
        population["learning"] = [
            {"epoch": epoch + 1,
             "training_return": float(np.mean([c[epoch]["training_mean_return"] for c in curves])),
             "validation_return": float(np.mean([c[epoch]["mean_return"] for c in curves]))
             if all(c[epoch]["evaluation_performed"] for c in curves) else None}
            for epoch in range(150)
        ]
        result["populations"].append(population)
    return result, curve_rows, summary_rows


def collect(dest):
    """Read, cross-check and copy only presentation evidence, never model weights."""
    registry = read(ROOT / "configs/pilot_experiments.json")["experiments"]
    entry = registry["tactical_adaptation"]
    job, baseline = ROOT / entry["job"], ROOT / entry["baseline"]
    plan, status, comp = (read(job / f"{name}.json") for name in ("experiment", "status", "comparison"))
    assert status["status"] == comp["status"] == "completed"
    inventory = {}

    def evidence(path):
        inventory[path.relative_to(ROOT).as_posix()] = sha(path)
        return read(path)

    for rel, digest in plan["baseline_hashes"].items():
        assert sha(baseline / rel) == digest, f"Changed parent: {rel}"
    for rel, digest in plan["source_hashes"].items():
        assert sha(job / "source" / rel) == digest, f"Changed frozen source: {rel}"
    data = {
        "title": "Ten pilots: from population PPO to tactical adaptation",
        "date": "11 October 2026", "completed_utc": status["updated_utc"],
        "started_utc": status["started_utc"], "job": entry["job"], "baseline": entry["baseline"],
        "audit": {"parent_files_unchanged": len(plan["baseline_hashes"]),
                  "frozen_source_files_unchanged": len(plan["source_hashes"])},
        "suites": {}, "pilots": [], "history": [],
    }
    paired = []
    pilot_ids = [f"pilot_{i:03d}" for i in range(1, 11)]
    for suite, count, seed, before_dir, after_dir, comparison_key in (
        ("assessment", 300, 1120000, "baseline_assessment", "adapted_assessment", "development"),
        ("confirmation", 100, 1220000, "baseline_confirmation", "training", "fresh_confirmation"),
    ):
        result = {"scenarios_per_pilot": count, "seed": seed, "pilots": []}
        combined = {"before": [], "after": []}
        scenario_deltas = []
        for pid in pilot_ids:
            before = evidence(job / before_dir / pid / "evaluation.json")
            after = evidence(job / after_dir / pid / ("test_episodes.json" if after_dir == "training" else "evaluation.json"))
            assert before["opponent_type"] == "tactical"
            if after_dir != "training":
                assert after["opponent_type"] == "tactical"
            episodes = {"before": before["episodes"], "after": after["episodes"]}
            for values in episodes.values():
                assert len(values) == count
                assert [e["seed"] for e in values] == list(range(seed, seed + count))
            for b, a in zip(episodes["before"], episodes["after"]):
                assert b["scenario"] == a["scenario"]
                paired.append({"suite": suite, "pilot": pid, "seed": b["seed"],
                               "before_outcome": b["outcome"], "after_outcome": a["outcome"],
                               "before_eliminated": int(b["eliminated"]), "after_eliminated": int(a["eliminated"]),
                               "before_survived": int(b["survived"]), "after_survived": int(a["survived"])})
            row = {"pilot": pid, **{k: summarize(v) for k, v in episodes.items()}}
            expected = next(p for p in comp[comparison_key]["pilots"] if p["pilot_id"] == pid)
            for label, prefix in (("before", "baseline"), ("after", "experiment")):
                assert math.isclose(row[label][OUTCOMES[0]], expected[f"{prefix}_clean_win_rate"])
                assert math.isclose(row[label]["survival"], expected[f"{prefix}_survival_rate"])
                assert math.isclose(row[label][OUTCOMES[2]], expected[f"{prefix}_mutual_destruction_rate"])
                combined[label].extend(episodes[label])
            scenario_deltas.append([
                int(a["outcome"] == OUTCOMES[0]) - int(b["outcome"] == OUTCOMES[0])
                for b, a in zip(episodes["before"], episodes["after"])
            ])
            result["pilots"].append(row)
        result.update({k: summarize(v) for k, v in combined.items()})
        result["clean_win_gain_pp"] = 100 * (result["after"][OUTCOMES[0]] - result["before"][OUTCOMES[0]])
        result["ci95_gain_pp"] = comp[comparison_key]["paired_scenario_bootstrap_ci95_percentage_points"]
        # Independent verification: resample shared scenarios, not individual flights.
        deltas = np.asarray(scenario_deltas).mean(axis=0)
        rng = np.random.default_rng(8128)
        boots = np.array([deltas[rng.integers(0, count, count)].mean() * 100 for _ in range(2000)])
        assert np.allclose(np.quantile(boots, [.025, .975]), result["ci95_gain_pp"])
        data["suites"][suite] = result
    for pid in pilot_ids:
        run = job / "training" / pid
        manifest = evidence(run / "manifest.json")
        config = evidence(run / "config.json")
        initial = evidence(run / "initialization.json")
        metrics_path = run / "training_metrics.jsonl"
        inventory[metrics_path.relative_to(ROOT).as_posix()] = sha(metrics_path)
        metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()]
        assert [m["epoch"] for m in metrics] == list(range(1, 151))
        assert all(m["episodes"] == 75 for m in metrics)
        training_seeds = set()
        for m in metrics:
            expected_first = 1000007 + (m["epoch"] - 1) * 75
            assert m["training_seed_first"] == expected_first
            assert m["training_seed_last"] == expected_first + 74
            training_seeds.update(range(expected_first, expected_first + 75))
        assert not training_seeds.intersection(range(1100000, 1100050))
        assert not training_seeds.intersection(range(1120000, 1120300))
        assert not training_seeds.intersection(range(1220000, 1220100))
        assert all(math.isfinite(v) for m in metrics for v in m.values() if isinstance(v, float))
        assert sha(run / "initial_model.pt") == sha(baseline / pid / "best_model.pt") == initial["checkpoint_sha256"]
        assert sha(run / "best_model.pt") == manifest["checkpoint_sha256"]
        assert manifest["reward"] == read(baseline / pid / "manifest.json")["reward"]
        selected = max((m for m in metrics if m["evaluation_performed"]), key=lambda m: m["mean_return"])
        assert math.isclose(selected["mean_return"], manifest["selection_return"])
        assert selected["mean_return"] > initial["initial_validation_return"]
        data["pilots"].append({
            "pilot": pid, "epochs": len(metrics), "selected_epoch": selected["epoch"],
            "simulator_retries": sum(m.get("simulator_batch_retries", 0) for m in metrics),
            "initial_sha256": initial["checkpoint_sha256"], "selected_sha256": manifest["checkpoint_sha256"],
            "reward_weights": manifest["reward"]["parameters"]["weights"],
            "skill_decision_fractions": manifest["test"]["skill_decision_fractions"],
            "initial_validation_return": initial["initial_validation_return"],
            "selected_validation_return": manifest["selection_return"],
            "validation": [{"epoch": m["epoch"], "return": m["mean_return"]} for m in metrics if m["evaluation_performed"]],
        })
        if pid == "pilot_001":
            excluded = {"output_dir", "mlflow_tracking_uri"}
            data["training"] = {k: v for k, v in config["training"].items() if k not in excluded}
            data["skill_names"] = config["skill_names"]
            data["parameter_names"] = config["parameter_names"]
            data["observation_size"] = config["observation_size"]
    for key, label in (("lower_learning_rate", "Lower learning rate"),
                       ("selected_skill_parameters", "Selected-skill probabilities"),
                       ("skill_conditioned_parameters", "Per-skill parameter heads")):
        c = evidence(ROOT / registry[key]["job"] / "comparison.json")
        fresh = c["fresh_confirmation"]
        combined = {"baseline": [], "experiment": []}
        for pid in pilot_ids:
            pair = {}
            for population in combined:
                path = ROOT / registry[key]["job"] / "confirmation" / population / pid / "evaluation.json"
                pair[population] = evidence(path)["episodes"]
                assert len(pair[population]) == c["confirmation_suite"]["scenarios"]
                combined[population].extend(pair[population])
            assert [(e["seed"], e["scenario"]) for e in pair["baseline"]] == [
                (e["seed"], e["scenario"]) for e in pair["experiment"]]
        before, after = (summarize(combined[name]) for name in ("baseline", "experiment"))
        assert math.isclose(before[OUTCOMES[0]], fresh["baseline_mean_clean_win_rate"])
        assert math.isclose(after[OUTCOMES[0]], fresh["experiment_mean_clean_win_rate"])
        data["history"].append({"label": label, "opponent": "simple", "pilots": 10,
                                "suite": c["confirmation_suite"], "role": "fresh confirmation",
                                "before": fresh["baseline_mean_clean_win_rate"],
                                "after": fresh["experiment_mean_clean_win_rate"],
                                "before_outcomes": before, "after_outcomes": after,
                                "ci95_gain_pp": fresh["paired_scenario_bootstrap_ci95_percentage_points"]})
    bonus = registry["clean_win_bonus"]
    before, after = [], []
    for i in range(1, 7):
        pid = f"pilot_{i:03d}"
        before.extend(evidence(ROOT / bonus["baseline"] / pid / "test_episodes.json")["episodes"])
        after.extend(evidence(ROOT / bonus["training"] / pid / "test_episodes.json")["episodes"])
    assert [(e["seed"], e["scenario"]) for e in before] == [(e["seed"], e["scenario"]) for e in after]
    data["history"].append({"label": "Terminal clean-win bonus (stopped)", "opponent": "simple", "pilots": 6,
                            "suite": {"seed": before[0]["seed"], "scenarios": 100}, "role": "development only",
                            "before": summarize(before)[OUTCOMES[0]], "after": summarize(after)[OUTCOMES[0]],
                            "before_outcomes": summarize(before), "after_outcomes": summarize(after)})
    data_dir = dest / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    data["development"], curves, summaries = collect_development(registry, evidence, inventory)
    for filename, rows in (("learning_curves.csv", curves), ("development_summary.csv", summaries)):
        with (data_dir / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    write_json(data_dir / "results.json", data)
    write_json(data_dir / "source_hashes.json", inventory)
    with (data_dir / "paired_outcomes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(paired)
    with (data_dir / "reward_profiles.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["pilot", *data["pilots"][0]["reward_weights"]], lineterminator="\n")
        writer.writeheader()
        writer.writerows({"pilot": p["pilot"], **p["reward_weights"]} for p in data["pilots"])
    summary_rows = []
    for suite_name, suite in data["suites"].items():
        for p in suite["pilots"]:
            summary_rows.append({"suite": suite_name, "pilot": p["pilot"],
                                 "scenarios": suite["scenarios_per_pilot"],
                                 **{f"{state}_{key}": value for state in ("before", "after")
                                    for key, value in p[state].items()}})
    with (data_dir / "pilot_outcomes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)
    return data


def render(dest, data):
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": FG,
                         "axes.labelcolor": FG, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.edgecolor": MUTED, "font.size": 12, "savefig.facecolor": BG,
                         "figure.max_open_warning": 0})  # Slides are closed after the batch export.
    slides = []

    def page(title, subtitle):
        fig = plt.figure(figsize=(16, 9), facecolor=BG)
        fig.text(.065, .929, "PILOT POPULATION  /  DATASET GENERATION  /  FORECASTING", color=TEAL, size=10, weight="bold")
        fig.text(.065, .849, title, size=30, weight="bold")
        fig.text(.065, .793, subtitle, size=14, color=MUTED)
        fig.text(.065, .041, "Saved experiment results  |  Scripted tactical observer  |  Forecasting final test remains reserved", size=9, color=MUTED)
        fig.text(.935, .041, str(len(slides) + 1).zfill(2), ha="right", size=11, color=TEAL)
        return fig

    def words(fig, x, y, title, body):
        fig.text(x, y, title, size=20, color=TEAL, weight="bold", va="top")
        fig.text(x, y - .065, body, size=17, linespacing=1.55, va="top")

    def chart(fig, bounds):
        ax = fig.add_axes(bounds, facecolor=BG)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        return ax

    def finish(fig, note):
        slides.append((fig, note))

    def outcome_table(fig, bounds, rows, labels, widths, fontsize=14):
        ax = fig.add_axes(bounds)
        ax.axis("off")
        table = ax.table(cellText=rows, colLabels=labels, colWidths=widths,
                         cellLoc="left", colLoc="left", bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(fontsize)
        for (row, _col), cell in table.get_celld().items():
            cell.set_facecolor("#1c324b" if row == 0 else BG)
            cell.set_edgecolor("#35485e")
            cell.get_text().set_color(TEAL if row == 0 else FG)

    def pipeline_box(x, y, width, height, title, body, colour=TEAL, fontsize=13):
        box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.007,rounding_size=0.009",
                             transform=f.transFigure, facecolor="#192d44", edgecolor=colour,
                             linewidth=1.1, zorder=1)
        f.add_artist(box)
        f.text(x + .012, y + height - .017, title, size=15, weight="bold", color=colour, va="top")
        if body:
            f.text(x + .012, y + height - .056, body, size=fontsize, va="top", linespacing=1.35)

    def pipeline_arrow(points, colour=TEAL, dashed=False):
        route = PlotPath(points, [PlotPath.MOVETO, *([PlotPath.LINETO] * (len(points) - 1))])
        f.add_artist(FancyArrowPatch(path=route, transform=f.transFigure, arrowstyle="-|>",
                                    mutation_scale=14, linewidth=1.5, color=colour,
                                    linestyle="--" if dashed else "-", zorder=2))

    f = page("Multi-pilot training as dataset preparation",
             "Project objective: forecast trained aircraft movement from one-sided observations")
    pipeline_box(.065, .46, .255, .25, "1  TRAIN PILOTS",
                 "Learn useful combat behaviour\nVary fixed reward priorities\nAdapt to tactical opposition\nRetain ten selected checkpoints", fontsize=14)
    pipeline_box(.375, .46, .255, .25, "2  RECORD ENCOUNTERS",
                 "Freeze the learned pilot policies\n300 encounters per pilot\n3,000 episodes in total\nObserve from the tactical aircraft", colour=BLUE, fontsize=14)
    pipeline_box(.685, .46, .25, .25, "3  FORECAST MOTION",
                 "Use the recorded observations\nPredict the trained pilot's motion\nPosition and velocity at +1 / 2 / 5 s\nAssess on held-out scenarios", fontsize=13)
    f.text(.065, .35, "Multi-pilot RL is an upstream pre-processing and data-generation stage for forecasting.",
           size=18, color=TEAL)
    f.text(.065, .27, "Combat evaluations establish the quality of the flight-generating policies.\n"
           "Different reward priorities provide behavioural variation for the observation dataset.",
           size=17, linespacing=1.6)
    f.text(.065, .13, "The final forecasting objective is learned from recorded movement; pilot rewards guide the earlier RL stage.",
           size=14, color=MUTED)
    finish(f, "The repository's end objective is forecasting aircraft manoeuvres from one-sided observations. Multi-pilot reinforcement learning, its controlled experiments and tactical adaptation prepare the policies used to generate that dataset. The learned pilots then fly new encounters with frozen checkpoints against the scripted tactical opponent. In the current forecasting corpus, the tactical aircraft B01 is the observer and trained aircraft A01 is the prediction target. Ten pilots each contribute 300 episodes from matched starting scenarios. Variation in rewards and observed behaviour broadens the sources of trajectories, without guaranteeing that all manoeuvres or styles are covered. Forecasting is a subsequent supervised learning problem using those recorded flights; it does not change the pilot policies. This deck first documents the upstream population training and then presents initial forecasting validation results.")

    f = page("Where RL, BVR Sim and JSBSim fit",
             "The learned policy chooses behaviour; the simulator and flight model determine what happens")
    pipeline_box(.065, .54, .255, .18, "OUR ACTOR + CRITIC",
                 "Transformer reads flight history\nActor selects skill + parameters\nCritic estimates future reward", fontsize=15)
    pipeline_box(.375, .54, .255, .18, "TACTICAL SKILLS",
                 "33 fixed tactical skills\nSelected skill produces guidance\nHeading / altitude / speed / fire", fontsize=15)
    pipeline_box(.685, .28, .25, .44, "BVR SIM",
                 "Aircraft and scripted opponent\nSensors, missiles and combat events\nControllers translate guidance\nto aircraft control inputs", colour=BLUE, fontsize=13.5)
    physics = FancyBboxPatch((.697, .299), .226, .17,
                             boxstyle="round,pad=0.005,rounding_size=0.006",
                             transform=f.transFigure, facecolor="#233e59", edgecolor=BLUE, zorder=1)
    f.add_artist(physics)
    f.text(.71, .445, "JSBSIM", size=17, color=BLUE, weight="bold", va="top")
    f.text(.71, .401, "Aircraft flight dynamics\nControls → forces and moments\nUpdated position, speed, attitude", size=12.5, va="top", linespacing=1.35)
    pipeline_box(.375, .28, .255, .20, "ENVIRONMENT ADAPTER",
                 "Builds the observation history\nCollects events across substeps\nReturns state and event records", fontsize=14.5)
    pipeline_box(.065, .28, .255, .20, "REWARD + PPO",
                 "Pilot weights score combat events\nExperience trains actor and critic\nSeparate model for each pilot", fontsize=14.5)
    pipeline_arrow([(.327, .62), (.367, .62)])
    pipeline_arrow([(.637, .62), (.677, .62)])
    pipeline_arrow([(.677, .375), (.637, .375)], colour=BLUE)
    pipeline_arrow([(.367, .375), (.327, .375)])
    pipeline_arrow([(.1875, .488), (.1875, .532)])
    pipeline_arrow([(.5025, .488), (.5025, .507), (.27, .507), (.27, .532)], colour=BLUE)
    f.text(.395, .51, "Next observation", size=10.5, color=BLUE, ha="center", va="bottom")
    f.text(.065, .20, "Current setup: F-16 versus F-16, 1-v-1 • scripted simple or tactical opponent • ten independent pilots",
           size=14, color=TEAL)
    f.text(.065, .135, "CPU workers advance independent duels. GPU batches policy inference and PPO updates.", size=15)
    f.text(.065, .082, "Scenario setup initializes the encounter. JSBSim supplies aircraft physics; BVR Sim supplies the combat environment.",
           size=12, color=MUTED)
    finish(f, "This is the architecture used by the repository's custom PPO pipeline, rather than the separate skrl training example bundled with BVR Sim. The policy chooses a tactical skill and its numeric parameters. SkillManager converts that choice into heading, altitude, speed and fire guidance, which BVR Sim routes through its aircraft controllers. JSBSim computes aircraft flight dynamics inside the environment. BVR Sim also maintains the opponent, sensors, missiles and engagement events. The adapter exposes flight observations and accumulated events to our training code, which builds history and applies each pilot's own reward definition. The critic supplies learning predictions; aircraft control comes from the actor's actions and fixed skill/controller implementations. Each worker advances its own simulation; pilots do not learn against one another in these experiments.")

    f = page("RL pipeline: decisions, simulation and learning",
             "Ten independent pilots • individual reward profiles and checkpoints • shared scenario schedules")
    f.text(.065, .755, "Seeded duels: initial geometry, speed and altitude • 75 fresh scenarios / epoch • 150 epochs / pilot",
           size=12, color=MUTED)
    pipeline_box(.065, .535, .245, .17, "1  Observation history",
                 "20 frames × 51 features\nPosition, motion, threats + energy\nTwo seconds of state at 10 Hz")
    pipeline_box(.365, .535, .245, .17, "2  Actor + critic (GPU)",
                 "Transformer processes history\nActor: skill + numeric parameters\nCritic: expected future return\nOne decision about every second")
    pipeline_box(.685, .535, .25, .17, "3  Skills → guidance",
                 "Choose from 33 fixed skills\nHeading / altitude / speed / fire\nGuidance passed to controllers")
    pipeline_box(.685, .27, .25, .20, "4  BVR Sim (CPU)",
                 "Scripted opponent, missiles, events\nControllers execute flight guidance", colour=BLUE)
    physics = FancyBboxPatch((.697, .283), .226, .055,
                             boxstyle="round,pad=0.005,rounding_size=0.006",
                             transform=f.transFigure, facecolor="#233e59", edgecolor=BLUE, zorder=1)
    f.add_artist(physics)
    f.text(.81, .311, "JSBSim: aircraft flight physics", size=12.5, ha="center", va="center", color=BLUE)
    pipeline_box(.365, .27, .245, .20, "5  Reward + experience",
                 "Combat events × pilot coefficients\nStore observations, actions, rewards,\nvalue estimates and probabilities\nCollect 75 flights before updating")
    pipeline_box(.065, .27, .245, .20, "6  PPO update (GPU)",
                 "Scaled rewards + critic → advantages\nClipped policy loss + value loss\n256 transitions per minibatch\nUpdate actor and critic; repeat", fontsize=12.5)

    pipeline_arrow([(.317, .62), (.357, .62)])
    pipeline_arrow([(.617, .62), (.677, .62)])
    pipeline_arrow([(.81, .527), (.81, .478)])
    pipeline_arrow([(.677, .37), (.617, .37)])
    pipeline_arrow([(.357, .37), (.317, .37)])
    pipeline_arrow([(.943, .41), (.96, .41), (.96, .73), (.1875, .73), (.1875, .713)], colour=BLUE)
    f.text(.57, .733, "Next observation from the simulation", size=10.5, color=BLUE, ha="center", va="bottom")
    pipeline_arrow([(.1875, .478), (.1875, .504), (.4875, .504), (.4875, .527)])
    f.text(.32, .507, "Updated actor + critic", size=10.5, color=TEAL, ha="center", va="bottom")

    pipeline_box(.065, .12, .245, .10, "Validation → checkpoint",
                 "50 fixed cases, periodically\nSave best own validation return", colour="#ffcf88", fontsize=11.5)
    pipeline_box(.365, .12, .245, .10, "Compare selected pilots",
                 "Matched development scenarios\nElimination • survival • clean win", colour="#ffcf88", fontsize=11.5)
    pipeline_box(.685, .12, .25, .10, "Tactical continuation",
                 "Load selected actor + critic\nSame rewards; tactical opponent", colour="#ffcf88", fontsize=11.5)
    pipeline_arrow([(.1875, .262), (.1875, .228)], colour="#ffcf88", dashed=True)
    pipeline_arrow([(.317, .17), (.357, .17)], colour="#ffcf88", dashed=True)
    pipeline_arrow([(.617, .17), (.677, .17)], colour="#ffcf88", dashed=True)
    f.text(.065, .075, "CPU workers parallelise flights within each pilot. Final testing uses untouched cases after development choices are frozen.",
           size=11, color=MUTED)
    finish(f, "The diagram describes this repository's custom PPO pipeline. Each pilot has its own fixed reward coefficients, actor/critic parameters, optimizer and checkpoint; pilots train independently and sequentially. Seeded scenarios specify the initial duel geometry, speeds and altitudes. The environment adapter builds a two-second observation history and accumulates combat events across 0.1-second simulator steps. The actor selects a skill and parameters roughly once per second. SkillManager converts that decision to guidance; BVR Sim manages the scripted opponent, missiles and engagement events, while aircraft controllers and JSBSim advance flight dynamics. The critic estimates discounted future own reward; it does not command the aircraft. Per-pilot event rewards, observations, actions, old action log probabilities, values and termination flags form the rollout. After 75 flights, scaled rewards and critic estimates produce GAE advantages and return targets. PPO updates the actor and critic using 256-transition minibatches, an entropy term and up to ten update passes with KL stopping. This collection/update cycle repeats for 150 epochs. Batched neural work runs on the GPU; persistent CPU workers run independent simulators. Validation uses 50 fixed, separate cases at epoch 1, every fifth epoch and the final epoch; the highest own validation return selects the checkpoint. Development cases compare those selected pilots using elimination, survival, clean wins and mutual destruction; they do not select checkpoints. Tactical continuation loads each selected actor and critic, starts a fresh Adam optimizer and adds 150 epochs against the tactical scripted opponent with the same reward profile. Future final testing is reserved for unused scenarios after development choices are fixed. Historical confirmation results in the later slides remain archived evidence.")

    f = page("Training, checkpoint selection and comparison",
             "Training changes weights • validation selects checkpoints • development compares experiments")
    pipeline_box(.065, .51, .255, .20, "COLLECT 75 DUELS",
                 "Fresh seeded cases each epoch\nParallel CPU flights; GPU inference\nStore all decision transitions", fontsize=14.5)
    pipeline_box(.375, .51, .255, .20, "PPO LEARNING STEP",
                 "Reward + value → GAE advantages\nClipped policy loss + value loss\nMinibatches: 256 transitions\nUp to 10 passes; KL early stopping", fontsize=13)
    pipeline_box(.685, .51, .25, .20, "REPEAT FOR 150 EPOCHS",
                 "11,250 training duels per pilot\n10 pilots trained independently\nMatched scenario schedules", fontsize=14)
    pipeline_arrow([(.327, .61), (.367, .61)])
    pipeline_arrow([(.637, .61), (.677, .61)])
    pipeline_arrow([(.81, .718), (.81, .741), (.1925, .741), (.1925, .718)])
    f.text(.5025, .745, "Next epoch: fresh cases and the updated policy", size=11, color=TEAL, ha="center", va="bottom")
    pipeline_box(.065, .23, .255, .19, "VALIDATION",
                 "50 fixed, separate scenarios\nEvaluate without weight updates\nBest own reward selects the epoch", colour="#ffcf88", fontsize=14)
    pipeline_box(.375, .23, .255, .19, "SELECTED CHECKPOINT",
                 "Save actor + critic weights\nOne model per reward profile\nBest epoch may precede the last", colour="#ffcf88", fontsize=14)
    pipeline_box(.685, .23, .25, .19, "SHARED DEVELOPMENT",
                 "100 fixed simple cases per pilot\nElimination, survival, clean win\nAlso report mutual destruction", colour="#ffcf88", fontsize=13)
    pipeline_arrow([(.81, .502), (.81, .46), (.1925, .46), (.1925, .428)], colour="#ffcf88", dashed=True)
    f.text(.5025, .465, "Validation at epoch 1, every fifth epoch and the final epoch", size=11, color="#ffcf88", ha="center", va="bottom")
    pipeline_arrow([(.327, .325), (.367, .325)], colour="#ffcf88", dashed=True)
    pipeline_arrow([(.637, .325), (.677, .325)], colour="#ffcf88", dashed=True)
    f.text(.065, .16, "Tactical stage: preserve selected parents → load actor + critic → fresh Adam → 150 further epochs against tactical.",
           size=13.5, color=TEAL)
    f.text(.065, .105, "Validation and development cases have no training updates. Final testing waits until experimental choices are frozen.",
           size=12, color=MUTED)
    finish(f, "A scenario is one seeded duel. One decision transition contains the observation, selected skill and parameters, reward, value estimate, action log probability and termination information. An epoch collects 75 duels, then optimizes using minibatches of 256 decision transitions; these are different units. Training rewards are scaled by 0.01 for GAE and critic targets, while reported returns stay in original units. GAE combines rewards and value predictions into advantages; PPO uses clipped policy updates, value fitting, entropy regularization and a KL stopping rule. Validation runs periodically without gradient updates. Checkpoints maximize each pilot's own validation reward, so differing reward profiles retain separate selection criteria. Shared development cases compare selected checkpoints using common combat outcomes and do not choose the checkpoint. Those inspected cases can still influence later experimental choices, so they are not untouched tests. The simple-opponent study uses seed 7 for training, 111257 for 50 validation cases and 211307 for 100 development cases. Tactical adaptation uses separate scenario schedules and loads each parent's actor and critic with a fresh Adam optimizer; the initial parent is also a validation candidate. The tactical stage adds 150 epochs and has its own matched before/after assessments. Future final testing uses unused cases shared across the final candidates.")

    f = page("Simple-opponent results", "Shared development benchmark • 10 pilots × 100 matched cases • seed 211307")
    rows = [[p["label"], *[f"{100*p[key]:.1f}%" for key in
             ("elimination_rate", "survival_rate", "clean_win_rate", "mutual_destruction_rate")]]
            for p in data["development"]["populations"]]
    outcome_table(f, [.065, .36, .87, .36], rows,
                  ["Population", "Elimination", "Survival", "Clean win", "Mutual loss"],
                  [.38, .155, .155, .155, .155])
    f.text(.065, .285, "Lower learning rate: the main observed gain was survival, with similar elimination frequency.", size=16)
    f.text(.065, .225, "Selected-skill probabilities: more eliminations, with a smaller further improvement in survival.", size=16)
    f.text(.065, .145, "Elimination and survival overlap. Clean win records both occurring in the same flight.", size=16, color=TEAL)
    finish(f, "Rates are equal-weight means over the same ten reward profiles. All four populations share training seed 7, validation seed 111257 and development seed 211307. Elimination means a confirmed missile elimination regardless of own survival; survival means the pilot remains alive regardless of opponent outcome. Neither is a substitute for their joint clean-win outcome. The separate parameter-head trial was not retained. The inspected benchmark is development evidence, not an untouched final test.")

    f = page("Experimental development", "Independent population training, controlled PPO experiments and tactical adaptation")
    entries = [
        ("04 → 05", "One PPO pilot became an arbitrary-size population with separate rewards and checkpoints."),
        ("Reward diversity", "Monotonic sweep → relative Halton profiles → seeded, independent uniform ranges."),
        ("Evaluation design", "Recorded training/validation and a fixed benchmark; final testing after development."),
        ("Optimisation", "Lower learning rate and selected-skill probabilities improved clean wins."),
        ("Bounded trials", "More parameter heads underperformed; the terminal-bonus trial stopped incomplete."),
        ("Tactical", "The stronger shared-head population was retained and adapted to scripted tactical."),
    ]
    for i, (label, body) in enumerate(entries):
        y = .68 - i * .097
        f.text(.065, y, label, color=TEAL, size=18, weight="bold")
        f.text(.275, y, body, size=15)
    finish(f, "Notebook 06 preserves the alternative evade/pursue/eliminate reward recipe. It is a separate experiment, not the reward used by the tactical population. This is not self-play: the opponent is scripted, not another member of the learning population.")

    f = page("Scenario collection and PPO updates", "75 scenarios per epoch for each pilot; shared scenario schedules support matched comparisons")
    boxes = [("OBSERVE", "20 frames × 51 features\n2 seconds of flight history"),
             ("CHOOSE", "One of 33 skills\n+ its numeric parameters"),
             ("SIMULATE", "Execute about 1 second\nRecord events and rewards"),
             ("UPDATE", "Estimate advantages\nPPO minibatches of 256")]
    for i, (title, body) in enumerate(boxes):
        x = .065 + .228 * i
        words(f, x, .66, title, body)
        if i < 3:
            f.text(x + .198, .575, "→", color=TEAL, size=26)
    words(f, .065, .38, "Scenarios and minibatches", "A scenario is one duel with seeded starting geometry, speeds and altitudes (up to 120 seconds).\n75 scenarios produce many decision transitions. A PPO minibatch contains 256 transitions.\n150 epochs × 75 scenarios = 11,250 training flights per pilot; 112,500 across the population.")
    finish(f, "CPU workers run the JSBSim aircraft and control simulation. The GPU batches policy inference and PPO updates. Pilots train independently and sequentially; simulator workers parallelise flights within a pilot. Each epoch uses fresh training seeds; fixed validation selects checkpoints.")

    f = page("Actor–critic network architecture", "Shared transformer: width 64 • 4 attention heads • 2 layers • shared parameter predictor")
    words(f, .065, .68, "ACTOR: policy distribution", "Predict a distribution over 33 skills\nand 16 numeric parameter slots.\nOnly parameters used by the chosen\nskill count in its PPO probability.")
    words(f, .535, .68, "CRITIC: value estimation", "Predict future discounted reward.\nCompare rewards + future value with\nthat prediction to estimate advantage.\nPositive advantage favours the action.")
    f.text(.065, .285, r"$\log\pi(a\mid s)=\log\pi(k\mid s)+\sum_{j\in\mathrm{active}(k)}\log p(z_j\mid s)$", size=24, color=TEAL)
    f.text(.065, .19, "The selected-skill correction changed the learning objective, without adding network capacity.", size=17)
    finish(f, "The actor changes action probabilities; the critic learns return targets. PPO clips probability-ratio changes and includes an entropy term. Smaller loss does not automatically mean better combat. Per-skill parameter heads were a separate architecture experiment and were not retained.")

    f = page("Reward profiles and common evaluation metrics", "Each pilot draws its coefficients once using a reproducible random seed (42)")
    ranges = [("Crash / being shot down", "Each independently uniform: −450 to −300"),
              ("Lock / launch with lock", "Each independently uniform: 0 to 2"),
              ("Avoid an incoming missile", "Uniform: 0 to 15"),
              ("Launch without lock", "Uniform: −12 to −4"),
              ("Opponent destroyed / clearance", "Fixed: +250 / +0.02")]
    for i, (a, b) in enumerate(ranges):
        y = .68 - i * .079
        f.text(.065, y, a, size=17, color=TEAL)
        f.text(.46, y, b, size=17)
    words(f, .065, .245, "Common outcome scoring", "Clean win +100  •  survive without elimination 0  •  mutual destruction −50  •  own loss −100")
    finish(f, "These are plausible ranges chosen as hypotheses, not calibrated optimal rewards. The population is not restricted to named aggression categories. Different weights encourage different tradeoffs but do not guarantee a particular style. Own reward chooses each pilot's checkpoint. Common benchmark reward and common outcome score are distinct summaries. Tactical adaptation reused the exact parent reward profiles; it did not add the terminal clean-win bonus.")

    f = page("Interpretation of simple-opponent results", "Changes on the fixed development benchmark • percentage points relative to the matching baseline")
    words(f, .065, .68, "LOWER LEARNING RATE", "Eliminations: +1.0 points.\nSurvival: +8.9 points.\nClean wins: +8.6 points.\nMutual destruction: −7.6 points.")
    words(f, .535, .68, "SELECTED-SKILL PROBABILITIES", "Eliminations: +11.4 points.\nSurvival: +2.3 points.\nClean wins: +13.2 points.\nMutual destruction: −1.8 points.")
    words(f, .065, .34, "Bounded trials", "Per-skill heads: eliminations fell 72.2% → 65.4%; survival was similar, 68.8% → 68.3%.\nBonus trial (6 matched pilots): eliminations 72.8% → 72.2%; survival 68.3% → 63.8%.\nBoth scored fewer clean wins; the shared-head selected-skill population was retained.")
    finish(f, "Lower LR changed 0.0003 to 0.00003. The selected-skill correction changed action probabilities without increasing network capacity. The first intervention mainly improved survival on these cases; the second mainly increased eliminations. Overall, original to selected-skill elimination rose 12.4 points, survival 11.2 points and clean wins 21.8 points. These are observed outcome changes, not proof of a unique behavioural mechanism. Per-skill-head clean wins fell 50.0 to 42.5 percent. Bonus clean wins fell 51.5 to 49.83 percent in only six completed pilots; this subset cannot be compared directly with full-population averages.")

    f = page("Recorded training and validation", "Population mean own return • identical reward profiles across runs • 150 epochs per pilot")
    for bounds, metric, title in (([.08, .29, .38, .40], "training_return", "Training: 75 fresh cases / epoch"),
                                  ([.56, .29, .38, .40], "validation_return", "Validation: 50 fixed cases, seed 111257")):
        ax = chart(f, bounds)
        for p, colour in zip(data["development"]["populations"], (BLUE, "#ffcf88", TEAL, RED)):
            points = [m for m in p["learning"] if m[metric] is not None]
            ax.plot([m["epoch"] for m in points], [m[metric] for m in points], color=colour,
                    label=p["label"], linewidth=1.6)
        ax.set(title=title, xlabel="Epoch", ylabel="Mean own return")
        ax.grid(alpha=.15)
    handles, labels = ax.get_legend_handles_labels()
    f.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .14), ncol=2,
             frameon=False, labelcolor=FG, fontsize=12)
    f.text(.065, .105, "Own return describes learning. Assess combat using elimination, survival, clean wins and mutual destruction.", size=14, color=MUTED)
    finish(f, "Curves read recorded epoch metrics only. Training uses stochastic actions and changing cases; validation uses fixed cases and evaluation actions, so their returns need not coincide. Population means are descriptive: the same ten pilot reward profiles are held fixed across all four runs. Raw own reward must not rank pilots with different coefficients. Per-pilot returns and PPO losses are included in learning_curves.csv; selected-epoch validation statistics are in development_summary.csv. Historical validation files do not record the joint clean-win outcome, so it cannot be reconstructed from separate survival and elimination rates.")

    f = page("Tactical adaptation: experimental design", "Same reward profiles, skills, physics and PPO settings; new scripted opponent behaviour")
    words(f, .065, .68, "BEFORE", "Freeze the 10 selected-skill parents.\nAssess all on 300 tactical cases.\nKeep the originals unchanged.")
    words(f, .535, .68, "FINE-TUNE", "Load each pilot's actor + critic.\nFresh Adam; LR 0.00003.\n150 epochs × 75 fresh tactical flights.")
    words(f, .065, .365, "SELECT", "50 separate validation cases.\nOwn reward selects the checkpoint;\ninitial weights are also eligible.")
    words(f, .535, .365, "COMPARE", "Repeat the same 300-case assessment.\nUse a separate 100-case confirmation\nfor both parent and adapted pilots.")
    finish(f, "Tactical evades incoming missiles and supports/cranks during its own missile flight; simple pursues and fires. Both use the same JSBSim aircraft. Seeds: training 1000007 onward, validation 1100000, assessment 1120000, confirmation 1220000. Report suites never select checkpoints. Retention against simple after adaptation has not been measured.")

    s = data["suites"]["confirmation"]
    f = page("Tactical adaptation outcomes", "Historical 100-case confirmation • seed 1220000 • the same cases before and after adaptation")
    rows = [[label, *[f"{100*s[state][key]:.1f}%" for key in
             ("elimination", "survival", OUTCOMES[0], OUTCOMES[2])]]
            for state, label in (("before", "Before"), ("after", "After"))]
    outcome_table(f, [.085, .55, .83, .17], rows,
                  ["Population", "Elimination", "Survival", "Clean win", "Mutual loss"], [.24, .19, .19, .19, .19])
    ax = chart(f, [.085, .25, .83, .23])
    labels = ["Clean win", "Survive, no elimination", "Mutual destruction", "Own loss, no elimination"]
    colours = [TEAL, BLUE, "#ffcf88", RED]
    left = np.zeros(2)
    for key, label, colour in zip(OUTCOMES, labels, colours):
        widths = np.array([100 * s[state][key] for state in ("before", "after")])
        ax.barh([1, 0], widths, left=left, color=colour, height=.45, label=label)
        for y, v, offset in zip([1, 0], widths, left):
            ax.text(offset + v / 2, y, f"{v:.1f}%", ha="center", va="center", color=BG, size=16, weight="bold")
        left += widths
    ax.set(yticks=[1, 0], yticklabels=["Before", "After"], xlim=(0, 100), xlabel="Share of flights (%)", ylim=(-.6, 1.6))
    ax.legend(loc="lower center", bbox_to_anchor=(.5, .93), ncol=4, frameon=False, labelcolor=FG, fontsize=10)
    f.text(.085, .145, "Clean-win gain: +5.9 pp  |  Paired scenario bootstrap 95% interval: +1.2 to +11.0 pp", size=17, color=TEAL)
    finish(f, "Eliminations rose 41.8 to 48.6 percent, survival 75.1 to 83.8 and clean wins 32.7 to 38.6. Losses without elimination fell 15.8 to 6.2. Mutual destruction rose 9.1 to 10.0 and remains a tradeoff. The table's elimination and survival rates overlap; the stacked bars below instead partition flights into exclusive outcomes. Intervals resample shared scenarios jointly across pilots and omit training-seed variability. The separate 300-case assessment records eliminations 35.73 to 42.60, survival 76.80 to 87.23 and clean wins 31.73 to 38.67 percent.")

    f = page("Per-pilot confirmation results", "Nine clean-win rates improved and one tied on the same 100 tactical confirmation cases")
    rows = [[p["pilot"].removeprefix("pilot_"), *[
        f"{100*p['before'][key]:.0f}% → {100*p['after'][key]:.0f}%"
        for key in ("elimination", "survival", OUTCOMES[0], OUTCOMES[2])]] for p in s["pilots"]]
    outcome_table(f, [.085, .22, .83, .50], rows,
                  ["Pilot", "Elimination", "Survival", "Clean win", "Mutual loss"], [.12, .22, .22, .22, .22], fontsize=13)
    f.text(.085, .13, "005 after adaptation: 52% eliminations, 86% survival, 44% clean wins. Small gaps remain uncertain.", size=15)
    finish(f, "005 has the highest observed confirmation clean-win rate, not a proven universal best policy. On the distinct 300-case assessment, pilot 010 leads. No pilot regressed in confirmation clean-win rate, but individual confidence intervals overlap and one tied. Selected epochs vary from 55 to 150; the final epoch is not always the saved best model.")

    f = page("Behavioural variation after adaptation", "Observed decision frequencies on confirmation flights • descriptive measures, not aggression rankings")
    skills = ["lag_pursuit", "turn_cold", "recommit", "extend", "pursue_target", "beam_target_left", "crank_target_left", "crank_target_right"]
    ax = chart(f, [.15, .22, .69, .49])
    values = [[100 * p["skill_decision_fractions"].get(k, 0) for k in skills] for p in data["pilots"]]
    im = ax.imshow(values, cmap="YlGnBu", vmin=0, vmax=50, aspect="auto")
    ax.set(yticks=np.arange(10), yticklabels=[f"{i:03d}" for i in range(1, 11)],
           xticks=np.arange(len(skills)), xticklabels=[k.replace("_", "\n") for k in skills], ylabel="Pilot")
    for i, row in enumerate(values):
        for j, val in enumerate(row):
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", size=11, color="white" if val > 28 else BG)
    cax = f.add_axes([.87, .22, .015, .49]); f.colorbar(im, cax=cax, label="Decisions (%)")
    f.text(.15, .12, "001 favours lag pursuit; 010 often turns cold; 005 mixes beaming, cranking and pursuit.", size=16)
    finish(f, "Only eight representative skills are shown; rows therefore need not sum to 100. These are aggregate decisions, not fractions of episodes. Different exposure times and situations affect the frequencies. They support behavioural variety but do not isolate reward weights as the sole cause of style.")

    f = page("Generalisation and evaluation limitations", "Matched performance improvements and the scope of the available evidence")
    words(f, .065, .68, "OBSERVED RESULTS", "100-case confirmation: +5.9 pp clean wins.\n300-case assessment: +6.9 pp clean wins.\nSurvival improved in both suites.\nAll 10 training runs completed; parents\nand frozen source files verified unchanged.")
    words(f, .535, .68, "EVALUATION LIMITATIONS", "One training seed per reward profile.\nNo post-adaptation test against simple.\nMutual destruction did not improve\non the confirmation suite.\nNo universal best pilot or style proven.")
    f.text(.065, .195, "No direct seed overlap was found across training, selection and tactical report suites.", size=17, color=TEAL)
    f.text(.065, .13, "Held-out gains do not establish the absence of overfitting to this simulator or opponent.", size=16)
    finish(f, "Own return and PPO loss are unsuitable for ranking pilots with different reward scales. The 300-case assessment improved from 31.73 to 38.67 percent, paired interval +3.87 to +9.90 points. Historical tactical and simple confirmation suites have already been inspected and cannot serve as untouched final tests in later experiments. Ongoing comparisons use recorded training, validation and fixed development cases. Final testing should occur after model and reward choices are frozen, on unused scenarios shared across all final candidates. Simulation results cannot establish real-world flight performance.")

    add_forecasting_slides(dest, page, finish, chart, outcome_table)

    f = page("Portable reports and supporting data", "PDF and self-contained HTML formats; no notebook runtime, Python, GPU or LFS download required")
    words(f, .065, .68, "REPORTS", "presentation.pdf — slide deck\npresentation.html — offline browser deck\nREADME.md — technical report")
    words(f, .535, .68, "DATA", "data/results.json — pilot results\ndata/forecasting/metrics.csv — forecast errors\ndata/forecasting/summary.json — provenance")
    words(f, .065, .345, "Principal findings", "Population training and tactical adaptation prepared ten flight-generating policies.\nTheir recorded encounters support forecasts from the tactical observer's perspective.\nInitial neural forecasts improve on constant-motion references on held-out scenarios.")
    finish(f, "The branch also contains executed read-only report notebooks for technical follow-up. Model weights remain in the existing local experiment directories; this presentation requires only ordinary small evidence files. Forecasting metrics, tolerance rates and a reproducibly selected validation example are packaged under data/forecasting. The deck is rebuildable from that bundle without loading models, the full corpus or simulator libraries. Standalone PNGs of the new introduction and forecasting slides are in figures for reuse in another slide pack. Historical shared MP4 demonstrations depict earlier populations and must not be labelled as tactical adaptation footage.")

    dest.mkdir(parents=True, exist_ok=True)
    sections, thumbs = [], []
    with PdfPages(dest / "presentation.pdf", metadata={"Title": TITLE, "Author": "Trajectory Classification project"}) as pdf:
        for i, (fig, note) in enumerate(slides):
            pdf.savefig(fig)
            buffer = io.BytesIO(); fig.savefig(buffer, format="png", dpi=110)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            sections.append(f'<section class="slide" aria-label="Slide {i+1}"><img alt="{html.escape(fig.texts[1].get_text())}" src="data:image/png;base64,{encoded}"><aside>{html.escape(note)}</aside></section>')
            thumbs.append(buffer.getvalue())
            if i == 0 or 16 <= i <= 20:
                figure_dir = dest / "figures"
                figure_dir.mkdir(exist_ok=True)
                fig.savefig(figure_dir / f"slide_{i + 1:02d}.png", dpi=160)
            plt.close(fig)
    markup = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>From multi-pilot training to motion forecasting</title><style>
body{margin:0;background:#101e30;color:#f2f5f9;font:18px system-ui}main{max-width:1600px;margin:auto}.slide{display:none}.slide.active{display:block}img{width:100%;height:calc(100vh - 80px);object-fit:contain;display:block}aside{display:none;padding:1em 3em;line-height:1.6;color:#d4e0ee}body.notes aside{display:block}nav{display:flex;gap:16px;justify-content:center;align-items:center;padding:14px;position:sticky;bottom:0;background:#15253b}button{background:#233d59;color:white;border:1px solid #8baeff;border-radius:5px;padding:9px 16px;cursor:pointer}button:focus-visible{outline:3px solid #62ddbc}#counter{min-width:70px;text-align:center}@media print{nav{display:none}.slide{display:block;break-after:page}img{height:auto}aside{display:none!important}@page{size:landscape;margin:0}}
</style><main>""" + "\n".join(sections) + """</main><nav aria-label="Slide controls"><button id="prev">← Previous</button><span id="counter" aria-live="polite"></span><button id="next">Next →</button><button id="notes">Supplementary notes (N)</button><button id="full">Fullscreen (F)</button></nav>
<script>
const slides=[...document.querySelectorAll('.slide')];let index=0;
function show(n){index=Math.max(0,Math.min(slides.length-1,n));slides.forEach((s,i)=>s.classList.toggle('active',i===index));document.getElementById('counter').textContent=`${index+1} / ${slides.length}`;location.hash=String(index+1)}
function notes(){document.body.classList.toggle('notes')}
function full(){if(document.fullscreenElement)document.exitFullscreen();else document.documentElement.requestFullscreen().catch(()=>{})}
document.getElementById('prev').onclick=()=>show(index-1);document.getElementById('next').onclick=()=>show(index+1);document.getElementById('notes').onclick=notes;document.getElementById('full').onclick=full;
document.addEventListener('keydown',e=>{if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();show(index+1)}if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();show(index-1)}if(e.key==='Home')show(0);if(e.key==='End')show(slides.length-1);if(e.key.toLowerCase()==='n')notes();if(e.key.toLowerCase()==='f')full()});show((parseInt(location.hash.slice(1))||1)-1);
</script></html>"""
    (dest / "presentation.html").write_text(markup, encoding="utf-8", newline="\n")
    # Preview sheets are local checks, not extra files needed by the other PC.
    from PIL import Image, ImageOps
    sheet = Image.new("RGB", (1200, math.ceil(len(thumbs) / 3) * 225), BG)
    for i, png in enumerate(thumbs):
        tile = ImageOps.contain(Image.open(io.BytesIO(png)).convert("RGB"), (400, 225))
        sheet.paste(tile, ((i % 3) * 400, (i // 3) * 225))
    preview = ROOT / "artifacts/checks/presentation"
    preview.mkdir(parents=True, exist_ok=True)
    sheet.save(preview / "contact_sheet.jpg")
    for i, png in enumerate(thumbs):
        (preview / f"slide_{i + 1:02d}.png").write_bytes(png)
    write_json(dest / "checksums.json", {
        p.relative_to(dest).as_posix(): sha(p) for p in sorted(dest.rglob("*"))
        if p.is_file() and p.name != "checksums.json"
    })
    print(f"Exported {len(slides)} slides to {dest}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true", help="Audit/copy existing local results before rendering.")
    parser.add_argument("--collect-forecast", action="store_true", help="Package saved forecasting validation evidence; no training or inference.")
    args = parser.parse_args()
    data = collect(DEST) if args.collect else read(DEST / "data/results.json")
    if args.collect_forecast:
        collect_forecasting(ROOT, DEST)
    render(DEST, data)


if __name__ == "__main__":
    main()
