"""Export existing pilot results as an offline presentation. Never runs a simulator.

Use --collect on the training PC to audit and package existing result files.
Without --collect, rebuild the PDF/HTML from the committed data on any PC with
Matplotlib and NumPy. Viewing the exported files needs neither Python nor Git LFS.
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

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "docs/presentations"
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
    result["survival"] = sum(bool(e["survived"]) for e in episodes) / n
    result["outcome_score"] = sum(SCORES[e["outcome"]] for e in episodes) / n
    return result


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
        data["history"].append({"label": label, "opponent": "simple", "pilots": 10,
                                "suite": c["confirmation_suite"], "role": "fresh confirmation",
                                "before": fresh["baseline_mean_clean_win_rate"],
                                "after": fresh["experiment_mean_clean_win_rate"],
                                "ci95_gain_pp": fresh["paired_scenario_bootstrap_ci95_percentage_points"]})
    bonus = registry["clean_win_bonus"]
    before, after = [], []
    for i in range(1, 7):
        pid = f"pilot_{i:03d}"
        before.extend(evidence(ROOT / bonus["baseline"] / pid / "test_episodes.json")["episodes"])
        after.extend(evidence(ROOT / bonus["training"] / pid / "test_episodes.json")["episodes"])
    assert [e["seed"] for e in before] == [e["seed"] for e in after]
    data["history"].append({"label": "Terminal clean-win bonus (stopped)", "opponent": "simple", "pilots": 6,
                            "suite": {"seed": before[0]["seed"], "scenarios": 100}, "role": "development only",
                            "before": summarize(before)[OUTCOMES[0]], "after": summarize(after)[OUTCOMES[0]]})
    data_dir = dest / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
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
                         "axes.edgecolor": MUTED, "font.size": 12, "savefig.facecolor": BG})
    slides = []

    def page(title, subtitle):
        fig = plt.figure(figsize=(16, 9), facecolor=BG)
        fig.text(.065, .929, "PILOT POPULATION  /  BRANCH UPDATE", color=TEAL, size=10, weight="bold")
        fig.text(.065, .849, title, size=30, weight="bold")
        fig.text(.065, .793, subtitle, size=14, color=MUTED)
        fig.text(.065, .041, "Existing results only  |  Tactical evaluation uses a scripted opponent  |  No new training", size=9, color=MUTED)
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

    s = data["suites"]["confirmation"]
    f = page("Tactical adaptation: population outcomes", "Ten independent pilots • 150 additional epochs each • all training runs completed")
    for x, value, label, detail in (
        (.065, "38.6%", "CLEAN WINS", "32.7% before  /  +5.9 points"),
        (.375, "83.8%", "SURVIVAL", "75.1% before  /  +8.7 points"),
        (.685, "10 / 10", "PILOTS COMPLETED", "150 epochs each  /  0 worker retries"),
    ):
        f.text(x, .59, value, size=53, weight="bold", color=TEAL)
        f.text(x, .515, label, size=15, weight="bold")
        f.text(x, .456, detail, size=14, color=MUTED)
    words(f, .065, .32, "Summary of findings", "On the same separate tactical cases, the adapted pilots won more often and survived more often.\nMutual destruction increased slightly: 9.1% → 10.0%.")
    finish(f, "The matched 100-case confirmation comprises 1,000 flights per population with 100 shared scenario IDs. Clean wins require a confirmed missile elimination and survival. The report was generated from existing results without additional training.")

    f = page("Experimental development", "Independent population training, controlled PPO experiments and tactical adaptation")
    entries = [
        ("04 → 05", "One PPO pilot became an arbitrary-size population with separate rewards and checkpoints."),
        ("Reward diversity", "Monotonic sweep → relative Halton profiles → seeded, independent uniform ranges."),
        ("Evaluation design", "Fresh training cases, common outcome metrics and separate confirmation suites."),
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

    f = page("Controlled experiment results", "Clean wins against simple • each row compares the same pilots on the same cases")
    ax = f.add_axes([.065, .28, .87, .44]); ax.axis("off")
    rows = []
    for e in data["history"]:
        rows.append([e["label"], f"{e['pilots']} / {e['suite']['scenarios']}",
                     f"{100*e['before']:.1f}% → {100*e['after']:.1f}%",
                     f"{100*(e['after']-e['before']):+.1f} pp",
                     str(e["suite"]["seed"])])
    table = ax.table(cellText=rows, colLabels=["Intervention", "Pilots / cases", "Before → after", "Change", "Suite seed"],
                     colWidths=[.38, .16, .20, .12, .14], cellLoc="left", colLoc="left", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False); table.set_fontsize(14)
    for (row, _col), cell in table.get_celld().items():
        cell.set_facecolor("#1c324b" if row == 0 else BG)
        cell.set_edgecolor("#35485e"); cell.get_text().set_color(TEAL if row == 0 else FG)
    f.text(.065, .19, "First three rows: fresh confirmation. Bonus: only 6 completed pilots, development cases; no confirmation.", size=14, color=MUTED)
    f.text(.065, .135, "Suite difficulty varies. Reported changes compare matched cases within each experiment.", size=16, color=TEAL)
    finish(f, "Lower LR was 0.0003 to 0.00003. Selected-skill probabilities improved clean wins without adding parameters. Separate per-skill heads increased parameters from 107,970 to 141,762 and underperformed under this budget. The +250 terminal bonus experiment stopped with six completed pilots and pilot 007 at 38 epochs. These findings do not establish that all larger models or terminal rewards are worse.")

    f = page("Tactical adaptation: experimental design", "Same reward profiles, skills, physics and PPO settings; new scripted opponent behaviour")
    words(f, .065, .68, "BEFORE", "Freeze the 10 selected-skill parents.\nAssess all on 300 tactical cases.\nKeep the originals unchanged.")
    words(f, .535, .68, "FINE-TUNE", "Load each pilot's actor + critic.\nFresh Adam; LR 0.00003.\n150 epochs × 75 fresh tactical flights.")
    words(f, .065, .365, "SELECT", "50 separate validation cases.\nOwn reward selects the checkpoint;\ninitial weights are also eligible.")
    words(f, .535, .365, "COMPARE", "Repeat the same 300-case assessment.\nUse a separate 100-case confirmation\nfor both parent and adapted pilots.")
    finish(f, "Tactical evades incoming missiles and supports/cranks during its own missile flight; simple pursues and fires. Both use the same JSBSim aircraft. Seeds: training 1000007 onward, validation 1100000, assessment 1120000, confirmation 1220000. Report suites never select checkpoints. Retention against simple after adaptation has not been measured.")

    f = page("Tactical confirmation outcomes", "Separate 100-case tactical confirmation • population means across all ten pilots")
    ax = chart(f, [.085, .29, .83, .40])
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
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=2, frameon=False, labelcolor=FG, fontsize=11)
    f.text(.085, .145, "Clean-win gain: +5.9 pp  |  Paired scenario bootstrap 95% interval: +1.2 to +11.0 pp", size=17, color=TEAL)
    finish(f, "Clean wins rose 32.7 to 38.6 percent. Survival rose 75.1 to 83.8, while losses without elimination fell 15.8 to 6.2. Mutual destruction rose 9.1 to 10.0 and remains a tradeoff. Intervals resample shared scenarios jointly across pilots, not all flights independently; they do not measure variation between training seeds.")

    f = page("Per-pilot confirmation results", "Nine clean-win rates improved and one tied on the same 100 tactical confirmation cases")
    ax = chart(f, [.085, .255, .83, .46])
    x = np.arange(10)
    for offset, state, colour, label in ((-.18, "before", BLUE, "Before"), (.18, "after", TEAL, "After")):
        values = [100 * p[state][OUTCOMES[0]] for p in s["pilots"]]
        bars = ax.bar(x + offset, values, width=.34, color=colour, label=label)
        ax.bar_label(bars, fmt="%.0f", padding=4, color=FG, fontsize=12)
    ax.set(xticks=x, xticklabels=[f"{i:03d}" for i in range(1, 11)], ylim=(0, 55), ylabel="Clean wins (%)", xlabel="Pilot")
    ax.legend(frameon=False, labelcolor=FG, loc="upper left", ncol=2)
    f.text(.085, .14, "Pilot 005: 44% clean wins, 86% survival. Pilot 007: 43%, 85%. Small gaps are uncertain.", size=17)
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
    finish(f, "Own return and PPO loss are unsuitable for ranking pilots with different reward scales. The 300-case assessment improved from 31.73 to 38.67 percent, paired interval +3.87 to +9.90 points. This assessment is distinct from confirmation. Simulation results cannot establish real-world flight performance. Potential further evaluation includes retention against simple, broader held-out geometries and opponents, and repeated training seeds; these were outside the reported experiments.")

    f = page("Portable reports and supporting data", "PDF and self-contained HTML formats; no notebook runtime, Python, GPU or LFS download required")
    words(f, .065, .68, "REPORTS", "presentation.pdf — slide deck\npresentation.html — offline browser deck\nREADME.md — technical report")
    words(f, .535, .68, "DATA", "data/results.json — verified summaries\ndata/paired_outcomes.csv — 4,000 paired rows\ndata/reward_profiles.csv — exact coefficients")
    words(f, .065, .345, "Principal findings", "Controlled comparisons improved PPO performance in a diverse pilot population.\nSubsequent tactical adaptation improved clean wins and survival on separate\nevaluation cases, with continued variation in observed behaviour.")
    finish(f, "The branch also contains executed read-only report notebooks for technical follow-up. Large model weights remain in the existing local experiment directories; the presentation evidence is committed as ordinary Git files. Historical shared MP4 demonstrations depict earlier populations and must not be labelled as tactical adaptation footage.")

    dest.mkdir(parents=True, exist_ok=True)
    sections, thumbs = [], []
    with PdfPages(dest / "presentation.pdf", metadata={"Title": data["title"], "Author": "Trajectory Classification project"}) as pdf:
        for i, (fig, note) in enumerate(slides):
            pdf.savefig(fig)
            buffer = io.BytesIO(); fig.savefig(buffer, format="png", dpi=110)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            sections.append(f'<section class="slide" aria-label="Slide {i+1}"><img alt="{html.escape(fig.texts[1].get_text())}" src="data:image/png;base64,{encoded}"><aside>{html.escape(note)}</aside></section>')
            thumbs.append(buffer.getvalue())
            plt.close(fig)
    markup = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Population PPO and tactical adaptation</title><style>
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
    sheet = Image.new("RGB", (1200, 4 * 225), BG)
    for i, png in enumerate(thumbs):
        tile = ImageOps.contain(Image.open(io.BytesIO(png)).convert("RGB"), (400, 225))
        sheet.paste(tile, ((i % 3) * 400, (i // 3) * 225))
    preview = ROOT / "artifacts/checks/presentation_20261011"
    preview.mkdir(parents=True, exist_ok=True)
    sheet.save(preview / "contact_sheet.jpg")
    (preview / "slide_03.png").write_bytes(thumbs[2])
    (preview / "slide_06.png").write_bytes(thumbs[5])
    (preview / "slide_08.png").write_bytes(thumbs[7])
    write_json(dest / "checksums.json", {
        p.relative_to(dest).as_posix(): sha(p) for p in sorted(dest.rglob("*"))
        if p.is_file() and p.name != "checksums.json"
    })
    print(f"Exported {len(slides)} slides to {dest}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true", help="Audit/copy existing local results before rendering.")
    args = parser.parse_args()
    data = collect(DEST) if args.collect else read(DEST / "data/results.json")
    render(DEST, data)


if __name__ == "__main__":
    main()
