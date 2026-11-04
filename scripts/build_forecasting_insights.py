"""Publish forecasting insights from saved evidence; no training or inference.

Use --collect on the experiment machine to package and audit the saved results.
Default builds read only the portable, committed data beside the exported slides.
"""

import argparse
import base64
import csv
import hashlib
import html
import io
import json
import math
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "docs/presentations/forecasting_insights"
BG, PANEL, FG, MUTED = "#101e30", "#1a3048", "#f3f6fb", "#b2c3d8"
TEAL, BLUE, GOLD, GREY = "#62ddbc", "#8baeff", "#ffc17a", "#8f9ba9"
HORIZONS = (1, 2, 5)
TITLE = "Forecasting trained pilots from a tactical observer's view"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def collect(dest, snapshot_date):
    data = dest / "data"
    data.mkdir(parents=True, exist_ok=True)
    sources = {}

    def record(relative):
        path = ROOT / relative
        sources[relative] = digest(path)
        return path

    def load(relative):
        return read_json(record(relative))

    def copy(relative, name):
        # Git normalizes text in this presentation directory to LF. Hash the
        # original source separately, but publish canonical portable text.
        text = record(relative).read_text(encoding="utf-8")
        (data / name).write_text(text, encoding="utf-8", newline="\n")

    skill = "artifacts/skill_forecasting/"
    improved = skill + "improvement_v3/"
    corpus = load("shared/corpora/trained-pilot-episodes-v1/config.json")
    manifest = load(skill + "data_v1/data_manifest.json")
    assert corpus["observer_id"] == "B01" and corpus["target_id"] == "A01"
    assert len(corpus["pilots"]) == 10 and corpus["episodes_per_pilot"] == 300
    assert len(manifest["classes"]) == 33 and manifest["counts"]["test"]["samples"] == 0
    frozen = load(improved + "frozen_selection.json")
    protocol = load(improved + "protocol.json")
    extension = load(improved + "ensemble_protocol.json")
    for item in (protocol, extension):
        for name, expected in item["source_hashes"].items():
            assert digest(record(name)) == expected, f"Changed experiment source: {name}"
    verification = load(improved + "verification.json")
    assert verification["ensemble_probabilities_exact"]
    assert verification["matched_validation_origins"] and not verification["test_evaluated"]
    assert len(verification["members"]) == 12
    preserved = load(improved + "preserved_initial_hashes.json")
    for name, expected in preserved.items():
        assert digest(record(name)) == expected, f"Changed initial artifact: {name}"
    normalization = load(improved + "normalization_audit.json")
    assert len(normalization["completed_runs_checked"]) == 12
    assert all(row["normalization_groups"] == 210 and not row["includes_outer_validation"]
               and not row["includes_test"] for row in normalization["completed_runs_checked"])
    windows = load(improved + "full_window_audit.json")
    assert windows["test_samples"] == 0
    assert all(row["future_frame_reads"] == row["cross_episode_reads"] == 0
               for row in windows["history_configs"])
    timing = load(improved + "notebook_check/verification.json")
    for model, filename in (("transformer", "skill_transformer.csv"),
                            ("gru", "skill_gru.csv"),
                            ("training_frequency", "skill_frequency.csv")):
        copy(skill + f"initial_v1/{model}/validation_metrics.csv", filename)
    copy(improved + "validation/ensemble/validation_metrics.csv", "skill_ensemble.csv")
    copy(improved + "paired_scenario_bootstrap.csv", "scenario_intervals.csv")
    copy(improved + "ensemble_confirmation.csv", "internal_confirmation.csv")
    motion = "artifacts/motion_forecasting/initial_v1/"
    copy(motion + "validation_comparison.csv", "motion_metrics.csv")
    for model in ("transformer", "gru"):
        assert not load(motion + f"{model}/summary.json")["test_evaluated"]
    components = [{**component, "settings": protocol["candidates"][component["candidate"]]}
                  for component in frozen["components"]]
    evidence = {
        "snapshot_date": snapshot_date,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                  text=True).strip(),
        "observer": "B01: scripted tactical F-16",
        "target": "A01: one of ten frozen trained F-16 policies",
        "episodes": 3000, "scenario_groups": 300, "pilots": 10,
        "observation_hz": 10, "decision_hz": 1,
        "classes": manifest["classes"], "skill_counts": manifest["counts"],
        "split_seed": manifest["split_seed"],
        "components": components,
        "internal_selection": frozen["paired_internal_results"],
        "verification": verification, "preserved_files_checked": len(preserved),
        "history_audit": windows,
        "normalization_members_checked": len(normalization["completed_runs_checked"]),
        "runtime_check": timing,
        "test_evaluated": False,
        "source_sha256": sources,
        "metric_definitions": {
            "motion": "3D Euclidean error averaged within episodes, then equally across episodes",
            "skill": "Equal weight per valid forecast origin at each horizon",
            "macro_f1": "Equal weight for all 33 catalogue labels; undefined F1 is zero",
            "changed": "Future label differs from true label at origin; not any intermediate switch",
        },
    }
    write_json(data / "evidence.json", evidence)
    write_json(data / "checksums.json", {
        path.name: digest(path) for path in sorted(data.iterdir())
        if path.is_file() and path.name != "checksums.json"
    })
    print(f"Collected evidence; {len(preserved)} original artifacts unchanged.")


def load_evidence(dest):
    data = dest / "data"
    for name, expected in read_json(data / "checksums.json").items():
        assert digest(data / name) == expected, f"Changed portable evidence: {name}"
    evidence = read_json(data / "evidence.json")
    skill = {name: read_csv(data / f"skill_{name}.csv")
             for name in ("frequency", "gru", "transformer", "ensemble")}
    motion = read_csv(data / "motion_metrics.csv")
    intervals = read_csv(data / "scenario_intervals.csv")
    for h in HORIZONS:
        counts = [int(row["count"]) for rows in skill.values() for row in rows
                  if row["pilot_id"] == "population" and row["cohort"] == "all"
                  and float(row["horizon_s"]) == h]
        assert len(counts) == 4 and len(set(counts)) == 1
        new = metric(skill, "ensemble", "top1_accuracy", h)
        old = metric(skill, "transformer", "top1_accuracy", h)
        interval = next(row for row in intervals if int(row["horizon_s"]) == h)
        np.testing.assert_allclose(new - old, float(interval["accuracy_gain"]), atol=1e-12)
        assert int(interval["scenario_groups"]) == 45
    return evidence, skill, motion, intervals


def metric(skill, model, field, horizon, cohort="all", pilot="population"):
    rows = [row for row in skill[model] if float(row["horizon_s"]) == horizon
            and row["cohort"] == cohort and row["pilot_id"] == pilot]
    assert len(rows) == 1
    return float(rows[0][field])


def render(dest):
    evidence, skill, motion, intervals = load_evidence(dest)
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": FG,
                         "axes.labelcolor": FG, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.edgecolor": MUTED, "font.size": 14, "pdf.fonttype": 42,
                         "savefig.facecolor": BG, "figure.max_open_warning": 0})
    slides = []

    def page(title, subtitle, source):
        fig = plt.figure(figsize=(16, 9), facecolor=BG)
        fig.text(.055, .94, "TACTICAL OBSERVER  /  TRAINED PILOTS  /  FORECASTING INSIGHTS",
                 fontsize=11, color=TEAL, weight="bold")
        fig.text(.055, .858, title, fontsize=29, weight="bold")
        fig.text(.055, .80, subtitle, fontsize=15, color=MUTED)
        fig.text(.055, .038, source + "  |  Development validation; test reserved",
                 fontsize=8.5, color=MUTED)
        fig.text(.945, .038, f"{len(slides) + 1:02}", ha="right", color=TEAL, fontsize=11)
        return fig

    def words(fig, x, y, title, body, color=TEAL, size=18):
        fig.text(x, y, title, fontsize=19, color=color, weight="bold", va="top")
        fig.text(x, y - .063, body, fontsize=size, va="top", linespacing=1.5)

    def box(fig, x, y, w, h, title, body, color=TEAL, size=17):
        fig.add_artist(FancyBboxPatch((x, y), w, h, transform=fig.transFigure,
                       boxstyle="round,pad=0.009,rounding_size=0.012", facecolor=PANEL,
                       edgecolor=color, linewidth=1.2))
        words(fig, x + .018, y + h - .025, title, body, color, size)

    def arrow(fig, start, end):
        fig.add_artist(FancyArrowPatch(start, end, transform=fig.transFigure,
                       arrowstyle="-|>", mutation_scale=21, linewidth=2, color=BLUE))

    def chart(fig, bounds):
        ax = fig.add_axes(bounds, facecolor=BG)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", alpha=.16, color=MUTED)
        ax.set_axisbelow(True)
        return ax

    def finish(fig, note):
        slides.append((fig, fig.texts[1].get_text(), note))

    def series(model, field, cohort="all"):
        return np.array([metric(skill, model, field, h, cohort) for h in HORIZONS])

    def motion_series(model, quantity):
        return [float(next(row["episode_mean_error"] for row in motion
                           if row["model"] == model and row["pilot_id"] == "population"
                           and row["quantity"] == quantity and float(row["horizon_s"]) == h))
                for h in HORIZONS]

    f = page("Forecasting trained pilots from a tactical viewpoint",
             "What the experiments establish so far  |  " + evidence["snapshot_date"],
             "Saved motion initial_v1 + skill initial_v1 / improvement_v3")
    five = motion_series("transformer", "position")[-1]
    gain = (series("ensemble", "top1_accuracy") - series("transformer", "top1_accuracy"))[-1]
    for x, number, title, body, colour in [
        (.065, f"{five:.1f} m", "MOTION AT +5 s", "Mean position error\nOriginal transformer", BLUE),
        (.37, f"{series('ensemble', 'top1_accuracy')[-1]:.1%}", "SELECTION AT +5 s",
         "Exact-label accuracy\nImproved ensemble", TEAL),
        (.675, f"+{gain * 100:.2f} pts", "ACCURACY GAIN", "Versus original transformer\nSame validation origins", GOLD),
    ]:
        f.text(x, .64, number, fontsize=43, color=colour, weight="bold")
        words(f, x, .53, title, body, colour)
    f.text(.065, .235, "One observation stream. Two distinct prediction tasks.", fontsize=24, weight="bold")
    f.text(.065, .165, "These are predictions of future motion and selected labels, not ratings of pilot competence.",
           fontsize=17, color=MUTED)
    finish(f, "The dataset records the tactical aircraft B01 observing frozen trained policy A01. "
           "Forecasting is a separate supervised task: no RL retraining was needed for these experiments. "
           "Motion errors and classification accuracy have different units and weighting, so they cannot be ranked against each other.")

    f = page("B01 supplies the observations. A01 is the forecast target.",
             "The trained policies generate behaviour; a separate model learns to forecast it.",
             "Corpus config + data contracts")
    box(f, .065, .49, .32, .23, "B01  /  SCRIPTED TACTICAL", "Own state + relative target state\nAvailable observations through time t", BLUE, 16)
    box(f, .615, .49, .32, .23, "A01  /  TRAINED PILOT", "One of ten frozen policies\nFuture state and selection are targets", TEAL, 16)
    arrow(f, (.40, .60), (.59, .60))
    f.text(.496, .64, "observes", ha="center", color=MUTED)
    box(f, .065, .17, .40, .22, "FORECAST INPUTS", "Observation history and causal geometry\nNo pilot ID or private skill decision", BLUE, 17)
    box(f, .535, .17, .40, .22, "SUPERVISED OUTPUTS", "Position / velocity at +1, +2, +5 seconds\nCatalogue selection at those same times", TEAL, 17)
    arrow(f, (.48, .28), (.52, .28))
    finish(f, "Both aircraft are simulated F-16s. The corpus uses idealised exact simulator tracks, without radar "
           "noise, occlusion or track loss. Future physical states and decisions supply training targets, never input "
           "features. True current labels are used only for evaluation subsets, or as targets in an unselected auxiliary candidate.")

    f = page("Keep complete scenarios together when splitting the data",
             "3,000 episodes = 300 starting scenarios × 10 trained pilots.", "Corpus + skill data manifest")
    for x, count, title, body, colour in [
        (.065, "210", "TRAINING GROUPS", "2,100 episodes\nModel fitting and internal selection", TEAL),
        (.37, "45", "VALIDATION GROUPS", "450 episodes\nReported development comparison", BLUE),
        (.675, "45", "RESERVED TEST GROUPS", "450 episodes\nNo reported test evaluation", GOLD),
    ]:
        f.text(x, .65, count, fontsize=48, weight="bold", color=colour)
        words(f, x, .55, title, body, colour, 16)
    words(f, .065, .32, "ONE GROUP CONTAINS ALL TEN PILOTS",
          "Matching encounters stay in one partition. Overlapping windows are never randomly split.", size=17)
    f.text(.065, .16, "10 Hz observations  ·  1 Hz selection labels  ·  Exact +1 / +2 / +5 s targets",
           fontsize=19, color=TEAL)
    finish(f, "The seed-1729 outer split is stratified by requested initial range. Initial models used validation "
           "for epoch selection, so this remains a development set. Skill validation has 30,802 / 30,466 / 29,458 "
           "valid origins across the horizons. Motion has 31,029 / 30,693 / 29,685; label availability differs. "
           "Missing future targets are masked, not forward-filled. The final skill experiment prepares no test rows.")

    f = page("Learned motion forecasts improve on kinematic baselines",
             "The GRU leads short-horizon position; the transformer leads at +5 seconds.",
             "motion_metrics.csv  |  Equal episode weight")
    for x, quantity, label in ((.08, "position", "Mean position error (m, log scale)"),
                               (.56, "velocity", "Mean velocity error (m/s, log scale)")):
        ax = chart(f, [x, .28, .37, .43])
        for model, name, color in (("constant_velocity", "Constant velocity", GREY),
                                    ("constant_acceleration", "Constant acceleration", GOLD),
                                    ("gru", "GRU", BLUE), ("transformer", "Transformer", TEAL)):
            ax.plot(HORIZONS, motion_series(model, quantity), "o-", color=color, label=name, lw=2.5)
        ax.set(yscale="log", xticks=HORIZONS, xlabel="Forecast horizon (s)", ylabel=label)
    handles, labels = ax.get_legend_handles_labels()
    f.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .075), ncol=4,
             facecolor=BG, edgecolor=MUTED, labelcolor=FG, fontsize=13)
    finish(f, "Errors are 3D Euclidean error averaged within an episode, then equally across 450 validation episodes. "
           f"At +5 s, transformer position error is {five:.3f} m versus "
           f"{motion_series('constant_acceleration', 'position')[-1]:.3f} m for constant acceleration. "
           "GRU / transformer position errors at +1 s are 0.440 / 0.484 m. This was one seed per architecture; "
           "it does not establish a universal architecture ranking. The log scale makes all four baselines visible.")

    f = page("A skill forecast predicts a selected catalogue entry",
             "It does not score competence, infer intent with certainty, or guarantee a distinct manoeuvre.",
             "Skill target contract + label support")
    box(f, .065, .48, .39, .23, "EXAMPLE ORIGIN: t = 6 s", "Observe the available history up to 6 s.\nPredict selections at 7, 8 and 11 s.", BLUE, 17)
    box(f, .545, .48, .39, .23, "33 POSSIBLE LABELS", "26 have truth support in validation.\nSome names share executed guidance.", TEAL, 17)
    words(f, .065, .35, "MOTION AND LABELS ANSWER DIFFERENT QUESTIONS",
          "Motion: where will the trained aircraft be?\nCatalogue: which entry will its policy select at that future instant?", size=20)
    f.text(.065, .14, "Accurate motion does not imply that every private catalogue choice is identifiable.",
           fontsize=18, color=GOLD)
    finish(f, "Selections are joined at exact recorded times. There is no nearest-time join or carry-forward "
           "of a missing selection. Catalogue aliases such as lag_pursuit, turn_cold, extend, recommit and "
           "pursue_target can share guidance implementations. The label task measures selected policy entries; "
           "it is not a validated taxonomy of distinct physical manoeuvres or tactical quality.")

    f = page("Exact-label accuracy rises at all three forecast horizons",
             "The new ensemble improves on both initial neural models and the frequency reference.",
             "skill_*.csv  |  Same origins and horizon masks")
    ax = chart(f, [.085, .25, .84, .46])
    x = np.arange(3)
    for i, (model, name, color) in enumerate((("frequency", "Training frequency", GREY),
                                              ("gru", "Original GRU", GOLD),
                                              ("transformer", "Original transformer", BLUE),
                                              ("ensemble", "Improved ensemble", TEAL))):
        values = series(model, "top1_accuracy") * 100
        bars = ax.bar(x + (i - 1.5) * .19, values, .18, color=color, label=name)
        ax.bar_label(bars, labels=[f"{v:.1f}" for v in values], padding=4, fontsize=12, color=FG)
    ax.set(xticks=x, xticklabels=["+1 second", "+2 seconds", "+5 seconds"],
           ylim=(0, 78), ylabel="Top-1 accuracy (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.15), ncol=4,
              facecolor=BG, edgecolor=MUTED, labelcolor=FG, fontsize=12)
    finish(f, "Each valid origin has equal weight. Ensemble top-1 scores are 65.62%, 65.17%, 63.48%; "
           "original transformer scores are 58.97%, 58.38%, 55.90%. The ensemble uses more history and computation. "
           "No labels were merged or dropped to increase accuracy. The train-frequency baseline predicts class "
           "frequencies learned only from training targets.")

    f = page("The harder changed-selection subset improves too",
             "Cases where the future selection differs from the true selection at the origin.",
             "skill_transformer.csv + skill_ensemble.csv  |  Changed cohort")
    ax = chart(f, [.08, .25, .53, .46])
    for model, name, color in (("transformer", "Original transformer", BLUE),
                                ("ensemble", "Improved ensemble", TEAL)):
        values = series(model, "top1_accuracy", "changed") * 100
        ax.plot(HORIZONS, values, "o-", color=color, label=name, lw=3, markersize=9)
        for h, value in zip(HORIZONS, values, strict=True):
            ax.annotate(f"{value:.1f}%", (h, value), xytext=(0, 12), textcoords="offset points",
                        ha="center", color=color, fontsize=15)
    ax.set(ylim=(0, 70), xticks=HORIZONS, xlabel="Forecast horizon (s)", ylabel="Top-1 accuracy (%)")
    ax.legend(loc="lower right", facecolor=BG, edgecolor=MUTED, labelcolor=FG, fontsize=12)
    words(f, .68, .67, "AT +5 SECONDS", "40.2% → 55.3%\n8,659 changed origins", size=23)
    words(f, .68, .40, "INTERPRETATION", "Useful gains beyond cases\nwith the same endpoint label.\nThis is not a switch detector.", GOLD, 16)
    finish(f, "Changed-subset support is 2,264 / 4,254 / 8,659 at +1 / +2 / +5 s. These are different subsets; "
           "the rising line is not evidence that longer horizons are intrinsically easier. A selection can change "
           "and return between endpoints. The true current label is used to define this diagnostic only and "
           "is not an inference input. No future switch time is predicted by this experiment.")

    f = page("All ten pilots improve, with uneven gain sizes",
             "Each cell is the ensemble's top-1 gain over the original transformer, in percentage points.",
             "skill_transformer.csv + skill_ensemble.csv  |  Per-pilot all cohort")
    pilots = [f"pilot_{i:03}" for i in range(1, 11)]
    gains = np.array([[100 * (metric(skill, "ensemble", "top1_accuracy", h, pilot=p)
                             - metric(skill, "transformer", "top1_accuracy", h, pilot=p))
                       for h in HORIZONS] for p in pilots])
    assert (gains > 0).all()
    ax = f.add_axes([.10, .17, .43, .55])
    ax.imshow(gains, cmap="YlGnBu", vmin=0, vmax=14, aspect="auto")
    ax.set(xticks=range(3), xticklabels=["+1 s", "+2 s", "+5 s"],
           yticks=range(10), yticklabels=[f"Pilot {i:03}" for i in range(1, 11)])
    ax.tick_params(length=0)
    for i in range(10):
        for j in range(3):
            ax.text(j, i, f"+{gains[i, j]:.1f}", ha="center", va="center",
                    color="white" if gains[i, j] > 7 else BG, fontsize=15, weight="bold")
    words(f, .61, .65, "BROAD DIRECTION OF IMPROVEMENT", "30 of 30 pilot / horizon cells improve.\nThe gains are not uniform.", size=18)
    words(f, .61, .40, "SCOPE OF THE CLAIM", "New scenarios for these ten pilots.\nNo unseen-pilot evaluation.\nPilot identity is not a model input.", GOLD, 17)
    finish(f, "These point estimates show the improvement is not confined to one pilot. They do not establish "
           "statistical significance for each cell individually, nor rank the pilots' tactical competence. "
           "Training and validation include the same ten trained policy identities under disjoint scenario groups.")

    f = page("The gain also appears in class balance and log-loss",
             "Overall top-3 accuracy improves; improvements are not universal across every subset.",
             "skill_transformer.csv + skill_ensemble.csv  |  Population all cohort")
    for bounds, field, title in (([.08, .29, .36, .40], "macro_f1_catalogue", "Macro-F1 across all 33 entries ↑"),
                                  ([.57, .29, .36, .40], "nll", "Negative log-likelihood ↓")):
        ax = chart(f, bounds)
        for model, name, color in (("transformer", "Original transformer", BLUE),
                                    ("ensemble", "Improved ensemble", TEAL)):
            ax.plot(HORIZONS, series(model, field), "o-", color=color, lw=3, markersize=8, label=name)
        ax.set(xticks=HORIZONS, xlabel="Forecast horizon (s)", title=title)
        if field == "macro_f1_catalogue":
            ax.set_ylim(0, .5)
        else:
            ax.set_ylim(1.0, 1.35)
        ax.legend(facecolor=BG, edgecolor=MUTED, labelcolor=FG, fontsize=11, loc="lower left")
    top3 = series("ensemble", "top3_accuracy") * 100
    f.text(.08, .13, f"Improved top-3 accuracy:  {top3[0]:.1f}% at +1 s  ·  {top3[1]:.1f}% at +2 s  ·  {top3[2]:.1f}% at +5 s",
           color=TEAL, fontsize=18)
    finish(f, "Macro-F1 assigns equal weight to all 33 labels and zero to undefined F1; 26 labels have validation "
           "truth support. NLL assesses probabilities assigned to the correct label; lower is better, but it is not "
           "a complete calibration assessment. One counterexample to universal improvement is the +5 s unchanged "
           "subset: top-3 accuracy falls from 91.12% to 90.81%, and NLL rises from 1.0575 to 1.0813.")

    f = page("The selected improvement combines four model variants",
             "Geometry, longer history, regularization and uniform probability averaging were explored together.",
             "Frozen selection + internal confirmation + local runtime check")
    box(f, .065, .49, .40, .22, "91 OBSERVABLE / DERIVED FEATURES", "Relative geometry, speed and turn cues\nCausal acceleration; no future trajectory", BLUE, 17)
    box(f, .535, .49, .40, .22, "20 s AND 60 s HISTORIES", "Three 20-second temporal MLP variants\nOne 60-second temporal MLP variant", TEAL, 17)
    box(f, .065, .18, .40, .22, "3 FIXED SEEDS PER VARIANT", "12 models; equal probability weights\nStop-group temperature calibration", TEAL, 17)
    runtime = evidence["runtime_check"]
    box(f, .535, .18, .40, .22, "THE COST OF THE IMPROVEMENT",
        f"29 million parameters; ~116 MB\nLocal CPU check: ~{runtime['median_inference_ms']:.0f} ms per forecast", GOLD, 17)
    finish(f, "Initial models use five seconds of observations; the new ensemble uses up to sixty. Full raw "
           "dependencies include an extra 0.5 s for backward acceleration. Each token uses only past or current "
           "observations, with episode-start masking. This is not an architecture-only comparison or a factorial "
           "ablation. The six-model wide/long alternative had almost identical internal accuracy (60.96% versus "
           "60.98%); the selection rule chose twelve. The CPU timing is 30 warmed calls with four threads on one "
           "machine, including feature construction; it is not a deployment latency guarantee.")

    f = page("All new model selection stayed inside the training groups",
             "Two internal partitions informed the final configuration before the validation comparison.",
             "Registered protocols + source snapshots + saved audits")
    for x, count, title, body, color in [
        (.065, "147", "FIT GROUPS", "Gradients + normalization", TEAL),
        (.37, "21", "STOP GROUPS", "Epoch choice + calibration", BLUE),
        (.675, "42", "SELECT GROUPS", "Candidate comparison", GOLD),
    ]:
        f.text(x, .65, count, fontsize=43, weight="bold", color=color)
        words(f, x, .55, title, body, color, 16)
    box(f, .065, .19, .87, .19, "THEN FREEZE → REFIT → COMPARE",
        "Fix components, epochs, temperatures and seeds; refit on all 210 training groups.\n"
        "Evaluate the frozen ensemble on the original 45 validation groups.", TEAL, 17)
    finish(f, "The search was adaptive within training: an initial nine-candidate screen, an accuracy-focused "
           "follow-up, and a recorded fixed-mixture extension. Two internal partitions reduce reliance on one "
           "split but are not independent replications. Eligibility guards constrain deterioration in NLL and "
           "macro-F1. Every final normalization uses only the 210 training groups. The all-origin audit found "
           "zero future-frame or cross-episode reads. Reloads reproduce all saved probabilities exactly, and "
           "76 preserved artifacts pass their hashes. These checks reduce specific leakage risks; they do not "
           "prove zero overfitting. The outer validation set was already used in the initial experiment.")

    f = page("Accuracy gains remain positive within the 95% intervals",
             "95% bootstrap intervals preserve all pilots and overlapping windows within each scenario.",
             "scenario_intervals.csv  |  45 groups; 5,000 paired bootstrap draws")
    ax = chart(f, [.08, .23, .49, .48])
    values = np.array([float(row["accuracy_gain"]) for row in intervals]) * 100
    lower = np.array([float(row["lower_95"]) for row in intervals]) * 100
    upper = np.array([float(row["upper_95"]) for row in intervals]) * 100
    ax.errorbar(HORIZONS, values, yerr=[values - lower, upper - values], fmt="o",
                color=TEAL, lw=3, markersize=10, capsize=7)
    ax.axhline(0, color=GREY, lw=1)
    ax.set(xticks=HORIZONS, ylim=(-.4, 10), xlabel="Forecast horizon (s)",
           ylabel="Top-1 gain (percentage points)")
    body = "\n".join(f"+{h} s:  +{v:.2f} points  [{lo:.2f}, {hi:.2f}]"
                     for h, v, lo, hi in zip(HORIZONS, values, lower, upper, strict=True))
    words(f, .63, .65, "GAINS AND 95% INTERVALS", body, TEAL, 18)
    words(f, .63, .38, "DEVELOPMENT EVIDENCE", "No correction for all model selection.\nNot an untouched-test claim.\nNot a guarantee for new pilots.", GOLD, 17)
    finish(f, "Bootstrap draws resample complete scenario groups with replacement. The paired statistic compares "
           "new and old predictions on the same valid origins, retaining the population window weighting. "
           "Individual windows are correlated and are not treated as independent replicates. These intervals "
           "describe this development comparison and do not account for all historical adaptive model selection.")

    f = page("Useful forecasts; clear limits on what is established",
             "Motion and exact-label forecasting both show learnable structure in these recorded encounters.",
             "Synthesis of the saved experiments  |  Snapshot " + evidence["snapshot_date"])
    words(f, .065, .69, "SUPPORTED BY THE RESULTS",
          "Neural motion forecasts beat kinematic baselines.\nThe skill ensemble gains ~7 accuracy points.\nGains reach every known pilot and changed cases.\nOriginal results remain preserved.", TEAL, 18)
    words(f, .565, .69, "STILL TO ESTABLISH",
          "Performance on the reserved test scenarios.\nGeneralisation to unseen trained policies.\nRobustness to noisy or missing observations.\nWhether fewer models retain the gain.", GOLD, 18)
    f.text(.065, .24, "Next: predeclare the final test comparison, then evaluate once.",
           fontsize=23, weight="bold", color=BLUE)
    f.text(.065, .15, "The companion notes and portable evidence keep each claim tied to its metric and scope.",
           fontsize=17, color=MUTED)
    finish(f, "These are offline simulation results for a known population under one scripted tactical observer. "
           "They do not validate real sensor deployment, tactical decision support, pilot competence ranking or "
           "unseen-policy generalisation. No new simulation, fitting, inference or test evaluation was performed "
           "to produce these slides. Suggested future work is not a completed experiment.")

    export(dest, slides, evidence)


def export(dest, slides, evidence, *, stem="forecasting_insights", deck_title=TITLE,
           short_title="Forecasting insights", check_name="forecasting_insights"):
    figures = dest / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    sections, notes, thumbnails, overflow = [], [], [], []
    metadata = {"Title": deck_title, "Author": "Trajectory Classification project",
                "CreationDate": None, "ModDate": None}
    with PdfPages(dest / f"{stem}.pdf", metadata=metadata) as pdf:
        for index, (fig, title, note) in enumerate(slides, 1):
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            for text in fig.texts:
                bounds = text.get_window_extent(renderer)
                if bounds.x0 < 0 or bounds.y0 < 0 or bounds.x1 > fig.bbox.width or bounds.y1 > fig.bbox.height:
                    overflow.append({"slide": index, "text": text.get_text()})
            pdf.savefig(fig)
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=120)
            png = buffer.getvalue()
            (figures / f"slide_{index:02}.png").write_bytes(png)
            thumbnails.append(png)
            encoded = base64.b64encode(png).decode("ascii")
            sections.append(f'<section class="slide" aria-label="Slide {index}: {html.escape(title)}">'
                            f'<img alt="{html.escape(title)}" src="data:image/png;base64,{encoded}">'
                            f'<aside>{html.escape(note)}</aside></section>')
            notes.append(f"## {index:02}. {title}\n\n{note}\n")
            plt.close(fig)
    assert not overflow, overflow
    markup = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Forecasting insights</title>
<style>body{margin:0;background:#101e30;color:#f3f6fb;font:17px system-ui}.slide{display:none}.slide.active{display:block}
img{width:100%;height:calc(100vh - 72px);object-fit:contain;display:block}aside{display:none;max-width:1100px;margin:auto;padding:24px;line-height:1.6}
body.notes aside{display:block}nav{display:flex;justify-content:center;align-items:center;gap:14px;padding:12px;background:#1a3048;position:sticky;bottom:0}
button{padding:10px 16px;border:1px solid #8baeff;border-radius:5px;background:#1a3048;color:white;cursor:pointer}
button:focus-visible{outline:3px solid #62ddbc}button:disabled{opacity:.4;cursor:default}
@media print{nav,aside{display:none!important}.slide{display:block;break-after:page}img{height:auto}@page{size:landscape;margin:0}}
</style></head><body><main>""" + "\n".join(sections) + """</main>
<nav aria-label="Slide controls"><button id="prev">Previous</button><span id="count" aria-live="polite"></span>
<button id="next">Next</button><button id="notes" aria-pressed="false">Notes (N)</button><button id="full">Fullscreen (F)</button></nav>
<script>
const slides=[...document.querySelectorAll('.slide')];let index=0;
const prev=document.getElementById('prev'),next=document.getElementById('next'),notes=document.getElementById('notes');
function show(n){index=Math.max(0,Math.min(slides.length-1,n));slides.forEach((s,i)=>s.classList.toggle('active',i===index));
document.getElementById('count').textContent=`${index+1} / ${slides.length}`;prev.disabled=index===0;next.disabled=index===slides.length-1;history.replaceState(null,'','#'+(index+1))}
function toggleNotes(){notes.setAttribute('aria-pressed',String(document.body.classList.toggle('notes')))}
function fullscreen(){if(document.fullscreenElement)document.exitFullscreen();else document.documentElement.requestFullscreen().catch(()=>{})}
prev.onclick=()=>show(index-1);next.onclick=()=>show(index+1);notes.onclick=toggleNotes;document.getElementById('full').onclick=fullscreen;
document.addEventListener('keydown',e=>{if(e.target.closest('input,textarea')||(e.key===' '&&e.target.closest('button')))return;
if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();show(index+1)}
if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();show(index-1)}
if(e.key==='Home')show(0);if(e.key==='End')show(slides.length-1);if(e.key.toLowerCase()==='n')toggleNotes();if(e.key.toLowerCase()==='f')fullscreen()});
show((parseInt(location.hash.slice(1))||1)-1);
</script></body></html>"""
    markup = markup.replace("<title>Forecasting insights</title>", f"<title>{html.escape(short_title)}</title>")
    (dest / "index.html").write_text(markup, encoding="utf-8", newline="\n")
    (dest / "speaker_notes.md").write_text(f"# {short_title}: speaker notes\n\n" + "\n".join(notes),
                                          encoding="utf-8", newline="\n")
    check_dir = ROOT / "artifacts/checks" / check_name
    check_dir.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGB", (1500, math.ceil(len(thumbnails) / 3) * 282), BG)
    for i, png in enumerate(thumbnails):
        tile = ImageOps.contain(Image.open(io.BytesIO(png)).convert("RGB"), (500, 282))
        sheet.paste(tile, ((i % 3) * 500, (i // 3) * 282))
    sheet.save(check_dir / "contact_sheet.jpg")
    write_json(check_dir / "render_check.json", {"slides": len(slides), "figure_text_overflow": overflow,
               "source_commit": evidence["source_commit"], "test_evaluated": False})
    write_json(dest / "checksums.json", {
        path.relative_to(dest).as_posix(): digest(path) for path in sorted(dest.rglob("*"))
        if path.is_file() and path != dest / "checksums.json"
    })
    print(f"Rendered {len(slides)} slides to {dest}; no figure text overflow.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true", help="Audit and package existing local evidence.")
    parser.add_argument("--snapshot-date", default="2026-11-02")
    parser.add_argument("--output", type=Path, default=DEST)
    args = parser.parse_args()
    if args.collect:
        collect(args.output, args.snapshot_date)
    render(args.output)


if __name__ == "__main__":
    main()
