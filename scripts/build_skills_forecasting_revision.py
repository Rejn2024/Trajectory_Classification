"""Build the focused revision edition from published, fixed forecasting evidence."""

import argparse
import hashlib
import html
import io
import json
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from skills_forecasting_revision_content import SLIDES
from skills_forecasting_revision_evidence import collect, load

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "docs/presentations/skills_forecasting_revision"
BG, PANEL, FG, MUTED = "#101e2e", "#20354b", "#edf3fa", "#b2c0ce"
TEAL, BLUE, GOLD, GREY = "#42d5b0", "#70aaff", "#ffc15c", "#91a1b1"
COLORS = {"original": GREY, "v3": BLUE, "v4": TEAL, "v5": GOLD}
LABELS = {"original": "Original Transformer", "v3": "12 MLPs (v3)",
          "v4": "MLP + TCN (v4)", "v5": "6 Transformers (v5)"}
MODELS = tuple(LABELS)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8", newline="\n")


def wrap(value, width):
    return "\n".join(textwrap.fill(line, width=width, break_long_words=False,
                                   break_on_hyphens=False) for line in str(value).split("\n"))


def render(dest, preview=None):
    dest = Path(dest)
    evidence, tables = load(dest / "data")
    metrics, skills = tables["metrics"], tables["skills"]
    population = metrics[metrics.pilot_id.eq("population")]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14, "text.color": FG,
                         "axes.labelcolor": FG, "axes.edgecolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED, "savefig.facecolor": BG, "pdf.fonttype": 42,
                         "svg.fonttype": "path", "svg.hashsalt": "skills-revision-20261104",
                         "figure.max_open_warning": 0})
    figures = []

    def page(number):
        content = SLIDES[number - 1]
        fig = plt.figure(figsize=(16, 9), facecolor=BG)
        fig.text(.05, .946, "SKILLS FORECASTING  /  REVISION EDITION", color=TEAL, fontsize=11, weight="bold")
        fig.text(.05, .865, content["title"], fontsize=28, weight="bold")
        fig.text(.05, .805, wrap(content["subtitle"], 128), fontsize=13, color=MUTED, va="top")
        fig.text(.05, .105, "CHECK  " + wrap(content["question"], 125), fontsize=12, color=GOLD, va="top")
        fig.text(.05, .035, "B01 observes ten frozen trained pilots  |  Development validation; reserved test untouched",
                 fontsize=9, color=MUTED)
        fig.text(.95, .035, f"{number:02} / {len(SLIDES)}", ha="right", fontsize=10, color=TEAL)
        figures.append(fig)
        return fig

    def paragraph(fig, x, y, text, width=66, size=17, color=FG):
        return fig.text(x, y, wrap(text, width), va="top", fontsize=size, linespacing=1.35, color=color)

    def heading(fig, x, y, text, color=TEAL):
        fig.text(x, y, text, va="top", fontsize=18, color=color, weight="bold")

    def box(fig, x, y, w, h, title, body="", color=TEAL, size=15):
        fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.008",
                                     transform=fig.transFigure, facecolor=PANEL, edgecolor=color, linewidth=1.2))
        fig.text(x + .015, y + h - .025, title, fontsize=17, color=color, weight="bold", va="top")
        if body:
            paragraph(fig, x + .015, y + h - .078, body, max(15, int(w * 103)), size)

    def arrow(fig, start, end, color=MUTED):
        fig.add_artist(FancyArrowPatch(start, end, transform=fig.transFigure, arrowstyle="-|>",
                                      color=color, mutation_scale=18, linewidth=1.5))

    def table(fig, bounds, headers, rows, widths=None, size=14):
        ax = fig.add_axes(bounds)
        ax.axis("off")
        result = ax.table(cellText=rows, colLabels=headers, colWidths=widths,
                          cellLoc="left", colLoc="left", bbox=[0, 0, 1, 1])
        result.auto_set_font_size(False)
        result.set_fontsize(size)
        for (r, _), cell in result.get_celld().items():
            cell.set_facecolor(PANEL if r == 0 else BG)
            cell.set_edgecolor("#3a5065")
            cell.set_linewidth(.5)
            cell.get_text().set_color(TEAL if r == 0 else FG)
            if r == 0:
                cell.get_text().set_weight("bold")
        return result

    def chart(fig, bounds):
        ax = fig.add_axes(bounds, facecolor=BG)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", alpha=.17, color=MUTED)
        ax.set_axisbelow(True)
        return ax

    def metric(model, horizon):
        rows = population[population.model.eq(model) & population.horizon_s.eq(horizon)]
        if len(rows) != 1:
            raise ValueError(f"Metric row not unique: {model}/{horizon}")
        return rows.iloc[0]

    # 1: A study route, with the central result established immediately.
    f = page(1)
    box(f, .06, .48, .87, .23, "Our question", "From the tactical observer's available history, which catalogue label will a trained pilot select at +1, +2 or +5 seconds?", size=22)
    for x, title, body, color in [(.06, "Understand", "Task, data and inputs", BLUE),
                                 (.36, "Explain", "Networks and training", TEAL),
                                 (.66, "Assess", "Results and uncertainty", GOLD)]:
        box(f, x, .25, .27, .15, title, body, color)
    paragraph(f, .06, .19, "Main finding: the current MLP/TCN reference retains higher measured accuracy than the pure Transformer challenger.", 118, 14, MUTED)

    # 2: The behaviour generator and the forecaster are different models.
    f = page(2)
    box(f, .06, .49, .38, .23, "1  Train the pilots with RL", "Simulator → observations → pilot actions\nReward teaches the policy how to act.\nFreeze ten trained policies.", BLUE)
    box(f, .56, .49, .38, .23, "2  Train a skill forecaster", "B01's observed tracks → forecast labels\nSupervised examples teach prediction.\nOne shared forecaster covers all ten pilots.", TEAL)
    arrow(f, (.45, .60), (.55, .60))
    box(f, .06, .22, .88, .18, "What a forecast means", "A01 is the observed aircraft; B01 is the scripted tactical observer/opponent. A predicted label describes the pilot's catalogue choice. Forecast accuracy does not rate combat competence.", GOLD)

    # 3: Flight counts and independent experimental units.
    f = page(3)
    box(f, .06, .49, .39, .23, "Pilot training", "112,500 simulated training flights\nA population-training stage develops the policies.\nThese are not 112,500 forecasting examples.", BLUE)
    box(f, .55, .49, .39, .23, "Forecasting corpus", "300 scenario groups × 10 frozen pilots\n= 3,000 observer–pilot encounters\nEach encounter supplies overlapping origins.", TEAL)
    table(f, [.06, .22, .88, .20], ["Partition", "Scenario groups", "Encounters", "Purpose"],
          [["Training", "210", "2,100", "Fit and internal model selection"],
           ["Validation", "45", "450", "Fixed development comparisons"],
           ["Reserved test", "45", "450", "Not evaluated"]], [.22, .20, .20, .38], 15)
    paragraph(f, .06, .18, "Keep all ten versions of a scenario together. Thousands of overlapping windows are not thousands of independent scenarios.", 120, 13, MUTED)

    # 4: History is past, forecast horizon is future, encounter age is different.
    f = page(4)
    ax = f.add_axes([.07, .40, .86, .29], facecolor=BG)
    ax.axvspan(-60, 0, color=BLUE, alpha=.18)
    ax.axvline(0, color=TEAL, lw=2)
    ax.plot([-60, 8], [0, 0], color=MUTED)
    ax.scatter([-60, -20, 0, 1, 2, 5], [0] * 6, color=[BLUE, BLUE, TEAL, GOLD, GOLD, GOLD], s=65)
    for x, label, y in [(-60, "t − 60 s", -.22), (-20, "t − 20 s", -.22), (0, "Origin t", -.48),
                         (1, "+1 s", .27), (2, "+2 s", .57), (5, "+5 s", .27)]:
        ax.annotate(label, (x, 0), (x, y), ha="center", color=TEAL if x == 0 else FG,
                    arrowprops={"arrowstyle": "-", "color": MUTED}, fontsize=13)
    ax.text(-42, .47, "Observed history", color=BLUE, fontsize=20)
    ax.set(xlim=(-65, 10), ylim=(-.7, .9))
    ax.axis("off")
    paragraph(f, .06, .33, "History length: how far back the model may look.\nForecast horizon: how far ahead the label lies.\nElapsed time: how old the encounter is at this origin.", 108, 19)
    paragraph(f, .06, .18, "Early origins remain in the evaluation. Missing earlier history is masked; attention never receives future tracks or future labels.", 125, 13, MUTED)

    # 5: Concrete inputs, with privileged labels explicitly outside the input.
    f = page(5)
    rows = [["Observer and relative tracks", "B01 position/velocity/heading; A01 relative position/velocity, range and bearing"],
            ["Missiles and energy", "Hostile-missile flag; four B01 missile slots; kinetic/potential-energy proxies"],
            ["Backward changes", "Target acceleration derived from observations 0.5 seconds apart"],
            ["37 geometry features", "Line-of-sight rates, aspect, closure/closest approach, relative energy, missile presence"]]
    table(f, [.06, .34, .88, .38], ["Feature group", "Examples used in forecasting"], rows, [.29, .71], 14)
    box(f, .06, .16, .88, .15, "54 original + 37 derived = 91 features", "Derived geometry uses available observations. Pilot ID and the private current skill label are not inputs.", GOLD, 14)

    # 6: Convert a familiar table into a batched history tensor.
    f = page(6)
    table(f, [.06, .42, .40, .29], ["Time", "Speed", "Range", "Valid"],
          [["t − 2", "250", "24,000", "1"], ["t − 1", "255", "23,600", "1"],
           ["t", "260", "23,200", "1"]], [.25, .25, .28, .22], 17)
    box(f, .55, .44, .39, .25, "One ordinary feature row", "[speed(t−2), range(t−2), valid(t−2), …, speed(t), range(t), valid(t)]\nTime keeps a fixed column position.", BLUE, 16)
    arrow(f, (.47, .55), (.54, .55))
    paragraph(f, .06, .34, "Batch × time × features is a stack of those per-origin tables.\n20-second MLP: 26 timestamps × (91 features + mask) = 2,392 columns.\n60-second MLP: 66 × 92 = 6,072 columns.", 114, 18)
    paragraph(f, .06, .18, "Toy values above are illustrative. The target is a separate three-column table of future catalogue labels.", 122, 13, MUTED)

    # 7: The MLP processes the whole ordered history, not isolated frames.
    f = page(7)
    stages = [("Flatten", "26 / 66 time slots"), ("Project", "256 / 512 width"),
              ("3 residual blocks", "Two linear layers each"), ("Classify", "3 × 33 logits")]
    for j, (title, body) in enumerate(stages):
        x = .06 + j * .225
        box(f, x, .51, .20, .18, title, body, TEAL if j != 0 else BLUE, 14)
        if j < 3:
            arrow(f, (x + .202, .60), (x + .222, .60))
    paragraph(f, .06, .43, "Eight linear layers overall: input projection + six inside the residual blocks + output projection. A residual connection adds a block's input back to its output.", 117, 17)
    box(f, .06, .19, .88, .15, "The v3 ensemble", "Four history/width/regularization recipes × three random training seeds = twelve members. Their calibrated probabilities are averaged equally.", GOLD, 16)

    # 8: TCN filters and probability averaging form the retained v4 reference.
    f = page(8)
    box(f, .06, .49, .40, .23, "Temporal convolutional network", "Six causal, dilated convolution blocks\nDilations: 1, 2, 4, 8, 16, 32\n127-token receptive field covers all 66 input tokens.", BLUE, 16)
    box(f, .55, .49, .39, .23, "Current reference: 15 members", "12 temporal MLPs → mean probabilities\n3 sixty-second TCNs → mean probabilities\nMix those family means 50 : 50.", TEAL, 16)
    paragraph(f, .06, .38, "P(final label) = 0.5 × mean(P from 12 MLPs) + 0.5 × mean(P from 3 TCNs)", 110, 19, TEAL)
    table(f, [.12, .19, .76, .12], ["Illustrative family predictions", "Label A", "Label B", "Label C"],
          [["MLPs / TCNs", "0.60 / 0.20", "0.30 / 0.60", "0.10 / 0.20"],
           ["50 : 50 mixture", "0.40", "0.45", "0.15"]], [.44, .19, .19, .18], 14)

    # 9: The current pure Transformer recipe has two complementary readouts.
    f = page(9)
    box(f, .06, .49, .40, .23, "Temporal Transformer", "66 observed tokens → attention\nRetain all contextual positions in readout\nThree training seeds = three members", BLUE, 16)
    box(f, .55, .49, .39, .23, "Patch Transformer", "66 observed tokens → 17 local patches\nAttention combines the patch representations\nThree training seeds = three members", GOLD, 16)
    paragraph(f, .06, .40, "Each encoder: 3 attention blocks • width 192 • 4 attention heads\nCurrent observed features also enter the classifier. Each member outputs 3 × 33 logits.\nThe final six-member ensemble averages calibrated probabilities equally.", 117, 18)
    paragraph(f, .06, .19, "Attention connects observations inside the available past. Standard feed-forward sublayers are part of a Transformer. These models are trained from scratch.", 119, 14, MUTED)

    # 10: Four milestones replace an exhaustive candidate catalogue.
    f = page(10)
    rows = []
    for model in MODELS:
        setting = evidence["models"][model]
        rows.append([LABELS[model], str(setting["features"]), setting["history_text"],
                     setting["tokens_text"], str(setting["members"]), f"{setting['parameters'] / 1e6:.2f} M"])
    table(f, [.06, .35, .88, .37], ["Main milestone", "Features", "History", "Input tokens", "Members", "Parameters"],
          rows, [.26, .10, .15, .17, .12, .20], 14)
    paragraph(f, .06, .28, "Original: 51 samples in five seconds. Later models: sparse sampling over 20/60 seconds. Longer history does not always mean more input tokens.", 116, 17)
    paragraph(f, .06, .18, "Several design choices changed together. The history, features, architecture and model count are explicit; these are complete-recipe comparisons.", 121, 13, MUTED)

    # 11: Partition roles and historic validation reuse.
    f = page(11)
    for x, title, body, color in [(.06, "147 fit groups", "Weights and normalization", BLUE),
                                 (.36, "21 stop groups", "Epochs and calibration", TEAL),
                                 (.66, "42 select groups", "Compare candidate recipes", GOLD)]:
        box(f, x, .52, .27, .18, title, body, color, 16)
    paragraph(f, .06, .46, "These groups all sit inside the original 210 training groups. Repeat across two internal partitions; each has all ten pilots for every included scenario.", 116, 17)
    box(f, .06, .23, .88, .14, "Freeze → refit on all 210 → evaluate on the same 45 validation groups", "Freeze members, seeds, epoch budgets and temperatures first. Keep the 45-group reserved test separate.", TEAL, 15)
    paragraph(f, .06, .18, "The original neural models used validation for stopping; validation has been inspected repeatedly. Current scores are development evidence.", 120, 13, MUTED)

    # 12: Actual loss curves and a probability-based interpretation of NLL.
    f = page(12)
    curves = tables["learning_curves"]
    part = curves[curves.model.eq("v5")].sort_values("epoch")
    ax = chart(f, [.08, .32, .51, .38])
    ax.plot(part.epoch, part.training_objective, color=TEAL, label="Training objective")
    ax.plot(part.epoch, part.stop_nll, color=GOLD, label="Stop NLL, before calibration")
    selected_epoch = evidence["learning_curves"]["v5"]["selected_epoch"]
    ax.axvline(selected_epoch, color=BLUE, linestyle="--", label=f"Selected by accuracy: epoch {selected_epoch}")
    ax.set(xlabel="Epoch", ylabel="Loss (natural log units)")
    ax.legend(fontsize=11, facecolor=BG, edgecolor=MUTED, labelcolor=FG)
    heading(f, .65, .70, "NLL = −ln(p of the true label)")
    table(f, [.65, .43, .29, .20], ["p(true label)", "NLL"],
          [["0.90", "0.105"], ["0.50", "0.693"], ["0.10", "2.303"]], [.6, .4], 15)
    paragraph(f, .65, .36, "Lower is better. A confident mistake is expensive. NLL is not a percentage.", 33, 16)
    paragraph(f, .06, .23, "Actual 60 s patch model, internal partition 0, seed 42. Training uses label smoothing and dropout; stopping NLL uses ordinary evaluation predictions. The loss gap is not directly comparable.", 124, 13)
    paragraph(f, .06, .16, "Forecaster curves pool pilots. Per-pilot forecaster epoch curves were not logged; the archived RL-return curves describe a different learner.", 124, 12, MUTED)

    # 13: Understand the denominator in each classification metric.
    f = page(13)
    box(f, .06, .48, .39, .24, "Across all predictions", "Top-1: was the highest-ranked label correct?\nTop-3: was the true label among three suggestions?", BLUE, 16)
    box(f, .55, .48, .39, .24, "For one catalogue label", "Precision = TP / (TP + FP)\nRecall = TP / (TP + FN)\nF1 = 2 × precision × recall / (precision + recall)", TEAL, 16)
    table(f, [.06, .24, .88, .16], ["Illustrative counts", "Precision", "Recall", "F1"],
          [["60 true positives; 20 false positives; 40 missed", "60 / 80 = 75%", "60 / 100 = 60%", "66.7%"]],
          [.49, .18, .18, .15], 15)
    paragraph(f, .06, .18, "Catalogue macro-F1 averages the 33 class F1 values equally, including zero-F1 absent labels. It does not weight frequent labels more heavily.", 123, 13, MUTED)

    # 14: The fixed evaluation task, four models only.
    f = page(14)
    ax = chart(f, [.08, .29, .48, .43])
    for model in MODELS:
        ax.plot([1, 2, 5], [metric(model, h).top1_accuracy * 100 for h in (1, 2, 5)],
                "o-", color=COLORS[model], label=LABELS[model], linewidth=2, markersize=5)
    ax.set(xlabel="Forecast horizon (seconds)", ylabel="Top-1 accuracy (%)", xticks=[1, 2, 5])
    ax.legend(fontsize=10, facecolor=BG, edgecolor=MUTED, labelcolor=FG)
    rows = [[LABELS[model], *[f"{metric(model, h).top1_accuracy * 100:.2f}%" for h in (1, 2, 5)],
             f"{population[population.model.eq(model)].top1_accuracy.mean() * 100:.2f}%"] for model in MODELS]
    table(f, [.61, .42, .33, .30], ["Model", "+1 s", "+2 s", "+5 s", "Mean"], rows,
          [.38, .155, .155, .155, .155], 11)
    paragraph(f, .61, .34, "Same origins, 33 labels and target masks.\nThe mean weights the three horizons equally.", 38, 16)
    paragraph(f, .06, .19, "Validation support: 30,802 / 30,466 / 29,458 eligible origins at +1 / +2 / +5 s. Missing future labels are masked identically.", 122, 13, MUTED)

    # 15: Confidence quality is a different axis from a correct argmax.
    f = page(15)
    rows = []
    for model in MODELS:
        row = metric(model, 5)
        rows.append([LABELS[model], f"{row.top3_accuracy * 100:.2f}%", f"{row.macro_f1_catalogue:.4f}",
                     f"{row.nll:.4f}", f"{row.brier:.4f}", f"{row.ece_10_bins * 100:.2f} pp"])
    table(f, [.06, .42, .88, .29], ["Model, +5 s", "Top-3 ↑", "F1 / 33 ↑", "NLL ↓", "Brier ↓", "ECE ↓"],
          rows, [.28, .14, .14, .14, .14, .16], 15)
    box(f, .06, .17, .42, .18, "What v5 improves", "NLL, Brier and ECE improve versus v4. It uses 18.9 M rather than 31.7 M parameters.", TEAL, 15)
    box(f, .53, .17, .41, .18, "What v5 loses", "Top-1 and macro-F1 fall at all three horizons. Better probabilities need not produce more correct top labels.", GOLD, 15)

    # 16: Every pilot remains visible, with paired whole-scenario intervals.
    f = page(16)
    pilots = tables["pilot_intervals"]
    p5 = pilots[pilots.horizon_s.eq(5)].sort_values("pilot_id")
    rows = [[str(row.pilot_id).replace("pilot_", ""), f"{int(row.count):,}",
             f"{row.baseline_top1 * 100:.2f}%", f"{row.candidate_top1 * 100:.2f}%", f"{row.gain * 100:+.2f}"]
            for row in p5.itertuples()]
    table(f, [.06, .26, .48, .46], ["Pilot", "Origins", "v4", "v5", "Gain pp"], rows,
          [.15, .22, .22, .22, .19], 12)
    ax = chart(f, [.64, .31, .29, .38])
    ys = np.arange(len(p5))
    ax.hlines(ys, p5.gain_lower * 100, p5.gain_upper * 100, color=BLUE, linewidth=2)
    ax.plot(p5.gain * 100, ys, "o", color=GOLD)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set(yticks=ys, yticklabels=[str(p).replace("pilot_", "") for p in p5.pilot_id], xlabel="v5 − v4 accuracy (pp)")
    ax.invert_yaxis()
    paragraph(f, .06, .19, "Four pilot point estimates improve at +5 s. The 95% paired scenario intervals are conditional and unadjusted for multiple comparisons. Accuracy is not pilot competence.", 124, 13, MUTED)

    # 17: A complete per-label F1 view, without repeating every historic model.
    f = page(17)
    selected = skills[skills.horizon_s.eq(5)]
    old = selected[selected.model.eq("v4")].set_index("skill")
    new = selected[selected.model.eq("v5")].set_index("skill")
    rows = [[name, f"{int(row.support):,}", f"{row.f1:.3f}" if row.support else "—",
             f"{new.loc[name].f1:.3f}" if row.support else "—"] for name, row in old.sort_index().iterrows()]
    for bounds, subset in [([.05, .27, .435, .46], rows[:17]), ([.515, .27, .435, .46], rows[17:])]:
        table(f, bounds, ["Skill", "Support", "v4 F1", "v5 F1"], subset, [.55, .17, .14, .14], 10)
    paragraph(f, .06, .21, "Dash = no true examples at +5 s. Those labels still count as zero in the all-33 macro-F1. Support counts overlapping origins, not independent flights.", 126, 13)
    paragraph(f, .06, .155, "Example: support_missile has only 19 origins; its F1 falls from 0.429 to 0.160. Interpret sparse-label changes cautiously.", 126, 12, MUTED)

    # 18: Temporal results must keep the changing population in view.
    f = page(18)
    temporal = tables["temporal"]
    rows = []
    phases = temporal[temporal.horizon_s.eq(5)]
    for row in phases[phases.model.eq("v4")].itertuples():
        other = phases[phases.model.eq("v5") & phases.elapsed_phase.eq(row.elapsed_phase)].iloc[0]
        if int(row.count) != int(other["count"]) or int(row.episodes) != int(other.episodes):
            raise ValueError("Temporal comparison populations differ")
        rows.append([str(row.elapsed_phase), str(int(row.episodes)), f"{int(row.count):,}",
                     f"{row.top1_accuracy * 100:.2f}%", f"{other.top1_accuracy * 100:.2f}%"])
    table(f, [.06, .36, .55, .35], ["Elapsed phase", "Encounters", "Origins", "v4 +5 s", "v5 +5 s"], rows,
          [.24, .20, .20, .18, .18], 13)
    persist = evidence["persistence"]
    heading(f, .66, .70, "Selected labels often persist")
    paragraph(f, .66, .63, f"{persist['fraction_adjacent_equal'] * 100:.2f}% of adjacent observed decisions repeat the same label.\nCompleted runs: median {persist['median_completed_run_s']:g} s, 90th percentile {persist['p90_completed_run_s']:g} s.", 32, 17)
    paragraph(f, .06, .27, "Common-origin analysis requires a valid +5 s target for every compared horizon. Later phases contain a different mix of surviving encounters and decisions.", 119, 16)
    paragraph(f, .06, .18, "Later accuracy does not isolate the value of extra history. Label persistence is descriptive: the private current true label remains unavailable to the forecaster.", 124, 13, MUTED)

    # 19: Uncertainty compares complete scenario groups, not independent windows.
    f = page(19)
    comparisons = tables["comparisons"]
    ax = chart(f, [.16, .41, .73, .25])
    yy = np.arange(len(comparisons))
    ax.hlines(yy, comparisons.lower_95 * 100, comparisons.upper_95 * 100, color=BLUE, linewidth=4)
    ax.plot(comparisons.gain * 100, yy, "o", color=TEAL, markersize=8)
    ax.axvline(0, color=MUTED)
    ax.set(yticks=yy, yticklabels=comparisons.label, xlabel="Mean accuracy change (percentage points)")
    ax.invert_yaxis()
    paragraph(f, .06, .31, "95% intervals from 5,000 paired resamples of whole scenario groups preserve the ten pilots and correlated horizons. They are conditional development intervals, not final-test evidence.", 116, 17)
    secondary = evidence["secondary"]
    paragraph(f, .06, .20, f"Secondary frozen blend: 75% v5 + 25% v4, 21 models. Mean gain {secondary['gain'] * 100:+.2f} pp; interval [{secondary['lower_95'] * 100:+.2f}, {secondary['upper_95'] * 100:+.2f}]. It establishes no accuracy gain or equivalence.", 122, 14, GOLD)

    # 20: Timers describe work performed, not an isolated architecture benchmark.
    f = page(20)
    costs = tables["costs"]
    rows = []
    for model in ("v3", "v4", "v5"):
        selected_cost = costs[costs.model.eq(model)]
        rows.append([LABELS[model], str(int(selected_cost.runs.sum())),
                     f"{selected_cost.seconds.sum() / 60:.2f} min",
                     str(int(selected_cost[selected_cost.phase.eq('internal')].runs.sum())),
                     str(int(selected_cost[selected_cost.phase.eq('refit')].runs.sum()))])
    table(f, [.06, .43, .88, .29], ["Study", "New completed fits", "Summed fit timers", "Internal", "Final refits"],
          rows, [.28, .21, .23, .14, .14], 15)
    paragraph(f, .06, .35, "A fit is one recipe × one seed × one partition (or a final refit). Cached reference models are reused and excluded from these new-fit counts.", 115, 18)
    paragraph(f, .06, .23, "The v3 row covers v3 alone; it excludes the original two neural fits and eleven earlier v2 fits. v4/v5 timers include shared-GPU contention and exclude loading/setup.", 120, 15)
    paragraph(f, .06, .16, "Summed fit minutes are not elapsed project time, simulation cost or an isolated speed comparison between architectures.", 123, 13, MUTED)

    # 21: Separate demonstrated evidence from hypotheses for future work.
    f = page(21)
    box(f, .06, .40, .42, .32, "Evidence boundaries", "Known pilots under new scenarios\nIdealised simulator tracks\nSelected catalogue labels, with some aliases\nRare/absent labels and uneven pilot effects\nRepeatedly inspected development validation", GOLD, 16)
    box(f, .54, .39, .40, .33, "Prospective experiments", "Stronger regularization and probability-aware stopping\nTraining-only representation learning\nControlled feature/history ablations\nPerformance-versus-cost objectives\nFinal test after choices settle", BLUE, 16)
    paragraph(f, .06, .29, "Freeze a new experiment before inspecting its validation results. Keep the existing task and baseline comparable, and report failed candidates and subgroup regressions.", 117, 17)
    paragraph(f, .06, .18, "These are future hypotheses. The current revision deck reads saved evidence only; it performs no new training, inference or test evaluation.", 124, 13, MUTED)

    # 22: A compact oral revision route.
    f = page(22)
    box(f, .06, .47, .88, .24, "What the experiments establish", "The task predicts future skill selections from observed history. MLP/TCN models can use temporal context effectively. The new pure Transformers are smaller and improve probability quality; they do not establish improved top-1 accuracy.", TEAL, 20)
    rows = [["Explain the task", "Observer → history → three future labels"],
            ["Explain the model", "Input shape → temporal computation → probabilities"],
            ["Judge the evidence", "Accuracy + F1 + NLL + pilot/skill support + uncertainty"]]
    table(f, [.06, .20, .88, .21], ["Revision step", "What you should be able to say"], rows, [.27, .73], 16)
    if len(figures) != 22 or len(SLIDES) != 22:
        raise ValueError("The revision edition must keep its 22-slide sequence")

    dest.mkdir(parents=True, exist_ok=True)
    pdf = dest / "skills_forecasting_revision.pdf"
    svgs = []
    with PdfPages(pdf, metadata={"Title": "Skills forecasting: revision edition", "Author": "Trajectory Classification",
                                 "CreationDate": None, "ModDate": None}) as output:
        for index, fig in enumerate(figures, 1):
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            texts = list(fig.texts)
            for ax in fig.axes:
                texts.extend([ax.title, ax.xaxis.label, ax.yaxis.label])
                for tab in ax.tables:
                    texts.extend(cell.get_text() for cell in tab.get_celld().values())
            for item in texts:
                if item.get_visible() and item.get_text().strip():
                    bounds = item.get_window_extent(renderer)
                    if bounds.x0 < -2 or bounds.y0 < -2 or bounds.x1 > fig.bbox.x1 + 2 or bounds.y1 > fig.bbox.y1 + 2:
                        raise ValueError(f"Text outside slide {index}: {item.get_text()}")
            output.savefig(fig)
            buffer = io.StringIO()
            fig.savefig(buffer, format="svg", metadata={"Date": None})
            svg = buffer.getvalue()
            svg = svg[svg.index("<svg"):]
            for name in set(re.findall(r'id="([^"]+)"', svg)):
                renamed = f"slide{index}_{name}"
                svg = svg.replace(f'id="{name}"', f'id="{renamed}"').replace(f'url(#{name})', f'url(#{renamed})')
                svg = svg.replace(f'href="#{name}"', f'href="#{renamed}"')
            svgs.append(svg)
            if preview:
                Path(preview).mkdir(parents=True, exist_ok=True)
                fig.savefig(Path(preview) / f"slide_{index:02}.png", dpi=95)
            plt.close(fig)
    write_documents(dest, evidence, svgs)
    write_json(dest / "checksums.json", {p.relative_to(dest).as_posix(): sha(p)
                                         for p in sorted(dest.rglob("*")) if p.is_file() and p != dest / "checksums.json"})
    return {"pdf": str(pdf), "pages": len(figures), "html": str(dest / "index.html"), "pdf_sha256": sha(pdf)}


def write_documents(dest, evidence, svgs):
    sections = []
    for index, (slide, svg) in enumerate(zip(SLIDES, svgs, strict=True), 1):
        sections.append(f'<section class="slide" id="slide-{index}" aria-label="Slide {index}: {html.escape(slide["title"])}" hidden>'
                        f'{svg}<aside class="notes"><p>{html.escape(slide["notes"])}</p>'
                        f'<details><summary>Revision answer</summary><p>{html.escape(slide["answer"])}</p></details></aside></section>')
    options = "".join(f'<option value="{i}">{i:02}. {html.escape(s["title"])}</option>' for i, s in enumerate(SLIDES, 1))
    document = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Skills forecasting — revision edition</title><style>
:root{color-scheme:dark}body{margin:0;background:#101e2e;color:#edf3fa;font:16px system-ui,sans-serif}
main{max-width:1600px;margin:auto}.slide svg{display:block;width:100%;height:auto;max-height:calc(100vh - 76px)}
.slide[hidden]{display:none}.notes{display:none;padding:1rem 2rem;background:#20354b;line-height:1.6}
body.show-notes .slide:not([hidden]) .notes{display:block}.notes p{max-width:105ch;margin:.5rem auto}
details{max-width:105ch;margin:auto}summary{cursor:pointer;color:#ffc15c}
nav{position:sticky;bottom:0;display:flex;gap:.7rem;align-items:center;justify-content:center;flex-wrap:wrap;padding:.7rem;background:#182b3e}
button,select,a{font:inherit}button,select{border:1px solid #7990a7;color:#edf3fa;background:#20354b;padding:.45rem .65rem;border-radius:.3rem}
button{cursor:pointer}button:disabled{opacity:.4;cursor:default}select{max-width:48vw}a{color:#42d5b0}button:focus-visible,select:focus-visible,a:focus-visible{outline:3px solid #ffc15c}
@media print{nav,.notes{display:none!important}.slide[hidden]{display:block}.slide{break-after:page}.slide svg{max-height:none}}</style></head><body><main>'''
    document += "\n".join(sections)
    document += f'''</main><nav aria-label="Slide navigation"><button id="previous" aria-label="Previous slide">←</button>
<label for="jump">Slide</label><select id="jump">{options}</select><span id="count" aria-live="polite"></span>
<button id="next" aria-label="Next slide">→</button><button id="notes" aria-pressed="false">Notes (N)</button>
<a href="skills_forecasting_revision.pdf">PDF</a><a href="study_guide.md">Study guide</a></nav>
<script>const slides=[...document.querySelectorAll('.slide')],jump=document.getElementById('jump');let current=1;
function show(n){{current=Math.max(1,Math.min(slides.length,n));slides.forEach((s,i)=>s.hidden=i!==current-1);jump.value=current;
document.getElementById('count').textContent=current+' / '+slides.length;document.getElementById('previous').disabled=current===1;
document.getElementById('next').disabled=current===slides.length;history.replaceState(null,'','#slide-'+current);}}
function notes(){{const on=document.body.classList.toggle('show-notes');document.getElementById('notes').setAttribute('aria-pressed',String(on));}}
document.getElementById('previous').onclick=()=>show(current-1);document.getElementById('next').onclick=()=>show(current+1);
jump.onchange=()=>show(Number(jump.value));document.getElementById('notes').onclick=notes;
document.addEventListener('keydown',e=>{{if(['SELECT','INPUT','TEXTAREA'].includes(e.target.tagName))return;
if(e.key==='ArrowRight'||e.key==='PageDown'){{e.preventDefault();show(current+1);}}
if(e.key==='ArrowLeft'||e.key==='PageUp'){{e.preventDefault();show(current-1);}}
if(e.key==='Home'){{e.preventDefault();show(1);}}if(e.key==='End'){{e.preventDefault();show(slides.length);}}
if(e.key.toLowerCase()==='n')notes();if(e.key.toLowerCase()==='f'){{if(document.fullscreenElement)document.exitFullscreen();else document.documentElement.requestFullscreen?.();}}}});
window.addEventListener('hashchange',()=>show(Number(location.hash.replace('#slide-',''))||1));show(Number(location.hash.replace('#slide-',''))||1);
</script></body></html>'''
    document = "\n".join(line.rstrip() for line in document.splitlines()) + "\n"
    (dest / "index.html").write_text(document, encoding="utf-8", newline="\n")
    notes = ["# Skills forecasting revision: speaker notes", ""]
    guide = ["# Skills forecasting: slide-by-slide revision", "", "Work through one numbered slide at a time. Explain the idea in your own words, then try its revision question. Answers are provided for checking after discussion.", ""]
    for index, slide in enumerate(SLIDES, 1):
        for lines in (notes, guide):
            lines.extend([f"## {index:02}. {slide['title']}", "", f"**Takeaway:** {slide['takeaway']}", "", slide["notes"], "",
                          f"**Revision question:** {slide['question']}", ""])
        notes.extend([f"**Answer:** {slide['answer']}", ""])
        guide.extend(["<details>", "<summary>Check your answer</summary>", "", slide["answer"], "", "</details>", ""])
    (dest / "speaker_notes.md").write_text("\n".join(notes), encoding="utf-8", newline="\n")
    (dest / "study_guide.md").write_text("\n".join(guide), encoding="utf-8", newline="\n")
    rows = [f"| {i:02} | {s['title']} | {s['takeaway']} |" for i, s in enumerate(SLIDES, 1)]
    readme = ["# Skills forecasting: revision edition", "", "**22 slides, four main model milestones, one revision question per slide.**", "",
              "[Download the PDF](skills_forecasting_revision.pdf) · [Offline HTML deck](index.html) · [Study guide](study_guide.md) · [Speaker notes](speaker_notes.md)", "",
              "This is the current teaching edition, dated **4 November 2026**. It brings the original experiment and the v3, v4 and v5 studies into one learning sequence. It uses fixed published evidence; building it does not train models, run inference or evaluate the reserved test.", "",
              "The comparison follows four main references: original Transformer, twelve temporal MLPs (v3), the fifteen-member MLP/TCN reference (v4), and six pure Transformers (v5). Full candidate searches remain in the research reports. The secondary 21-member blend is discussed once on the uncertainty slide.", "",
              "The main finding remains: v4 averages **66.19%** validation accuracy and the pure Transformers **65.88%**. The Transformers improve probability quality and use fewer parameters; their accuracy difference interval crosses zero. This is neither proof of an accuracy improvement nor equivalence.", "",
              "## Use the deck for revision", "", "Start at slide 1 and progress in order. Each slide has a check question; notes and the study guide explain the answer. The HTML is self-contained and supports arrows, Home/End, N for notes and F for fullscreen. The PDF contains vector diagrams and selectable text.", "",
              "| Slide | Topic | Main idea |", "|---:|---|---|", *rows, "", "## Evidence and previous editions", "",
              "- [Compact evidence and source provenance](data/evidence.json), [main metrics](data/metrics.csv), [per-skill F1](data/skills.csv), [pilot intervals](data/pilot_intervals.csv).",
              "- [Learning curves](data/learning_curves.csv), [temporal analysis](data/temporal.csv), [study costs](data/costs.csv), [paired comparisons](data/comparisons.csv), [checksums](checksums.json).",
              "- [Historical 41-slide catalogue deck](../skills_catalogue/README.md), including the per-pilot RL-return curves and full original comparisons.",
              "- [v4 sequence-model experiment](../../experiments/skill_forecasting_v4/README.md) and [v5 Transformer experiment](../../experiments/skill_forecasting_v5_transformers/README.md), with full configurations, searches and preserved results.", "",
              "The historical decks and experiment bundles remain unchanged. The new edition replaces them as the main revision route, while retaining their detailed evidence. All figures describe known pilots under new scenario groups and idealised simulator observations. The validation set has been inspected during development; the reserved test remains unevaluated.", "",
              "## Rebuild", "", "```powershell", ".venv\\Scripts\\python.exe -X utf8 scripts/build_skills_forecasting_revision.py", "", "# Refresh the compact snapshot from the committed, fixed evidence bundles:", ".venv\\Scripts\\python.exe -X utf8 scripts/build_skills_forecasting_revision.py --collect", "```", "",
              "The default rebuild uses only this edition's portable data snapshot and the tracked builder/content/helper scripts. It does not need checkpoints, raw observations or network access.", ""]
    (dest / "README.md").write_text("\n".join(readme), encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true", help="Snapshot published portable evidence only")
    parser.add_argument("--output", type=Path, default=DEST)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    protected = [ROOT / "docs/presentations/skills_catalogue", ROOT / "docs/experiments/skill_forecasting_v4",
                 ROOT / "docs/experiments/skill_forecasting_v5_transformers"]
    if any(output == p or p in output.parents for p in protected):
        raise ValueError("Preserve historical bundles; use the new revision destination")
    if args.collect:
        collect(ROOT, output / "data")
    print(json.dumps(render(output, args.preview_dir), indent=2))


if __name__ == "__main__":
    main()
