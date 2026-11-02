"""Build a detailed original-versus-improved catalogue comparison from saved evidence."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from build_forecasting_insights import BG, BLUE, FG, GOLD, GREY, MUTED, TEAL, export
from skill_catalogue_evidence import (
    HORIZONS,
    LABELS,
    MODELS,
    collect,
    load,
    write_comparison,
)

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "docs/presentations/skills_catalogue"
COLORS = {"frequency": GREY, "gru": GOLD, "transformer": BLUE, "ensemble": TEAL}
TITLE = "Skills catalogue forecasting: original experiments versus improved ensemble"


def render(dest):
    evidence, tables, matrices = load(dest)
    write_comparison(dest, evidence, tables)
    summary, class_data = tables["summary_metrics"], tables["class_metrics"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.color": FG,
            "axes.labelcolor": FG,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": MUTED,
            "font.size": 14,
            "pdf.fonttype": 42,
            "savefig.facecolor": BG,
            "figure.max_open_warning": 0,
        }
    )
    slides = []

    def page(title, subtitle, source="Saved validation evidence"):
        fig = plt.figure(figsize=(16, 9), facecolor=BG)
        fig.text(
            0.055,
            0.94,
            "SKILLS CATALOGUE  /  ORIGINAL EXPERIMENTS → IMPROVED ENSEMBLE",
            fontsize=11,
            color=TEAL,
            weight="bold",
        )
        fig.text(0.055, 0.858, title, fontsize=29, weight="bold")
        fig.text(0.055, 0.80, subtitle, fontsize=15, color=MUTED)
        fig.text(
            0.055,
            0.038,
            source + "  |  Development validation; test reserved",
            fontsize=8.5,
            color=MUTED,
        )
        fig.text(0.945, 0.038, f"{len(slides) + 1:02}", ha="right", fontsize=11, color=TEAL)
        return fig

    def finish(fig, note):
        slides.append((fig, fig.texts[1].get_text(), note))

    def words(fig, x, y, title, body, color=TEAL, size=18):
        fig.text(x, y, title, color=color, fontsize=19, weight="bold", va="top")
        fig.text(x, y - 0.062, body, fontsize=size, va="top", linespacing=1.5)

    def chart(fig, bounds):
        ax = fig.add_axes(bounds, facecolor=BG)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color=MUTED, alpha=0.16)
        ax.set_axisbelow(True)
        return ax

    def table(fig, bounds, rows, headers, widths=None, size=16, highlight=None):
        ax = fig.add_axes(bounds)
        ax.axis("off")
        t = ax.table(
            cellText=rows,
            colLabels=headers,
            colWidths=widths,
            cellLoc="left",
            colLoc="left",
            bbox=[0, 0, 1, 1],
        )
        t.auto_set_font_size(False)
        t.set_fontsize(size)
        for (r, _), cell in t.get_celld().items():
            cell.set_facecolor("#1a3048" if r == 0 else BG)
            cell.set_edgecolor("#35485e")
            cell.get_text().set_color(TEAL if r == 0 or r == highlight else FG)
        return t

    def row(model, h, cohort="all", pilot="population"):
        return summary[
            summary.model.eq(model)
            & summary.horizon_s.eq(h)
            & summary.cohort.eq(cohort)
            & summary.pilot_id.eq(pilot)
        ].iloc[0]

    def values(model, metric, cohort="all"):
        return np.array([row(model, h, cohort)[metric] for h in HORIZONS])

    def legend(ax, **kwargs):
        ax.legend(facecolor=BG, edgecolor=MUTED, labelcolor=FG, fontsize=12, **kwargs)

    f = page(
        "What changed in skill forecasting—and how much?",
        "Tactical observer B01 → frozen trained pilot A01  |  Exact +1, +2 and +5 s selections",
    )
    for x, number, title, body in [
        (
            0.065,
            "55.90 → 63.48%",
            "+5 s TOP-1",
            "+7.58 percentage points\n17.19% fewer top-1 errors",
        ),
        (
            0.53,
            "0.274 → 0.366",
            "+5 s MACRO-F1",
            "Equal weight across all 33 labels\nThe gain extends beyond overall accuracy",
        ),
    ]:
        f.text(x, 0.64, number, fontsize=35, color=TEAL, weight="bold")
        words(f, x, 0.53, title, body, size=18)
    words(
        f,
        0.065,
        0.28,
        "THE COMPARISON",
        "Training frequency → original GRU / transformer → improved 12-model ensemble",
        BLUE,
        19,
    )
    finish(
        f,
        "Headline changes compare the original transformer with the frozen improved ensemble on identical "
        "validation origins. The task predicts catalogue choices, not pilot competence. Motion forecasting is "
        "a separate task covered in the earlier overview. This deck adds descriptive metrics from saved "
        "probabilities without refitting or changing model selection.",
    )

    f = page(
        "The prediction target and evaluation population stay fixed",
        "Same 33-label vocabulary, same pilots, same scenarios and same missing-label masks.",
    )
    words(
        f,
        0.065,
        0.70,
        "ONE-SIDED INPUT",
        "B01 observations available through time t.\nNo pilot ID or private selected label as input.\nLabels are the exact choices at t+1, t+2, t+5.",
        BLUE,
        18,
    )
    words(
        f,
        0.56,
        0.70,
        "GROUPED DATA",
        "3,000 episodes: 300 scenarios × 10 pilots.\n210 train / 45 validation / 45 test groups.\nAll ten pilot encounters stay in one group.",
        TEAL,
        18,
    )
    table(
        f,
        [0.065, 0.17, 0.87, 0.22],
        [[f"+{h} s", f"{int(row('ensemble', h)['count']):,}", "26 of 33"] for h in HORIZONS],
        ["Forecast horizon", "Valid validation origins", "Labels with truth support"],
        [0.30, 0.35, 0.35],
        17,
    )
    finish(
        f,
        "All scores weight valid forecast windows equally within each horizon. Overlapping windows are "
        "correlated. The reserved test is untouched. Same known pilots appear under disjoint scenario groups, "
        "so these are not unseen-pilot results. Idealised simulator tracks have no radar noise or track loss. "
        "Missing future selections are masked rather than carried forward. Some catalogue names share guidance.",
    )

    f = page(
        "The improved system has a larger information budget",
        "The experiment compares complete forecasting procedures, not architecture alone.",
        "Run metadata + frozen selection",
    )
    r = evidence["original_runs"]
    table(
        f,
        [0.065, 0.24, 0.87, 0.48],
        [
            ["Observation history", "5 seconds", "5 seconds", "20 / 60 seconds"],
            ["Features per token", "54", "54", "91"],
            [
                "Neural architecture",
                "2-layer GRU",
                "2-layer transformer",
                "4 temporal MLP variants",
            ],
            [
                "Parameters",
                f"{r['gru']['parameters']:,}",
                f"{r['transformer']['parameters']:,}",
                "28,977,060 total",
            ],
            ["Seeds / fitted models", "Seed 42 / one", "Seed 42 / one", "42, 43, 44 / twelve"],
            [
                "Checkpoint selection",
                "Validation NLL",
                "Validation NLL",
                "Training-only partitions",
            ],
            ["Selected epochs", "15", "20", "46 / 32 / 40 / 40"],
        ],
        ["Setting", "Original GRU", "Original transformer", "Improved ensemble"],
        [0.25, 0.23, 0.23, 0.29],
        15,
    )
    f.text(
        0.065,
        0.13,
        "Frequency reference: training-label priors only; no observation history or neural network.",
        fontsize=17,
        color=MUTED,
    )
    finish(
        f,
        "Initial neural models share width 128, two layers, dropout 0.1, seed 42, AdamW and no class weighting. "
        "Both use 51 tokens spanning 5 s; acceleration needs 0.5 s extra raw history. The ensemble has three "
        "20-second variants and a 60-second variant, each with three seeds. Its total parameter count is about "
        "66 times the original transformer's; this is not a measured latency ratio. Initial selection used outer "
        "validation; the new search used only original training groups before a frozen outer comparison.",
    )

    f = page(
        "Read accuracy, class balance and confidence together",
        "Each metric answers a different question; a single ranking can hide important trade-offs.",
    )
    table(
        f,
        [0.065, 0.18, 0.87, 0.54],
        [
            ["Top-1 / top-3 ↑", "Is the true label first / among the three highest probabilities?"],
            [
                "Macro-F1 (33) ↑",
                "How well are labels recovered when every catalogue entry counts equally?",
            ],
            [
                "Supported macro-F1 (26) ↑",
                "The same F1 average, restricted to labels with truth support.",
            ],
            ["Negative log-likelihood ↓", "How much probability is assigned to the actual label?"],
            [
                "Multiclass Brier score ↓",
                "How far is the entire probability vector from the correct one-hot target?",
            ],
            [
                "Calibration error (ECE) ↓",
                "How close is top-label confidence to observed accuracy in fixed bins?",
            ],
        ],
        ["Metric", "Question it answers"],
        [0.31, 0.69],
        16,
    )
    finish(
        f,
        "F1 is the harmonic mean of precision and recall. Undefined F1 is zero; seven absent labels remain "
        "in macro-F1 (33). Supported F1 changes the averaging set, not the vocabulary. NLL uses natural logs and "
        "a 1e-12 probability floor. Brier sums squared errors across 33 classes (range 0 to 2, no division by 33). "
        "ECE is weighted absolute confidence–accuracy difference in ten equal-width top-label confidence bins. "
        "ECE depends on bins and may be small for an uninformative predictor. See comparison.md for definitions.",
    )

    for h in HORIZONS:
        f = page(
            f"+{h} second{'s' if h != 1 else ''}: all four methods on the same origins",
            f"{int(row('ensemble', h)['count']):,} valid origins  |  Higher accuracy / F1, lower NLL are better.",
            "summary_metrics.csv  |  Population all cohort",
        )
        rows = []
        for model in MODELS:
            a = row(model, h)
            rows.append(
                [
                    LABELS[model],
                    f"{a.top1_accuracy * 100:.2f}%",
                    f"{a.top3_accuracy * 100:.2f}%",
                    f"{a.macro_f1_catalogue:.4f}",
                    f"{a.macro_f1_supported:.4f}",
                    f"{a.nll:.4f}",
                ]
            )
        table(
            f,
            [0.065, 0.37, 0.87, 0.34],
            rows,
            ["Model", "Top-1 ↑", "Top-3 ↑", "F1 (33) ↑", "F1 (26) ↑", "NLL ↓"],
            [0.30, 0.14, 0.14, 0.14, 0.14, 0.14],
            18,
            highlight=4,
        )
        a, b = row("transformer", h), row("ensemble", h)
        words(
            f,
            0.065,
            0.27,
            "IMPROVEMENT OVER THE ORIGINAL TRANSFORMER",
            f"Top-1: +{(b.top1_accuracy - a.top1_accuracy) * 100:.2f} points   ·   Top-3: +{(b.top3_accuracy - a.top3_accuracy) * 100:.2f} points\n"
            f"Macro-F1: +{b.macro_f1_catalogue - a.macro_f1_catalogue:.4f}   ·   NLL: {b.nll - a.nll:.4f}",
            size=20,
        )
        finish(
            f,
            "These values reproduce the original saved metrics on matched origins and masks. The frequency "
            "reference uses training-only per-horizon class priors. F1 (33) and F1 (26) have different denominators "
            "and should not be interchanged. The ensemble changes history and compute as well as model design. "
            "This is development validation; the original baselines used it to choose checkpoints.",
        )

    f = page(
        "Percentage-point gain is not relative error reduction",
        "Both improve, but their denominators differ.",
        "Derived from matched top-1 accuracies",
    )
    gain = (values("ensemble", "top1_accuracy") - values("transformer", "top1_accuracy")) * 100
    reduction = gain / (1 - values("transformer", "top1_accuracy"))
    for bounds, v, title in (
        ([0.09, 0.28, 0.35, 0.40], gain, "Accuracy gain (percentage points)"),
        ([0.57, 0.28, 0.35, 0.40], reduction, "Relative top-1 error reduction (%)"),
    ):
        ax = chart(f, bounds)
        bars = ax.bar(range(3), v, color=TEAL, width=0.55)
        ax.bar_label(bars, labels=[f"{x:.2f}" for x in v], padding=7, color=FG, fontsize=19)
        ax.set(
            xticks=range(3),
            xticklabels=["+1 s", "+2 s", "+5 s"],
            ylim=(0, max(v) * 1.27),
            title=title,
        )
    f.text(
        0.065,
        0.13,
        "+5 s: error rate falls from 44.10% to 36.52%; 7.58 ÷ 44.10 = 17.19% relative reduction.",
        fontsize=18,
        color=MUTED,
    )
    finish(
        f,
        "Relative reduction refers specifically to the top-1 error rate, not a 17.19-point gain in accuracy "
        "and not the fraction of all individual errors corrected. Corrections and new errors can coexist; the next "
        "slide reports those paired outcomes explicitly. All calculations use unrounded values.",
    )

    f = page(
        "The ensemble fixes errors and introduces new ones",
        "Paired outcomes on the same forecast origins reveal what the net accuracy gain hides.",
        "paired_changes.csv",
    )
    changes = tables["paired_changes"]
    table(
        f,
        [0.065, 0.35, 0.87, 0.36],
        [
            [
                f"+{int(r.horizon_s)} s",
                f"{int(r['count']):,}",
                f"{int(r.both_correct):,}",
                f"{int(r.corrected):,}",
                f"{int(r.regressed):,}",
                f"{int(r.both_wrong):,}",
            ]
            for _, r in changes.iterrows()
        ],
        ["Horizon", "Origins", "Both correct", "Error corrected", "New error", "Both wrong"],
        [0.11, 0.15, 0.18, 0.19, 0.17, 0.20],
        18,
    )
    words(
        f,
        0.065,
        0.26,
        "AT +5 SECONDS",
        "4,144 corrections − 1,911 regressions = 2,233 additional correct forecasts.\n8,847 origins are still wrong under both models.",
        size=20,
    )
    finish(
        f,
        "Corrected means the original transformer was wrong and the ensemble right; a new error means "
        "the reverse. Both-wrong predictions may choose the same or different incorrect labels. These counts "
        "are correlated forecast windows, not independent encounters. The difference between corrections and "
        "regressions exactly reproduces the top-1 gain numerator at every horizon.",
    )

    f = page(
        "Changed selections gain most; unchanged cases also matter",
        "True current labels define these diagnostic subsets only; they are not inputs.",
        "summary_metrics.csv  |  Endpoint cohorts",
    )
    for bounds, cohort, title in (
        ([0.08, 0.32, 0.36, 0.38], "same", "Same endpoint selection"),
        ([0.57, 0.32, 0.36, 0.38], "changed", "Different endpoint selection"),
    ):
        ax = chart(f, bounds)
        for model in ("gru", "transformer", "ensemble"):
            ax.plot(
                HORIZONS,
                values(model, "top1_accuracy", cohort) * 100,
                "o-",
                color=COLORS[model],
                lw=2.5,
                label=LABELS[model],
            )
        ax.set(
            xticks=HORIZONS,
            ylim=(0, 80),
            title=title,
            ylabel="Top-1 accuracy (%)",
            xlabel="Horizon (s)",
        )
        legend(ax, loc="lower right")
    f.text(
        0.065,
        0.18,
        "+5 s unchanged: 62.44% → 66.87%    |    changed: 40.18% → 55.33%",
        fontsize=20,
        color=TEAL,
    )
    f.text(
        0.065,
        0.11,
        "Unchanged +5 s top-3 falls 91.12% → 90.81%; its NLL worsens 1.0575 → 1.0813.",
        fontsize=17,
        color=GOLD,
    )
    finish(
        f,
        "Changed support is 2,264 / 4,254 / 8,659 and unchanged support is 28,538 / 26,212 / 20,799. "
        "Different horizons have different subsets, so rising changed accuracy does not mean longer horizons "
        "are easier. Endpoint equality does not rule out an intermediate change and return. Full top-1, top-3, "
        "F1 and NLL for each cohort and neural model are in comparison.md. Improvements are not universal.",
    )

    f = page(
        "Every pilot improves; absolute difficulty still varies",
        "+5-second top-1 accuracy for the known pilot population.",
        "summary_metrics.csv  |  Per-pilot all cohort",
    )
    ax = chart(f, [0.12, 0.18, 0.47, 0.54])
    ax.grid(axis="x", alpha=0.16)
    for i in range(10):
        pilot = f"pilot_{i + 1:03}"
        old, new = (
            row("transformer", 5, pilot=pilot).top1_accuracy * 100,
            row("ensemble", 5, pilot=pilot).top1_accuracy * 100,
        )
        ax.plot([old, new], [i, i], color=MUTED, lw=2)
        for model in ("gru", "transformer", "ensemble"):
            ax.scatter(
                row(model, 5, pilot=pilot).top1_accuracy * 100,
                i,
                color=COLORS[model],
                s=58,
                label=LABELS[model] if i == 0 else None,
            )
        ax.text(new + 0.8, i, f"+{new - old:.1f}", va="center", fontsize=13, color=TEAL)
    ax.set(
        yticks=range(10),
        yticklabels=[f"Pilot {i:03}" for i in range(1, 11)],
        xlabel="Top-1 accuracy (%)",
        xlim=(25, 85),
    )
    ax.invert_yaxis()
    legend(ax, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=3)
    words(
        f,
        0.66,
        0.67,
        "GAIN AT EVERY HORIZON",
        "30 of 30 pilot / horizon cells improve.\nAt +5 s, gains range from\n+0.77 to +12.86 points.",
        size=18,
    )
    words(
        f,
        0.66,
        0.39,
        "WHAT THIS DOES NOT MEASURE",
        "Pilot competence or tactical quality.\nGeneralisation to unseen policies.\nSignificance for each pilot separately.",
        GOLD,
        16,
    )
    finish(
        f,
        "Pilot IDs identify evaluation subsets only. The same ten policies appear in training and validation, "
        "under disjoint scenarios. Absolute accuracy and gain vary: for example some already high-accuracy pilots "
        "gain little, while other pilots gain more than ten points. All three-horizon per-pilot tables are portable.",
    )

    classes5 = class_data[class_data.horizon_s.eq(5)]
    new = (
        classes5[classes5.model.eq("ensemble")]
        .set_index("skill")
        .sort_values("support", ascending=False)
    )
    common = new[new.support.ge(500)]
    rare = new[(new.support > 0) & new.support.lt(500)]
    f = page(
        "A small set of labels dominates the validation windows",
        "Class support explains why overall accuracy and macro-F1 tell different stories.",
        "class_metrics.csv  |  +5 s truth support",
    )
    ax = chart(f, [0.23, 0.18, 0.39, 0.54])
    ax.grid(axis="x", alpha=0.16)
    ax.barh(range(len(common)), common.support, color=BLUE)
    ax.set(
        yticks=range(len(common)),
        yticklabels=[s.replace("_", " ") for s in common.index],
        xlabel="True occurrences at +5 s",
    )
    ax.invert_yaxis()
    words(
        f,
        0.68,
        0.66,
        "11 FREQUENT LABELS",
        f"{common.support.sum() / new.support.sum():.2%} of valid windows.\nAt least 500 occurrences each.",
        size=19,
    )
    words(
        f,
        0.68,
        0.42,
        "THE REMAINDER",
        f"{len(rare)} supported labels below 500.\n7 entries with no truth support.\nRare-label estimates are fragile.",
        GOLD,
        17,
    )
    finish(
        f,
        "The 500-occurrence cutoff is a display rule, chosen by support rather than score gains. All 33 labels "
        "remain in evaluation. The five most frequent labels include turn_cold, lag_pursuit, recommit, extend and "
        "pursue_target, which can share guidance implementations. Predictability of private catalogue choices "
        "must not be interpreted as a validated set of distinct manoeuvres.",
    )

    f = page(
        "Frequent labels: compare F1 with the original models",
        "All labels with at least 500 true +5 s occurrences; selected by support, not improvement.",
        "class_metrics.csv",
    )
    ax = chart(f, [0.24, 0.17, 0.64, 0.55])
    ax.grid(axis="x", alpha=0.16)
    for model, offset in (("gru", -0.18), ("transformer", 0), ("ensemble", 0.18)):
        values5 = classes5[classes5.model.eq(model)].set_index("skill").loc[common.index]
        ax.scatter(
            values5.f1,
            np.arange(len(common)) + offset,
            label=LABELS[model],
            color=COLORS[model],
            s=55,
        )
    ax.set(
        yticks=range(len(common)),
        yticklabels=[
            f"{s.replace('_', ' ')}  (n={int(common.loc[s, 'support']):,})" for s in common.index
        ],
        xlim=(0, 1),
        xlabel="Per-label F1",
    )
    ax.invert_yaxis()
    legend(ax, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=3)
    finish(
        f,
        "The improved ensemble raises +5 s F1 for every frequent class compared with the original transformer. "
        "This is a descriptive point-estimate comparison, not a significance claim for every label. Precision, "
        "recall and F1 for all 33 entries are tabulated in comparison.md; CSVs include all models and horizons.",
    )

    f = page(
        "Sparse labels: show the gains alongside the small counts",
        "Every supported +5 s label below 500 occurrences. Do not infer reliable rare-class generalisation.",
        "class_metrics.csv",
    )
    old = classes5[classes5.model.eq("transformer")].set_index("skill")
    rows = [
        [
            s.replace("_", " "),
            str(int(r.support)),
            f"{old.loc[s, 'f1']:.3f}",
            f"{r.f1:.3f}",
            f"{r.precision:.3f}",
            f"{r.recall:.3f}",
        ]
        for s, r in rare.iterrows()
    ]
    table(
        f,
        [0.065, 0.18, 0.87, 0.55],
        rows,
        ["Label", "Support", "Old F1", "New F1", "New precision", "New recall"],
        [0.33, 0.12, 0.12, 0.12, 0.16, 0.15],
        13,
        highlight=None,
    )
    f.text(
        0.065,
        0.10,
        "Three labels with only 1–2 true occurrences still have zero F1; seven further entries have zero support.",
        fontsize=16,
        color=GOLD,
    )
    finish(
        f,
        "The smallest samples cannot establish stable accuracy. Beam_missile_left has six occurrences; "
        "support_missile nineteen. A larger F1 change there is not stronger evidence than a smaller change on "
        "thousands of examples. Precision and recall help expose trade-offs such as beam_missile_right's low "
        "precision despite higher recall. No labels were merged or removed for scoring.",
    )

    f = page(
        "Confusion matrices reveal the remaining label mixtures",
        "+5 s, row-normalized over all predictions. The eight most frequent true labels are shown individually.",
        "confusions.json",
    )
    names = evidence["classes"]
    top = list(new.index[:8])
    groups = [[names.index(s)] for s in top]
    groups.append([i for i, s in enumerate(names) if s not in top])
    short = [
        "turn cold",
        "lag pursuit",
        "recommit",
        "extend",
        "pursue",
        "beam left",
        "beam right",
        "second shot",
        "Other 25",
    ]
    for bounds, model, title in (
        ([0.15, 0.29, 0.32, 0.40], "transformer", "Original transformer"),
        ([0.62, 0.29, 0.32, 0.40], "ensemble", "Improved ensemble"),
    ):
        ax = f.add_axes(bounds)
        matrix = matrices[f"{model}_5"]
        grouped = np.array([[matrix[np.ix_(a, b)].sum() for b in groups] for a in groups])
        norm = grouped / grouped.sum(1, keepdims=True)
        ax.imshow(norm, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set(
            xticks=range(9),
            xticklabels=short,
            yticks=range(9),
            yticklabels=short,
            title=title,
            xlabel="Predicted label group",
            ylabel="True label group",
        )
        ax.tick_params(axis="x", labelrotation=55, labelsize=10)
        ax.tick_params(axis="y", labelsize=11)
        for i in range(9):
            ax.text(
                i,
                i,
                f"{norm[i, i] * 100:.0f}%",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if norm[i, i] > 0.55 else BG,
            )
    f.text(
        0.065,
        0.10,
        "Darker = larger share of a true-label row. “Other 25” is for display only; scores retain all 33 labels.",
        fontsize=16,
        color=MUTED,
    )
    finish(
        f,
        "The top eight labels are chosen by pooled truth support. All other true and predicted classes are "
        "aggregated into Other 25, retaining all counts in each denominator. Its diagonal includes confusions "
        "between different labels inside that group and is not exact-label recall. Individual displayed-class "
        "diagonals are recall. The full ungrouped count matrices are included for all methods and horizons.",
    )

    f = page(
        "The ensemble improves probability scores over both networks",
        "Brier measures the entire distribution; ECE measures binned top-label calibration.",
        "probability_metrics.csv",
    )
    probs = tables["probability_metrics"]
    for bounds, field, title, scale in (
        ([0.08, 0.30, 0.36, 0.39], "brier", "Multiclass Brier ↓", 1),
        ([0.57, 0.30, 0.36, 0.39], "ece_10_bins", "Calibration error (percentage points) ↓", 100),
    ):
        ax = chart(f, bounds)
        for model in MODELS:
            a = probs[probs.model.eq(model)].sort_values("horizon_s")
            ax.plot(
                a.horizon_s,
                a[field] * scale,
                "o-",
                lw=2.5,
                label=LABELS[model],
                color=COLORS[model],
            )
        ax.set(
            xticks=HORIZONS,
            xlabel="Forecast horizon (s)",
            title=title,
            ylim=(0, 1 if scale == 1 else 11),
        )
    handles, labels = ax.get_legend_handles_labels()
    f.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.17),
        ncol=4,
        facecolor=BG,
        edgecolor=MUTED,
        labelcolor=FG,
        fontsize=12,
    )
    f.text(
        0.065,
        0.10,
        "Low ECE alone is not enough: the frequency reference has low ECE but poor accuracy.",
        fontsize=18,
        color=GOLD,
    )
    finish(
        f,
        "At +5 s, transformer to ensemble Brier changes 0.5830 to 0.4916 and ECE changes 8.74 to 3.42 points. "
        "The frequency baseline has ECE around 0.42 points because its low constant confidence nearly matches "
        "its low hit rate, despite much worse accuracy, NLL and Brier. ECE depends on binning and finite support; "
        "it should never be used alone to rank these forecasters. No validation temperature was fitted here.",
    )

    f = page(
        "Confidence is closer to observed accuracy after improvement",
        "+5 s reliability diagram using ten fixed confidence bins; marker size reflects bin support.",
        "reliability.csv",
    )
    ax = chart(f, [0.10, 0.20, 0.46, 0.51])
    ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1.5, label="Confidence = accuracy")
    for model in ("gru", "transformer", "ensemble"):
        bins = tables["reliability"]
        bins = bins[bins.model.eq(model) & bins.horizon_s.eq(5) & bins["count"].gt(0)]
        ax.plot(bins.mean_confidence, bins.accuracy, color=COLORS[model], lw=2, label=LABELS[model])
        ax.scatter(
            bins.mean_confidence,
            bins.accuracy,
            s=15 + bins["count"] / 100,
            color=COLORS[model],
            alpha=0.9,
        )
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean top-label confidence",
        ylabel="Observed top-1 accuracy",
    )
    legend(ax, loc="upper left")
    words(
        f,
        0.64,
        0.66,
        "AT +5 SECONDS",
        "Original transformer\nMean confidence 64.64% / accuracy 55.90%\n\nImproved ensemble\nMean confidence 60.83% / accuracy 63.48%",
        size=17,
    )
    words(
        f,
        0.64,
        0.29,
        "INTERPRETATION",
        "Less overconfidence in aggregate.\nCalibration still varies by confidence bin.",
        GOLD,
        17,
    )
    finish(
        f,
        "Confidence is the maximum predicted class probability. Accuracy is averaged within each fixed bin, "
        "not across classes. Zero-count bins are omitted from the plot and retained with count zero in the CSV. "
        "Marker area is a monotone function of support. Sparse-bin points are descriptive, without independent "
        "bin confidence intervals. The final ensemble uses temperatures fixed from internal stop groups.",
    )

    f = page(
        "The development search also records unsuccessful changes",
        "First internal screen: mean top-1 across horizons. These are not the outer-validation scores.",
        "initial_search.csv  |  Internal split 0",
    )
    screen = tables["initial_search"].query("fold == 0").sort_values("mean_accuracy")
    labels = {
        "control": "Original encoder control",
        "geometry_mlp": "Geometry MLP, 5 s",
        "geometry": "Geometry transformer",
        "history20_aux": "20 s transformer + auxiliary",
        "history20_mlp": "20 s MLP",
        "history20_mlp_wide": "Wider 20 s MLP",
        "history20": "20 s transformer",
        "history20_wide": "Wider 20 s transformer",
        "regularized": "Regularized transformer",
    }
    ax = chart(f, [0.27, 0.20, 0.37, 0.52])
    ax.grid(axis="x", alpha=0.16)
    bars = ax.barh(
        range(len(screen)),
        screen.mean_accuracy * 100,
        color=[
            TEAL if n == "history20_mlp" else BLUE if n == "control" else GREY for n in screen.name
        ],
    )
    ax.bar_label(
        bars,
        labels=[f"{v * 100:.2f}%" for v in screen.mean_accuracy],
        padding=5,
        color=FG,
        fontsize=12,
    )
    ax.set(
        yticks=range(len(screen)),
        yticklabels=[labels[n] for n in screen.name],
        xlim=(0, 65),
        xlabel="Internal mean top-1 (%)",
    )
    words(
        f,
        0.70,
        0.66,
        "WHAT THE SCREEN SAYS",
        "More geometry or a wider\ntransformer did not reliably help.\nThe 20 s MLP warranted follow-up.",
        size=17,
    )
    words(
        f,
        0.70,
        0.38,
        "WHAT IT DOES NOT SAY",
        "An isolated causal contribution\nfor every change. Settings and\ntraining procedures differ.",
        GOLD,
        16,
    )
    finish(
        f,
        "The first split fitted 147 groups, stopped on 21 and compared on 42. Two-split confirmation improved "
        "mean top-1 from 51.63% for control to 54.75% for the NLL-selected 20 s MLP. A targeted follow-up explored "
        "accuracy-based stopping, smoothing and 60-second history. Individual accuracy-selected models failed "
        "the NLL guard; fixed mixtures improved accuracy and NLL. All stages were internal and adaptive, not "
        "independent replications. Detailed original and mixture scores remain in comparison.md.",
    )

    f = page(
        "The final gain comes with more history and computation",
        "Freeze the selected procedure before evaluating outer validation.",
        "Frozen selection + original run metadata",
    )
    table(
        f,
        [0.065, 0.40, 0.87, 0.31],
        [
            [
                "Fit / stop / select groups",
                "147 / 21 / 42, inside the original 210 training groups",
            ],
            ["Confirmation", "Two internal partitions; NLL and macro-F1 eligibility guards"],
            ["Final refit", "Four variants × seeds 42, 43, 44; fixed epochs and temperatures"],
            ["Ensemble weights", "Uniform arithmetic probability average across 12 members"],
        ],
        ["Step", "Protocol"],
        [0.28, 0.72],
        16,
    )
    words(
        f,
        0.065,
        0.29,
        "ORIGINAL TRANSFORMER → ENSEMBLE",
        "5 s → up to 60 s history\n0.44M → 28.98M total parameters",
        BLUE,
        18,
    )
    words(
        f,
        0.56,
        0.29,
        "A SMALLER ALTERNATIVE IS STILL PLAUSIBLE",
        "Internal accuracy: 60.96% vs 60.98%.\nSix versus twelve models was not decisive.",
        GOLD,
        18,
    )
    finish(
        f,
        "Three 20-second variants use widths 256, 512 and 256; the 60-second variant uses 256. Final epochs "
        "are 46, 32, 40, 40. No outer validation was used for their epoch count, temperature, weights or seeds. "
        "Total checkpoints are approximately 116 MB. A local warmed CPU check measured median 20.3 ms, four "
        "threads, 30 calls; no comparable original-model latency measurement is available. The source audit "
        "does not establish zero overfitting, and near-equal internal alternatives should not be overinterpreted.",
    )

    f = page(
        "The gain remains positive within scenario-level intervals",
        "Resample whole scenarios, preserving all ten pilots and overlapping windows.",
        "scenario_intervals.csv",
    )
    ci = tables["scenario_intervals"]
    ax = chart(f, [0.09, 0.23, 0.46, 0.48])
    ax.errorbar(
        ci.horizon_s,
        ci.accuracy_gain * 100,
        yerr=np.array([ci.accuracy_gain - ci.lower_95, ci.upper_95 - ci.accuracy_gain]) * 100,
        fmt="o",
        color=TEAL,
        capsize=7,
        lw=3,
        markersize=9,
    )
    ax.axhline(0, color=GREY)
    ax.set(
        xticks=HORIZONS,
        ylim=(-0.4, 10),
        xlabel="Horizon (s)",
        ylabel="Top-1 gain (percentage points)",
    )
    words(
        f,
        0.62,
        0.67,
        "PAIRED 95% INTERVALS",
        "+1 s: +6.65 points  [5.60, 7.84]\n+2 s: +6.79 points  [5.69, 8.04]\n+5 s: +7.58 points  [6.39, 8.95]",
        size=19,
    )
    words(
        f,
        0.62,
        0.40,
        "THE EVIDENCE LIMIT",
        "45 scenario groups; 5,000 draws.\nOriginal validation already informed baselines.\nReserved test remains unevaluated.",
        GOLD,
        17,
    )
    finish(
        f,
        "Bootstrap intervals describe the paired population accuracy difference and retain window weighting. "
        "They do not correct for the entire historical model search. New Brier/ECE and class comparisons are "
        "descriptive analyses of fixed saved predictions; they are not a fresh held-out confirmation. No interval "
        "is claimed for each individual class, pilot or confidence bin.",
    )

    f = page(
        "A stronger forecaster, with specific remaining limitations",
        "Use the complete comparison to judge the gain—not only the headline accuracy.",
    )
    words(
        f,
        0.065,
        0.69,
        "WHAT IMPROVED",
        "Top-1 and top-3 at every horizon.\nCatalogue macro-F1 and probability scores.\nAll ten pilots; changed-selection accuracy.\nNet corrections exceed new errors.",
        size=19,
    )
    words(
        f,
        0.56,
        0.69,
        "WHAT REMAINS UNRESOLVED",
        "Rare and absent labels.\nSome unchanged-subset metrics worsen.\nCompute cost and minimal ensemble size.\nUnseen pilots and realistic sensor inputs.",
        GOLD,
        19,
    )
    f.text(
        0.065,
        0.24,
        "Next: freeze the comparison, evaluate the reserved test once, and report all metrics.",
        fontsize=21,
        color=BLUE,
        weight="bold",
    )
    f.text(
        0.065,
        0.14,
        "Complete scorecards, every catalogue entry and reproducible evidence accompany this deck.",
        fontsize=18,
        color=MUTED,
    )
    finish(
        f,
        "No new model fitting, model selection, simulation, inference or test evaluation was performed for "
        "this presentation. Additional diagnostics reuse fixed saved probabilities. Original experiments, "
        "training artifacts and earlier presentations remain preserved. Future work is proposed, not completed.",
    )

    export(
        dest,
        slides,
        evidence,
        stem="skills_catalogue",
        deck_title=TITLE,
        short_title="Skills catalogue comparison",
        check_name="skills_catalogue",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Audit saved local predictions and package derived evidence.",
    )
    parser.add_argument("--output", type=Path, default=DEST)
    args = parser.parse_args()
    if args.collect:
        collect(args.output)
    render(args.output)


if __name__ == "__main__":
    main()
