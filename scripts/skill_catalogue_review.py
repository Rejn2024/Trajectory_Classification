"""Deeper descriptive review of frozen skill forecasts; never fit or read test targets."""

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from build_forecasting_insights import BG, BLUE, FG, GOLD, MUTED, TEAL
from skill_catalogue_evidence import diagnostic_metrics, markdown_table, read_json, sha, write_json

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (1, 2, 5)
METHOD_SOURCES = [
    (
        "Selection bias",
        "Cawley & Talbot (2010)",
        "https://www.jmlr.org/beta/papers/v11/cawley10a.html",
    ),
    (
        "Dependence-aware evaluation",
        "Roberts et al. (2017)",
        "https://nsojournals.onlinelibrary.wiley.com/doi/10.1111/ecog.02881",
    ),
    (
        "Cluster bootstrap",
        "Field & Welsh (2007)",
        "https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-9868.2007.00593.x",
    ),
    ("Calibration", "Guo et al. (2017)", "https://proceedings.mlr.press/v70/guo17a.html"),
    ("Supplementary segment metrics", "Lea et al. (2017)", "https://arxiv.org/abs/1611.05267"),
]


def paired_group_intervals(old_correct, new_correct, groups, draws=5000, seed=20261103):
    """Resample whole groups; estimate the original pooled-window accuracy ratio."""
    old, new = np.asarray(old_correct, dtype=float), np.asarray(new_correct, dtype=float)
    groups = np.asarray(groups)
    if old.ndim != 1 or new.shape != old.shape or groups.shape != old.shape or not len(old):
        raise ValueError("Require nonempty aligned correctness vectors and group IDs")
    if not np.isin(old, [0, 1]).all() or not np.isin(new, [0, 1]).all():
        raise ValueError("Correctness must be binary")
    unique, inverse = np.unique(groups, return_inverse=True)
    count = np.bincount(inverse)
    old_count = np.bincount(inverse, weights=old)
    new_count = np.bincount(inverse, weights=new)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(unique), size=(draws, len(unique)))
    denominator = count[indices].sum(1)
    gain = (new_count - old_count)[indices].sum(1) / denominator
    accuracy = new_count[indices].sum(1) / denominator
    return {
        "scenario_groups": len(unique),
        "gain": float(new.mean() - old.mean()),
        "gain_lower": float(np.quantile(gain, 0.025)),
        "gain_upper": float(np.quantile(gain, 0.975)),
        "accuracy_lower": float(np.quantile(accuracy, 0.025)),
        "accuracy_upper": float(np.quantile(accuracy, 0.975)),
    }


def accuracy_weightings(correct, episode_ids, pilot_ids):
    """Three estimands: equal origins, equal eligible episodes, equal eligible pilots."""
    frame = pd.DataFrame({"correct": correct, "episode": episode_ids, "pilot": pilot_ids})
    if frame.empty:
        raise ValueError("No eligible origins")
    return {
        "pooled_windows": float(frame.correct.mean()),
        "equal_episodes": float(frame.groupby("episode").correct.mean().mean()),
        "equal_pilots": float(frame.groupby("pilot").correct.mean().mean()),
        "episodes": frame.episode.nunique(),
        "pilots": frame.pilot.nunique(),
    }


def collect_review(dest):
    from skill_catalogue_architecture import collect_architecture
    from skill_catalogue_temporal import collect_temporal

    data = dest / "data"
    for name, expected in read_json(data / "checksums.json").items():
        assert sha(data / name) == expected, name
    sources = {}

    def record(path):
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
        return path

    base = ROOT / "artifacts/skill_forecasting"
    prepared = base / "data_v1"
    samples = pd.read_parquet(record(prepared / "samples.parquet"))
    assert set(samples.split.unique()) == {"train", "validation"}
    valid = samples.split.eq("validation").to_numpy()
    targets = np.load(record(prepared / "targets.npy"))[valid]
    current = np.load(record(prepared / "current.npy"))[valid]
    samples = samples[valid].reset_index(drop=True)
    classes = read_json(record(prepared / "classes.json"))
    assert len(samples) == 30802 and samples.scenario_group.nunique() == 45
    directories = {
        "transformer": base / "initial_v1/transformer",
        "ensemble": base / "improvement_v3/validation/ensemble",
    }
    probabilities, skill_frames = {}, []
    for model, path in directories.items():
        pd.testing.assert_frame_equal(
            samples, pd.read_parquet(record(path / "validation_samples.parquet"))
        )
        p = np.load(record(path / "validation_probabilities.npy"))
        assert p.shape == (len(samples), 3, 33)
        np.testing.assert_allclose(p.sum(-1), 1, atol=1e-6)
        probabilities[model] = p
        skill_frames.append(pd.read_csv(record(path / "per_class_metrics.csv")).assign(model=model))
    pd.concat(skill_frames).to_csv(
        data / "pilot_skill_metrics.csv", index=False, lineterminator="\n"
    )

    pilot_rows, weighting_rows, support_rows = [], [], []
    for j, horizon in enumerate(HORIZONS):
        available = targets[:, j] >= 0
        for model, p in probabilities.items():
            correct = p[available, j].argmax(1) == targets[available, j]
            weighting_rows.append(
                {
                    "model": model,
                    "horizon_s": horizon,
                    "count": int(available.sum()),
                    **accuracy_weightings(
                        correct,
                        samples.loc[available, "episode_id"],
                        samples.loc[available, "pilot_id"],
                    ),
                }
            )
        for pilot in sorted(samples.pilot_id.unique()):
            mask = available & samples.pilot_id.eq(pilot).to_numpy()
            y = targets[mask, j]
            old = probabilities["transformer"][mask, j].argmax(1) == y
            new = probabilities["ensemble"][mask, j].argmax(1) == y
            score, _ = diagnostic_metrics(probabilities["ensemble"][mask, j], y)
            counts = np.bincount(y, minlength=33)
            frequencies = counts[counts > 0] / len(y)
            known_current = current[mask] >= 0
            pilot_rows.append(
                {
                    "pilot_id": pilot,
                    "horizon_s": horizon,
                    "count": len(y),
                    "episodes": samples.loc[mask, "episode_id"].nunique(),
                    "old_top1": float(old.mean()),
                    "new_top1": float(new.mean()),
                    **paired_group_intervals(
                        old, new, samples.loc[mask, "scenario_group"].to_numpy()
                    ),
                    "dominant_skill": classes[int(counts.argmax())],
                    "dominant_share": float(counts.max() / len(y)),
                    "supported_classes": int((counts > 0).sum()),
                    "label_entropy_bits": float(-np.sum(frequencies * np.log2(frequencies))),
                    "endpoint_change_rate": float(
                        (y[known_current] != current[mask][known_current]).mean()
                    ),
                    "nll": score["nll"],
                    "brier": score["brier"],
                    "ece": score["ece_10_bins"],
                    "mean_confidence": score["mean_confidence"],
                }
            )
        for c, skill in enumerate(classes):
            mask = available & (targets[:, j] == c)
            part = samples[mask]
            support_rows.append(
                {
                    "horizon_s": horizon,
                    "skill": skill,
                    "support": len(part),
                    "episodes": part.episode_id.nunique(),
                    "scenario_groups": part.scenario_group.nunique(),
                    "pilots": part.pilot_id.nunique(),
                }
            )
    for filename, rows in (
        ("pilot_review.csv", pilot_rows),
        ("accuracy_weightings.csv", weighting_rows),
        ("skill_support_context.csv", support_rows),
    ):
        pd.DataFrame(rows).to_csv(data / filename, index=False, lineterminator="\n")

    # Preserve the original metrics and source code as evidence, without changing labels.
    record(ROOT / "bvr_sim_source/bvr_sim/agents/skill_manager.py")
    record(ROOT / "bvr_behavior_prediction/skill_forecasting/evaluation.py")
    record(ROOT / "bvr_behavior_prediction/skill_forecasting/improved_features.py")
    collect_architecture(dest)
    collect_temporal(dest)
    write_json(
        data / "review_evidence.json",
        {
            "review_date": "2026-11-03",
            "source_sha256": sources,
            "bootstrap": {
                "unit": "scenario_group",
                "draws": 5000,
                "seed": 20261103,
                "interval": "pointwise percentile 95%; no multiple-comparison or selection-bias correction",
            },
            "contract": "Descriptive reuse of fixed saved predictions and permitted validation decisions; no model fitting or selection.",
            "test_evaluated": False,
            "inference_performed": False,
            "method_sources": [
                {"topic": topic, "title": title, "url": url} for topic, title, url in METHOD_SOURCES
            ],
        },
    )
    write_json(
        data / "checksums.json",
        {
            p.name: sha(p)
            for p in sorted(data.iterdir())
            if p.is_file() and p.name != "checksums.json"
        },
    )
    print("Audited pilot, skill, temporal and training-cost review evidence; test excluded.")


def append_review_intro(dest, page, finish, words, chart, table):
    f = page(
        "Two learning problems, two sets of networks",
        "The forecaster learns the behaviour of frozen trained pilots; it does not control the aircraft",
        "Recorded pipeline and corpus configuration",
    )
    words(
        f,
        0.065,
        0.70,
        "1  TRAIN THE PILOTS",
        "PPO actor–critic learns flight decisions.\n10 independent reward profiles.\n150 × 75 × 10 = 112,500 training flights\nper complete population training stage.",
        BLUE,
        17,
    )
    words(
        f,
        0.54,
        0.70,
        "2  RECORD FROZEN PILOTS",
        "Select a checkpoint for each pilot.\n300 new scenarios × 10 pilots = 3,000 encounters.\nB01 observes; A01 selects catalogue skills.\nRecord observations and selected labels.",
        TEAL,
        17,
    )
    words(
        f,
        0.065,
        0.32,
        "3  TRAIN THE FORECASTER",
        "Supervised cross-entropy; no PPO or critic.\nOne shared model learns across all ten pilots.\nInputs stop at t; labels are at t+1, t+2, t+5.",
        GOLD,
        17,
    )
    words(
        f,
        0.54,
        0.32,
        "WHAT THE SCORE MEANS",
        "Predictability of the selected skill label.\nIt is not an RL reward or combat-success score.\nCurrent evidence: new scenarios, known pilots.",
        TEAL,
        17,
    )
    finish(
        f,
        (
            "The 112,500 figure is training flights for one complete ten-pilot, 150-epoch stage, not the "
            "total project simulation count and not the size of the forecasting corpus. Selected frozen "
            "policies subsequently generated 3,000 encounters, using 300 shared starting scenarios. "
            "The forecasting split is 210/45/45 scenario groups, equivalent to 2,100/450/450 episodes. "
            "Only training and validation forecast targets are prepared. The 146,690 training origins "
            "and 30,802 validation origins are overlapping windows, not independent flights. The pilot "
            "actor's transformer and the forecasting model are different trained networks with different "
            "inputs, objectives and outputs. Skill forecasting estimates an exact symbolic policy choice; "
            "it does not directly assess tactical competence, manoeuvre quality or survival."
        ),
    )


def append_review_slides(dest, page, finish, words, chart, table):
    import matplotlib.pyplot as plt

    data = dest / "data"
    pilots = pd.read_csv(data / "pilot_review.csv")
    skills = pd.read_csv(data / "pilot_skill_metrics.csv")
    support = pd.read_csv(data / "skill_support_context.csv")
    pilot_ids = sorted(pilots.pilot_id.unique())
    short_ids = [p.removeprefix("pilot_") for p in pilot_ids]
    p5 = pilots[pilots.horizon_s.eq(5)].set_index("pilot_id").loc[pilot_ids]

    f = page(
        "Every pilot: forecast accuracy and uncertainty",
        "Ensemble accuracy at all three horizons; paired +5 s gain against the original transformer",
    )
    ax = f.add_axes([0.095, 0.21, 0.36, 0.52], facecolor=BG)
    grid = (
        pilots.pivot(index="pilot_id", columns="horizon_s", values="new_top1")
        .loc[pilot_ids, list(HORIZONS)]
        .to_numpy()
        * 100
    )
    ax.imshow(grid, aspect="auto", cmap="YlGnBu", vmin=35, vmax=85)
    ax.set_xticks(range(3), ["+1 s", "+2 s", "+5 s"])
    ax.set_yticks(range(10), short_ids)
    ax.set_ylabel("Target pilot")
    for i in range(10):
        for j in range(3):
            ax.text(
                j,
                i,
                f"{grid[i, j]:.1f}%",
                ha="center",
                va="center",
                color="white" if grid[i, j] > 63 else "#101e30",
                fontsize=14,
            )
    ax = chart(f, [0.58, 0.21, 0.34, 0.52])
    y = np.arange(10)
    ax.hlines(y, p5.gain_lower * 100, p5.gain_upper * 100, color=TEAL, linewidth=2)
    ax.scatter(p5.gain * 100, y, color=GOLD, s=40, zorder=3)
    ax.axvline(0, color=MUTED, linestyle="--", linewidth=1)
    ax.set_yticks(y, short_ids)
    ax.invert_yaxis()
    ax.set_xlabel("+5 s gain (percentage points)")
    ax.set_title("Paired scenario-bootstrap 95% intervals", color=FG, fontsize=13)
    f.text(
        0.065,
        0.10,
        "All point estimates improve; uncertainty differs by pilot. Each pilot has 45 validation encounters.",
        fontsize=16,
        color=GOLD,
    )
    finish(
        f,
        (
            "Left: top-1 accuracy using each horizon's available labels, not a common-origin subset. "
            "Right: paired difference between ensemble and original-transformer correctness on the same "
            "+5-second origins. Each pilot's interval resamples its 45 scenario groups 5,000 times, "
            "keeping all overlapping windows within an encounter together. These are pointwise descriptive "
            "95% percentile intervals, not simultaneous intervals across ten pilots or correction for "
            "development selection. Do not claim a statistically established improvement for every pilot "
            "simply because all point estimates are positive. Supports and all three horizon intervals "
            "are published in pilot_review.csv. These pilots are present in training; this is not a "
            "held-out-pilot generalisation assessment."
        ),
    )

    f = page(
        "Pilots differ in skill mix and forecast confidence",
        "A high forecasting score is not a rating of pilot competence | +5 s validation",
    )
    ax = chart(f, [0.08, 0.30, 0.37, 0.40])
    ax.scatter(
        p5.dominant_share * 100,
        p5.new_top1 * 100,
        color=TEAL,
        s=95,
        edgecolor=FG,
    )
    for pid, row in p5.iterrows():
        ax.annotate(
            pid[-3:],
            (row.dominant_share * 100, row.new_top1 * 100),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=11,
        )
    ax.set_xlabel("Most common true skill (% of origins)", fontsize=12)
    ax.set_ylabel("Forecast accuracy (%)", fontsize=12)
    ax.set_title("Different skill mixtures", color=FG, fontsize=17)
    ax = chart(f, [0.57, 0.30, 0.35, 0.40])
    ax.plot([40, 85], [40, 85], color=MUTED, linestyle="--")
    ax.scatter(p5.mean_confidence * 100, p5.new_top1 * 100, color=TEAL, s=75)
    for pid, row in p5.iterrows():
        ax.annotate(
            pid[-3:],
            (row.mean_confidence * 100, row.new_top1 * 100),
            xytext={
                "pilot_004": (-24, 8),
                "pilot_009": (7, 12),
                "pilot_010": (9, -8),
                "pilot_006": (4, -12),
            }.get(pid, (5, 5)),
            textcoords="offset points",
            fontsize=11,
        )
    ax.set_xlabel("Mean top-label confidence (%)", fontsize=12)
    ax.set_ylabel("Forecast accuracy (%)", fontsize=12)
    ax.set_title("Calibration also varies by pilot", color=FG, fontsize=17)
    f.text(
        0.065,
        0.18,
        f"Per-pilot ECE ranges from {p5.ece.min() * 100:.2f} to {p5.ece.max() * 100:.2f} points; pooled ECE alone hides this variation.",
        fontsize=18,
        color=GOLD,
    )
    f.text(
        0.065,
        0.11,
        "Compare label support, change rate and F1 alongside accuracy. These descriptive relationships do not establish causation.",
        fontsize=14,
        color=MUTED,
    )
    finish(
        f,
        (
            "Each point is one of the ten known RL pilots. Dominant-label fraction is computed from "
            "validation truth to describe the label mixture; it is not a fitted baseline or inference "
            "input. Different rewards, episode durations and policy behaviours may affect forecast "
            "difficulty, but this plot does not identify their causal contribution. The "
            "right diagonal denotes equality of aggregate confidence and accuracy, not perfect "
            "calibration at every confidence level. Below it means aggregate overconfidence; above "
            "means underconfidence. ECE uses ten fixed bins separately for each pilot, so pooled ECE "
            "can conceal opposing errors. Per-pilot bin counts and uncertainty would be needed before "
            "considering a dedicated calibration intervention. Pilot identity is not a model input."
        ),
    )

    f = page(
        "Which skills can we forecast for each pilot?",
        "Recall at +5 s for the eleven most-supported labels | Cell text: percentage; dash: no truth support",
    )
    population = skills[
        skills.model.eq("ensemble") & skills.pilot_id.eq("population") & skills.horizon_s.eq(5)
    ]
    common = (
        population[population.support.ge(500)]
        .sort_values("support", ascending=False)
        .skill.tolist()
    )
    frame = skills[
        skills.model.eq("ensemble") & skills.pilot_id.ne("population") & skills.horizon_s.eq(5)
    ]
    recall = (
        frame.pivot(index="pilot_id", columns="skill", values="recall")
        .loc[pilot_ids, common]
        .to_numpy()
    )
    counts = (
        frame.pivot(index="pilot_id", columns="skill", values="support")
        .loc[pilot_ids, common]
        .to_numpy()
    )
    values = np.ma.masked_where(counts == 0, recall * 100)
    ax = f.add_axes([0.10, 0.31, 0.82, 0.42], facecolor=BG)
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#35485e")
    ax.imshow(values, aspect="auto", cmap=cmap, vmin=0, vmax=100)
    ax.set_xticks(
        range(len(common)),
        [s.replace("_", " ") for s in common],
        rotation=35,
        ha="right",
        fontsize=10,
    )
    ax.set_yticks(range(10), short_ids)
    for i in range(10):
        for j in range(len(common)):
            label = (
                "–"
                if counts[i, j] == 0
                else f"{recall[i, j] * 100:.0f}" + ("*" if counts[i, j] < 50 else "")
            )
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=10,
                color="white" if counts[i, j] == 0 or recall[i, j] > 0.6 else "#101e30",
            )
    f.text(
        0.065,
        0.12,
        "* Fewer than 50 true origins in that pilot/skill cell. Full precision, recall, F1 and support are in the companion tables.",
        fontsize=14,
        color=GOLD,
    )
    finish(
        f,
        (
            "Recall is the fraction of true occurrences of this skill that receive the correct label. "
            "It is not one-versus-rest accuracy, which would be dominated by true negatives for rare "
            "skills. A dash means zero truth support and does not mean measured failure or success. "
            "An asterisk marks fewer than 50 true forecast origins; 50 is a visibility flag, not an "
            "adequacy guarantee, because many origins belong to the same flight or label run. The "
            "eleven columns are selected solely by pooled truth support of at least 500, matching "
            "slide 13. All 33 labels for both models and all horizons are published, including "
            "predicted counts and precision. Equal supported-class macro-F1 across pilots averages "
            "different class sets; retain the fixed-33 result and supports as well."
        ),
    )

    f = page(
        "Per-skill results need precision, recall and support",
        "The same +5 s ensemble predictions can look different under each metric",
    )
    chosen = population.set_index("skill").loc[common]
    context = support[support.horizon_s.eq(5)].set_index("skill")
    rows = [
        [
            s.replace("_", " "),
            f"{int(r.support):,}",
            int(context.loc[s, "episodes"]),
            f"{r.precision:.3f}",
            f"{r.recall:.3f}",
            f"{r.f1:.3f}",
        ]
        for s, r in chosen.iterrows()
    ]
    table(
        f,
        [0.065, 0.25, 0.87, 0.48],
        rows,
        ["True skill", "Origins", "Encounters", "Precision", "Recall", "F1"],
        widths=[0.32, 0.13, 0.13, 0.14, 0.14, 0.14],
        size=14,
    )
    words(
        f,
        0.065,
        0.17,
        "ASK THREE QUESTIONS",
        "Does it miss this skill? → recall     Does it over-predict it? → precision     Are there enough independent cases? → support",
        GOLD,
        13,
    )
    finish(
        f,
        (
            "Precision divides correct skill predictions by all predictions of that skill. Recall "
            "divides correct skill predictions by all true occurrences. F1 is their harmonic mean. "
            "Origin support and encounter support answer different questions: many consecutive windows "
            "in one encounter can repeat the same label. skill_support_context.csv also supplies "
            "scenario-group and pilot counts. Frequent-label rows are a readable extract; the existing "
            "rare-label slide and complete tables retain all labels. For zero truth support, a stored "
            "zero F1 is a fixed-catalogue scoring convention rather than evidence that recall was "
            "measured. Never advertise high one-versus-rest accuracy as rare-skill recognition."
        ),
    )

    # Temporal evidence is produced by the validation-only row-group collector.
    append_temporal_review_slides(dest, page, finish, words, chart, table)

    f = page(
        "Exact catalogue labels can share visible guidance",
        "The task predicts the selected symbolic label; visible motion may not uniquely identify it",
        "Inspected TacticalSkill.execute and saved label contract",
    )
    table(
        f,
        [0.065, 0.40, 0.87, 0.30],
        [
            [
                "lag pursuit / extend / turn cold / recommit / pursue target",
                "Can emit the same zero-change guidance fields",
            ],
            ["beam target left / crank target left", "Both emit a −90° heading-change command"],
            [
                "launch / secondary shot / short range attack",
                "All set the shoot flag; selected labels remain distinct",
            ],
        ],
        ["Catalogue examples", "What the inspected execution code has in common"],
        widths=[0.54, 0.46],
        size=14,
    )
    words(
        f,
        0.065,
        0.29,
        "INTERPRETATION",
        "A label error can occur even when predicted physical behaviour is similar.\nThis is an observability question, not proof of a known accuracy ceiling.",
        GOLD,
        18,
    )
    f.text(
        0.065,
        0.10,
        "Keep exact-label scores. Any behaviour-family score needs a separately justified, frozen mapping.",
        fontsize=17,
        color=TEAL,
    )
    finish(
        f,
        (
            "The named cases share emitted guidance components in the inspected implementation; they "
            "are not a claim that entire trajectories, parameters, lifecycle semantics or tactical "
            "consequences are identical. The original corpus records exact selections, which remain "
            "the supervised labels. No entries were merged or rescored for this review. Such overlap "
            "helps explain why tactical catalogue names must not automatically be interpreted as "
            "separable manoeuvres from B01 observations. Establishing an observability ceiling would "
            "require a separate study. A supplementary behaviour-family or motion-consistency metric "
            "could be informative only after a semantic mapping is justified independently of which "
            "mapping raises validation performance."
        ),
    )

    f = page(
        "A reporting structure that makes the claim reviewable",
        "Lead with the task and independent cases; then show the headline and its important breakdowns",
    )
    rows = [
        [
            "Scope",
            "Known frozen pilots; new scenario groups; idealised B01 tracks; exact future selections",
        ],
        [
            "Headline",
            "Top-1 / top-3, macro-F1 (33 and supported), NLL, Brier, ECE; supports and paired intervals",
        ],
        [
            "Pilot breakdown",
            "All 10 pilots × 3 horizons; class mix, support, calibration and gain uncertainty",
        ],
        [
            "Skill breakdown",
            "Precision / recall / F1, predicted counts, origin and encounter support; absent labels explicit",
        ],
        [
            "Temporal breakdown",
            "Common-origin horizons, endpoint changes, elapsed time, repeated-label runs and censoring",
        ],
        [
            "Reproducibility",
            "Split IDs, source hashes, selection chronology, seeds, cost and preserved failed candidates",
        ],
    ]
    table(
        f,
        [0.065, 0.31, 0.87, 0.41],
        rows,
        ["Layer", "What to report"],
        widths=[0.20, 0.80],
        size=13,
    )
    weighting = pd.read_csv(data / "accuracy_weightings.csv")
    w = weighting[weighting.model.eq("ensemble") & weighting.horizon_s.eq(5)].iloc[0]
    f.text(
        0.065,
        0.21,
        f"+5 s accuracy: pooled origins {w.pooled_windows * 100:.2f}%  |  equal encounters {w.equal_episodes * 100:.2f}%  |  equal pilots {w.equal_pilots * 100:.2f}%",
        fontsize=18,
        color=TEAL,
    )
    f.text(
        0.065,
        0.12,
        "The chosen weighting changes the question. Keep one primary definition and publish the alternatives.",
        fontsize=17,
        color=GOLD,
    )
    finish(
        f,
        (
            "The established primary metric weights valid forecast origins equally. Equal-encounter "
            "accuracy first averages within an eligible encounter, then across encounters; equal-pilot "
            "accuracy first averages all eligible origins within a pilot, then across pilots. These "
            "need not agree because encounter lengths and label mixtures differ. The companion table "
            "reports all horizons and the original transformer as well. Use scenario-group resampling "
            "rather than independent-window standard errors. Per-pilot intervals are pointwise, and "
            "selection bias is not removed by a bootstrap. Validation remains development evidence; "
            "the reserved test should be assessed only after the next protocol is frozen. Method "
            "references: Cawley & Talbot 2010, Roberts et al. 2017, Field & Welsh 2007 and Guo et al. "
            "2017; links and their application are in study_guide.md."
        ),
    )

    f = page(
        "Next experiments: isolate causes before adding complexity",
        "Proposed work only | Preserve this snapshot and use internal training-group partitions",
    )
    rows = [
        [
            "1",
            "Data-size learning curves",
            "Vary fit groups; fixed stop/select sets; repeat seeds. Test whether more diverse cases still help.",
        ],
        [
            "2",
            "Matched feature/history ablations",
            "54 vs 91 features; 5/20/60 s histories. Keep model and training budget controlled.",
        ],
        [
            "3",
            "Selection and calibration",
            "NLL vs accuracy stopping, before/after calibration. Report accuracy, F1 and probability scores.",
        ],
        [
            "4",
            "Six models versus twelve",
            "Two vs four variants nearly tied internally (one seed each). Test the three-seed refit recipes.",
        ],
        [
            "5",
            "Rare skills and transition errors",
            "Broader targeted training cases; cautious reweighting. Preserve natural-prevalence evaluation.",
        ],
    ]
    table(
        f,
        [0.065, 0.29, 0.87, 0.43],
        [[r[0], r[1], textwrap.fill(r[2], width=80)] for r in rows],
        ["Order", "Question", "Controlled test"],
        widths=[0.07, 0.29, 0.64],
        size=13,
    )
    f.text(
        0.065,
        0.20,
        "Acceptance: paired accuracy gain, no material F1 / NLL regression, stable pilot breakdowns, and stated compute cost.",
        fontsize=15,
        color=TEAL,
    )
    f.text(
        0.065,
        0.115,
        "Broader claims need separate tests: unseen pilots, realistic sensor errors and genuinely new scenario conditions.",
        fontsize=15,
        color=GOLD,
    )
    finish(
        f,
        (
            "This is a ranked experimental agenda, not a claim that any proposed change has improved "
            "the saved results. Fix success criteria and compute budgets before each comparison, keep "
            "all ten versions of a scenario together, fit preprocessing only on the fitting partition, "
            "and retain failed runs. Data-size learning curves should use nested fitting-group subsets "
            "with common stop/select groups and multiple seeds. Matched ablations are required because "
            "the original-to-improved experiment changed architecture, features, history and ensemble "
            "size together. The measured internal comparison used two versus four variants with "
            "one seed each; a six-versus-twelve-member final comparison remains proposed. "
            "Rare-skill interventions may harm calibration or "
            "common-class accuracy, so natural-prevalence evaluation and all metrics remain necessary. "
            "Any future temporal-segment F1 must have separately defined alignment, matching and "
            "censoring; it is different from the per-label F1 already reported. No new training, "
            "calibration, model selection or reserved-test evaluation was performed for this review."
        ),
    )


def append_temporal_review_slides(dest, page, finish, words, chart, table):
    data = dest / "data"
    common = pd.read_csv(data / "temporal_common_origins.csv")
    endpoint = pd.read_csv(data / "temporal_endpoint_changes.csv")
    f = page(
        "Forecast horizon: compare the same starting observations",
        "29,458 origins have labels at all three horizons | History length and forecast horizon are different",
    )
    ax = chart(f, [0.085, 0.30, 0.38, 0.42])
    for model, label, color in (
        ("transformer", "Original transformer", BLUE),
        ("ensemble", "Improved ensemble", TEAL),
    ):
        frame = common[common.model.eq(model)].sort_values("horizon_s")
        ax.plot(
            frame.horizon_s,
            frame.top1_accuracy * 100,
            marker="o",
            color=color,
            linewidth=2.5,
            label=label,
        )
        for row in frame.itertuples():
            ax.annotate(
                f"{row.top1_accuracy * 100:.2f}%",
                (row.horizon_s, row.top1_accuracy * 100),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=13,
            )
    ax.set_xticks(HORIZONS)
    ax.set_ylim(52, 71)
    ax.set_xlabel("Seconds ahead")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.legend(facecolor=BG, labelcolor=FG, fontsize=11, loc="lower left")
    rows = [
        [
            f"+{int(r.horizon_s)} s",
            f"{int(r.available_count):,}",
            f"{r.changed_endpoint_fraction * 100:.2f}%",
        ]
        for r in endpoint.itertuples()
    ]
    table(
        f,
        [0.54, 0.43, 0.39, 0.25],
        rows,
        ["Horizon", "Eligible origins", "Changed label"],
        widths=[0.23, 0.39, 0.38],
        size=14,
    )
    f.text(
        0.54,
        0.35,
        "Right: each horizon's full eligible set.\nChange means endpoint label differs from current.",
        fontsize=13,
        color=MUTED,
        linespacing=1.5,
    )
    f.text(
        0.065,
        0.16,
        "On matched origins, longer-horizon accuracy falls. Changed-label subsets have different members at each horizon.",
        fontsize=16,
        color=GOLD,
    )
    finish(
        f,
        (
            "Left uses the intersection of origins with all three future labels, so the forecast start "
            "times are identical across horizons. Ensemble accuracy is 65.6019%, 65.1436% and 63.4802%; "
            "original-transformer accuracy is 58.8703%, 58.2898% and 55.8999%. Targets still differ "
            "because the future time changes. Right uses all eligible origins at each horizon, whose "
            "counts are 30,802, 30,466 and 29,458. Endpoint change is 7.35%, 13.96% and 29.39%. "
            "The current true label is privileged evaluation information, never a feature. A repeat-"
            "current-true-label reference would be an oracle diagnostic, not a deployable baseline. "
            "Same endpoints can hide a switch and return in between. A longer history is more past "
            "input; a longer horizon is a later future target. Neither should be confused with elapsed "
            "time since the encounter began. Common-origin supported-class counts are 25/25/26, so "
            "fixed-33 macro-F1 is the consistent vocabulary comparison in the published table."
        ),
    )

    phases = pd.read_csv(data / "temporal_elapsed_phases.csv")
    phases = phases[phases.horizon_s.eq(5)]
    order = ["6-20 s", "21-60 s", "61+ s"]
    old = phases[phases.model.eq("transformer")].set_index("origin_time_bin").loc[order]
    new = phases[phases.model.eq("ensemble")].set_index("origin_time_bin").loc[order]
    f = page(
        "Later encounter accuracy is higher—but the cases change",
        "+5 s forecasts, using the common-origin population | Elapsed time is measured at the forecast origin",
    )
    ax = chart(f, [0.085, 0.33, 0.37, 0.38])
    x = np.arange(3)
    ax.bar(x - 0.18, old.top1_accuracy * 100, 0.36, color=BLUE, label="Original transformer")
    ax.bar(x + 0.18, new.top1_accuracy * 100, 0.36, color=TEAL, label="Improved ensemble")
    for i, value in enumerate(new.top1_accuracy * 100):
        ax.text(i + 0.18, value + 1, f"{value:.1f}%", ha="center", fontsize=13)
    ax.set_xticks(x, order)
    ax.set_ylim(0, 82)
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.legend(facecolor=BG, labelcolor=FG, fontsize=10, loc="upper left")
    rows = [
        [
            label,
            int(r.episodes),
            int(r.scenario_groups),
            f"{r.changed_endpoint_fraction * 100:.1f}%",
        ]
        for label, r in new.iterrows()
    ]
    table(
        f,
        [0.54, 0.40, 0.39, 0.28],
        rows,
        ["Origin time", "Flights", "Groups", "Changed"],
        widths=[0.31, 0.21, 0.23, 0.25],
        size=13,
    )
    words(
        f,
        0.065,
        0.23,
        "WHAT CHANGES WITH TIME?",
        "Available history, skill mix and surviving encounters. Supported labels fall from 24 to 23 to 17.",
        GOLD,
        16,
    )
    f.text(
        0.065,
        0.10,
        "Late top-3 slightly worsens: 90.48% → 90.06%. This is not an isolated test of the benefit of longer history.",
        fontsize=16,
        color=MUTED,
    )
    finish(
        f,
        (
            "Bins are fixed around when a full 20-second and 60-second history becomes available: "
            "21 and 61 seconds, accounting for acceleration lookback, reset exclusion and one-second "
            "forecast origins. Counts are 6,411, 11,531 and 11,516 origins. The early bin covers all "
            "450 encounters and 45 scenario groups; the middle 389/39, and late only 225/24. These "
            "are selected surviving subsets, not repeated measurements of an unchanged evaluation "
            "population. Endpoint-change rates decline 40.01% to 32.70% to 20.17%, and supported "
            "labels decline 24 to 23 to 17. Accuracy rises, but late top-3 decreases from 90.48% "
            "to 90.06% relative to the original transformer. The complete table retains all horizons, "
            "NLL and both F1 definitions. A controlled history ablation on the same origins, models "
            "and training protocol is needed to attribute any effect to history length."
        ),
    )

    dynamics = pd.read_csv(data / "temporal_pilot_dynamics.csv")
    population = dynamics[dynamics.pilot_id.eq("population")].iloc[0]
    per_pilot = dynamics[dynamics.pilot_id.ne("population")]
    transitions = pd.read_csv(data / "temporal_transition_counts.csv")
    switches = (
        transitions[transitions.from_skill.ne(transitions.to_skill)]
        .sort_values(["count", "from_skill", "to_skill"], ascending=[False, True, True])
        .head(5)
    )
    f = page(
        "Skill selections persist, then switch",
        "Full validation decision streams: 38,181 choices at 1 Hz | These are label runs, not physical manoeuvre durations",
        "450 validation decision row groups decoded; zero test groups",
    )
    ax = chart(f, [0.085, 0.32, 0.35, 0.39])
    ax.bar(np.arange(10), per_pilot.one_second_repeat_fraction * 100, color=TEAL)
    ax.set_xticks(np.arange(10), [p[-3:] for p in per_pilot.pilot_id], rotation=45, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Same label one second later (%)", fontsize=12)
    ax.set_xlabel("Target pilot", fontsize=12)
    rows = [
        [r.from_skill.replace("_", " "), r.to_skill.replace("_", " "), int(r.count)]
        for r in switches.itertuples()
    ]
    table(
        f,
        [0.49, 0.39, 0.445, 0.29],
        rows,
        ["From", "To", "Switches"],
        widths=[0.42, 0.42, 0.16],
        size=12,
    )
    f.text(
        0.49,
        0.33,
        "Five most frequent observed changes; self-transitions excluded.",
        fontsize=11,
        color=MUTED,
    )
    f.text(
        0.065,
        0.21,
        f"{population.one_second_repeat_fraction * 100:.2f}% repeat at adjacent decisions  |  {int(population.switches):,} switches  |  {int(population.right_censored_runs)} episode-ending runs censored",
        fontsize=18,
        color=TEAL,
    )
    f.text(
        0.065,
        0.13,
        f"Completed runs only: median {population.completed_run_median_s:.0f} s, 90th percentile {population.completed_run_p90_s:.0f} s. These omit incomplete endings.",
        fontsize=17,
        color=GOLD,
    )
    finish(
        f,
        (
            "This analysis reads full decision streams for validation episodes only, skipping every "
            "training/test row group before decoding. There are 37,731 adjacent one-second pairs, "
            "91.7733% with the same label, 3,104 switches and 3,554 runs. A run means consecutive "
            "identical catalogue selections. The 3,104 fully observed completed runs have median "
            "5 seconds and 90th percentile 16 seconds; the 450 final runs are right-censored and "
            "are excluded from those duration quantiles. These are conditional summaries, not "
            "unbiased estimates of all label durations. Gaps would break and censor runs; no gaps "
            "were found in these streams. Full streams include early/terminal choices outside "
            "eligible forecast targets and contain 27 labels versus 26 in the forecast scorecards. "
            "The repeat fraction is a truth statistic, not a model's forecasting accuracy. Current "
            "labels remain unavailable at inference. Transition counts reveal common changes but "
            "do not establish that every label change is a distinct physical manoeuvre."
        ),
    )


def write_study_guide(dest):
    data = dest / "data"
    architecture = read_json(data / "architecture_review.json")
    pilots = pd.read_csv(data / "pilot_review.csv")
    weights = pd.read_csv(data / "accuracy_weightings.csv")
    phases = pd.read_csv(data / "temporal_elapsed_phases.csv")
    dynamics = pd.read_csv(data / "temporal_pilot_dynamics.csv")
    p5 = pilots[pilots.horizon_s.eq(5)].copy()
    ptable = p5[
        [
            "pilot_id",
            "count",
            "new_top1",
            "gain",
            "gain_lower",
            "gain_upper",
            "dominant_skill",
            "dominant_share",
            "endpoint_change_rate",
            "ece",
        ]
    ].copy()
    for name in (
        "new_top1",
        "gain",
        "gain_lower",
        "gain_upper",
        "dominant_share",
        "endpoint_change_rate",
        "ece",
    ):
        ptable[name] *= 100
    variants = pd.DataFrame(
        [
            {
                "Variant": c["label"],
                "History s": c["history_seconds"],
                "Input columns": c["input_columns_with_flags"],
                "Width": c["width"],
                "Blocks": c["residual_blocks"],
                "Parameters/member": c["parameters_per_member"],
                "Epochs": c["epochs"],
                "Temperature": round(c["temperature"], 4),
            }
            for c in architecture["components"]
        ]
    )
    times = pd.DataFrame(architecture["timing_stages"])[
        ["stage", "runs", "epochs_sum", "elapsed_seconds_sum"]
    ]
    times["minutes_sum"] = times.elapsed_seconds_sum / 60
    phase_table = phases[phases.horizon_s.eq(5) & phases.model.eq("ensemble")][
        [
            "origin_time_bin",
            "count",
            "episodes",
            "scenario_groups",
            "supported_skills",
            "top1_accuracy",
            "changed_endpoint_fraction",
        ]
    ].copy()
    phase_table[["top1_accuracy", "changed_endpoint_fraction"]] *= 100
    weight_table = weights.copy()
    weight_table[["pooled_windows", "equal_episodes", "equal_pilots"]] *= 100
    dynamics_table = dynamics[
        [
            "pilot_id",
            "decisions",
            "one_second_repeat_fraction",
            "switches",
            "completed_runs",
            "right_censored_runs",
            "completed_run_median_s",
            "completed_run_p90_s",
        ]
    ].copy()
    dynamics_table.one_second_repeat_fraction *= 100
    text = [
        """# Skill catalogue forecasting: study guide and experiment review

Review date: **3 November 2026**. This guide accompanies the **41-slide PDF**.
Slides 1–26 retain their numbering and results; slide 19 clarifies the internal
ensemble-size comparison. Slides 27–41 add architecture, training cost, experiment
evolution and descriptive pilot/skill/time analyses.

This review inspected source code, completed training metadata and fixed saved
validation predictions. It fitted no models, changed no checkpoints or
calibration, and performed no reserved-test evaluation. Full temporal decisions
were decoded from **450 validation row groups only**, with zero test groups read.

## A useful reading sequence

1. Slides 2, 26–29: define the forecasting task and follow one input through the network.
2. Slides 23–25, 30–31: learn what the losses mean, what training cost, and why the experiments changed.
3. Slides 5–17, 32–35: inspect headline metrics, each pilot, each skill and calibration.
4. Slides 36–39: separate forecast horizon, elapsed encounter time, label persistence and physical behaviour.
5. Slides 40–41: decide how to report the claim and what to test next.

## 1. What is trained, and what is predicted?

There are two learning problems. PPO trains ten independent pilot policies to
select actions using different reward profiles. The pilot actor–critic is part
of the aircraft controller. A complete 150-epoch × 75-flight × ten-pilot stage
uses **112,500 training flights**, excluding its validation/evaluation flights
and other experiment stages. Pilots change during that training.

After selecting and freezing their checkpoints, a separate recorder generated
**300 new starting scenarios × ten pilots = 3,000 encounters**. B01 is the scripted
tactical observer; A01 is the trained target pilot. The forecaster is a separate
supervised network learning from B01's observations to predict the exact selected
catalogue entry at +1, +2 and +5 seconds. It has no PPO critic and does not update
the pilots. A high forecast score describes predictability, not combat competence.

The outer split is 210/45/45 scenario groups, or 2,100/450/450 encounters. All ten
pilot versions of a scenario stay together. The 146,690 training and 30,802
validation origins are overlapping windows, not independent encounters. The
reserved test targets remain unprepared. Every pilot appears in training and
validation: the claim is **new scenarios for known pilots**, not unseen policies.

## 2. From observations to an MLP prediction

The original GRU/transformer use 54 features per timestamp: 51 recorded observation
values and three backward target-acceleration estimates. The improved models add
37 causal geometry features, for 91 values. These include aircraft position and
velocity, relative geometry, missile information, energy proxies, speed/course,
closure/aspect, turn-rate estimates and constant-velocity closest approach.
Closest approach is computed from current position/velocity, never a future track.

The 20-second model samples 26 timestamps; the 60-second model samples 66. Older
observations are one second apart; the latest five seconds are half a second apart.
The acceleration estimate needs a further half-second of raw lookback. A flag per
timestamp identifies missing early history. Features are normalised using fitting
data only; invalid timestamps are zeroed and flagged.

For N examples, the organised input is `[N, timestamps, 91]`. Appending flags and
flattening gives `[N, 2392]` or `[N, 6072]`: an ordinary 2D feature table. The column
position identifies the lag, so `speed(t-2)` and `speed(t)` are distinct columns.
Future skill labels belong in y, not X. Pilot ID, checkpoint ID, private skill
choices, private continuous action parameters and final outcomes are excluded.

For the standard model, the exact computation is:

```text
2,392 input values
  → Linear(2392, 256) → GELU
  → three residual blocks
  → LayerNorm(256) → Linear(256, 99)
  → reshape to 3 horizons × 33 logits
  → divide logits by the fixed temperature
  → softmax separately for each horizon
```

Each residual block computes:

```text
output = x + [LayerNorm → Linear(256,512) → GELU → Dropout
              → Linear(512,256) → Dropout](x)
```

Linear layers learn weighted combinations. GELU supplies nonlinearity. LayerNorm
controls activation scale. Dropout disables sampled activations during training,
and the residual addition carries the original representation around the block.
The wider variant uses width 512 and expands to 1,024 inside each block.
**Three residual blocks means eight linear layers in the whole network**, including
input and output projections. There is no recurrent state or attention in these MLPs.

Each member emits all three horizons; there is no one-member-per-pilot arrangement
and no explicit constraint enforcing consistency between the three future labels.
The twelve members' calibrated probabilities are averaged uniformly: each member
has weight 1/12, each four-variant family 1/4. Nine members use 20 seconds and three
use 60 seconds. Seeds 42/43/44 change initialization and training order.
""",
        markdown_table(variants),
        """
The ensemble has **28,977,060 parameters**, versus 439,651 for the original
transformer and 234,723 for the original GRU. Input budget and compute both changed;
the result does not isolate an architecture-only improvement. The existing local
four-thread CPU timing was 20.3 ms median / 22.0 ms p95 over 30 warm single-origin
calls, including features. This is not a deployment guarantee or a matched runtime
comparison with the original networks.

## 3. Training, loss and timing

Training uses AdamW, mini-batches of 1,024 origins, learning rate 0.0003,
gradient-norm clipping at 1.0 and CUDA BF16 autocasting. Improved runs use a cosine
schedule toward 0.00003 over 60 epochs, stopping at their saved epoch budgets. Label
smoothing is 0.02/0.03/0.08/0.02; dropout 0.20/0.30/0.30/0.20 and weight decay
0.03/0.05/0.05/0.03 across the four variants. All are mechanisms intended to control
fitting, not guarantees against overfitting.

With a hard correct label, cross-entropy / NLL is `-ln(p_correct)`: p=0.9 gives
0.105, p=0.5 gives 0.693, and p=0.1 gives 2.303. Zero is perfect; the values are
not percentages. Training loss is recorded during weight updates with dropout;
validation NLL is measured using fixed end-of-epoch weights and no dropout.
Improved training also uses smoothed targets, while NLL uses hard labels, so the
absolute train/validation gap is not a pure generalisation-gap measurement.

The loss averages valid examples within a horizon, then the represented horizons.
Missing future decisions are masked, never carried forward. Accuracy cares about
the highest-probability label; NLL cares about probability assigned to truth.
Accuracy can rise while NLL worsens—for example, if remaining mistakes receive
increasingly confident predictions. The accuracy/NLL divergence was observed
internally and motivated calibrated probability mixtures; the precise error-level
cause was not decomposed in this review. Positive scalar temperature scaling preserves a single
model's class order, but calibrating members before averaging can change the
ensemble's final order. [Guo et al., 2017](https://proceedings.mlr.press/v70/guo17a.html)

Original checkpoints were selected at GRU epoch 15 and transformer epoch 20,
using validation NLL and a minimum-improvement threshold. Improved candidates
used internal stop groups; final refits used fixed epoch budgets. Per-pilot
forecast losses were not logged each epoch, so no such curves were reconstructed.

The following totals count each completed fit once, excluding copied validation
summaries. They are sums of recorded fit-stage elapsed timers:
""",
        markdown_table(times.round(3)),
        f"\nTotal: **{architecture['all_logged_fit_seconds'] / 60:.2f} minutes across 36 runs**.\n",
        """
Final refits account for 15.42 minutes. This is not complete project wall time.
Improved timers exclude loading, feature construction and model/optimizer setup;
they include the training loop, internal stopping evaluation, checkpoint writes
and temperature fitting, and stop before selection-set prediction. Original
timers additionally include final validation/report work. These boundaries differ.
Simulation, dataset preparation, broader evaluation, audits and presentation work
add separate costs. RTX 4090/CUDA 13.0 is recorded for the original/v3 experiments;
v2 has no separate environment file. See training_run_inventory.csv for all runs.

## 4. How the experiments evolved

1. **Initial experiment:** frequency baseline, GRU and transformer; 5 s / 54 features.
   The transformer led the initial validation comparisons, but its training loss
   continued falling after validation NLL ceased improving.
2. **v2 internal search:** nine first-split candidates explored regularisation,
   geometry, auxiliary current-label supervision, longer history and MLPs. Two
   further second-split runs provided confirmation. Unsuccessful candidates remain
   in the record; the 20-second MLP was promising.
3. **v3 targeted search:** accuracy-selected checkpoints, temperature scaling,
   wider/smoothed networks and 60-second history. Eleven fits across the two internal
   partitions. Individual accuracy gains could breach the NLL guard.
4. **Recorded mixture extension:** fixed uniform mixtures were compared internally.
   Averaging improved accuracy and probability quality. Four variants averaged
   60.9817% internal accuracy versus 60.9598% for the wide/long pair; NLL favoured
   four variants (1.2581 versus 1.2807), while macro-F1 slightly favoured the pair.
   Internal confirmation used one seed per variant; the final recipe uses three.
5. **Freeze and refit:** variants, epoch counts, temperatures and seeds were fixed,
   then twelve models fitted on all 210 training groups. Outer validation was used
   for the preserved original-versus-improved comparison. The reserved test was not used.

Internal groups comprise 147 fit / 21 stop / 42 select, wholly within the original
training partition. Two split seeds reduce reliance on one favourable partition,
but the search was adaptive. Internal and outer-validation scores must not be
drawn as a single comparable improvement curve. The evidence supports the combined
procedure; it does not isolate how much each change contributed.

The outer validation set already informed the original models. This review adds
diagnostics to that same development set; it does not turn it into an untouched
test. Repeated model selection can overfit the selection criterion itself.
[Cawley & Talbot, 2010](https://www.jmlr.org/beta/papers/v11/cawley10a.html)

## 5. Pilot-by-pilot interpretation

All 30 pilot/horizon top-1 point estimates improve over the original transformer.
That does not establish a significant improvement for every pilot. The table
below uses +5-second origins; accuracy/share/change/ECE values are percentages or
percentage points, and gain intervals are paired, pointwise 95% intervals.
""",
        markdown_table(ptable.round(3)),
        """
Intervals resample whole scenario groups 5,000 times, keeping windows together;
they are not corrected for 30 comparisons or historical model selection. Pilot
identity was not provided as a feature. Different dominant labels, switch rates,
episode lengths and reward profiles influence the interpretation of difficulty.
The table describes association, not a causal attribution of accuracy to any one
factor. A predictable pilot is not necessarily a better or worse combat pilot.

Per-pilot ECE ranges from about 2.00 to 10.91 points despite pooled ECE of 3.42.
Opposing overconfidence and underconfidence can cancel when pooled. Mean confidence
matching accuracy is not enough to establish calibration across confidence bins.
Conditional calibration is a reporting gap to investigate internally before
considering any new calibration method.

## 6. Per-skill results: terminology and support

- Precision = TP / (TP + FP): when we predict this skill, how often are we right?
- Recall = TP / (TP + FN): when this skill actually occurs, how often do we recognise it?
- F1 = 2TP / (2TP + FP + FN): balances precision and recall.
- Top-1 accuracy: fraction of all eligible origins with the correct selected label.
- Top-3 accuracy: fraction with truth among three highest-scored labels.
- Catalogue macro-F1 averages all 33 entries, using zero for undefined stored class scores.
- Supported macro-F1 averages only labels with truth support; the set can change by pilot/horizon.

Per-skill 'accuracy' is ambiguous. If the denominator is true occurrences of that
skill, the metric is recall. One-versus-rest accuracy includes true negatives and
can be high for an almost-never-predicted rare class. Keep precision, recall, F1,
truth support and predicted count together. Seven labels have no validation target
support, so performance on them is not measured. Eleven labels account for 96.42%
of +5-second target origins; the remaining fifteen supported labels are sparse.

pilot_skill_metrics.csv includes every pilot, all 33 labels, all horizons and both
original-transformer/ensemble results. skill_support_context.csv adds distinct
encounter, scenario and pilot counts per label. Many adjacent origins can come
from one sustained label run; raw support is not the number of independent cases.
Use the pilot × skill heatmap to locate misses, then inspect precision and
predicted counts to distinguish under-prediction from broad over-prediction.

## 7. Three meanings of time

**Observation history** is how far backwards X extends. **Forecast horizon** is
how far ahead the target lies. **Elapsed encounter time** is when the forecast is
issued. Increasing one does not imply increasing either of the others.

On the same 29,458 origins, ensemble top-1 is 65.60 / 65.14 / 63.48% at +1/+2/+5 s,
versus 58.87 / 58.29 / 55.90% for the original transformer. Thus longer-horizon
accuracy decreases even when the eligible starting observations are held fixed.
Supported class sets are 25/25/26, so use fixed-33 macro-F1 for equal-vocabulary
comparisons. Different future targets still have different label distributions.

On each horizon's full eligible set, endpoint-change fractions are 7.35 / 13.96 /
29.39%. These are different cohorts. Same endpoint labels can conceal an intervening
switch and return. Repeating the current true label is a privileged diagnostic,
not an observation-only forecasting baseline, because that label is unavailable.

Elapsed-time results for the ensemble at +5 seconds are:
""",
        markdown_table(phase_table.round(3)),
        """
Later cases have more history, fewer changes and a different surviving population.
Only 225 encounters and 24 scenario groups remain in the late bin. Supported labels
fall from 24 to 23 to 17. The higher late top-1 cannot isolate the benefit of more
history. Late top-3 actually falls slightly versus the transformer: 90.48% to 90.06%.
The complete table includes both models, all horizons, NLL and F1.

The full validation decision streams contain 38,181 choices and 37,731 adjacent
one-second pairs. 91.77% repeat the same label, with 3,104 switches. There are 3,554
runs, including 450 episode-ending runs whose durations are incomplete. A run means
consecutive identical selections, not a physical manoeuvre. Completed runs have
median duration 5 s and p90 16 s; excluding censored endings makes these conditional
summaries, not unbiased estimates of every duration. Gaps would also censor runs;
none occurred here. Full streams contain 27 labels because early/terminal choices
outside forecasting eligibility are included. Per-pilot temporal summaries:
""",
        markdown_table(dynamics_table.round(3)),
        """
## 8. Catalogue semantics and observability

The execution code allows several labels to share guidance components. For example,
lag_pursuit, extend, turn_cold, recommit and pursue_target can emit the same zero-change
commands; beam_target_left and crank_target_left both emit a −90° heading change;
launch, secondary_shot and short_range_attack all set shoot. This does not prove
identical trajectories or identical semantics in every context. It does show why
exact symbolic label prediction is stricter than coarse physical-motion prediction.

No labels were merged. A supplementary behaviour-family score would need a justified
mapping fixed independently of which grouping raises validation scores. The present
evidence does not establish an irreducible accuracy ceiling.

## 9. Report the estimand, not just a percentage

The primary skill score weights eligible forecast origins equally. Equal-encounter
accuracy averages within each eligible encounter first; equal-pilot accuracy averages
within each pilot first. Long encounters contribute more under the primary definition.
The three accuracy columns are percentages; horizon is in seconds and the remaining
numeric columns are counts:
""",
        markdown_table(weight_table.round(3)),
        """
A practical report should contain: task/observation boundary; groups/episodes/origins;
primary top-1/top-3/F1 plus NLL/Brier/calibration; paired uncertainty; all-pilot and
all-skill breakdowns; temporal strata and negative results; selection chronology,
seeds and cost. State units, averaging and denominators in each table. Keep test
results separate from development history when the final assessment is performed.

Dependent observations require evaluation blocks suited to the intended use. Here,
scenario blocking targets new encounters for known pilots; unseen pilots require a
different blocked design. [Roberts et al., 2017](https://nsojournals.onlinelibrary.wiley.com/doi/10.1111/ecog.02881)
Cluster resampling preserves within-group structure under assumptions about the
clusters; it does not make windows independent or undo selection bias.
[Field & Welsh, 2007](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-9868.2007.00593.x)

## 10. Proposed experiments, with measurable questions

1. **Data-size learning curves.** Use nested fitting-group subsets with fixed internal
   stop/select groups and repeated seeds. Determine whether gains plateau before
   committing to broader scenario collection. Preserve natural-prevalence evaluation.
2. **Matched ablations.** Compare 54/91 features and 5/20/60-second histories while
   controlling model, training protocol and compute. Separate architecture changes
   from feature/history changes rather than attributing the combined gain to one.
3. **Checkpoint objective and calibration.** Compare NLL- and accuracy-selected
   checkpoints, with/without calibration, using internal groups. Report accuracy,
   macro-F1 and probability quality together; reject materially harmful trade-offs.
4. **Ensemble size.** Test the six- versus twelve-member refit recipes with paired
   scores, training and inference cost. The measured 0.02-point internal difference
   was between two and four variants using one seed each; it was not a final
   six-versus-twelve evaluation and does not establish that twelve are necessary.
5. **Rare skills and changes.** Collect broader relevant training cases and test
   cautious reweighting/current-skill-estimation plus transition models. Current
   true skill must remain a target or diagnostic, never an inference feature.
   Measure common-skill regressions, NLL and calibration alongside rare-skill gains.

Freeze acceptance criteria before comparisons, retain failed runs, and keep the
reserved test for the final frozen protocol. Any future use of the present validation
diagnostics to choose models remains development, not independent confirmation.
Unseen-pilot and sensor-noise evaluations would support broader claims and should
have their own explicit design.

Segmental metrics can reveal errors hidden by pointwise accuracy in action-sequence
tasks. [Lea et al., 2017](https://arxiv.org/abs/1611.05267) motivates a supplementary
future analysis, not a replacement or reinterpretation of the existing per-label
F1. Define future-time alignment, segment matching, gaps and censored endings first.

## Reproducibility and files

- [Full PDF](skills_catalogue.pdf), [speaker notes](speaker_notes.md), [existing comparison](comparison.md).
- [Per-pilot review](data/pilot_review.csv), [pilot/skill metrics](data/pilot_skill_metrics.csv), [label support context](data/skill_support_context.csv).
- [Weighting comparison](data/accuracy_weightings.csv), [common origins](data/temporal_common_origins.csv), [elapsed-time strata](data/temporal_elapsed_phases.csv).
- [Temporal dynamics](data/temporal_pilot_dynamics.csv), [run records](data/temporal_validation_runs.csv), [transition counts](data/temporal_transition_counts.csv).
- [Architecture and timing provenance](data/architecture_review.json), [all fit timers](data/training_run_inventory.csv), [temporal audit](data/temporal_evidence.json), [review contract](data/review_evidence.json).

The default builder reads this portable bundle only. `--collect-review` regenerates
the review from fixed local metadata/predictions and validation-only decision rows.
It does not train, run model inference, alter selected checkpoints or evaluate test
targets. Bundle and source SHA-256 checksums make the evidence auditable.
""",
    ]
    (dest / "study_guide.md").write_text("\n\n".join(text), encoding="utf-8", newline="\n")
