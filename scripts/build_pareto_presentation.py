"""Plot observed Pareto frontiers from saved evaluations; never runs a simulator.

--collect audits local saved episode reports and packages their outcome flags.
The default rebuilds all plots and reports using only the portable CSV package.
"""

import argparse
import base64
import csv
import hashlib
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "docs/presentations/pareto"
SOURCE = ROOT / "docs/presentations/data"
BG, FG, MUTED, GRID = "#101e30", "#f2f5f9", "#aebdd0", "#304158"
LABELS = {
    "original": "Original baseline",
    "lower_learning_rate": "Reduced learning rate",
    "selected_skill_parameters": "Selected-skill probabilities",
    "skill_conditioned_parameters": "Per-skill parameter heads",
    "clean_win_bonus": "Clean-win bonus (six completed pilots)",
    "before": "Before tactical adaptation",
    "after": "After tactical adaptation",
}
FULL = tuple(list(LABELS)[:4])
MARKERS = dict(zip(FULL, ("o", "s", "D", "^")))
SHORT = dict(zip(FULL, ("O", "L", "S", "H")))
METRICS = ("elimination", "survival", "clean_win", "mutual_destruction")
EPISODE_FIELDS = ("suite", "population", "pilot", "seed", *METRICS)
REPLICATES, BOOTSTRAP_SEED = 2000, 8128


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def nondominated(values):
    """Maximise every column. Equal points are retained, not mutually dominated."""
    values = np.asarray(values)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty finite matrix of candidate objectives")
    p, q = values[:, None, :], values[None, :, :]
    return ~(np.all(q >= p, axis=-1) & np.any(q > p, axis=-1)).any(axis=1)


def bootstrap_fronts(values):
    """values: replicate x candidate x objective; integer counts avoid tie drift."""
    p, q = values[:, :, None, :], values[:, None, :, :]
    return ~(np.all(q >= p, axis=-1) & np.any(q > p, axis=-1)).any(axis=2)


def collect(dest):
    registry = read(ROOT / "configs/pilot_experiments.json")["experiments"]
    prior_hashes = read(SOURCE / "source_hashes.json")
    sources, rows, case_reference = {}, [], None

    def audit(path):
        relative = path.relative_to(ROOT).as_posix()
        actual = digest(path)
        if relative in prior_hashes and prior_hashes[relative] != actual:
            raise ValueError(f"Saved evidence changed since presentation audit: {relative}")
        sources[relative] = actual
        return read(path)

    audit(SOURCE / "source_hashes.json")
    audit(ROOT / "configs/pilot_experiments.json")
    paths = {
        "original": registry["lower_learning_rate"]["baseline"],
        **{key: registry[key]["training"] for key in FULL[1:]},
        "clean_win_bonus": registry["clean_win_bonus"]["training"],
    }
    completed = [p["pilot_id"] for p in audit(ROOT / paths["clean_win_bonus"] / "pilots.json")["pilots"]
                 if p["status"] == "completed"]
    if completed != [f"pilot_{i:03d}" for i in range(1, 7)]:
        raise ValueError("Bonus population has changed; review the partial-trial scope")
    for population, directory in paths.items():
        pilots = completed if population == "clean_win_bonus" else [f"pilot_{i:03d}" for i in range(1, 11)]
        for pilot in pilots:
            run = ROOT / directory / pilot
            report = audit(run / "test_episodes.json")
            manifest = audit(run / "manifest.json")
            config = audit(run / "config.json")["training"]
            assert report["checkpoint_sha256"] == manifest["checkpoint_sha256"]
            assert config["test_seed"] == 211307 and config["test_scenarios_per_pilot"] == 100
            assert config["seed"] == 7 and config["epochs"] == 150 and config["scenarios_per_epoch"] == 75
            cases = [(e["seed"], e["scenario"]) for e in report["episodes"]]
            assert [seed for seed, _ in cases] == list(range(211307, 211407))
            if case_reference is None:
                case_reference = cases
            assert cases == case_reference, f"Unmatched simple scenarios: {population}/{pilot}"
            for episode in report["episodes"]:
                eliminated, survived = bool(episode["eliminated"]), bool(episode["survived"])
                clean = eliminated and survived
                assert clean == (episode["outcome"] == "surviving_elimination")
                rows.append(dict(zip(EPISODE_FIELDS, (
                    "simple", population, pilot, episode["seed"], int(eliminated),
                    int(survived), int(clean), int(episode["outcome"] == "mutual_destruction"),
                ))))

    path = SOURCE / "paired_outcomes.csv"
    sources[path.relative_to(ROOT).as_posix()] = digest(path)
    for episode in read_csv(path):
        for state in ("before", "after"):
            eliminated, survived = int(episode[f"{state}_eliminated"]), int(episode[f"{state}_survived"])
            clean = eliminated * survived
            assert clean == int(episode[f"{state}_outcome"] == "surviving_elimination")
            rows.append(dict(zip(EPISODE_FIELDS, (
                "tactical_" + episode["suite"], state, episode["pilot"], int(episode["seed"]),
                eliminated, survived, clean, int(episode[f"{state}_outcome"] == "mutual_destruction"),
            ))))

    # Cross-check every complete population against the already published tables.
    for source_name in ("development_summary.csv", "pilot_outcomes.csv", "results.json"):
        path = SOURCE / source_name
        sources[path.relative_to(ROOT).as_posix()] = digest(path)
    grouped = {}
    for row in rows:
        grouped.setdefault((row["suite"], row["population"], row["pilot"]), []).append(row)

    def check(key, expected):
        actual = np.mean([[r[m] for m in METRICS] for r in grouped[key]], axis=0)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)

    for row in read_csv(SOURCE / "development_summary.csv"):
        check(("simple", row["population"], row["pilot"]), [float(row["development_" + k])
              for k in ("elimination", "survival", "surviving_elimination", "mutual_destruction")])
    for row in read_csv(SOURCE / "pilot_outcomes.csv"):
        for state in ("before", "after"):
            check(("tactical_" + row["suite"], state, row["pilot"]), [float(row[state + "_" + k])
                  for k in ("elimination", "survival", "surviving_elimination", "mutual_destruction")])
    bonus_expected = next(r for r in read(SOURCE / "results.json")["history"]
                          if r["label"] == "Terminal clean-win bonus (stopped)")["after_outcomes"]
    bonus_rows = [r for r in rows if r["population"] == "clean_win_bonus"]
    np.testing.assert_allclose(np.mean([[r[m] for m in METRICS] for r in bonus_rows], axis=0),
                               [bonus_expected[m] for m in ("elimination", "survival", "surviving_elimination", "mutual_destruction")])

    data_dir = dest / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    write_csv(data_dir / "episodes.csv", rows)
    write_json(data_dir / "provenance.json", {
        "description": "Recorded episode outcomes only; no model inference or simulation.",
        "simple_case_geometry_sha256": hashlib.sha256(json.dumps(case_reference, sort_keys=True).encode()).hexdigest(),
        "episode_csv_sha256": digest(data_dir / "episodes.csv"),
        "source_hashes": sources,
        "scopes": {
            "simple": "Four complete populations on 100 identical development cases; bonus trial includes only completed pilots 001-006.",
            "tactical_assessment": "300 matched cases before/after adaptation per pilot.",
            "tactical_confirmation": "100 matched historical confirmation cases per pilot; a separate, previously inspected suite.",
        },
    })


def analyse(dest):
    data_dir = dest / "data"
    provenance = read(data_dir / "provenance.json")
    assert digest(data_dir / "episodes.csv") == provenance["episode_csv_sha256"]
    grouped = {}
    for row in read_csv(data_dir / "episodes.csv"):
        key = (row["suite"], row["population"], row["pilot"])
        grouped.setdefault(key, []).append(row)
    points, groups, frontier_rows = [], {}, []
    for suite_index, suite in enumerate(("simple", "tactical_assessment", "tactical_confirmation")):
        keys = sorted(k for k in grouped if k[0] == suite)
        candidates, arrays, reference_seeds = [], [], None
        for key in keys:
            rows = sorted(grouped[key], key=lambda r: int(r["seed"]))
            seeds = [int(r["seed"]) for r in rows]
            if reference_seeds is None:
                reference_seeds = seeds
            assert seeds == reference_seeds and len(set(seeds)) == len(seeds)
            flags = np.array([[int(r[m]) for m in METRICS] for r in rows], dtype=np.int32)
            assert np.isin(flags, [0, 1]).all()
            assert np.array_equal(flags[:, 2], flags[:, 0] * flags[:, 1])
            assert not np.any(flags[:, 3] & flags[:, 1])
            arrays.append(flags)
            candidates.append({"id": "/".join(key), "suite": suite, "population": key[1],
                               "label": LABELS[key[1]], "pilot": key[2], "scenarios": len(seeds),
                               **dict(zip(METRICS, flags.mean(axis=0).tolist()))})
        n = len(reference_seeds)
        assert n == (300 if suite == "tactical_assessment" else 100)
        expected_start = {"simple": 211307, "tactical_assessment": 1120000, "tactical_confirmation": 1220000}[suite]
        assert reference_seeds == list(range(expected_start, expected_start + n))
        arrays = np.array(arrays)
        totals = arrays.sum(axis=1)
        rng = np.random.default_rng(BOOTSTRAP_SEED + suite_index)
        sampled = rng.integers(0, n, size=(REPLICATES, n))
        weights = np.stack([np.bincount(index, minlength=n) for index in sampled]).astype(np.int32)
        # The same scenario resample applies to every candidate and all four outcomes.
        boot = np.einsum("rn,pnm->rpm", weights, arrays, optimize=True)
        intervals = np.quantile(boot / n, [0.025, 0.975], axis=0)
        for i, candidate in enumerate(candidates):
            for m, metric in enumerate(METRICS):
                candidate[metric + "_ci_low"] = float(intervals[0, i, m])
                candidate[metric + "_ci_high"] = float(intervals[1, i, m])
        sets = {suite + "/" + population: [i for i, c in enumerate(candidates) if c["population"] == population]
                for population in dict.fromkeys(c["population"] for c in candidates)}
        if suite == "simple":
            sets["simple/full_comparison"] = [i for i, c in enumerate(candidates) if c["population"] in FULL]
            sets["simple/bonus_matched_subset"] = [i for i, c in enumerate(candidates)
                                                   if c["population"] in ("selected_skill_parameters", "clean_win_bonus")
                                                   and int(c["pilot"].split("_")[-1]) <= 6]
        else:
            sets[suite + "/comparison"] = list(range(len(candidates)))
        for name, indices in sets.items():
            observed = nondominated(totals[indices, :2])
            observed_3d = nondominated(totals[indices, :3])
            stability = bootstrap_fronts(boot[:, indices, :2]).mean(axis=0)
            stability_3d = bootstrap_fronts(boot[:, indices, :3]).mean(axis=0)
            groups[name] = {"candidates": [candidates[i]["id"] for i in indices],
                            "frontier_2d": [candidates[i]["id"] for j, i in enumerate(indices) if observed[j]],
                            "frontier_3d": [candidates[i]["id"] for j, i in enumerate(indices) if observed_3d[j]]}
            for j, i in enumerate(indices):
                frontier_rows.append({"comparison": name, "id": candidates[i]["id"],
                                      "observed_frontier_2d": int(observed[j]),
                                      "scenario_bootstrap_frontier_frequency_2d": float(stability[j]),
                                      "observed_frontier_3d": int(observed_3d[j]),
                                      "scenario_bootstrap_frontier_frequency_3d": float(stability_3d[j])})
        points.extend(candidates)
    assert len(points) == 86
    write_csv(data_dir / "points.csv", points)
    write_csv(data_dir / "frontiers.csv", frontier_rows)
    write_json(data_dir / "analysis.json", {
        "objectives": {"2d": ["maximise elimination", "maximise survival"],
                       "3d": ["maximise elimination", "maximise survival", "maximise clean win"]},
        "bootstrap": {"replicates": REPLICATES, "seed": BOOTSTRAP_SEED,
                      "sampling_unit": "Scenario ID, jointly across every pilot, population and metric within each suite.",
                      "interpretation": "Descriptive resampling stability, not probability of true optimality or a significance test. Does not measure training-seed variation or correct adaptive benchmark reuse."},
        "groups": groups,
    })
    return points, groups, frontier_rows


def style_axes(ax, xlim, ylim):
    ax.set_facecolor(BG)
    ax.set(xlim=xlim, ylim=ylim, xlabel="Survival (%)", ylabel="Elimination (%)")
    ax.grid(color=GRID, alpha=0.7, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def place_labels(ax, items, fontsize=10):
    """Greedily place labels away from all points and preceding labels."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    occupied = []
    for x, y, _ in items:
        px, py = ax.transData.transform((x, y))
        occupied.append(Bbox.from_extents(px - 9, py - 9, px + 9, py + 9))
    offsets = [(9, 8), (9, -15), (-9, 8), (-9, -15), (0, 19), (0, -26),
               (24, 0), (-24, 0), (22, 20), (-22, 20), (22, -28), (-22, -28),
               (0, 34), (0, -40), (35, 24), (-35, 24)]
    offsets += [(dx, dy) for radius in (45, 60, 75)
                for dx, dy in ((radius, 0), (-radius, 0), (0, radius), (0, -radius),
                               (radius, 25), (-radius, 25), (radius, -25), (-radius, -25))]
    for x, y, label in items:
        best, score = None, float("inf")
        for dx, dy in offsets:
            annotation = ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points",
                                     ha="left" if dx > 0 else "right" if dx < 0 else "center",
                                     va="center", fontsize=fontsize, color=FG, zorder=8)
            box = annotation.get_window_extent(renderer).expanded(1.12, 1.12)
            overflow = max(0, ax.bbox.x0 - box.x0) + max(0, box.x1 - ax.bbox.x1) + max(0, ax.bbox.y0 - box.y0) + max(0, box.y1 - ax.bbox.y1)
            penalty = 1000 * overflow + sum(box.overlaps(other) for other in occupied) * 100
            if penalty < score:
                best, score = (dx, dy, box), penalty
            annotation.remove()
            if penalty == 0:
                break
        dx, dy, box = best
        ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha="left" if dx > 0 else "right" if dx < 0 else "center", va="center",
                    fontsize=fontsize, color=FG, zorder=8,
                    arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.5, "alpha": 0.6,
                                "shrinkA": 2, "shrinkB": 8} if abs(dx) + abs(dy) > 30 else None)
        occupied.append(box)


def scatter(ax, points, *, pooled=False):
    totals = np.array([[round(p["survival"] * p["scenarios"]), round(p["elimination"] * p["scenarios"])] for p in points])
    frontier = nondominated(totals)
    leaders = np.array([p["clean_win"] for p in points])
    leaders = leaders == leaders.max()
    for i, point in enumerate(points):
        x, y = 100 * point["survival"], 100 * point["elimination"]
        marker = MARKERS[point["population"]] if pooled else "o"
        ax.scatter(x, y, c=[100 * point["clean_win"]], cmap="viridis", vmin=0, vmax=70,
                   marker=marker, s=100 if pooled else 115,
                   edgecolors=FG if frontier[i] else BG, linewidths=2.0 if frontier[i] else 0.8, zorder=5)
        if leaders[i]:
            if frontier[i]:
                ax.scatter(x, y, facecolors="none", edgecolors=FG, marker="o",
                           s=420, linewidths=1.4, zorder=4)
            ax.scatter(x, y, c=[100 * point["clean_win"]], cmap="viridis", vmin=0, vmax=70,
                       marker="*", s=280, edgecolors=FG, linewidths=1.1, zorder=6)
    coords = sorted(set(map(tuple, totals[frontier] / points[0]["scenarios"] * 100)))
    if len(coords) > 1:
        x, y = np.array(coords).T
        ax.step(x, y, where="pre", color=FG, linewidth=1.5, linestyle="--", alpha=0.8, zorder=2)
    items = []
    for i, point in enumerate(points):
        if not pooled or frontier[i] or leaders[i]:
            pid = point["pilot"].split("_")[-1]
            label = (SHORT[point["population"]] + "-" if pooled else "") + pid
            items.append((point["survival"] * 100, point["elimination"] * 100, label))
    place_labels(ax, items, fontsize=11 if pooled else 10)
    return frontier


def figure_base(title, subtitle):
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    fig.text(0.06, 0.93, title, fontsize=25, weight="bold")
    fig.text(0.06, 0.883, subtitle, fontsize=12.5, color=MUTED)
    cax = fig.add_axes([0.927, 0.26, 0.015, 0.49])
    scalar = plt.cm.ScalarMappable(norm=plt.Normalize(0, 70), cmap="viridis")
    bar = fig.colorbar(scalar, cax=cax)
    bar.set_label("Clean wins (%)", fontsize=12, color=FG)
    bar.ax.tick_params(labelsize=11, colors=FG)
    fig.text(0.06, 0.075, "White rings: observed frontier   |   Star: highest clean-win rate   |   Better outcomes: upper right", fontsize=12, color=FG)
    fig.text(0.06, 0.042, "Dashed steps are visual guides, not achievable intermediate policies. Axes are zoomed; use the labelled percentages.", fontsize=10.5, color=MUTED)
    return fig


def render(dest, points):
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": FG, "axes.labelcolor": FG,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 15,
                         "axes.labelsize": 12, "pdf.fonttype": 42, "savefig.facecolor": BG})
    figures = []
    simple = [p for p in points if p["suite"] == "simple" and p["population"] in FULL]
    fig = figure_base("Simple opponent: each experiment's observed frontier",
                      "Ten pilots per panel | Same 100 development scenarios per pilot | Same reward profiles and training budget")
    for population, bounds in zip(FULL, ([.07, .54, .36, .28], [.53, .54, .36, .28], [.07, .17, .36, .28], [.53, .17, .36, .28])):
        ax = fig.add_axes(bounds)
        style_axes(ax, (35, 85), (30, 95))
        ax.set_title(LABELS[population], loc="left", pad=9)
        scatter(ax, [p for p in simple if p["population"] == population])
    figures.append(("simple_experiments", fig, "Each frontier is relative to the ten pilots in that panel. Membership is not comparable as a score between panels."))

    fig = figure_base("Simple opponent: all four complete experiments",
                      "Forty selected checkpoints | Same 100 development scenarios | One frontier across all candidates")
    ax = fig.add_axes([.07, .27, .81, .54])
    style_axes(ax, (35, 85), (30, 95))
    scatter(ax, simple, pooled=True)
    handles = [Line2D([], [], marker=MARKERS[p], linestyle="none", color=MUTED,
                      markersize=8, label=SHORT[p] + ": " + LABELS[p]) for p in FULL]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.48, .105), ncol=2,
               fontsize=11, frameon=False, labelcolor=FG)
    figures.append(("simple_comparison", fig, "Only selected-skill pilots 002 and 010 are non-dominated among the forty observed points. This is a point-estimate comparison, not proof of statistical superiority."))

    for suite, title, subtitle in (
        ("tactical_assessment", "Tactical adaptation: matched assessment",
         "Ten pilots before and after | Same 300 scenarios per pilot | Frontiers shown separately for each population"),
        ("tactical_confirmation", "Tactical adaptation: historical confirmation",
         "Supplementary evidence | Same 100 scenarios before and after | Different suite from the assessment"),
    ):
        fig = figure_base(title, subtitle)
        for state, bounds in zip(("before", "after"), ([.07, .24, .36, .56], [.53, .24, .36, .56])):
            ax = fig.add_axes(bounds)
            style_axes(ax, (60, 100), (25, 60))
            ax.set_title(LABELS[state], loc="left", pad=10)
            scatter(ax, [p for p in points if p["suite"] == suite and p["population"] == state])
        note = ("Assessment: pilot 010 has the highest clean-win rate, yet is dominated on survival and elimination."
                if suite == "tactical_assessment" else
                "Confirmation: pilot 005 has the highest clean-win rate and is the sole observed after-adaptation frontier point.")
        fig.text(.07, .145, note, fontsize=12, color=FG)
        figures.append((suite, fig, note + " Opponent conditions and scenario suites must not be pooled with simple-opponent results."))

    fig = figure_base("Clean-win bonus: matched six-pilot subset",
                      "Incomplete trial: only pilots 001-006 completed | Same 100 development scenarios | No inference about all ten pilots")
    for population, title, bounds in zip(("selected_skill_parameters", "clean_win_bonus"),
                                          ("Matched selected-skill baseline", "Terminal clean-win bonus"),
                                          ([.07, .24, .36, .56], [.53, .24, .36, .56])):
        ax = fig.add_axes(bounds)
        style_axes(ax, (35, 85), (30, 95))
        ax.set_title(title, loc="left", pad=10)
        scatter(ax, [p for p in points if p["suite"] == "simple" and p["population"] == population
                     and int(p["pilot"].split("_")[-1]) <= 6])
    fig.text(.07, .145, "Population composition is matched within this figure. The bonus trial is excluded from the forty-model comparison.", fontsize=12, color=FG)
    figures.append(("clean_win_bonus", fig, "Partial experiment: the six completed pilots are compared only with their six matching baseline pilots."))

    with PdfPages(dest / "pareto_comparisons.pdf", metadata={"Title": "Observed Pareto frontiers across pilot experiments"}) as pdf:
        for name, fig, note in figures:
            pdf.savefig(fig)
            fig.savefig(dest / (name + ".png"), dpi=180)
            fig.savefig(dest / (name + ".pdf"))
            plt.close(fig)
    return [(name, note) for name, _, note in figures]


def report(dest, points, groups, frontier_rows, figures):
    by_id = {p["id"]: p for p in points}
    frequencies = {(r["comparison"], r["id"]): r for r in frontier_rows}
    simple_stability = [frequencies[("simple/full_comparison", f"simple/selected_skill_parameters/pilot_{i:03d}")]
                        ["scenario_bootstrap_frontier_frequency_2d"] for i in (2, 10)]
    md = ["# Observed Pareto frontiers across pilot experiments", "",
          "[Combined PDF](pareto_comparisons.pdf) · [Offline figure gallery](index.html)", "",
          "These plots use saved selected-checkpoint outcomes. No training, model inference or simulation was performed.", "",
          "## Interpretation", "",
          "Each point is one pilot checkpoint. Both survival and elimination are maximised. A point is dominated when another candidate is at least as good on both measures and strictly better on at least one. Equal points are both retained. White rings identify the observed non-dominated set; dashed steps are visual guides only, not interpolated or achievable policies. Colour encodes clean wins and a star identifies the highest clean-win rate in each plotted candidate set. A star without a surrounding ring can be off the survival/elimination frontier.", "",
          "This is an empirical frontier among tested checkpoints, not a proof of the global Pareto optimum. Reward coefficients and neural-network losses are not the plotted objectives. The four complete simple-opponent populations share the same reward profiles, training seed schedule, budget and 100 development cases. Tactical comparisons use matched before/after cases but are separate tasks and suites. The bonus trial includes only its six completed pilots and their matching baseline.", "",
          "## Principal findings", "",
          "- **Complete simple-opponent experiments:** selected-skill pilots 002 and 010 form the pooled observed frontier. Pilot 002 records 83% elimination, 73% survival and 60% clean wins; pilot 010 records 67%, 74% and 46%, respectively. The apparent survival benefit is only one percentage point.",
          f"- **Sensitivity to scenario sampling:** those two pilots remain on the pooled frontier in {100*simple_stability[0]:.1f}% and {100*simple_stability[1]:.1f}% of paired scenario resamples, respectively. This supports caution about interpreting pilot 010's small survival advantage as a stable specialisation. These frequencies are descriptive, not posterior probabilities or significance levels.",
          "- **Tactical assessment:** adapted pilots 005 and 007 form the observed two-objective frontier. Their survival rates differ by only one case out of 300 (90.33% versus 90.00%). Adapted pilot 010 has the highest clean-win rate (43.33%) despite being off that frontier; adding clean wins as a third objective retains it.",
          "- **Partial bonus experiment:** the matched selected-skill pilot 002 dominates all six completed bonus-trial pilots on the two plotted point estimates. The full ten-pilot bonus population was not completed.", "",
          "## Figures", ""]
    for name, note in figures:
        md.extend([f"### {name.replace('_', ' ').capitalize()}", "",
                   f"[PNG]({name}.png) · [Vector PDF]({name}.pdf)", "", note, "",
                   f"![{name.replace('_', ' ')}]({name}.png)", ""])
    md.extend(["## Numerical frontiers", "",
               "All figures below are percentages. Frontier frequency is the percentage of 2,000 paired scenario resamples in which the point remains non-dominated within the stated candidate set. It is a sensitivity measure, not the probability that a pilot is truly optimal.", ""])
    for group in [*("simple/" + p for p in FULL), "simple/full_comparison", "simple/bonus_matched_subset",
                  "tactical_assessment/before", "tactical_assessment/after", "tactical_assessment/comparison",
                  "tactical_confirmation/before", "tactical_confirmation/after", "tactical_confirmation/comparison"]:
        md.extend(["### " + group.replace("_", " "), "",
                   "| Population | Pilot | Elimination | Survival | Clean win | Mutual destruction | Frontier frequency |",
                   "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
        for identity in groups[group]["frontier_2d"]:
            p = by_id[identity]
            freq = frequencies[(group, identity)]["scenario_bootstrap_frontier_frequency_2d"]
            md.append(f"| {p['label']} | {p['pilot'].split('_')[-1]} | " +
                      " | ".join(f"{100*p[m]:.2f}%" for m in METRICS) + f" | {100*freq:.1f}% |")
        md.append("")
    md.extend(["## Validity and limitations", "",
               "- **Useful for marginal outcome trade-offs.** The axes compare confirmed elimination and own survival. They are overlapping outcomes; neither is a reward total. Mutual destruction is reported separately and is not used to infer eliminations.",
               "- **Clean wins remain a separate objective.** Marginal elimination and survival do not determine how often both occur together. On tactical assessment, adapted pilot 010 has the highest clean-win rate (43.33%) but is dominated on both plotted axes by pilot 007. Discarding every point off this two-dimensional frontier would therefore discard the best observed clean-win pilot. `data/frontiers.csv` also includes the non-dominated set when clean-win rate is added as a third maximisation objective; mutual destruction is still reported rather than included as a fourth objective.",
               "- **Finite-scenario uncertainty.** Each suite uses the same cases for every pilot. There are 100 distinct scenario IDs in each simple or historical-confirmation suite, and 300 in tactical assessment, not ten times those counts. We resample scenario IDs jointly across every candidate and all outcomes. `data/points.csv` contains marginal percentile intervals; `data/frontiers.csv` contains frontier-membership frequencies. Small differences and point-estimate dominance are not statistical superiority claims, and marginal intervals are not a joint dominance test.",
               "- **Conditional bootstrap interpretation.** Resampling measures sensitivity to this empirical mix of scenarios under an exchangeable-scenario assumption. It does not measure training-seed variation, uncertainty across aircraft or opponents, or selection bias from repeated development-benchmark inspection. Checkpoints remain fixed in every replicate.",
               "- **Candidate-set dependence.** A pilot can be on its own experiment's frontier while being dominated by another experiment's pilot. A larger frontier is not automatically better. The pooled figure compares the forty complete simple-opponent checkpoints directly; the partial bonus trial is kept separate.",
               "- **Previously inspected development evidence.** These reports are useful for interpretation and further development, not an untouched final test. A final evaluation should use common unused cases after experimental choices are fixed.", "",
               "## Reproduction and evidence", "",
               "View the PDF, PNG files or standalone HTML on a presentation PC without Python, model weights or Git LFS. The original slide pack remains unchanged.", "",
               "Rebuild from packaged outcomes with `python scripts/build_pareto_presentation.py`. The optional `--collect` mode refreshes the package from local saved reports, verifies checkpoint identifiers and exact simple-scenario geometry, and cross-checks all complete populations against the published outcome tables. It never runs the simulator.", "",
               "- [Episode outcome flags](data/episodes.csv)",
               "- [Per-pilot rates and 95% marginal scenario-bootstrap intervals](data/points.csv)",
               "- [Two- and three-objective frontier membership and resampling frequencies](data/frontiers.csv)",
               "- [Analysis definitions](data/analysis.json)",
               "- [Source hashes and collection scope](data/provenance.json)",
               "- [Package checksums](checksums.json)", ""])
    (dest / "README.md").write_text("\n".join(md), encoding="utf-8", newline="\n")
    sections = []
    for name, note in figures:
        encoded = base64.b64encode((dest / (name + ".png")).read_bytes()).decode("ascii")
        sections.append(f'<section><h2>{html.escape(name.replace("_", " ").capitalize())}</h2><p>{html.escape(note)}</p>'
                        f'<img alt="{html.escape(name)}" src="data:image/png;base64,{encoded}"></section>')
    markup = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    markup += '<title>Pilot Pareto comparisons</title><style>body{background:#101e30;color:#f2f5f9;font:18px system-ui;max-width:1600px;margin:auto;padding:24px}img{width:100%}p{line-height:1.6;max-width:1000px}section{margin:48px 0}a{color:#8baeff}@media print{section{break-before:page}}</style>'
    markup += '<h1>Observed Pareto frontiers across pilot experiments</h1><p>Each point is a selected pilot checkpoint. Higher survival and elimination are preferred. White rings mark the observed frontier; stars identify clean-win leaders. A star without a surrounding ring can be off the frontier. These are finite-sample comparisons, not proofs of optimality. The two axes do not replace the joint clean-win objective.</p>'
    markup += '<p>Simple, tactical assessment and historical confirmation use separate comparison sets. The bonus trial is incomplete. See README.md for outcome tables, scenario-bootstrap sensitivity and evidence. All figures are embedded; this page works offline.</p>'
    (dest / "index.html").write_text(markup + "".join(sections) + "</html>", encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    if args.collect:
        collect(DEST)
    points, groups, frontier_rows = analyse(DEST)
    figures = render(DEST, points)
    report(DEST, points, groups, frontier_rows, figures)
    write_json(DEST / "checksums.json", {p.relative_to(DEST).as_posix(): digest(p)
                                       for p in sorted(DEST.rglob("*")) if p.is_file() and p.name != "checksums.json"})
    # Keep the enclosing presentation manifest consistent without rebuilding its slides.
    if (DEST.parent / "checksums.json").exists():
        write_json(DEST.parent / "checksums.json", {
            p.relative_to(DEST.parent).as_posix(): digest(p)
            for p in sorted(DEST.parent.rglob("*")) if p.is_file() and p.name != "checksums.json"
        })
    print(json.dumps({"plots": len(figures), "candidates": len(points), "output": str(DEST),
                      "pooled_simple_frontier": groups["simple/full_comparison"]["frontier_2d"]}, indent=2))


if __name__ == "__main__":
    main()
