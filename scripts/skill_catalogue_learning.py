"""Package recorded learning curves and explain the temporal MLP without refitting."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from build_forecasting_insights import BLUE, FG, GOLD, MUTED, TEAL
from skill_catalogue_evidence import read_json, sha, write_json

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "history20_mlp_accuracy": "20 s / standard",
    "history20_mlp_wide_accuracy": "20 s / wider",
    "history20_mlp_smooth_accuracy": "20 s / more smoothing",
    "history60_mlp_accuracy": "60 s / standard",
}


def collect_learning(dest):
    """Copy existing epoch metrics; read no trajectories, targets or model weights."""
    data = dest / "data"
    sources, runs, frames = {}, {}, []

    def record(path):
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
        return path

    for name, expected in read_json(data / "checksums.json").items():
        assert sha(data / name) == expected, name
    evidence = read_json(data / "evidence.json")
    experiment = ROOT / "artifacts/skill_forecasting"

    def add_run(path, phase, variant, fold, seed):
        history = pd.read_csv(record(path / "history.csv"))
        summary = read_json(record(path / "summary.json"))
        config = read_json(record(path / "run.json"))
        np.testing.assert_array_equal(history.epoch, np.arange(1, len(history) + 1))
        assert len(history) == summary["epochs_completed"]
        assert 1 <= summary["best_epoch"] <= len(history)
        name = f"{phase}/{path.name}" if fold < 0 else f"inner_{fold}/{path.name}"
        runs[name] = {
            "phase": phase,
            "variant": variant,
            "fold": fold,
            "seed": seed,
            "epochs": len(history),
            "selected_epoch": summary["best_epoch"],
        }
        if phase == "original":
            assert config["class_weights"] is None
            history = history.rename(columns={"train_loss": "training_objective"})
            runs[name]["selection"] = "outer validation NLL with recorded min_delta"
        else:
            assert config["candidate"]["auxiliary_weight"] == 0
            runs[name]["selection"] = (
                "fixed epochs" if phase == "refit" else "internal stop accuracy"
            )
            runs[name]["smoothing"] = config["candidate"]["smoothing"]
            if phase == "refit":
                assert list(history.columns) == ["epoch", "training_objective"]
                assert summary["best_epoch"] == len(history)
            else:
                history = history.rename(columns={"stop_nll": "validation_nll"})
        history = history.assign(run=name, phase=phase, variant=variant, fold=fold, seed=seed)
        frames.append(
            history[
                [
                    column
                    for column in (
                        "run",
                        "phase",
                        "variant",
                        "fold",
                        "seed",
                        "epoch",
                        "training_objective",
                        "validation_nll",
                        "stop_mean_accuracy",
                    )
                    if column in history
                ]
            ]
        )

    for model in ("gru", "transformer"):
        add_run(experiment / "initial_v1" / model, "original", model, -1, 42)
    for component in evidence["components"]:
        name = component["candidate"]
        assert name in VARIANTS
        for fold in (0, 1):
            add_run(
                experiment / f"improvement_v3/inner_{fold}/{name}_s42", "internal", name, fold, 42
            )
        for seed in component["seeds"]:
            add_run(experiment / f"improvement_v3/refit/{name}_s{seed}", "refit", name, -1, seed)
    pd.concat(frames, ignore_index=True).to_csv(
        data / "forecast_learning_curves.csv", index=False, lineterminator="\n"
    )

    # Match the upstream pilot training logs to the exact frozen corpus policies.
    presentation = ROOT / "docs/presentations/data"
    original = read_json(record(presentation / "results.json"))
    inventory = read_json(record(presentation / "source_hashes.json"))
    corpus = read_json(record(ROOT / "shared/corpora/trained-pilot-episodes-v1/config.json"))
    corpus_hashes = {p["pilot_id"]: p["checkpoint_sha256"] for p in corpus["pilots"]}
    pilot_rows, pilot_metadata = [], []
    for pilot in original["pilots"]:
        pid = pilot["pilot"]
        assert pilot["selected_sha256"] == corpus_hashes[pid]
        source = ROOT / original["job"] / "training" / pid / "training_metrics.jsonl"
        assert sha(source) == inventory[source.relative_to(ROOT).as_posix()]
        metrics = [json.loads(line) for line in record(source).read_text().splitlines()]
        assert [m["epoch"] for m in metrics] == list(range(1, 151))
        for m in metrics:
            pilot_rows.append(
                {
                    "pilot": pid,
                    "epoch": m["epoch"],
                    "training_return": m["training_mean_return"],
                    "validation_return": m["mean_return"] if m["evaluation_performed"] else None,
                }
            )
        recorded_validation = [
            {"epoch": m["epoch"], "return": m["mean_return"]}
            for m in metrics
            if m["evaluation_performed"]
        ]
        assert recorded_validation == pilot["validation"]
        pilot_metadata.append(
            {
                "pilot": pid,
                "selected_epoch": pilot["selected_epoch"],
                "checkpoint_sha256": corpus_hashes[pid],
            }
        )
    pd.DataFrame(pilot_rows).to_csv(
        data / "pilot_learning_curves.csv", index=False, lineterminator="\n"
    )
    write_json(
        data / "learning_evidence.json",
        {
            "runs": runs,
            "pilots": pilot_metadata,
            "source_sha256": sources,
            "per_pilot_forecast_epoch_metrics_available": False,
            "training_performed": False,
            "test_evaluated": False,
            "contract": "Recorded losses/returns only; no new fitting, inference or checkpoint selection.",
        },
    )
    write_json(
        data / "checksums.json",
        {
            path.name: sha(path)
            for path in sorted(data.iterdir())
            if path.is_file() and path.name != "checksums.json"
        },
    )
    print(f"Packaged {len(runs)} forecast histories and ten matched RL pilot histories.")


def append_learning_slides(dest, page, finish, words, chart, table):
    data = dest / "data"
    evidence = read_json(data / "learning_evidence.json")
    curves = pd.read_csv(data / "forecast_learning_curves.csv")
    pilots = pd.read_csv(data / "pilot_learning_curves.csv")

    f = page(
        "How each RL pilot learned: training and validation",
        "Upstream tactical adaptation of the ten frozen corpus pilots | Own episode return; higher is better",
        "Recorded RL training logs; corpus checkpoint hashes matched",
    )
    for i, pilot in enumerate(evidence["pilots"]):
        ax = chart(f, [0.065 + (i % 5) * 0.184, 0.48 - (i // 5) * 0.29, 0.153, 0.20])
        frame = pilots[pilots.pilot.eq(pilot["pilot"])]
        validation = frame.dropna(subset=["validation_return"])
        ax.plot(frame.epoch, frame.training_return, color=BLUE, linewidth=1, alpha=0.8)
        ax.plot(
            validation.epoch,
            validation.validation_return,
            color=TEAL,
            linewidth=1.6,
            marker=".",
            markersize=3,
        )
        selected = validation[validation.epoch.eq(pilot["selected_epoch"])].iloc[0]
        ax.scatter(
            [selected.epoch], [selected.validation_return], color=GOLD, marker="*", s=75, zorder=5
        )
        ax.set_title(
            f"Pilot {i + 1:03}  |  selected {pilot['selected_epoch']}", color=FG, fontsize=11
        )
        ax.tick_params(labelsize=9)
        ax.set_xticks([1, 75, 150])
        if i >= 5:
            ax.set_xlabel("RL epoch", fontsize=10)
    f.text(
        0.065,
        0.735,
        "BLUE: training (75 fresh cases / epoch)     GREEN: validation (50 fixed cases)     STAR: selected checkpoint",
        fontsize=13,
    )
    f.text(
        0.065,
        0.09,
        "Panels use individual reward scales: compare each pilot with itself. These are flight-learning curves, not forecasting accuracy.",
        fontsize=14,
        color=GOLD,
    )
    finish(
        f,
        (
            "These are the ten RL pilots' tactical-adaptation training histories, before they were frozen "
            "to generate forecasting encounters. All ten checkpoint hashes match the corpus configuration. "
            "Blue is raw mean training return each epoch; green shows only actual validation evaluations "
            "(epoch 1 and every fifth epoch), joined for readability. Stars mark the recorded selected "
            "checkpoints. Training uses stochastic actions on 75 fresh cases, validation uses evaluation "
            "actions on 50 fixed cases. Their gap is not a direct supervised generalisation gap. Each "
            "pilot has different reward weights; separate y scales preserve within-pilot trends but cannot "
            "rank pilot competence. No smoothing, new simulation or new selection was performed."
        ),
    )

    f = page(
        "How the original skill forecasters learned",
        "One shared forecaster across all ten pilots | Training loss versus development-validation NLL",
        "Original saved epoch histories",
    )
    for i, variant in enumerate(("gru", "transformer")):
        ax = chart(f, [0.085 + i * 0.47, 0.30, 0.38, 0.42])
        frame = curves[curves.variant.eq(variant)]
        info = evidence["runs"][f"original/{variant}"]
        ax.plot(
            frame.epoch,
            frame.training_objective,
            color=BLUE,
            label="Training cross-entropy",
            linewidth=2,
        )
        ax.plot(frame.epoch, frame.validation_nll, color=TEAL, label="Validation NLL", linewidth=2)
        ax.axvline(
            info["selected_epoch"],
            color=GOLD,
            linestyle=":",
            label=f"Selected epoch {info['selected_epoch']}",
        )
        ax.set_title(variant.upper() if i == 0 else "Transformer", color=FG, fontsize=18)
        ax.set_xlabel("Forecast-training epoch")
        ax.set_ylabel("Loss / NLL (lower is better)")
        ax.legend(fontsize=10, facecolor="#101e30", labelcolor=FG)
    words(
        f,
        0.065,
        0.19,
        "READ THE DIVERGENCE",
        "Training loss continues down; validation stops improving. Keep the selected checkpoint, not the final epoch.",
        GOLD,
        15,
    )
    finish(
        f,
        (
            "Original GRU and transformer logs pool all ten pilots and average the three forecast horizons. "
            "Selected epochs are 15 and 20 respectively, using validation NLL and the configured minimum "
            "improvement threshold. Therefore the selected epoch need not be the absolute numeric minimum. "
            "Training loss is measured during parameter updates with dropout; validation uses a fixed "
            "end-of-epoch model without dropout. The widening gap is consistent with overfitting, not a "
            "test for information leakage. These are recorded uncalibrated learning losses, not the "
            "ensemble's final probability scores. Per-pilot forecast losses were not logged each epoch; "
            "slide 11 shows final per-pilot validation accuracy only."
        ),
    )

    f = page(
        "Improved MLPs: learning inside the training split",
        "Four selected variants, both internal partitions | Seed 42 | 147 fit groups and 21 stopping groups",
        "Saved internal histories; outer validation excluded",
    )
    for i, (variant, label) in enumerate(VARIANTS.items()):
        ax = chart(f, [0.085 + (i % 2) * 0.47, 0.48 - (i // 2) * 0.29, 0.38, 0.20])
        for fold, style in ((0, "-"), (1, "--")):
            frame = curves[
                curves.variant.eq(variant) & curves.phase.eq("internal") & curves.fold.eq(fold)
            ]
            info = evidence["runs"][f"inner_{fold}/{variant}_s42"]
            ax.plot(
                frame.epoch, frame.training_objective, color=BLUE, linestyle=style, linewidth=1.6
            )
            ax.plot(frame.epoch, frame.validation_nll, color=TEAL, linestyle=style, linewidth=1.6)
            selected = frame[frame.epoch.eq(info["selected_epoch"])].iloc[0]
            ax.scatter(
                [selected.epoch], [selected.validation_nll], color=GOLD, marker="*", s=80, zorder=5
            )
        ax.set_title(label, color=FG, fontsize=14)
        ax.set_ylabel("Objective / NLL", fontsize=10)
        ax.tick_params(labelsize=10)
        if i >= 2:
            ax.set_xlabel("Forecast-training epoch", fontsize=10)
    f.text(
        0.065,
        0.735,
        "BLUE: training objective    GREEN: internal validation NLL    SOLID / DASHED: splits 1 / 2    STAR: accuracy-selected epoch",
        fontsize=12,
    )
    f.text(
        0.065,
        0.09,
        "Training uses label smoothing; validation NLL does not. Their vertical gap is not a like-for-like loss comparison.",
        fontsize=14,
        color=GOLD,
    )
    finish(
        f,
        (
            "All four selected variants are shown on both internal partitions, using the saved seed-42 "
            "search runs. No favourable run is singled out. These stopping groups are within the original "
            "training partition, separate from the outer development validation. Stars mark recorded "
            "accuracy-selected epochs, not minima of the plotted NLL. Each training objective uses its "
            "variant's label smoothing (0.02, 0.03, 0.08 or 0.02) and training-mode dropout, whereas stopping "
            "NLL is ordinary unsmoothed evaluation NLL before temperature calibration. Falling training "
            "objective with rising stopping NLL can flag deterioration in probability quality even when "
            "accuracy is stable. These pooled curves neither prove absence of overfitting nor establish "
            "per-pilot learning behaviour. Exact stop accuracy is also included in the portable CSV."
        ),
    )

    f = page(
        "Final ensemble refits: twelve training curves",
        "Four variants × three seeds | All 210 training groups | Fixed epoch counts",
        "Saved final-refit histories; no validation callbacks",
    )
    for i, (variant, label) in enumerate(VARIANTS.items()):
        ax = chart(f, [0.085 + (i % 2) * 0.47, 0.48 - (i // 2) * 0.29, 0.38, 0.20])
        for seed, color in zip((42, 43, 44), (BLUE, TEAL, GOLD)):
            frame = curves[
                curves.variant.eq(variant) & curves.phase.eq("refit") & curves.seed.eq(seed)
            ]
            ax.plot(frame.epoch, frame.training_objective, color=color, linewidth=1.6)
        ax.set_title(f"{label} / {int(frame.epoch.max())} epochs", color=FG, fontsize=14)
        ax.set_ylabel("Training objective", fontsize=10)
        ax.tick_params(labelsize=10)
        if i >= 2:
            ax.set_xlabel("Forecast-training epoch", fontsize=10)
    f.text(
        0.065,
        0.735,
        "BLUE: seed 42                       GREEN: seed 43                       GOLD: seed 44",
        fontsize=14,
    )
    f.text(
        0.065,
        0.09,
        "No final-refit validation curve was logged: epoch budgets were fixed in advance. No per-pilot forecast epoch curves were saved.",
        fontsize=13,
        color=GOLD,
    )
    finish(
        f,
        (
            "All twelve final component runs are shown, with no newly trained model. Their epoch counts "
            "were fixed from internal development before refitting on all 210 training groups: 46, 32, "
            "40 and 40. Refits deliberately had no outer-validation callback or early stopping. Training "
            "objectives include variant-specific label smoothing, so compare trajectories within a panel "
            "rather than treating heights across variants as a ranking. There is no single jointly "
            "trained ensemble loss curve: its members were fitted separately, calibrated with previously "
            "fixed temperatures and their probabilities averaged. Final per-pilot forecast evaluation "
            "exists, but per-pilot epoch histories do not. Recovering complete such histories would "
            "require a separate instrumented experiment, preserving the existing results."
        ),
    )

    f = page(
        "Temporal MLP: a history window becomes one table row",
        "Temporal = past values in ordered columns | Feed forward = input → hidden layers → output",
        "Implementation: ImprovedForecaster.forward; causal multiscale history",
    )
    words(
        f,
        0.065,
        0.71,
        "ONE ORIGIN: THREE TIMESTAMPS",
        "Toy example with two features; values are illustrative.",
        BLUE,
        14,
    )
    table(
        f,
        [0.065, 0.43, 0.37, 0.17],
        [["t − 2 s", "250", "12"], ["t − 1 s", "255", "11"], ["t", "260", "10"]],
        ["Time", "Speed (m/s)", "Range (km)"],
        size=15,
    )
    words(
        f,
        0.49,
        0.71,
        "FLATTEN → ONE ORDINARY ROW",
        "[250, 12, 255, 11, 260, 10]\n\nColumns: speed(t−2), range(t−2), …, range(t)",
        TEAL,
        16,
    )
    f.text(
        0.065,
        0.36,
        "Many origins: [N, 3 timestamps, 2 features]  →  X = [N rows, 6 columns]",
        fontsize=20,
        weight="bold",
    )
    f.text(
        0.065,
        0.275,
        "Actual 20 s model:  [N, 26, 91] + availability flags  →  [N, 2,392]",
        fontsize=19,
        color=TEAL,
    )
    f.text(
        0.065,
        0.22,
        "Actual 60 s model:  [N, 66, 91] + availability flags  →  [N, 6,072]",
        fontsize=19,
        color=TEAL,
    )
    f.text(
        0.065,
        0.135,
        "Flat row → dense residual layers → 99 scores → 3 horizons × 33 skill probabilities",
        fontsize=19,
        weight="bold",
    )
    f.text(
        0.065,
        0.08,
        "Inputs stop at t. Targets are labels at t+1, t+2 and t+5. The MLP has no recurrent state.",
        fontsize=15,
        color=MUTED,
    )
    finish(
        f,
        (
            "N is the number of forecast origins, or the mini-batch size during training. The 3D array "
            "organises examples, timestamps and features; it does not mean three spatial dimensions. "
            "Flattening keeps all feature values and their time positions as distinct columns. In the "
            "implementation each timestamp also has a validity flag, so 26 × (91 + 1) = 2,392 and "
            "66 × 92 = 6,072 columns. Older samples are one second apart; the last five seconds use "
            "half-second spacing including the present. Unavailable history at episode start is "
            "zero-padded after normalisation and identified by the flag. The toy row omits flags and "
            "normalisation for clarity. Each ordinary feed-forward residual MLP outputs 99 logits, "
            "reshaped into three sets of 33; softmax operates separately per horizon. Twelve members' "
            "calibrated probabilities are then averaged. Features at time t or earlier form X; future "
            "skill labels form three target columns y and are never appended to X. Scenario-group "
            "splitting keeps overlapping history windows and all pilots from a scenario together."
        ),
    )


def append_learning_report(dest):
    text = """
## Learning curves and what was actually logged

Slides 22–26 extend the original 21 slides without renumbering them.

| Slide | Evidence | Meaning / limit |
| --- | --- | --- |
| 22 | Ten RL pilots' tactical-adaptation histories | Training and validation episode return, not skill-forecast accuracy. Checkpoints match the corpus. |
| 23 | Original GRU and transformer histories | Population-pooled training cross-entropy and outer development-validation NLL. Selected epochs 15 / 20. |
| 24 | Four MLP variants on both internal partitions | Training objective and stopping-set NLL; stars use accuracy-selected epochs. |
| 25 | All twelve final refits | Training objective only; epoch budgets fixed before outer validation. |
| 26 | Temporal MLP worked example | The time/feature dimensions are flattened into one ordinary feature row. |

RL return is higher-is-better; forecasting loss/NLL is lower-is-better. RL training
and validation use different cases and action selection. Pilots have different
reward scales, so their returns cannot rank competence. Validation points are
only plotted at actual evaluation epochs; no missing evaluation is fabricated.

The forecasters pool all ten pilots. **Per-pilot forecasting losses or accuracies
were not saved each epoch.** Final per-pilot validation scores exist (slide 11),
but are not learning curves. Full per-pilot curves require a new, separately
logged experiment; the existing artifacts cannot supply them.

Original training losses are recorded during updates with dropout; validation
NLL is computed after each epoch with dropout disabled. For the improved MLPs,
training also uses label smoothing while validation NLL does not. Do not interpret
the vertical gap as a pure generalisation gap. Internal checkpoint selection used
accuracy, so stars need not coincide with NLL minima. Final refits have no validation
callbacks and no validation curves. A declining training loss alone does not prove
generalisation, and learning curves cannot prove absence of information leakage.

The portable CSVs contain every recorded epoch for the plotted runs, including
internal stopping accuracy. `learning_evidence.json` records source hashes and
the original selected epochs. Plotting performed no fitting, inference, checkpoint
selection, or reserved-test evaluation.

## Temporal MLP for someone used to 2D tables

An ordinary table has shape **[examples, features]**. A history-window dataset is
conveniently stored as **[examples, timestamps, features per timestamp]**. For the
MLP, the latter two axes are flattened: each lag gets its own feature columns.

For example, speed/range at three times becomes one row:
`[speed(t-2), range(t-2), speed(t-1), range(t-1), speed(t), range(t)]`.
The model can learn changes by comparing values in these fixed columns.

| Model history | Stored observations | Add a validity flag per time | Actual 2D MLP input |
| --- | --- | --- | --- |
| 20 s | N × 26 × 91 | N × 26 × 92 | N × 2,392 |
| 60 s | N × 66 × 91 | N × 66 × 92 | N × 6,072 |

The most recent five seconds use half-second spacing, while older observations
use one-second spacing. A validity flag distinguishes real observations from
padding when an episode is younger than the history window. Values are normalised
using training data only. N can denote the full dataset or a mini-batch.

**Feed forward** means the row passes through dense hidden layers to outputs,
without recurrent state. **Temporal** describes the ordered past observations
used as input. The network emits 99 scores, reshaped into three distributions
over 33 skills, for +1 / +2 / +5 seconds. Its three future labels are targets y;
they are not input features X. Twelve separately trained members' calibrated
probabilities are averaged for the final ensemble. Scenario-group splits keep
overlapping windows from the same encounter on the same side of the split.
"""
    path = dest / "comparison.md"
    path.write_text(path.read_text(encoding="utf-8") + text, encoding="utf-8", newline="\n")
