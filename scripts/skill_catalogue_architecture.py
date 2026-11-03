"""Audit saved skill-model architecture, fitting timers and experiment evolution."""

from pathlib import Path

import numpy as np
import pandas as pd
from build_forecasting_insights import BLUE, FG, GOLD, TEAL
from matplotlib.patches import FancyBboxPatch
from skill_catalogue_evidence import read_json, sha, write_json

ROOT = Path(__file__).resolve().parents[1]
VARIANT_LABELS = {
    "history20_mlp_accuracy": "20 s / standard",
    "history20_mlp_wide_accuracy": "20 s / wider",
    "history20_mlp_smooth_accuracy": "20 s / smoothing",
    "history60_mlp_accuracy": "60 s / standard",
}


def collect_architecture(dest):
    """Read completed-run metadata only; never fit or load a model checkpoint."""
    data = dest / "data"
    data.mkdir(parents=True, exist_ok=True)
    sources = {}

    def record(path):
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
        return path

    experiment = ROOT / "artifacts/skill_forecasting"
    rows = []
    for version in ("initial_v1", "improvement_v2", "improvement_v3"):
        for path in sorted((experiment / version).glob("**/summary.json")):
            relative = path.relative_to(experiment / version)
            if "validation" in relative.parts or "notebook_check" in relative.parts:
                continue
            summary = read_json(path)
            if "elapsed_seconds" not in summary:
                continue
            summary = read_json(record(path))
            run = read_json(record(path.parent / "run.json"))
            original = version == "initial_v1"
            model = run["model"]
            phase = "original" if original else relative.parts[0]
            stage = version if original else ("final_refits" if phase == "refit" else version)
            rows.append(
                {
                    "version": version,
                    "phase": phase,
                    "stage": stage,
                    "run": path.parent.relative_to(experiment).as_posix(),
                    "variant": model["architecture"] if original else summary["name"],
                    "seed": run["training"]["seed"] if original else summary["seed"],
                    "architecture": model["architecture"],
                    "input_features": model["input_size"],
                    "history_tokens": model["history_steps"] if original else model["tokens"],
                    "width": model["width"],
                    "layers_or_residual_blocks": model["layers"],
                    "parameters": run["parameters"],
                    "epochs_completed": summary["epochs_completed"],
                    "selected_epoch": summary["best_epoch"],
                    "elapsed_seconds": summary["elapsed_seconds"],
                }
            )
    inventory = pd.DataFrame(rows)
    assert len(inventory) == 36 and inventory.run.is_unique
    assert np.isfinite(inventory.elapsed_seconds).all() and (inventory.elapsed_seconds > 0).all()
    inventory.to_csv(data / "training_run_inventory.csv", index=False, lineterminator="\n")

    final_dir = experiment / "improvement_v3"
    frozen = read_json(record(final_dir / "frozen_selection.json"))
    protocol = read_json(record(final_dir / "protocol.json"))
    environment = read_json(record(final_dir / "environment.json"))
    for name in ("design.json", "ensemble_protocol.json"):
        record(final_dir / name)
    components = []
    for item in frozen["components"]:
        name = item["candidate"]
        records = inventory[inventory.stage.eq("final_refits") & inventory.variant.eq(name)]
        assert set(records.seed) == set(item["seeds"])
        assert records.parameters.nunique() == 1
        assert records.epochs_completed.eq(item["epochs"]).all()
        run = read_json(experiment / records.iloc[0].run / "run.json")
        model, settings = run["model"], run["candidate"]
        assert model["architecture"] == "mlp" and not model["auxiliary"]
        columns = model["tokens"] * (model["input_size"] + 1)
        width = model["width"]
        # Input projection + three pre-norm residual blocks + output norm + 99 logits.
        expected_parameters = columns * width + 12 * width**2 + 117 * width + 99
        assert model["layers"] == 3 and run["parameters"] == expected_parameters
        components.append(
            {
                **item,
                "label": VARIANT_LABELS[name],
                "features_per_timestamp": model["input_size"],
                "history_tokens": model["tokens"],
                "input_columns_with_flags": columns,
                "history_seconds": settings["history_seconds"],
                "width": width,
                "block_expansion_width": 2 * width,
                "residual_blocks": model["layers"],
                "parameters_per_member": run["parameters"],
                "dropout": settings["dropout"],
                "label_smoothing": settings["smoothing"],
                "weight_decay": settings["weight_decay"],
                "refit_seconds_sum": float(records.elapsed_seconds.sum()),
                "refit_seconds_min": float(records.elapsed_seconds.min()),
                "refit_seconds_max": float(records.elapsed_seconds.max()),
            }
        )

    timing = []
    for stage in ("initial_v1", "improvement_v2", "improvement_v3", "final_refits"):
        frame = inventory[inventory.stage.eq(stage)]
        timing.append(
            {
                "stage": stage,
                "runs": len(frame),
                "epochs_sum": int(frame.epochs_completed.sum()),
                "elapsed_seconds_sum": float(frame.elapsed_seconds.sum()),
                "elapsed_seconds_min": float(frame.elapsed_seconds.min()),
                "elapsed_seconds_max": float(frame.elapsed_seconds.max()),
            }
        )

    v2 = pd.read_csv(record(experiment / "improvement_v2/search_results.csv"))
    v3 = pd.read_csv(record(final_dir / "search_results.csv"))
    mixtures = pd.read_csv(record(final_dir / "ensemble_confirmation.csv"))
    evolution = []

    def outer(label, path, description):
        scores = pd.read_csv(record(path))
        population = scores[scores.pilot_id.eq("population") & scores.cohort.eq("all")]
        assert set(population.horizon_s) == {1, 2, 5}
        evolution.append(
            {
                "label": label,
                "partition": "Outer validation",
                "groups": 45,
                "mean_accuracy": float(population.top1_accuracy.mean()),
                "mean_nll": float(population.nll.mean()),
                "change": description,
            }
        )

    def inner(label, frame, name, description):
        score = frame[frame.name.eq(name) & frame.fold.eq(0)].iloc[0]
        evolution.append(
            {
                "label": label,
                "partition": "Internal split 1",
                "groups": 42,
                "mean_accuracy": float(score.mean_accuracy),
                "mean_nll": float(score.nll),
                "change": description,
            }
        )

    outer("Original GRU", experiment / "initial_v1/gru/validation_metrics.csv", "5 s / 54 features")
    outer(
        "Original transformer",
        experiment / "initial_v1/transformer/validation_metrics.csv",
        "5 s / 54 features",
    )
    inner("v2 control", v2, "control", "Matched internal reference")
    inner("v2 temporal MLP", v2, "history20_mlp", "20 s / geometry; NLL stopping")
    inner("v3 long-history MLP", v3, "history60_mlp_accuracy", "Accuracy stopping + calibration")
    inner("v3 four-variant mixture", mixtures, "accuracy_all", "Uniform probability averaging")
    outer(
        "Final 12-model ensemble",
        final_dir / "validation/ensemble/validation_metrics.csv",
        "Frozen recipe; three seeds each",
    )

    for path in (
        "bvr_behavior_prediction/skill_forecasting/improved_models.py",
        "bvr_behavior_prediction/skill_forecasting/improved_features.py",
        "bvr_behavior_prediction/skill_forecasting/improvement.py",
        "bvr_behavior_prediction/skill_forecasting/ensemble_search.py",
        "bvr_behavior_prediction/skill_forecasting/temperature.py",
        "bvr_behavior_prediction/skill_forecasting/train.py",
        "bvr_behavior_prediction/skill_forecasting/models.py",
        "bvr_behavior_prediction/forecasting/models.py",
    ):
        record(ROOT / path)
    write_json(
        data / "architecture_review.json",
        {
            "components": components,
            "total_parameters": int(inventory[inventory.stage.eq("final_refits")].parameters.sum()),
            "timing_stages": timing,
            "all_logged_fit_seconds": float(inventory.elapsed_seconds.sum()),
            "environment": environment,
            "optimizer": {
                "name": "AdamW",
                "fused": True,
                "learning_rate": protocol["learning_rate"],
                "batch_size": protocol["batch_size"],
                "gradient_clip_norm": 1.0,
                "schedule": "cosine over 60 epochs to 0.00003",
                "autocast": "CUDA bfloat16",
            },
            "experiment_evolution": evolution,
            "internal_confirmation": frozen["paired_internal_results"],
            "timing_contract": (
                "Sum of each completed run's elapsed_seconds once, excluding copied validation summaries. "
                "Improved timers start after data loading, feature construction, model/optimizer creation; "
                "include fitting, stopping evaluation where used, checkpoint writes, and temperature fitting; "
                "stop before select-group prediction. Original timers include final validation/report writing. "
                "Not end-to-end workflow time, GPU kernel time, or simulation time."
            ),
            "source_sha256": sources,
            "training_performed": False,
            "model_inference_performed": False,
            "test_evaluated": False,
        },
    )
    print(f"Audited {len(inventory)} saved fit timers and {len(components)} frozen MLP variants.")


def append_architecture_slides(dest, page, finish, words, chart, table):
    evidence = read_json(dest / "data/architecture_review.json")
    components = evidence["components"]

    f = page(
        "Inside one skill-forecasting network",
        "Worked example: standard 20-second MLP | A supervised predictor of frozen pilots' future selections",
        "Audited model implementation and saved run configurations",
    )
    stages = [
        ("HISTORY", "26 × 91 values\n+ 26 validity flags"),
        ("FLATTEN", "One row\n2,392 columns"),
        ("PROJECT", "Linear 2,392 → 256\nGELU activation"),
        ("REFINE", "3 residual blocks\nwidth 256"),
        ("PREDICT", "LayerNorm + Linear\n256 → 99 scores"),
    ]
    for i, (title, body) in enumerate(stages):
        x = 0.065 + i * 0.18
        f.add_artist(
            FancyBboxPatch(
                (x, 0.545),
                0.158,
                0.17,
                transform=f.transFigure,
                boxstyle="round,pad=0.008",
                linewidth=1.2,
                edgecolor=BLUE,
                facecolor="#1a3048",
            )
        )
        f.text(x + 0.008, 0.68, title, fontsize=15, color=TEAL, weight="bold")
        f.text(x + 0.008, 0.625, body, fontsize=12, color=FG, linespacing=1.5, va="top")
        if i < 4:
            f.text(x + 0.164, 0.615, "→", fontsize=22, color=BLUE)
    f.text(
        0.065,
        0.465,
        "Inside each residual block:  x + [LayerNorm → Linear 256→512 → GELU → Dropout → Linear 512→256 → Dropout]",
        fontsize=13,
        color=FG,
    )
    f.text(
        0.065,
        0.385,
        "99 scores → 3 horizons × 33 skills → temperature scaling → softmax per horizon",
        fontsize=20,
        weight="bold",
        color=TEAL,
    )
    words(
        f,
        0.065,
        0.285,
        "WHAT LEARNS?",
        "Weights learn to predict logged future skills.\nCross-entropy averages the three horizons;\nunavailable future labels contribute no loss.",
        BLUE,
        16,
    )
    words(
        f,
        0.55,
        0.285,
        "WHICH NETWORK IS THIS?",
        "This is the observer's forecasting MLP.\nThe RL pilot's actor–critic is a separate model.\nPilot weights stay frozen during forecasting.",
        GOLD,
        16,
    )
    finish(
        f,
        (
            "The worked standard model uses 26 observations of 91 features plus one validity flag per "
            "timestamp, flattened into 2,392 columns. The input linear layer projects to width 256 and "
            "applies GELU. Three residual blocks each compute x + f(x), where f is LayerNorm, Linear "
            "256 to 512, GELU, dropout, Linear 512 to 256, dropout. A final LayerNorm and Linear 256 "
            "to 99 generate logits reshaped into three categorical outputs over 33 labels. Thus the "
            "configuration field layers=3 means three residual blocks; the complete MLP contains eight "
            "linear layers, not merely three dense layers. It has 1,428,835 trainable parameters. "
            "For width 512, each block expands to 1,024 internally. Time positions are encoded by "
            "distinct input columns; there is no recurrent state, attention, or positional embedding "
            "in the selected MLPs. All history ends at the prediction origin. The loss uses smoothed "
            "cross-entropy on each available future label and averages the horizons. All four final "
            "variants have auxiliary_weight=0. Forecast training updates only the supervised predictor, "
            "not the RL pilot, simulator, skill catalogue, or tactical opponent. The earlier GRU and "
            "transformer are comparison forecasters; neither is included in the selected ensemble."
        ),
    )

    f = page(
        "Four variants, twelve separately trained members",
        "Same 91 features and three residual blocks | Each variant uses seeds 42, 43 and 44",
        "Frozen selection + final-refit run metadata",
    )
    rows = [
        [
            c["label"],
            f"{c['history_tokens']} × 92\n= {c['input_columns_with_flags']:,}",
            f"{c['width']} → {c['block_expansion_width']}",
            f"{c['parameters_per_member'] / 1e6:.3f} M",
            str(c["epochs"]),
            f"{c['temperature']:.3f}",
        ]
        for c in components
    ]
    table(
        f,
        [0.065, 0.42, 0.87, 0.31],
        rows,
        ["Variant", "Input columns", "Block width", "Params / model", "Epochs", "Temperature"],
        widths=[0.23, 0.17, 0.16, 0.17, 0.10, 0.17],
        size=14,
    )
    f.text(
        0.065,
        0.345,
        f"Total: {evidence['total_parameters'] / 1e6:.2f} million parameters across twelve models",
        fontsize=23,
        weight="bold",
        color=TEAL,
    )
    words(
        f,
        0.065,
        0.265,
        "CALIBRATE, THEN AVERAGE",
        "Temperature softens each model's confidence.\nAverage probabilities equally: 1/12 per member.\nEach variant contributes 25% of the ensemble.",
        BLUE,
        16,
    )
    words(
        f,
        0.55,
        0.265,
        "DIVERSITY HAS A COST",
        "Widths, regularisation and history differ.\n9 members use 20 s; 3 use 60 s.\nMore members are not proven necessary.",
        GOLD,
        16,
    )
    finish(
        f,
        (
            "The exact frozen members are standard20, wide20, smooth20 and standard60, each with seeds "
            "42, 43 and 44. Feature counts include 91 numeric features and one validity flag at each "
            "timestamp. Parameters per member are 1,428,835; 4,430,435; 1,428,835; and 2,370,915. "
            "Final epochs are 46, 32, 40 and 40. Each variant's epoch budget is the rounded median "
            "of its two internal best epochs, fixed before refitting on all 210 training groups. "
            "Each temperature is the median of the two stop-set-fitted values, not fitted on outer "
            "validation. A single positive temperature rescales all 99 logits in a member before "
            "softmax independently over each 33-label horizon; it changes confidence but not that "
            "member's class ordering. Temperatures are 1.465412, 1.401350, 1.169110 and 1.423757. "
            "Dropout/smoothing/weight-decay are respectively .20/.02/.03, .30/.03/.05, .30/.08/.05, "
            "and .20/.02/.03. Each member predicts all pilots and all horizons, not one pilot or "
            "horizon. Uniform probability averaging gives 75% of members to the 20-second variants "
            "and 25% to the 60-second variant. The internal wide/long pair was nearly tied with "
            "the four-variant mixture, so twelve members are not demonstrated to be the minimum "
            "necessary deployment size."
        ),
    )

    f = page(
        "Training time: what the saved timers actually measure",
        "Recorded fitting stages on the local GPU setup | Model fitting is separate from simulated flight generation",
        "36 completed run summaries; copied validation summaries excluded",
    )
    labels = {
        "initial_v1": "Original GRU + transformer",
        "improvement_v2": "v2 internal search + confirmation",
        "improvement_v3": "v3 internal search + confirmation",
        "final_refits": "Final ensemble refits",
    }
    rows = [
        [
            labels[s["stage"]],
            str(s["runs"]),
            str(s["epochs_sum"]),
            f"{s['elapsed_seconds_sum'] / 60:.2f} min",
            f"{s['elapsed_seconds_min']:.1f}–{s['elapsed_seconds_max']:.1f} s",
        ]
        for s in evidence["timing_stages"]
    ]
    table(
        f,
        [0.065, 0.43, 0.87, 0.30],
        rows,
        ["Stage", "Runs", "Epochs summed", "Timers summed", "Per-run range"],
        widths=[0.39, 0.08, 0.16, 0.18, 0.19],
        size=14,
    )
    f.text(
        0.065,
        0.35,
        f"Recorded fit-stage total: {evidence['all_logged_fit_seconds'] / 60:.2f} min; final twelve refits: 15.42 min",
        fontsize=23,
        weight="bold",
        color=TEAL,
    )
    words(
        f,
        0.065,
        0.27,
        "RECORDED ENVIRONMENT",
        "NVIDIA GeForce RTX 4090; CUDA 13.0.\nAdamW, batch 1,024, CUDA bfloat16.\nFinal members take approximately 55–100 s each.",
        BLUE,
        16,
    )
    words(
        f,
        0.55,
        0.27,
        "TIMING BOUNDARY",
        "Not the complete experiment duration.\nExcludes simulation, preparation and data loading;\nreporting, audits and selection add further time.",
        GOLD,
        16,
    )
    finish(
        f,
        (
            "All 36 unique completed run summaries are included once: two original fits; nine v2 "
            "screening fits plus two v2 second-split fits; six v3 first-split fits plus five second-split "
            "fits; twelve final refits. Their elapsed_seconds sum to 1,900.371772 seconds, or 31.672863 "
            "minutes. Stage sums are 89.737079, 352.325676, 533.309299 and 924.999717 seconds. Original "
            "GRU/transformer timers are 31.114268 and 58.622811 seconds. Final refit totals by variant "
            "are 237.733035, 166.544699, 228.977007 and 291.744976 seconds. The RTX4090/CUDA13 "
            "environment is explicitly recorded in v3/environment.json and the original run metadata; "
            "the v2 directory has no separate environment file. Improved timers start after feature "
            "construction, loading, model setup and optimizer creation. They cover the epoch loop, "
            "internal stopping predictions where applicable, checkpoint writes, and temperature "
            "fitting, but stop before select-group prediction. Original timers additionally include "
            "their final validation/report work. Summing these timers is an inventory of fit-stage "
            "elapsed time, not GPU kernel time or a measured end-to-end project duration. Pilot "
            "PPO training and the 3,000 corpus-generation flights are separate costs. Short training "
            "time does not establish low deployment latency or good generalisation."
        ),
    )

    f = page(
        "Experiment evolution: why the recipe changed",
        "Top-1 and NLL below are means across +1 / +2 / +5 s | Compare rows only within the same evaluation partition",
        "Saved v2/v3 searches, mixture confirmation and fixed validation reports",
    )
    rows = [
        [
            r["label"],
            r["partition"],
            f"{r['mean_accuracy'] * 100:.2f}%",
            f"{r['mean_nll']:.3f}",
            r["change"],
        ]
        for r in evidence["experiment_evolution"]
    ]
    table(
        f,
        [0.065, 0.285, 0.87, 0.445],
        rows,
        ["Experiment", "Evaluation", "Mean top-1", "Mean NLL", "Change"],
        widths=[0.25, 0.19, 0.12, 0.11, 0.33],
        size=12,
    )
    f.text(
        0.065,
        0.215,
        "Individual accuracy gains came with worse NLL; probability averaging improved both internally.",
        fontsize=17,
        color=TEAL,
        weight="bold",
    )
    f.text(
        0.065,
        0.155,
        "Two-partition confirmation: four-variant mixture 60.98% vs wide/long pair 60.96% mean top-1.",
        fontsize=16,
        color=FG,
    )
    f.text(
        0.065,
        0.095,
        "This was adaptive development with several changes together; it is not a clean ablation of architecture alone.",
        fontsize=15,
        color=GOLD,
    )
    finish(
        f,
        (
            "The chronology begins with five-second, 54-feature original GRU and transformer "
            "forecasters, selected on outer development validation by NLL. The v2 search then used "
            "only the original training groups, divided into 147 fit, 21 stop and 42 select groups. "
            "Nine first-split candidates tested geometry, regularisation, longer history, a current-skill "
            "auxiliary target, and temporal MLPs. The 20-second MLP improved first-split mean accuracy "
            "from 51.87% to 53.98%; many other candidates deteriorated. The targeted v3 search tested "
            "accuracy-selected epochs, stop-only calibration, wider and more-smoothed MLPs, and "
            "60-second history. Single models gained accuracy but worsened NLL, motivating the "
            "registered extension to uniform probability mixtures. The long-history single model "
            "shown here is the highest-accuracy individual on internal split 1; all candidate results "
            "remain available in the existing search tables. First-split four-variant mixture accuracy "
            "was 59.09% with NLL 1.309, versus control 51.87% and 1.384. Two shortlisted mixtures "
            "and control were confirmed on a second grouped partition. Mean accuracy across both "
            "partitions was 60.9817% for four variants and 60.9598% for the wide/long pair; four "
            "variants had lower mean NLL (1.2581 versus 1.2807), while the pair had slightly higher "
            "macro-F1. The chosen recipe, epoch budgets, temperatures and three seeds were frozen "
            "before fitting all training groups and comparing against the original models on outer "
            "validation. Internal mixture confirmation used one seed per variant; the final ensemble "
            "uses three. Internal and outer populations differ and must not form one uninterrupted "
            "performance trend. The adaptive search supports the combined procedure, not isolated "
            "causal credit to longer history, geometry, calibration or MLP architecture. No new model "
            "training, checkpoint selection or reserved-test evaluation was performed for this review."
        ),
    )
