"""Portable diagnostics and matched comparisons for notebooks 18--21.

These functions only inspect saved predictions.  The partition named ``test``
selected checkpoints; validation comparisons are descriptive, never model selection.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ARCHITECTURES = ("transformer", "missile_set", "typed_graph")
SPLITS = ("train", "test", "validation")
HORIZONS = (1, 2, 5)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _architecture(run):
    return run.get("architecture", run.get("model", {}).get("architecture",
        run.get("training", {}).get("architecture", "transformer")))


def _pyplot():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    return plt


def _figure(fig, path):
    plt = _pyplot()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _probability_metrics(targets, probabilities, classes, split):
    from .paired_skill_forecasting import metric_tables

    summary, per_class, confusion = metric_tables(targets, probabilities, classes, split)
    rows = []
    for h, horizon in enumerate(HORIZONS):
        valid = targets[:, h] >= 0
        y, p = targets[valid, h], probabilities[valid, h].astype(np.float64)
        confidence = p.max(1)
        correct = p.argmax(1) == y
        bins = np.minimum((confidence * 10).astype(int), 9)
        ece = 0.
        for b in range(10):
            selected = bins == b
            count = int(selected.sum())
            mean_confidence = float(confidence[selected].mean()) if count else np.nan
            accuracy = float(correct[selected].mean()) if count else np.nan
            if count:
                ece += count / len(y) * abs(mean_confidence - accuracy)
            rows.append({"split": split, "horizon_s": horizon, "bin": b,
                         "count": count, "mean_confidence": mean_confidence,
                         "accuracy": accuracy})
        # Multiclass Brier: sum over classes, then mean over examples.
        brier = np.square(p).sum(1) - 2 * p[np.arange(len(y)), y] + 1
        top = np.argsort(p, axis=1)[:, -min(3, len(classes)):]
        summary.loc[summary.horizon_s.eq(horizon), ["brier", "ece", "top3_accuracy"]] = [
            brier.mean(), ece, np.any(top == y[:, None], axis=1).mean()]
    return summary, per_class, pd.DataFrame(rows), confusion


def _load_prediction(run, split, classes):
    path = run / f"final_{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; finish final_evaluation before diagnostics.")
    with np.load(path, allow_pickle=False) as stored:
        required = {"probabilities", "targets", "sample_indices", "confusion"}
        if not required.issubset(stored.files):
            raise ValueError(f"{path} lacks saved prediction keys {sorted(required)}")
        result = {k: stored[k] for k in required}
    p, y, ids = [result[k] for k in ("probabilities", "targets", "sample_indices")]
    if (p.shape != (len(ids), 3, len(classes)) or y.shape != (len(ids), 3)
            or len(np.unique(ids)) != len(ids) or ids.ndim != 1
            or not np.issubdtype(ids.dtype, np.integer)
            or not np.issubdtype(y.dtype, np.integer)
            or np.any(ids < 0) or np.any(y < -1) or np.any(y >= len(classes))
            or not np.isfinite(p).all() or np.any(p < 0)
            or not np.allclose(p.sum(-1), 1, atol=1e-5)):
        raise ValueError(f"Invalid predictions, sample IDs or labels in {path}")
    if not (y >= 0).any(0).all():
        raise ValueError(f"A horizon in {path} has no valid targets")
    return result


def _plot_diagnostics(directory, history, summary, per_class, reliability, results, classes, selected):
    plt = _pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for split in ("train", "test"):
        key = f"{split}_nll"
        if key in history:
            axes[0].plot(history.epoch, history[key], label=split)
    for key in ("train_data_loss", "train_l1_penalty", "train_objective"):
        if key in history:
            axes[1].plot(history.epoch, history[key], label=key)
    if "generalization_gap" in history:
        axes[2].plot(history.epoch, history.generalization_gap, label="test minus train NLL")
    for ax, title in zip(axes, ("Unweighted NLL", "Training objective components", "Generalization gap")):
        ax.axvline(selected, color="black", linestyle=":")
        ax.set(title=title, xlabel="Epoch")
        if ax.lines:
            ax.legend(fontsize=7)
    _figure(fig, directory / "learning_curves.png")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, measure in zip(axes, ("accuracy", "nll", "macro_f1")):
        summary.pivot(index="horizon_s", columns="split", values=measure).reindex(
            columns=SPLITS).plot.bar(ax=ax, rot=0, title=measure)
        ax.set_xlabel("Forecast horizon (s)")
    _figure(fig, directory / "final_metrics.png")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for h, ax in zip(HORIZONS, axes):
        ax.plot([0, 1], [0, 1], "k:")
        for split in SPLITS:
            table = reliability.loc[(reliability.split == split) &
                                    (reliability.horizon_s == h) & (reliability["count"] > 0)]
            ax.plot(table.mean_confidence, table.accuracy, "o-", label=split)
        ax.set(title=f"+{h}s reliability", xlabel="Mean confidence", ylabel="Accuracy",
               xlim=(0, 1), ylim=(0, 1))
        ax.legend()
    _figure(fig, directory / "reliability.png")
    for split in SPLITS:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for h, ax, cm in zip(HORIZONS, axes, results[split]["confusion"]):
            normal = np.divide(cm, cm.sum(1, keepdims=True), out=np.zeros_like(cm, dtype=float),
                               where=cm.sum(1, keepdims=True) > 0)
            ax.imshow(normal, vmin=0, vmax=1, cmap="Blues")
            ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=6)
            ax.set_yticks(range(len(classes)), classes, fontsize=6)
            ax.set(title=f"{split}, +{h}s", xlabel="Predicted", ylabel="True")
        _figure(fig, directory / f"confusion_{split}.png")
    fig, axes = plt.subplots(2, 3, figsize=(15, max(6, len(classes) * .25)))
    for h, upper, lower in zip(HORIZONS, axes[0], axes[1]):
        rows = per_class.loc[per_class.horizon_s.eq(h)]
        for ax, measure in ((upper, "f1"), (lower, "support")):
            table = rows.pivot(index="skill", columns="split", values=measure).reindex(
                index=classes, columns=SPLITS)
            values = table.to_numpy() if measure == "f1" else np.log1p(table.to_numpy())
            ax.imshow(values, aspect="auto", vmin=0, cmap="viridis")
            ax.set_xticks(range(3), SPLITS)
            ax.set_yticks(range(len(classes)), classes, fontsize=6)
            ax.set_title(f"+{h}s {measure}" + (" (log1p)" if measure == "support" else ""))
    _figure(fig, directory / "per_class_f1_support.png")


def _preserve_logit_nll(run, summary, per_class):
    """Keep the exact FP32-logit CE saved by evaluation, even if p underflows.

    Saved probability NPZs cannot recover a log probability once it became zero.
    Original tables must describe exactly the same split/horizon/class supports.
    """
    for frame, filename, keys, value, count, probability_name in (
        (summary, "final_metrics.csv", ["split", "horizon_s"], "nll", "count", "probability_nll"),
        (per_class, "final_per_class.csv", ["split", "horizon_s", "skill"],
         "class_nll", "support", "probability_class_nll"),
    ):
        original = pd.read_csv(run / filename)
        if not set([*keys, value, count]).issubset(original.columns):
            raise ValueError(f"Missing diagnostic identity/NLL columns in {run / filename}")
        original = original.set_index(keys)
        calculated = frame.set_index(keys)
        if (original.index.duplicated().any() or calculated.index.duplicated().any()
                or len(original) != len(calculated)
                or len(original.index.difference(calculated.index))
                or len(calculated.index.difference(original.index))):
            raise ValueError(f"Saved diagnostic split/horizon/class identities differ: {run / filename}")
        aligned = original.reindex(calculated.index)
        if not np.array_equal(aligned[count], calculated[count]):
            raise ValueError(f"Saved diagnostic counts/support differ from predictions: {run / filename}")
        positive = aligned[count].to_numpy() > 0
        if (not np.isfinite(aligned.loc[positive, value]).all()
                or (aligned.loc[positive, value] < 0).any()):
            raise ValueError(f"Invalid authoritative NLL in {run / filename}")
        frame[probability_name] = frame[value]
        frame[value] = aligned[value].to_numpy()


def save_diagnostics(run, data=None):
    """Persist portable CSV/PNG/JSON diagnostics for one completed selected run.

    ``data`` is optional for report-only execution, but supplying the prepared
    loader records sample/scenario alignment and origin-time missile cohorts.
    Prediction NPZs remain the authoritative per-window probabilities.
    """
    run = Path(run).expanduser().resolve()
    required = ("run.json", "history.csv", "final_metrics.csv", "final_per_class.csv",
                "selection.json", "training_status.json")
    for name in required:
        if not (run / name).exists():
            raise FileNotFoundError(f"Missing {run / name}; finish this run before diagnostics.")
    metadata, selection, status = [_read(run / name) for name in
                                   ("run.json", "selection.json", "training_status.json")]
    if status.get("status") not in ("completed", "early_stopped"):
        raise ValueError(f"Run is not completed or early-stopped: {run}")
    if selection.get("validation_used_for_selection", False):
        raise ValueError("Expected test-selected checkpoint; validation was used for selection")
    classes = metadata["classes"]
    directory = run / "diagnostics"
    directory.mkdir(exist_ok=True)
    sample_path = directory / "samples.csv"
    if data is not None:
        from .paired_skill_forecasting import sha256
        if metadata["data_manifest_sha256"] != sha256(Path(data.directory) / "manifest.json"):
            raise ValueError("Diagnostics loader belongs to another prepared dataset")
        ids = np.unique(np.concatenate([data.indices[split] for split in SPLITS]))
        samples = data.samples.iloc[ids].copy()
        samples.insert(0, "sample_index", ids)
        # Only origin-time occupancy; bounded gathers keep the large frame file on disk.
        if data.features.shape[1] == 115:
            friendly, enemy = np.zeros(len(ids), bool), np.zeros(len(ids), bool)
            for start in range(0, len(ids), 4096):
                origins = data.features[data.anchors[ids[start:start + 4096]]]
                friendly[start:start + len(origins)] = np.any(origins[:, 19:47] != 0, axis=1)
                enemy[start:start + len(origins)] = np.any(origins[:, 54:82] != 0, axis=1)
            samples["friendly_missiles_present"] = friendly
            samples["enemy_missiles_present"] = enemy
        samples.to_csv(sample_path, index=False)
    samples = pd.read_csv(sample_path).set_index("sample_index") if sample_path.exists() else None
    summaries, scores, bins, cohort_rows, results = [], [], [], [], {}
    for split in SPLITS:
        # At most one split's prediction arrays are retained while making diagnostics.
        result = _load_prediction(run, split, classes)
        summary, per_class, reliability, confusion = _probability_metrics(
            result["targets"], result["probabilities"], classes, split)
        if not np.array_equal(confusion, result["confusion"]):
            raise ValueError(f"Saved confusion differs from predictions for {run}/{split}")
        summaries.append(summary)
        scores.append(per_class)
        bins.append(reliability)
        results[split] = {"confusion": confusion}
        if samples is not None:
            rows = samples.loc[result["sample_indices"]]
            for column in ("friendly_missiles_present", "enemy_missiles_present"):
                if column not in rows:
                    continue
                for present in (False, True):
                    chosen = rows[column].to_numpy() == present
                    for h, horizon in enumerate(HORIZONS):
                        valid = chosen & (result["targets"][:, h] >= 0)
                        y, p = result["targets"][valid, h], result["probabilities"][valid, h]
                        cohort_rows.append({"split": split, "horizon_s": horizon,
                            "cohort": column, "present": present, "count": int(valid.sum()),
                            "accuracy": float((p.argmax(1) == y).mean()) if len(y) else np.nan,
                            "nll": float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-30)).mean())
                            if len(y) else np.nan})
        del result
    summary, per_class, reliability = [pd.concat(tables, ignore_index=True)
                                     for tables in (summaries, scores, bins)]
    _preserve_logit_nll(run, summary, per_class)
    summary.to_csv(directory / "summary.csv", index=False)
    per_class.to_csv(directory / "per_class.csv", index=False)
    reliability.to_csv(directory / "reliability.csv", index=False)
    # Existing final_metrics can retain FP32-logit NLL beyond probability precision;
    # preserve those authoritative original diagnostics untouched.
    pd.DataFrame(cohort_rows, columns=["split", "horizon_s", "cohort", "present", "count", "accuracy", "nll"]).to_csv(
        directory / "missile_cohorts.csv", index=False)
    history = pd.read_csv(run / "history.csv")
    _plot_diagnostics(directory, history, summary, per_class, reliability, results,
                      classes, selection["epoch"])
    sources = list(required) + [f"final_{s}.npz" for s in SPLITS]
    for optional in ("best.pt", "class_weights.json", "class_weights.csv"):
        if (run / optional).exists():
            sources.append(optional)
    manifest = {"format": "enemy-missile-diagnostics-v1", "architecture": _architecture(metadata),
        "seed": metadata["training"]["seed"], "parameter_count": metadata.get("parameter_count"),
        "data_manifest_sha256": metadata["data_manifest_sha256"], "selected_epoch": selection["epoch"],
        "metric_note": "Primary nll/class_nll preserve authoritative FP32-logit CE after exact identity/support checks; probability_nll/probability_class_nll are recomputed from saved probabilities",
        "partition_roles": {"train": "fit", "test": "checkpoint/model selection",
                            "validation": "final descriptive evaluation; never choose seeds or recipes"},
        "sources": {name: _sha(run / name) for name in sources},
        "artifacts": {p.name: _sha(p) for p in directory.iterdir() if p.is_file() and p.name != "manifest.json"}}
    _write(directory / "manifest.json", manifest)
    return {"directory": directory, "summary": summary, "per_class": per_class,
            "reliability": reliability, "manifest": manifest}


def _protocol(run):
    training = dict(run["training"])
    for key in ("seed", "architecture", "graph_width", "graph_layers"):
        training.pop(key, None)
    return {"training": training, "classes": run["classes"],
            "feature_names": run.get("feature_names"), "data_contract": run.get("data_contract"),
            "horizons_s": run.get("horizons_s", list(HORIZONS)),
            "data_manifest_sha256": run["data_manifest_sha256"],
            "class_weighting": run.get("class_weighting")}


def _bootstrap(reference, candidate, sample_groups, count, rng):
    groups, inverse = np.unique(np.asarray(sample_groups).astype(str), return_inverse=True)
    rows = []
    values = []
    for h in range(3):
        y = reference["targets"][:, h]
        valid = y >= 0
        a, b = reference["probabilities"][:, h], candidate["probabilities"][:, h]
        delta_accuracy = ((b.argmax(1) == y).astype(float) - (a.argmax(1) == y).astype(float)) * valid
        indices = np.flatnonzero(valid)
        delta_nll = np.zeros(len(y))
        delta_nll[valid] = -np.log(np.maximum(b[indices, y[valid]], 1e-30)) + np.log(
            np.maximum(a[indices, y[valid]], 1e-30))
        denominator = np.bincount(inverse, weights=valid, minlength=len(groups))
        values.append((np.bincount(inverse, weights=delta_accuracy, minlength=len(groups)),
                       np.bincount(inverse, weights=delta_nll, minlength=len(groups)), denominator))
    draws = np.empty((count, 3, 2))
    for i in range(count):
        frequencies = np.bincount(rng.integers(len(groups), size=len(groups)), minlength=len(groups))
        for h, (acc, nll, denominator) in enumerate(values):
            total = frequencies @ denominator
            draws[i, h] = [(frequencies @ acc) / total, (frequencies @ nll) / total] if total else np.nan
    for h, horizon in enumerate((*HORIZONS, "mean")):
        samples = draws[:, h] if h < 3 else draws.mean(1)
        for j, metric in enumerate(("accuracy", "probability_nll")):
            low, high = np.nanquantile(samples[:, j], [.025, .975])
            point = []
            for acc, nll, denominator in values:
                point.append((acc.sum() if j == 0 else nll.sum()) / denominator.sum())
            estimate = point[h] if h < 3 else np.mean(point)
            rows.append({"horizon_s": horizon, "metric": metric, "estimate": estimate,
                         "lower": low, "upper": high, "scenario_groups": len(groups), "resamples": count})
    return rows


def compare_architectures(runs, output, bootstrap_resamples=1000):
    """Compare all three architectures on exactly matched saved examples/seeds.

    No network is loaded or fitted. Validation is descriptive and never ranked.
    Scenario bootstrap intervals are conditional on each saved seed/model pair.
    """
    if set(runs) != set(ARCHITECTURES):
        raise ValueError(f"Provide exactly these architectures: {ARCHITECTURES}")
    if not isinstance(bootstrap_resamples, int) or bootstrap_resamples < 0:
        raise ValueError("bootstrap_resamples must be a nonnegative integer")
    records, prediction_cache, baseline_protocol, seed_sets, class_tables = [], {}, None, {}, {}
    source_hashes, baseline_samples, metadata_available = {}, None, None
    for architecture in ARCHITECTURES:
        paths = [Path(p).expanduser().resolve() for p in runs[architecture]]
        if not paths or len(paths) != len(set(paths)):
            raise ValueError(f"Missing or duplicate run directories for {architecture}")
        seed_sets[architecture] = set()
        for run in paths:
            meta = _read(run / "run.json")
            if _architecture(meta) != architecture:
                raise ValueError(f"Architecture metadata does not match {architecture}: {run}")
            seed = meta["training"]["seed"]
            if seed in seed_sets[architecture]:
                raise ValueError(f"Duplicate seed {seed} for {architecture}; choose one run explicitly")
            seed_sets[architecture].add(seed)
            protocol = _protocol(meta)
            if baseline_protocol is None:
                baseline_protocol = protocol
            elif protocol != baseline_protocol:
                raise ValueError(f"Dataset, feature order or training protocol mismatch: {run}")
            diagnostic_path = run / "diagnostics" / "manifest.json"
            if not diagnostic_path.exists():
                raise FileNotFoundError(f"Missing {diagnostic_path}; run save_diagnostics(run, data) first")
            saved = _read(diagnostic_path)
            for relative, digest in saved["sources"].items():
                if _sha(run / relative) != digest:
                    raise ValueError(f"Saved run changed since diagnostic export: {run / relative}")
            for relative, digest in saved["artifacts"].items():
                if _sha(run / "diagnostics" / relative) != digest:
                    raise ValueError(f"Diagnostic artifact changed: {run / 'diagnostics' / relative}")
            source_hashes[str(run)] = _sha(diagnostic_path)
            sample_path = run / "diagnostics" / "samples.csv"
            if metadata_available is None:
                metadata_available = sample_path.exists()
            elif metadata_available != sample_path.exists():
                raise ValueError("Sample/scenario metadata must be supplied for every run or none")
            if sample_path.exists():
                sample_table = pd.read_csv(sample_path)
                required = {"sample_index", "split", "scenario_group"}
                if (not required.issubset(sample_table.columns)
                        or sample_table[list(required)].isna().any().any()
                        or sample_table.sample_index.duplicated().any()
                        or not np.issubdtype(sample_table.sample_index.dtype, np.integer)
                        or (sample_table.sample_index < 0).any()
                        or not set(sample_table.split).issubset(SPLITS)
                        or sample_table.groupby("scenario_group").split.nunique().gt(1).any()):
                    raise ValueError(f"Malformed sample/scenario metadata: {sample_path}")
                sample_table = sample_table.sort_values("sample_index").set_index("sample_index").sort_index(axis=1)
                if baseline_samples is None:
                    baseline_samples = sample_table
                elif not sample_table.equals(baseline_samples):
                    raise ValueError(f"Sample/scenario metadata differs across runs: {sample_path}")
                del sample_table
            prediction_cache[architecture, seed] = run
            class_tables[architecture, seed] = pd.read_csv(run / "diagnostics" / "per_class.csv")
            history = pd.read_csv(run / "history.csv")
            seconds = float(history.seconds.sum()) if "seconds" in history else np.nan
            records.append(pd.read_csv(run / "diagnostics" / "summary.csv").assign(
                architecture=architecture, seed=seed, selected_epoch=saved["selected_epoch"],
                parameter_count=meta.get("parameter_count"), logged_fit_evaluation_seconds=seconds,
                completed_epochs=len(history), run=str(run)))
    if any(seed_sets[a] != seed_sets[ARCHITECTURES[0]] for a in ARCHITECTURES):
        raise ValueError("All architectures must have exactly the same training seeds")
    bootstrap_rows, deltas = [], []
    # Confirm one canonical ordering/target vector across *all* runs and seeds,
    # as well as the per-seed pairings used below. Metadata alone cannot establish it.
    canonical_run = prediction_cache["transformer", min(seed_sets["transformer"])]
    for split in SPLITS:
        with np.load(canonical_run / f"final_{split}.npz", allow_pickle=False) as stored:
            canonical_ids, canonical_targets = stored["sample_indices"], stored["targets"]
        if baseline_samples is not None:
            if not np.isin(canonical_ids, baseline_samples.index).all():
                raise ValueError("Saved sample metadata does not cover the prediction IDs")
            rows = baseline_samples.loc[canonical_ids]
            if not rows.split.eq(split).all():
                raise ValueError("Saved sample split metadata does not match prediction splits")
        for run in prediction_cache.values():
            with np.load(run / f"final_{split}.npz", allow_pickle=False) as stored:
                if (not np.array_equal(canonical_ids, stored["sample_indices"])
                        or not np.array_equal(canonical_targets, stored["targets"])):
                    raise ValueError(f"Saved examples/targets differ across architectures/seeds: {run}/{split}")
    if baseline_samples is not None:
        expected_ids = []
        for split in SPLITS:
            with np.load(canonical_run / f"final_{split}.npz", allow_pickle=False) as stored:
                expected_ids.append(stored["sample_indices"])
        joined = np.concatenate(expected_ids)
        if len(np.unique(joined)) != len(joined) or not np.array_equal(np.sort(joined), baseline_samples.index):
            raise ValueError("Saved sample metadata does not exactly cover disjoint prediction splits")
    contrasts = (("transformer", "missile_set"), ("transformer", "typed_graph"),
                 ("missile_set", "typed_graph"))
    for seed in sorted(seed_sets[ARCHITECTURES[0]]):
        for reference_architecture, architecture in contrasts:
            reference_run = prediction_cache[reference_architecture, seed]
            candidate_run = prediction_cache[architecture, seed]
            for split in SPLITS:
                a = _load_prediction(reference_run, split, baseline_protocol["classes"])
                b = _load_prediction(candidate_run, split, baseline_protocol["classes"])
                ca = class_tables[reference_architecture, seed]
                ca = ca.loc[ca.split.eq(split)]
                cb = class_tables[architecture, seed]
                cb = cb.loc[cb.split.eq(split)]
                difference = cb.merge(ca, on=["split", "horizon_s", "skill"], suffixes=("", "_reference"))
                for measure in ("f1", "recall", "precision", "class_nll", "probability_class_nll"):
                    difference[f"delta_{measure}"] = difference[measure] - difference[f"{measure}_reference"]
                deltas.append(difference.assign(architecture=architecture,
                    reference_architecture=reference_architecture, seed=seed))
                if bootstrap_resamples and baseline_samples is not None:
                    groups = baseline_samples.loc[a["sample_indices"], "scenario_group"]
                    # Identical seed resamples are shared across architecture contrasts
                    # for a seed/split, and every draw preserves horizon dependence.
                    rng = np.random.default_rng(1729 + int(seed) * 3 + SPLITS.index(split))
                    for row in _bootstrap(a, b, groups, bootstrap_resamples, rng):
                        bootstrap_rows.append({**row, "architecture": architecture,
                            "reference_architecture": reference_architecture, "seed": seed, "split": split})
                del a, b
    output = Path(output).expanduser().resolve()
    if any(output == path or path in output.parents for path in prediction_cache.values()):
        raise ValueError("Comparison output must be outside the source run directories")
    output.mkdir(parents=True, exist_ok=True)
    metrics = pd.concat(records, ignore_index=True)
    measures = ["accuracy", "nll", "probability_nll", "macro_f1", "supported_macro_f1", "brier", "ece", "top3_accuracy"]
    aggregate = metrics.groupby(["architecture", "split", "horizon_s"], sort=False)[measures].agg(["mean", "std"])
    aggregate.columns = ["_".join(c) for c in aggregate.columns]
    aggregate = aggregate.reset_index()
    per_class = pd.concat(deltas, ignore_index=True)
    bootstrap = pd.DataFrame(bootstrap_rows, columns=["architecture", "reference_architecture", "seed", "split", "horizon_s", "metric", "estimate", "lower", "upper", "scenario_groups", "resamples"])
    horizon_means = metrics.groupby(["architecture", "seed", "split"], sort=False)[measures].mean().reset_index()
    horizon_mean_aggregate = horizon_means.groupby(["architecture", "split"], sort=False)[measures].agg(["mean", "std"])
    horizon_mean_aggregate.columns = ["_".join(c) for c in horizon_mean_aggregate.columns]
    horizon_mean_aggregate = horizon_mean_aggregate.reset_index()
    metrics.to_csv(output / "metrics.csv", index=False)
    aggregate.to_csv(output / "aggregate.csv", index=False)
    per_class.to_csv(output / "per_class_deltas.csv", index=False)
    bootstrap.to_csv(output / "paired_bootstrap.csv", index=False)
    horizon_means.to_csv(output / "horizon_means.csv", index=False)
    horizon_mean_aggregate.to_csv(output / "horizon_mean_aggregate.csv", index=False)
    plt = _pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for architecture in ARCHITECTURES:
        for seed in sorted(seed_sets[architecture]):
            run = prediction_cache[architecture, seed]
            curve = pd.read_csv(run / "history.csv")
            axes[0].plot(curve.epoch, curve.test_nll, label=f"{architecture}, seed {seed}")
    axes[0].set(title="Selection NLL", xlabel="Epoch", ylabel="Unweighted NLL")
    axes[0].legend(fontsize=6)
    for ax, split in zip(axes[1:], ("test", "validation")):
        table = metrics.loc[metrics.split.eq(split)].groupby(["architecture", "horizon_s"], sort=False).accuracy.mean().unstack(0)
        table.reindex(columns=ARCHITECTURES).plot.bar(ax=ax, rot=0)
        ax.set(title=f"{split} accuracy (seed mean)", xlabel="Horizon (s)", ylabel="Accuracy")
    _figure(fig, output / "architecture_comparison.png")
    manifest = {"format": "enemy-missile-comparison-v1", "sources": source_hashes,
        "protocol": baseline_protocol, "seeds": sorted(seed_sets[ARCHITECTURES[0]]),
        "validation_policy": "Descriptive only; no best seed/architecture is selected",
        "timing_note": "logged_fit_evaluation_seconds sums history.seconds; excludes later checkpoint/table/MLflow I/O, not full wall time",
        "bootstrap": "Paired whole-scenario resampling, conditional on each fitted seed; omitted without saved scenario metadata. probability_nll is reconstructed from NPZ probabilities (clipped at 1e-30), not authoritative FP32-logit NLL",
        "nll_note": "Primary nll/class_nll retain original evaluation CSV values; probability_nll/probability_class_nll expose probability-derived counterparts",
        "artifacts": {p.name: _sha(p) for p in output.iterdir() if p.is_file() and p.name != "manifest.json"}}
    _write(output / "manifest.json", manifest)
    return {"metrics": metrics, "aggregate": aggregate, "per_class_deltas": per_class,
            "paired_bootstrap": bootstrap, "horizon_means": horizon_means,
            "horizon_mean_aggregate": horizon_mean_aggregate, "directory": output}
