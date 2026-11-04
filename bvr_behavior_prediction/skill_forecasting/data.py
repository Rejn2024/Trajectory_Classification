"""Exact decision-time labels, causal histories and an untouched test partition."""

import inspect
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from bvr_behavior_prediction.forecasting.data import (
    DT,
    FEATURE_NAMES,
    HORIZONS,
    OBS_COLUMNS,
    grouped_split,
    observed_kinematics,
    read_json,
    sha256,
    write_json,
)

DECISION_COLUMNS = ["episode_id", "step", "time_s", "skill", "aircraft_id"]
KEYS = ["episode_id", "step", "time_s"]


def verify_dataset(directory):
    directory = Path(directory)
    manifest = read_json(directory / "data_manifest.json")
    for relative, expected in manifest["files"].items():
        if sha256(directory / relative) != expected:
            raise ValueError(f"Prepared data changed: {relative}")
    return manifest


def decision_series(decisions, classes, frame_count):
    """Index actual selections by 10 Hz tick; never forward-fill missing records."""
    if decisions.empty:
        return pd.Series(dtype="int64")
    steps = decisions.step.to_numpy()
    if (not np.array_equal(steps, np.rint(steps)) or np.any(steps < 0)
            or np.any(steps >= frame_count) or decisions.step.duplicated().any()
            or not np.allclose(decisions.time_s, steps * DT, atol=1e-7, rtol=0)):
        raise ValueError("Invalid or duplicate decision timestamps")
    if set(decisions.aircraft_id) != {"A01"}:
        raise ValueError("Expected A01 target decisions")
    mapping = {name: index for index, name in enumerate(classes)}
    labels = decisions.skill.map(mapping)
    if labels.isna().any():
        raise ValueError("Decision label is absent from the frozen catalogue")
    return pd.Series(labels.to_numpy(dtype=np.int64), index=steps.astype(np.int64))


def episode_examples(observations, tracks, decisions, classes, history_steps=51, stride_steps=10):
    observations = np.asarray(observations, dtype=np.float32)
    if observations.shape != (len(tracks), 51) or not np.isfinite(observations).all():
        raise ValueError("Invalid observations")
    if history_steps < 2 or stride_steps < 1:
        raise ValueError("Invalid history or stride")
    labels = decision_series(decisions, classes, len(tracks))
    _, _, acceleration = observed_kinematics(observations)
    features = np.concatenate((observations, acceleration / 100), axis=1).astype(np.float32)
    # Exclude reset and require all five extra acceleration-lookback frames.
    required = history_steps + 5
    first = ((required + stride_steps - 1) // stride_steps) * stride_steps
    ends = np.arange(first, len(tracks), stride_steps)
    live = tracks.observer_alive.to_numpy() & tracks.target_alive.to_numpy()
    invalid_prefix = np.r_[0, np.cumsum(~live)]
    history_ok = invalid_prefix[ends + 1] == invalid_prefix[ends - required + 1]
    ends = ends[history_ok]
    future_steps = ends[:, None] + np.rint(HORIZONS / DT).astype(np.int64)
    target = labels.reindex(future_steps.ravel(), fill_value=-1).to_numpy().reshape(-1, 3)
    keep = (target >= 0).any(axis=1)
    ends, target = ends[keep], target[keep]
    # Privileged diagnostic only. This array is never returned by model windows().
    current = labels.reindex(ends, fill_value=-1).to_numpy()
    return features, ends, target, current


def support_table(targets, samples, classes):
    rows = []
    for split in ("train", "validation"):
        for pilot in ["population", *sorted(samples.pilot_id.unique())]:
            selected = samples.split.eq(split).to_numpy(copy=True)
            if pilot != "population":
                selected &= samples.pilot_id.eq(pilot).to_numpy()
            for h, horizon in enumerate(HORIZONS):
                values = targets[selected, h]
                counts = np.bincount(values[values >= 0], minlength=len(classes))
                rows.extend({"split": split, "pilot_id": pilot, "horizon_s": float(horizon),
                             "skill": name, "count": int(count)}
                            for name, count in zip(classes, counts, strict=True))
    return pd.DataFrame(rows)


def prepare_dataset(corpus, output, reference_splits, history_steps=51, stride_steps=10,
                    split_seed=1729):
    from bvr_sim.agents.skill_manager import SkillManager

    corpus, output = Path(corpus), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Preparation output must be empty; choose a new directory")
    config, manifest = read_json(corpus / "config.json"), read_json(corpus / "manifest.json")
    if (config["observer_id"] != "B01" or config["target_id"] != "A01"
            or config["sample_interval_s"] != DT or not manifest["complete"]):
        raise ValueError("Expected complete B01-observer/A01-target corpus at 10 Hz")
    groups = grouped_split(config["cases"], split_seed)
    if groups != read_json(reference_splits):
        raise ValueError("Scenario partition differs from existing motion experiment")
    episodes = pd.read_parquet(corpus / "episodes.parquet").sort_values(["pilot_id", "episode_id"])
    if episodes.episode_id.nunique() != len(episodes) or len(episodes) != manifest["episodes"]:
        raise ValueError("Invalid episode index")
    episodes["split"] = episodes.scenario_group.map(lambda group: groups[group]["split"])
    classes = sorted(SkillManager().list_skills())
    used = ["config.json", "schema.json", "episodes.parquet"] + [
        f"{part}/{pilot}.parquet" for pilot in sorted(episodes.pilot_id.unique())
        for part in ("observations", "tracks", "target_decisions")]
    for relative in used:
        if sha256(corpus / relative) != manifest["files"][relative]["sha256"]:
            raise ValueError(f"Corpus checksum mismatch: {relative}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "scenario_splits.json", groups)
    write_json(output / "classes.json", classes)
    episodes[["episode_id", "pilot_id", "scenario_group", "split"]].to_csv(
        output / "split_assignments.csv", index=False)
    development = episodes.split.ne("test")
    features = np.lib.format.open_memmap(
        output / "features.npy", mode="w+", dtype="float32",
        shape=(int(episodes.loc[development, "frames"].sum()), len(FEATURE_NAMES)))
    anchors, targets, current_labels, metadata, decision_counts, cadence = [], [], [], [], [], []
    total, squares, count, offset, max_time_error = np.zeros(54), np.zeros(54), 0, 0, 0.0
    for pilot, pilot_episodes in episodes.groupby("pilot_id", sort=True):
        files = [pq.ParquetFile(corpus / part / f"{pilot}.parquet")
                 for part in ("observations", "tracks", "target_decisions")]
        if any(file.num_row_groups != len(pilot_episodes) for file in files):
            raise ValueError("Expected one row group per episode")
        for row_group, episode in enumerate(pilot_episodes.itertuples()):
            if episode.split == "test":
                continue  # Do not read test observation, track or decision rows.
            obs = files[0].read_row_group(row_group, columns=KEYS + OBS_COLUMNS).to_pandas()
            track = files[1].read_row_group(
                row_group, columns=KEYS + ["observer_alive", "target_alive"]).to_pandas()
            decisions = files[2].read_row_group(row_group, columns=DECISION_COLUMNS).to_pandas()
            if (set(obs.episode_id) != {episode.episode_id}
                    or set(decisions.episode_id) != {episode.episode_id}
                    or not obs[KEYS].equals(track[KEYS])
                    or not np.array_equal(obs.step, np.arange(len(obs)))
                    or not np.allclose(obs.time_s, obs.step * DT, atol=1e-7, rtol=0)
                    or len(obs) != episode.frames):
                raise ValueError("Invalid episode boundaries or frame timestamps")
            frame, ends, target, current = episode_examples(
                obs[OBS_COLUMNS].to_numpy(), track, decisions, classes, history_steps, stride_steps)
            features[offset:offset + len(frame)] = frame
            if episode.split == "train" and len(ends):
                selected = np.zeros(len(frame), dtype=bool)
                for end in ends:
                    selected[end - history_steps + 1:end + 1] = True
                values = frame[selected].astype(np.float64)
                total += values.sum(0)
                squares += np.square(values).sum(0)
                count += len(values)
            anchors.append(ends + offset)
            targets.append(target)
            current_labels.append(current)
            metadata.extend({"episode_id": episode.episode_id, "pilot_id": pilot,
                             "scenario_group": episode.scenario_group, "split": episode.split,
                             "step": int(end), "time_s": float(end * DT)} for end in ends)
            for name, n in decisions.skill.value_counts().items():
                decision_counts.append({"split": episode.split, "pilot_id": pilot,
                                        "skill": name, "count": int(n)})
            cadence.extend(np.diff(np.sort(decisions.step.to_numpy())).tolist())
            max_time_error = max(max_time_error, float(np.abs(decisions.time_s - decisions.step * DT).max()))
            offset += len(frame)
        print(f"Prepared {pilot}: {offset:,} development frames", flush=True)
    features.flush()
    del features
    if not count:
        raise ValueError("No training histories")
    target, current, samples = np.concatenate(targets), np.concatenate(current_labels), pd.DataFrame(metadata)
    for name, value in (("anchors", np.concatenate(anchors)), ("targets", target), ("current", current)):
        np.save(output / f"{name}.npy", value)
    samples.to_parquet(output / "samples.parquet", index=False)
    mean = total / count
    write_json(output / "scaling.json", {
        "feature_mean": mean.tolist(),
        "feature_std": np.maximum(np.sqrt(np.maximum(squares / count - mean ** 2, 0)), 0.01).tolist(),
        "fitted_on": "train only", "training_feature_frames": int(count)})
    support_table(target, samples, classes).to_csv(output / "label_support.csv", index=False)
    raw_support = pd.DataFrame(decision_counts).groupby(["split", "pilot_id", "skill"], as_index=False)["count"].sum()
    raw_support.to_csv(output / "decision_support.csv", index=False)
    audit = {
        "development_decisions": int(raw_support["count"].sum()),
        "development_observed_classes": int(raw_support.skill.nunique()),
        "catalogue_classes": len(classes), "max_decision_timestamp_error_s": max_time_error,
        "decision_step_differences": {str(k): int(v) for k, v in pd.Series(cadence).value_counts().items()},
        "duplicate_decision_timestamps": 0, "test_evaluated": False,
        "alignment": "Exact decision at origin time + horizon; no nearest/as-of join or forward fill",
        "current_label_usage": "Privileged change/stay evaluation strata only; never model input",
    }
    write_json(output / "alignment_audit.json", audit)
    # Retain the exact code that built the labels, features and catalogue.
    source_paths = {"skill_data.py": Path(__file__),
                    "motion_data.py": Path(inspect.getfile(observed_kinematics)),
                    "skill_manager.py": Path(inspect.getfile(SkillManager))}
    (output / "preparation_source").mkdir()
    for name, path in source_paths.items():
        shutil.copy2(path, output / "preparation_source" / name)
    summary = {
        "corpus": str(corpus.resolve()), "corpus_manifest_sha256": sha256(corpus / "manifest.json"),
        "reference_splits_sha256": sha256(reference_splits), "split_seed": split_seed,
        "classes": classes, "features": FEATURE_NAMES, "history_steps": history_steps,
        "raw_history_frames": history_steps + 5, "stride_steps": stride_steps,
        "horizons_s": HORIZONS.tolist(), "target_contract": audit["alignment"],
        "input_contract": "B01 observations through origin plus backward acceleration; no actions or IDs",
        "test_policy": "Reserved; no test rows prepared or evaluated",
        "preparation_source_sha256": sha256(__file__),
        "counts": {split: {
            "episodes": int(episodes.split.eq(split).sum()),
            "scenario_groups": int(episodes.loc[episodes.split.eq(split), "scenario_group"].nunique()),
            "samples": int(samples.split.eq(split).sum()),
            "valid_by_horizon": (target[samples.split.eq(split)] >= 0).sum(0).tolist(),
        } for split in ("train", "validation", "test")},
        "files": {p.relative_to(output).as_posix(): sha256(p)
                  for p in sorted(output.rglob("*")) if p.is_file()},
    }
    write_json(output / "data_manifest.json", summary)
    return summary
