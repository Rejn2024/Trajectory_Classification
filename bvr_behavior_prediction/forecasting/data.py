"""Prepare grouped, causal trajectory windows without materialising repeated histories."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DT = 0.1
HORIZONS = np.array([1.0, 2.0, 5.0], dtype=np.float32)
OBS_COLUMNS = [f"obs_{i:02d}" for i in range(51)]
FEATURE_NAMES = OBS_COLUMNS + [f"observed_target_acceleration_{axis}" for axis in "nwu"]
SPLITS = ("train", "validation", "test")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for attempt in range(20):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def grouped_split(cases, seed=1729):
    """70/15/15 scenario groups, stratified by requested initial range."""
    rng = np.random.default_rng(seed)
    by_range = {}
    for case in cases:
        by_range.setdefault(case["scenario"]["range_m"], []).append(case["scenario_group"])
    result = {}
    for distance, groups in sorted(by_range.items()):
        groups = np.asarray(sorted(groups))
        rng.shuffle(groups)
        train_end, validation_end = round(len(groups) * 0.7), round(len(groups) * 0.85)
        if not 0 < train_end < validation_end < len(groups):
            raise ValueError("Too few matched scenarios for three nonempty partitions")
        for split, subset in zip(SPLITS, np.split(groups, [train_end, validation_end]), strict=True):
            for group in subset:
                if group in result:
                    raise ValueError("Duplicate scenario group")
                result[str(group)] = {"split": split, "requested_range_m": float(distance)}
    return result


def observed_kinematics(observations):
    """Reconstruct target kinematics and a backward 0.5 s acceleration estimate.

    Only the observation vector is used. Neither private target actions nor the
    physical ground-truth track table is consulted for forecasting inputs.
    """
    obs = np.asarray(observations, dtype=np.float64)
    position = (obs[:, :3] + obs[:, 9:12]) * [50000.0, 50000.0, 10000.0]
    velocity = (obs[:, 3:6] + obs[:, 12:15]) * 600.0
    previous = np.maximum(0, np.arange(len(obs)) - 5)
    elapsed = np.maximum(1, np.arange(len(obs)) - previous) * DT
    acceleration = (velocity - velocity[previous]) / elapsed[:, None]
    return position, velocity, acceleration


def episode_examples(observations, tracks, history_steps=51, stride_steps=10):
    """Build anchors and masked future residuals entirely within one episode."""
    observations = np.asarray(observations, dtype=np.float32)
    if observations.shape != (len(tracks), 51) or not np.isfinite(observations).all():
        raise ValueError("Invalid observation shape or values")
    if history_steps < 2 or stride_steps < 1:
        raise ValueError("Invalid history or stride")
    position, velocity, acceleration = observed_kinematics(observations)
    features = np.concatenate((observations, acceleration / 100.0), axis=1).astype(np.float32)
    # Skip reset frame 0; require a complete chronological history, with no padding.
    first = ((history_steps + stride_steps - 1) // stride_steps) * stride_steps
    ends = np.arange(first, len(tracks), stride_steps)
    alive = tracks.observer_alive.to_numpy() & tracks.target_alive.to_numpy()
    invalid_prefix = np.r_[0, np.cumsum(~alive)]
    history_ok = (invalid_prefix[ends + 1] - invalid_prefix[ends - history_steps + 1]) == 0
    future_steps = ends[:, None] + np.rint(HORIZONS / DT).astype(int)
    in_bounds = future_steps < len(tracks)
    bounded = np.minimum(future_steps, len(tracks) - 1)
    mask = in_bounds & tracks.target_alive.to_numpy()[bounded] & history_ok[:, None]
    keep = mask.any(axis=1)
    ends, bounded, mask = ends[keep], bounded[keep], mask[keep]
    truth = tracks[[f"target_{name}" for name in ("x", "y", "z", "vx", "vy", "vz")]].to_numpy()
    if not np.isfinite(truth).all():
        raise ValueError("Nonfinite target track")
    baseline = np.concatenate((position[ends, None] + velocity[ends, None] * HORIZONS[None, :, None],
                               np.broadcast_to(velocity[ends, None], (len(ends), 3, 3))), axis=-1)
    residual = truth[bounded] - baseline
    residual[~mask] = 0
    ca_residual = np.concatenate((0.5 * acceleration[ends, None] * HORIZONS[None, :, None] ** 2,
                                  acceleration[ends, None] * HORIZONS[None, :, None]), axis=-1)
    return features, ends, residual.astype(np.float32), mask, ca_residual.astype(np.float32)


def fit_scaling(feature_sum, feature_square_sum, feature_count, residual, mask):
    """All arguments must contain training observations/targets only."""
    mean = feature_sum / feature_count
    std = np.sqrt(np.maximum(feature_square_sum / feature_count - mean ** 2, 0))
    std = np.maximum(std, 0.01)
    # No target centering: a zero network output must reproduce constant velocity.
    # One scale per horizon and quantity; all three spatial axes retain equal weight.
    scales = np.ones((3, 6), dtype=np.float32)
    for h in range(3):
        valid = residual[mask[:, h], h]
        if not len(valid):
            raise ValueError("Training partition has no valid labels for a horizon")
        for start in (0, 3):
            rms = np.sqrt(np.mean(valid[:, start:start + 3].astype(np.float64) ** 2))
            scales[h, start:start + 3] = max(1.0, rms)
    return {"feature_mean": mean.tolist(), "feature_std": std.tolist(),
            "residual_scale": scales.tolist(), "fitted_on": "train only",
            "training_feature_frames": int(feature_count)}


def prepare_dataset(corpus, output, history_steps=51, stride_steps=10, split_seed=1729):
    corpus, output = Path(corpus), Path(output)
    if (output / "data_manifest.json").exists():
        raise ValueError("Prepared dataset already exists; reuse it or choose another directory")
    output.mkdir(parents=True, exist_ok=True)
    config, manifest = read_json(corpus / "config.json"), read_json(corpus / "manifest.json")
    if config["observer_id"] != "B01" or config["target_id"] != "A01":
        raise ValueError("Expected tactical observer B01 and trained target A01")
    if config["sample_interval_s"] != DT or not manifest["complete"]:
        raise ValueError("Incomplete corpus or unexpected sample rate")
    groups = grouped_split(config["cases"], split_seed)
    episodes = pd.read_parquet(corpus / "episodes.parquet").sort_values(["pilot_id", "episode_id"])
    if episodes.episode_id.nunique() != len(episodes) or len(episodes) != manifest["episodes"]:
        raise ValueError("Invalid episode index")
    episodes["split"] = episodes.scenario_group.map(lambda g: groups[g]["split"])
    episodes[["episode_id", "pilot_id", "scenario_group", "seed", "split"]].to_csv(
        output / "split_assignments.csv", index=False)
    write_json(output / "scenario_splits.json", groups)
    used = ["episodes.parquet", "config.json", "schema.json"] + [
        f"{part}/{pilot}.parquet" for pilot in sorted(episodes.pilot_id.unique())
        for part in ("observations", "tracks")]
    for relative in used:
        if sha256(corpus / relative) != manifest["files"][relative]["sha256"]:
            raise ValueError(f"Corpus checksum mismatch: {relative}")
    feature_array = np.lib.format.open_memmap(output / "features.npy", mode="w+", dtype="float32",
                                             shape=(int(episodes.frames.sum()), len(FEATURE_NAMES)))
    feature_sum, feature_square_sum, feature_count = np.zeros(54), np.zeros(54), 0
    anchors, residuals, masks, ca_residuals, metadata = [], [], [], [], []
    offset = 0
    for pilot_id, pilot_episodes in episodes.groupby("pilot_id", sort=True):
        obs_file, track_file = [pq.ParquetFile(corpus / part / f"{pilot_id}.parquet")
                                for part in ("observations", "tracks")]
        if obs_file.num_row_groups != len(pilot_episodes) or track_file.num_row_groups != len(pilot_episodes):
            raise ValueError("Expected one row group per episode")
        for row_group, episode in enumerate(pilot_episodes.itertuples()):
            obs, track = [file.read_row_group(row_group).to_pandas() for file in (obs_file, track_file)]
            if (set(obs.episode_id) != {episode.episode_id} or
                    not obs[["episode_id", "step", "time_s"]].equals(track[["episode_id", "step", "time_s"]])):
                raise ValueError("Episode boundaries or timestamps do not align")
            if (not np.array_equal(track.step, np.arange(len(track))) or
                    not np.allclose(track.time_s, track.step * DT) or len(track) != episode.frames):
                raise ValueError("Invalid frame cadence")
            features, ends, residual, mask, ca_residual = episode_examples(
                obs[OBS_COLUMNS].to_numpy(), track, history_steps, stride_steps)
            feature_array[offset:offset + len(features)] = features
            if episode.split == "train" and len(ends):
                # Fit on the union of frames actually used by training histories.
                used_frames = np.zeros(len(features), dtype=bool)
                for end in ends:
                    used_frames[end - history_steps + 1:end + 1] = True
                values = features[used_frames].astype(np.float64)
                feature_sum += values.sum(axis=0)
                feature_square_sum += np.square(values).sum(axis=0)
                feature_count += len(values)
            anchors.append(ends + offset)
            residuals.append(residual)
            masks.append(mask)
            ca_residuals.append(ca_residual)
            metadata.extend({"episode_id": episode.episode_id, "pilot_id": pilot_id,
                             "scenario_group": episode.scenario_group, "split": episode.split,
                             "step": int(end), "time_s": float(end * DT)} for end in ends)
            offset += len(features)
        print(f"Prepared {pilot_id}: {offset:,} frames", flush=True)
    feature_array.flush()
    del feature_array
    residual, mask, metadata = np.concatenate(residuals), np.concatenate(masks), pd.DataFrame(metadata)
    train = metadata.split.to_numpy() == "train"
    scaling = fit_scaling(feature_sum, feature_square_sum, feature_count, residual[train], mask[train])
    for name, value in (("anchors", np.concatenate(anchors)), ("residuals", residual),
                        ("valid", mask), ("constant_acceleration", np.concatenate(ca_residuals))):
        np.save(output / f"{name}.npy", value)
    metadata.to_parquet(output / "samples.parquet", index=False, compression="zstd")
    write_json(output / "scaling.json", scaling)
    summary = {"corpus": str(corpus.resolve()), "corpus_manifest_sha256": sha256(corpus / "manifest.json"),
               "history_steps": history_steps, "history_span_s": (history_steps - 1) * DT,
               "anchor_stride_s": stride_steps * DT, "horizons_s": HORIZONS.tolist(),
               "features": FEATURE_NAMES, "split_seed": split_seed,
               "input_contract": "Current/past B01 observations and backward acceleration only; no target actions or IDs",
               "target_contract": "A01 world NWU future position and velocity, represented as residuals to observed constant velocity",
               "test_policy": "Locked; no test performance is computed during model development",
               "counts": {split: {"episodes": int((episodes.split == split).sum()),
                                  "scenario_groups": int(episodes.loc[episodes.split == split, "scenario_group"].nunique()),
                                  "samples": int((metadata.split == split).sum()),
                                  "valid_by_horizon": mask[metadata.split == split].sum(axis=0).tolist()}
                          for split in SPLITS},
               "exclusions": "Reset frame and incomplete/dead histories; unavailable/dead future targets are masked per horizon",
               "files": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}}
    write_json(output / "data_manifest.json", summary)
    return summary
