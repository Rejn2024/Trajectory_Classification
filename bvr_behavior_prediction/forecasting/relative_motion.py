"""Prepare 91-feature NN-duel histories for relative aircraft motion forecasting."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DT = 0.1
HORIZONS = np.asarray([1.0, 2.0, 5.0], dtype=np.float32)
HORIZON_STEPS = np.rint(HORIZONS / DT).astype(np.int64)
HISTORY_STEPS = 20
STRIDE_STEPS = 10
NONOVERLAP_STRIDE_STEPS = 20
SIDES = ("A01", "B01")
SPLITS = ("train", "test", "validation")
OBSERVATION_COLUMNS = tuple(f"obs_{index:02d}" for index in range(91))
KEY_COLUMNS = ("episode_id", "step", "time_s")
STATE_COLUMNS = tuple(
    f"{role}_{field}"
    for role in ("observer", "target")
    for field in ("x", "y", "z", "vx", "vy", "vz")
)
TARGET_FIELDS = tuple(
    f"relative_{field}"
    for field in ("north_m", "west_m", "up_m", "vn_mps", "vw_mps", "vu_mps")
)
DATA_CONTRACT = "nn-duel-relative-position-velocity-91-v1"
TARGET_CONTRACT = (
    "At each exact +1/+2/+5 second timestamp: opposing aircraft position minus observer "
    "position and opposing aircraft velocity minus observer velocity in world North/West/Up"
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(source, manifest, relative):
    path = Path(source) / relative
    expected = manifest.get("files", {}).get(relative, {}).get("sha256")
    if expected is None or not path.is_file() or sha256(path) != expected:
        raise ValueError(f"Missing, pointer-only, or changed corpus file: {path}")


def discover_corpus(root, source=None):
    """Resolve an explicit notebook-32 corpus, or the newest complete 91-feature corpus."""
    root = Path(root).resolve()
    if source is not None:
        candidate = Path(source).expanduser()
        candidate = candidate if candidate.is_absolute() else root / candidate
        manifests = [candidate / "manifest.json"]
    else:
        manifests = []
        for parent in (root / "shared/corpora", root / "artifacts"):
            if parent.exists():
                manifests.extend(parent.glob("**/manifest.json"))
    ranked = []
    for manifest_path in set(manifests):
        schema_path = manifest_path.with_name("schema.json")
        if not manifest_path.is_file() or not schema_path.is_file():
            continue
        manifest = read_json(manifest_path)
        schema = read_json(schema_path)
        if (
            manifest.get("complete")
            and manifest.get("physical_flights") == manifest.get("requested_flights")
            and schema.get("observation_columns") == list(OBSERVATION_COLUMNS)
        ):
            ranked.append(
                (
                    manifest_path.stat().st_mtime_ns,
                    int(manifest.get("physical_flights", 0)),
                    str(manifest_path.parent),
                    manifest_path.parent.resolve(),
                )
            )
    if not ranked:
        detail = " at the requested path" if source is not None else ""
        raise ValueError(
            "No complete 91-feature NN-duel corpus found"
            f"{detail}. Run notebook 32 to completion or set BVR_RELATIVE_MOTION_CORPUS."
        )
    return max(ranked)[-1]


def source_index(source):
    """Validate the packaged two-perspective corpus and return its immutable index."""
    source = Path(source).resolve()
    manifest = read_json(source / "manifest.json")
    for relative in ("config.json", "schema.json", "episodes.parquet"):
        verify_source(source, manifest, relative)
    config, schema = read_json(source / "config.json"), read_json(source / "schema.json")
    if not manifest.get("complete"):
        raise ValueError("The notebook-32 corpus is incomplete")
    if manifest.get("physical_flights") != manifest.get("requested_flights"):
        raise ValueError("The completed corpus flight counts disagree")
    if manifest.get("config_sha256") != sha256(source / "config.json"):
        raise ValueError("Corpus configuration digest mismatch")
    if config.get("sample_interval_s") != DT or schema.get("schema_version") != "nn-duels-v1":
        raise ValueError("Expected notebook-32 NN duels sampled at 10 Hz")
    columns = schema.get("observation_columns")
    features = schema.get("features")
    if columns != list(OBSERVATION_COLUMNS) or not isinstance(features, list) or len(features) != 91:
        raise ValueError("Expected the 91-feature notebooks 30/31 observation schema")
    if [feature.get("column") for feature in features] != columns:
        raise ValueError("Feature schema order differs from the Parquet observation columns")
    names = [feature.get("name") for feature in features]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != 91:
        raise ValueError("The 91 policy features require unique nonempty names")
    episodes = pd.read_parquet(source / "episodes.parquet").sort_values(
        ["shard", "observer_id", "episode_id"]
    ).reset_index(drop=True)
    required = {
        "episode_id", "scenario_group", "seed", "observer_id", "target_id", "pilot_id",
        "frames", "shard",
    }
    if not required.issubset(episodes.columns):
        raise ValueError("Episode index is missing required identity columns")
    if (
        len(episodes) != manifest.get("perspective_episodes")
        or len(episodes) != 2 * manifest.get("physical_flights")
        or episodes.duplicated(["episode_id", "observer_id"]).any()
        or episodes.episode_id.nunique() != manifest.get("physical_flights")
        or set(episodes.observer_id) != set(SIDES)
    ):
        raise ValueError("Invalid two-perspective episode index")
    for _, pair in episodes.groupby("episode_id", sort=False):
        if (
            len(pair) != 2
            or set(pair.observer_id) != set(SIDES)
            or pair.scenario_group.nunique() != 1
            or pair.shard.nunique() != 1
            or pair.frames.nunique() != 1
            or any(
                row.target_id != SIDES[1 - SIDES.index(row.observer_id)]
                for row in pair.itertuples()
            )
        ):
            raise ValueError("Missing, misidentified, or misgrouped aircraft perspective")
    cases = {case["episode_id"]: case for case in config.get("cases", [])}
    if set(cases) != set(episodes.episode_id):
        raise ValueError("Configuration cases and packaged episodes differ")
    if any(cases[row.episode_id]["scenario_group"] != row.scenario_group
           for row in episodes.itertuples()):
        raise ValueError("Scenario groups differ between configuration and episode index")
    return source, manifest, config, schema, episodes


def validate_frames(frame, episode_id, frame_count):
    if (
        len(frame) != frame_count
        or set(frame.episode_id) != {episode_id}
        or not np.array_equal(frame.step.to_numpy(), np.arange(frame_count))
        or not np.allclose(frame.time_s, frame.step * DT, atol=1e-7, rtol=0)
    ):
        raise ValueError(f"Invalid frame cadence or episode: {episode_id}")


def _validate_geometry(observations, tracks):
    raw = observations[list(OBSERVATION_COLUMNS)].to_numpy(dtype=np.float64)
    own = tracks[[f"observer_{field}" for field in ("x", "y", "z", "vx", "vy", "vz")]].to_numpy()
    target = tracks[[f"target_{field}" for field in ("x", "y", "z", "vx", "vy", "vz")]].to_numpy()
    own_scale = np.asarray([50000, 50000, 10000, 600, 600, 600], dtype=np.float64)
    live_observer = tracks.observer_alive.to_numpy(dtype=bool)
    live_pair = live_observer & tracks.target_alive.to_numpy(dtype=bool)
    expected_own = np.where(live_observer[:, None], own / own_scale, 0)
    relative = target - own
    expected_relative = np.where(live_pair[:, None], relative / own_scale, 0)
    expected_range = np.where(live_pair, np.linalg.norm(relative[:, :3], axis=1) / 50000, 0)
    comparisons = ((raw[:, :6], expected_own), (raw[:, 9:15], expected_relative),
                   (raw[:, 15], expected_range))
    if any(not np.allclose(actual, expected, rtol=1e-6, atol=1e-7)
           for actual, expected in comparisons):
        raise ValueError("Observation geometry does not match the declared aircraft roles")


def _eligible_origins(tracks, history_steps=HISTORY_STEPS, stride_steps=STRIDE_STEPS):
    if history_steps < 2 or stride_steps < 1:
        raise ValueError("History and stride must be positive, with at least two history frames")
    live = (
        tracks.observer_alive.to_numpy(dtype=bool)
        & tracks.target_alive.to_numpy(dtype=bool)
    )
    # Align origins to the one-second controller grid and skip the reset sample,
    # matching the full 20-frame policy history at steps 20, 30, ... .
    first = ((history_steps + stride_steps - 1) // stride_steps) * stride_steps
    ends = np.arange(first, len(tracks), stride_steps, dtype=np.int64)
    invalid_prefix = np.r_[0, np.cumsum(~live)]
    history_valid = (
        invalid_prefix[ends + 1] - invalid_prefix[ends - history_steps + 1]
    ) == 0
    future = ends[:, None] + HORIZON_STEPS
    in_bounds = future < len(tracks)
    bounded = np.minimum(future, len(tracks) - 1)
    valid = in_bounds & live[bounded] & history_valid[:, None]
    keep = valid.any(axis=1)
    return ends[keep], bounded[keep], valid[keep]


def episode_examples(observations, tracks, history_steps=HISTORY_STEPS,
                     stride_steps=STRIDE_STEPS):
    """Return exact policy frames and future same-time opponent-relative six-vectors."""
    episode_id, frames = tracks.episode_id.iloc[0], len(tracks)
    validate_frames(observations, episode_id, frames)
    validate_frames(tracks, episode_id, frames)
    if not observations[list(KEY_COLUMNS)].equals(tracks[list(KEY_COLUMNS)]):
        raise ValueError("Observation and track keys differ")
    raw = observations[list(OBSERVATION_COLUMNS)].to_numpy(dtype=np.float32)
    states = tracks[list(STATE_COLUMNS)].to_numpy(dtype=np.float64)
    if raw.shape != (frames, 91) or not np.isfinite(raw).all() or not np.isfinite(states).all():
        raise ValueError("Expected finite 91-feature observations and physical tracks")
    for column in ("observer_alive", "target_alive"):
        if np.any(np.diff(tracks[column].to_numpy(dtype=np.int8)) > 0):
            raise ValueError("Aircraft resurrected within an episode")
    _validate_geometry(observations, tracks)
    ends, future, valid = _eligible_origins(tracks, history_steps, stride_steps)
    relative_position = raw[ends, 9:12].astype(np.float64) * [50000, 50000, 10000]
    relative_velocity = raw[ends, 12:15].astype(np.float64) * 600
    baseline = np.concatenate(
        (
            relative_position[:, None] + relative_velocity[:, None] * HORIZONS[None, :, None],
            np.broadcast_to(relative_velocity[:, None], (len(ends), len(HORIZONS), 3)),
        ),
        axis=-1,
    )
    observer = tracks[[f"observer_{field}" for field in ("x", "y", "z", "vx", "vy", "vz")]].to_numpy()
    target = tracks[[f"target_{field}" for field in ("x", "y", "z", "vx", "vy", "vz")]].to_numpy()
    truth = target[future] - observer[future]
    residual = truth - baseline
    for values in (truth, residual, baseline):
        values[~valid] = 0
    return (
        raw,
        ends,
        truth.astype(np.float32),
        residual.astype(np.float32),
        valid,
        baseline.astype(np.float32),
    )


def _row_groups(files, side, subset):
    track_file = files[side, "tracks"]
    expected = len(subset[subset.observer_id.eq(side)])
    if track_file.num_row_groups != expected:
        raise ValueError("Expected one Parquet row group per perspective episode")
    result = {}
    for index in range(track_file.num_row_groups):
        ids = track_file.read_row_group(index, columns=["episode_id"]).column(0)
        if not len(ids):
            raise ValueError("Empty episode row group")
        result[ids[0].as_py()] = index
    if len(result) != expected:
        raise ValueError("Duplicate episode row group")
    return result


def _episodes(source, manifest, episodes, observations=True):
    """Stream row groups from each immutable shard without materialising the corpus."""
    for shard, subset in episodes.groupby("shard", sort=True):
        files, groups = {}, {}
        for side in SIDES:
            parts = ("observations", "tracks") if observations else ("tracks",)
            for part in parts:
                relative = f"{side}/{part}/{shard}.parquet"
                verify_source(source, manifest, relative)
                files[side, part] = pq.ParquetFile(source / relative)
            groups[side] = _row_groups(files, side, subset)
        for index, row in subset.iterrows():
            group = groups[row.observer_id].get(row.episode_id)
            if group is None:
                raise ValueError("Episode index and Parquet row groups differ")
            tracks = files[row.observer_id, "tracks"].read_row_group(group).to_pandas()
            validate_frames(tracks, row.episode_id, int(row.frames))
            observation = None
            if observations:
                observation = files[row.observer_id, "observations"].read_row_group(group).to_pandas()
            yield index, row, observation, tracks
        print(f"Read {shard}: {len(subset):,} perspective episodes", flush=True)


def analyze_corpus(source):
    """Survey eligible windows and future states before assigning a partition."""
    source, manifest, config, schema, episodes = source_index(source)
    counts = np.zeros((len(episodes), len(HORIZONS)), dtype=np.int64)
    for index, _, _, tracks in _episodes(source, manifest, episodes, observations=False):
        _, _, valid = _eligible_origins(tracks)
        counts[index] = valid.sum(axis=0)
    return {
        "source": str(source),
        "source_manifest_sha256": sha256(source / "manifest.json"),
        "episodes": episodes,
        "cases": config["cases"],
        "features": [feature["name"] for feature in schema["features"]],
        "eligible_by_horizon": counts,
    }


def split_scenarios(analysis, seed=1729):
    """Seeded 40/30/30 physical-flight split, stratified by initial range quantiles."""
    episodes = analysis["episodes"]
    names = sorted(episodes.scenario_group.unique())
    if len(names) < 3:
        raise ValueError("At least three physical scenario groups are required")
    cases = {case["scenario_group"]: case for case in analysis["cases"]}
    if set(cases) != set(names):
        raise ValueError("Scenario groups and source cases differ")
    ranges = {}
    for name in names:
        initial = cases[name]["initial"]
        ranges[name] = float(
            np.linalg.norm(
                np.asarray(initial["A01"]["position"], dtype=float)
                - np.asarray(initial["B01"]["position"], dtype=float)
            )
        )
    distance = np.asarray([ranges[name] for name in names])
    strata = pd.qcut(
        distance,
        q=min(10, max(1, len(names) // 10)),
        labels=False,
        duplicates="drop",
    )
    strata = np.nan_to_num(np.asarray(strata), nan=0).astype(int)
    while len(np.unique(strata)) > 1:
        labels, sizes = np.unique(strata, return_counts=True)
        sparse = labels[sizes < 3]
        if not len(sparse):
            break
        label = sparse[0]
        alternatives = labels[labels != label]
        center = distance[strata == label].mean()
        neighbour = min(
            alternatives,
            key=lambda other: abs(distance[strata == other].mean() - center),
        )
        strata[strata == label] = neighbour
    rng = np.random.default_rng(seed)
    assignment = np.empty(len(names), dtype=np.int8)
    for stratum in np.unique(strata):
        indices = rng.permutation(np.flatnonzero(strata == stratum))
        if len(indices) < 3:
            raise ValueError("Range stratum has too few groups for three partitions")
        train_count = max(1, round(len(indices) * 0.4))
        test_count = max(1, round(len(indices) * 0.3))
        if train_count + test_count >= len(indices):
            train_count, test_count = len(indices) - 2, 1
        for split, subset in enumerate(
            np.split(indices, [train_count, train_count + test_count])
        ):
            assignment[subset] = split
    return {
        name: {
            "split": SPLITS[int(assignment[index])],
            "initial_range_m": ranges[name],
            "range_stratum": int(strata[index]),
        }
        for index, name in enumerate(names)
    }


def nonoverlapping_train_mask(ends, episode_id, observer_id, seed=1729):
    """Alternate adjacent two-second history slots across both aircraft perspectives."""
    if observer_id not in SIDES:
        raise ValueError("Unknown observer aircraft")
    ends = np.asarray(ends)
    first = ((HISTORY_STEPS + STRIDE_STEPS - 1) // STRIDE_STEPS) * STRIDE_STEPS
    start_view = hashlib.sha256(f"{seed}:{episode_id}".encode()).digest()[0] % len(SIDES)
    slot = (ends - first) // NONOVERLAP_STRIDE_STEPS
    return (
        (ends >= first)
        & ((ends - first) % NONOVERLAP_STRIDE_STEPS == 0)
        & ((slot + start_view) % len(SIDES) == SIDES.index(observer_id))
    )


def nonoverlap_audit(samples):
    train = samples.loc[samples.split.eq("train")].sort_values(
        ["episode_id", "step", "observer_id"]
    )
    minimum_gap = None
    intervals = 0
    for _, rows in train.groupby("episode_id", sort=False):
        previous_end = -1
        for row in rows.itertuples():
            start = int(row.step) - HISTORY_STEPS + 1
            if start <= previous_end:
                raise ValueError("Training histories overlap across aircraft perspectives")
            if previous_end >= 0:
                gap = start - previous_end - 1
                minimum_gap = gap if minimum_gap is None else min(minimum_gap, gap)
            previous_end = int(row.step)
            intervals += 1
    return {
        "verified": True,
        "training_intervals": intervals,
        "history_steps": HISTORY_STEPS,
        "minimum_gap_frames": 0 if minimum_gap is None else int(minimum_gap),
        "scope": "both observer perspectives on each physical flight time axis",
    }


def _fit_target_scale(residuals, valid):
    scale = np.ones((len(HORIZONS), 6), dtype=np.float32)
    for horizon in range(len(HORIZONS)):
        values = residuals[valid[:, horizon], horizon].astype(np.float64)
        if not len(values):
            raise ValueError("Training partition has no valid label for a horizon")
        for start in (0, 3):
            rms = np.sqrt(np.mean(np.square(values[:, start:start + 3])))
            scale[horizon, start:start + 3] = max(1.0, rms)
    return scale


def prepare_dataset(root, output, source, split_seed=1729, analysis=None):
    """Write immutable raw frames, grouped windows, physical targets, and scaling."""
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty prepared-data directory or reuse its manifest")
    analysis = analyze_corpus(source) if analysis is None else analysis
    source, manifest, _, schema, episodes = source_index(source)
    if analysis["source_manifest_sha256"] != sha256(source / "manifest.json"):
        raise ValueError("Source manifest changed after analysis; rerun the survey")
    groups = split_scenarios(analysis, split_seed)
    episodes = episodes.copy()
    episodes["split"] = episodes.scenario_group.map(lambda name: groups[name]["split"])
    output.mkdir(parents=True, exist_ok=True)
    feature_array = np.lib.format.open_memmap(
        output / "features.npy",
        mode="w+",
        dtype="float32",
        shape=(int(episodes.frames.sum()), 91),
    )
    feature_sum = np.zeros(91, dtype=np.float64)
    feature_square_sum = np.zeros(91, dtype=np.float64)
    feature_count = 0
    anchors, targets, residuals, valid_masks, baselines = [], [], [], [], []
    metadata, sampling_rows = [], []
    offset = 0
    for _, row, observations, tracks in _episodes(source, manifest, episodes):
        frame, ends, target, residual, valid, baseline = episode_examples(observations, tracks)
        feature_array[offset:offset + len(frame)] = frame
        eligible = len(ends)
        if row.split == "train":
            keep = nonoverlapping_train_mask(ends, row.episode_id, row.observer_id, split_seed)
            ends, target, residual, valid, baseline = (
                value[keep] for value in (ends, target, residual, valid, baseline)
            )
        sampling_rows.append(
            {
                "episode_id": row.episode_id,
                "observer_id": row.observer_id,
                "split": row.split,
                "eligible_windows": eligible,
                "retained_windows": len(ends),
                "dropped_windows": eligible - len(ends),
            }
        )
        if row.split == "train" and len(ends):
            used = np.zeros(len(frame), dtype=bool)
            for end in ends:
                used[end - HISTORY_STEPS + 1:end + 1] = True
            values = frame[used].astype(np.float64)
            feature_sum += values.sum(axis=0)
            feature_square_sum += np.square(values).sum(axis=0)
            feature_count += len(values)
        anchors.append(ends + offset)
        targets.append(target)
        residuals.append(residual)
        valid_masks.append(valid)
        baselines.append(baseline)
        metadata.extend(
            {
                "episode_id": row.episode_id,
                "observer_id": row.observer_id,
                "target_id": row.target_id,
                "pilot_id": row.pilot_id,
                "scenario_group": row.scenario_group,
                "split": row.split,
                "step": int(end),
                "time_s": float(end * DT),
            }
            for end in ends
        )
        offset += len(frame)
    feature_array.flush()
    del feature_array
    if offset != int(episodes.frames.sum()) or not feature_count:
        raise ValueError("No usable training frames or incomplete corpus traversal")
    samples = pd.DataFrame(metadata)
    target = np.concatenate(targets)
    residual = np.concatenate(residuals)
    valid = np.concatenate(valid_masks)
    baseline = np.concatenate(baselines)
    if len(samples) != len(target):
        raise ValueError("Prepared sample metadata and target arrays differ")
    for split in SPLITS:
        selected = samples.split.eq(split).to_numpy()
        if not selected.any() or not (valid[selected].sum(axis=0) > 0).all():
            raise ValueError(f"No valid {split} labels for one or more horizons")
    mean = feature_sum / feature_count
    std = np.maximum(
        np.sqrt(np.maximum(feature_square_sum / feature_count - np.square(mean), 0)),
        0.01,
    )
    training = samples.split.eq("train").to_numpy()
    target_scale = _fit_target_scale(residual[training], valid[training])
    arrays = {
        "anchors": np.concatenate(anchors),
        "targets": target,
        "residuals": residual,
        "valid": valid,
        "constant_relative_velocity": baseline,
    }
    for name, values in arrays.items():
        np.save(output / f"{name}.npy", values)
    samples.to_parquet(output / "samples.parquet", index=False, compression="zstd")
    episodes.to_csv(output / "split_assignments.csv", index=False)
    pd.DataFrame(sampling_rows).to_csv(output / "window_sampling_audit.csv", index=False)
    write_json(output / "scenario_splits.json", groups)
    audit = nonoverlap_audit(samples)
    write_json(output / "nonoverlap_audit.json", audit)
    write_json(
        output / "scaling.json",
        {
            "feature_mean": mean.tolist(),
            "feature_std": std.tolist(),
            "target_residual_scale": target_scale.tolist(),
            "fitted_on": "unique frames in retained training histories and valid training targets only",
            "training_feature_frames": int(feature_count),
        },
    )
    support_rows = []
    for split in SPLITS:
        selected = samples.split.eq(split).to_numpy()
        for horizon, seconds in enumerate(HORIZONS):
            mask = selected & valid[:, horizon]
            support_rows.append(
                {
                    "split": split,
                    "horizon_s": float(seconds),
                    "windows": int(mask.sum()),
                    "physical_flights": int(samples.loc[mask, "episode_id"].nunique()),
                    "scenario_groups": int(samples.loc[mask, "scenario_group"].nunique()),
                }
            )
    pd.DataFrame(support_rows).to_csv(output / "target_support_by_split.csv", index=False)
    try:
        source_name = source.relative_to(root).as_posix()
    except ValueError:
        source_name = str(source)
    summary = {
        "data_contract": DATA_CONTRACT,
        "source": source_name,
        "source_manifest_sha256": analysis["source_manifest_sha256"],
        "source_physical_flights": int(manifest["physical_flights"]),
        "source_perspective_episodes": int(manifest["perspective_episodes"]),
        "source_schema_sha256": sha256(source / "schema.json"),
        "source_code_sha256": sha256(__file__),
        "features": [feature["name"] for feature in schema["features"]],
        "feature_columns": list(OBSERVATION_COLUMNS),
        "input_contract": (
            "The exact 91 scaled per-frame policy features used by notebooks 30 and 31; "
            "20 chronological frames at 10 Hz; no appended acceleration, skill, identity, or future data"
        ),
        "history_steps": HISTORY_STEPS,
        "history_span_s": (HISTORY_STEPS - 1) * DT,
        "candidate_stride_steps": STRIDE_STEPS,
        "horizons_s": HORIZONS.tolist(),
        "target_fields": list(TARGET_FIELDS),
        "target_contract": TARGET_CONTRACT,
        "target_representation": (
            "targets.npy contains physical six-vectors; residuals.npy subtracts the current "
            "constant-relative-velocity baseline saved in constant_relative_velocity.npy"
        ),
        "target_units": ["m", "m", "m", "m/s", "m/s", "m/s"],
        "split_seed": split_seed,
        "split_strategy": "scenario-group range-quantile stratified 40/30/30",
        "split_fractions": dict(zip(SPLITS, (0.4, 0.3, 0.3))),
        "window_sampling": {
            "train": "disjoint 20-frame histories across both views, alternating by physical time",
            "train_origin_stride_steps": NONOVERLAP_STRIDE_STEPS,
            "test_stride_steps": STRIDE_STEPS,
            "validation_stride_steps": STRIDE_STEPS,
        },
        "missing_target_policy": "valid.npy masks each horizon independently after termination/end",
        "counts": {
            split: {
                "groups": int(episodes.loc[episodes.split.eq(split), "scenario_group"].nunique()),
                "perspective_episodes": int(episodes.split.eq(split).sum()),
                "physical_flights": int(episodes.loc[episodes.split.eq(split), "episode_id"].nunique()),
                "windows": int(samples.split.eq(split).sum()),
                "valid_by_horizon": valid[samples.split.eq(split)].sum(axis=0).tolist(),
            }
            for split in SPLITS
        },
        "nonoverlap_audit": audit,
    }
    summary["files"] = {
        path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()
    }
    write_json(output / "manifest.json", summary)
    return summary


class RelativeMotionData:
    """Integrity-check and load notebook-33 relative-motion training data."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = read_json(self.directory / "manifest.json")
        if self.manifest.get("data_contract") != DATA_CONTRACT:
            raise ValueError("Expected notebook-33 91-feature relative-motion data")
        for name, digest in self.manifest.get("files", {}).items():
            if sha256(self.directory / name) != digest:
                raise ValueError(f"Prepared data changed: {name}")
        self.features = np.load(self.directory / "features.npy", mmap_mode="r")
        self.anchors = np.load(self.directory / "anchors.npy")
        self.targets = np.load(self.directory / "targets.npy")
        self.residuals = np.load(self.directory / "residuals.npy")
        self.valid = np.load(self.directory / "valid.npy")
        self.baseline = np.load(self.directory / "constant_relative_velocity.npy")
        self.samples = pd.read_parquet(self.directory / "samples.parquet")
        scaling = read_json(self.directory / "scaling.json")
        self.mean = np.asarray(scaling["feature_mean"], dtype=np.float32)
        self.std = np.asarray(scaling["feature_std"], dtype=np.float32)
        self.target_scale = np.asarray(scaling["target_residual_scale"], dtype=np.float32)
        expected = (len(self.samples), len(HORIZONS), len(TARGET_FIELDS))
        if (
            self.features.ndim != 2
            or self.features.shape[1] != 91
            or self.mean.shape != (91,)
            or self.std.shape != (91,)
            or self.target_scale.shape != (3, 6)
            or self.targets.shape != expected
            or self.residuals.shape != expected
            or self.valid.shape != expected[:2]
            or self.baseline.shape != expected
            or len(self.anchors) != len(self.samples)
            or not np.isfinite(self.mean).all()
            or not np.isfinite(self.std).all()
            or not np.isfinite(self.target_scale).all()
            or (self.std <= 0).any()
            or (self.target_scale <= 0).any()
        ):
            raise ValueError("Prepared array or scaling shape is invalid")
        if not np.allclose(
            self.targets[self.valid],
            (self.baseline + self.residuals)[self.valid],
            rtol=1e-5,
            atol=1e-3,
        ):
            raise ValueError("Physical targets do not equal baseline plus residual")
        for left in SPLITS:
            for right in SPLITS:
                if left < right and set(self.samples.loc[self.samples.split.eq(left), "scenario_group"]) & set(
                    self.samples.loc[self.samples.split.eq(right), "scenario_group"]
                ):
                    raise ValueError("Scenario leakage across partitions")
        nonoverlap_audit(self.samples)
        self.indices = {
            split: np.flatnonzero(self.samples.split.eq(split).to_numpy()) for split in SPLITS
        }

    def history(self, ids, standardize=True):
        ids = np.asarray(ids, dtype=np.int64)
        steps = self.anchors[ids, None] + np.arange(1 - HISTORY_STEPS, 1)
        histories = np.asarray(self.features[steps])
        return (histories - self.mean) / self.std if standardize else histories

    def batch(self, ids, device="cpu"):
        import torch

        ids = np.asarray(ids, dtype=np.int64)
        return (
            torch.from_numpy(self.history(ids).astype(np.float32)).to(device),
            torch.from_numpy((self.residuals[ids] / self.target_scale).astype(np.float32)).to(device),
            torch.from_numpy(self.valid[ids]).to(device),
        )
