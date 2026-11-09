"""Own-skill-conditioned forecasting from notebook 11's two-view NN duels.

The observer's current selected skill is recovered from the *other* view's
privileged target decisions. Opponent decisions remain future-only labels.
No simulator import, regeneration, or modification of notebook 09b is required.
"""

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .paired_skill_forecasting import (
    BASE_FEATURES, DT, HISTORY_STEPS, HORIZONS, KEYS, OBS_COLUMNS,
    RAW_HISTORY_STEPS, SPLITS, STRIDE_STEPS, PairedData, catalogue,
    download_paired_corpora, observation_features, read_json, sha256,
    skill_support_table, validate_frames, verify_source, write_json,
)

DATA_CONTRACT = "nn-duel-own-skill-horizon-v1"
NONOVERLAP_DATA_CONTRACT = "nn-duel-own-skill-nonoverlap-horizon-v1"
ENEMY_MISSILE_DATA_CONTRACT = "nn-duel-own-skill-enemy-missile-nonoverlap-horizon-v1"
MISSILE_FIELDS = ("relative_position_n", "relative_position_w", "relative_position_u",
                  "relative_velocity_n", "relative_velocity_w", "relative_velocity_u", "range")
ENEMY_MISSILE_FEATURES = [f"observer_enemy_missile_{slot}_{field}"
                          for slot in range(1, 5) for field in MISSILE_FIELDS]
ENEMY_MISSILE_POLICY = {
    "slots": 4, "fields": list(MISSILE_FIELDS),
    "source": "same-timestamp opposite-view friendly missile slots, obs_19 through obs_46",
    "ordering": "preserve opposing friendly-slot order; no nearest-range reorder",
    "empty_slots": "all seven raw values zero, matching the compact source convention",
    "position_scales_m": [50000, 50000, 10000], "velocity_scale_mps": 1200,
    "range_scale_m": 50000, "coordinates": "observer-relative world North/West/Up",
}
SIDES = ("A01", "B01")
DECISION_STEPS = 10
# Round the complete raw input footprint up to the decision grid. Alternating
# aircraft views on these slots also prevents repeated physical-flight intervals.
NONOVERLAP_STRIDE_STEPS = ((RAW_HISTORY_STEPS + STRIDE_STEPS - 1)
                          // STRIDE_STEPS) * STRIDE_STEPS
NONOVERLAP_POLICY = {
    "train": "disjoint raw histories across both aircraft views of each physical flight",
    "train_origin_stride_steps": NONOVERLAP_STRIDE_STEPS,
    "raw_history_steps": RAW_HISTORY_STEPS,
    "views": "alternate A01/B01; seeded stable per-flight starting view",
    "test_stride_steps": STRIDE_STEPS, "validation_stride_steps": STRIDE_STEPS,
}


def feature_names(classes, enemy_missiles=False):
    return ([f"observer_{name}" for name in BASE_FEATURES]
            + (ENEMY_MISSILE_FEATURES if enemy_missiles else [])) + [
        f"own_skill::{name}" for name in classes]


def reconstruct_enemy_missiles(observer_raw, opposite_raw):
    """Translate opposing friendly slots into observer-relative enemy slots.

    Both arrays are contemporaneous 51-channel compact+energy recordings. Only
    the opponent's friendly slots and the observer's relative aircraft geometry
    enter this calculation. Empty slots stay zero instead of becoming phantom
    missiles at the opponent's aircraft position. Range is recomputed in metres.
    """
    raw, other = np.asarray(observer_raw, dtype=np.float64), np.asarray(opposite_raw, dtype=np.float64)
    if (raw.ndim != 2 or raw.shape[1] != len(OBS_COLUMNS) or other.shape != raw.shape
            or not np.isfinite(raw).all() or not np.isfinite(other).all()):
        raise ValueError("Expected aligned finite (frames, 51) observation arrays")
    friendly = other[:, 19:47].reshape(-1, 4, 7)
    present = np.any(friendly != 0, axis=-1)
    enemy = np.zeros_like(friendly)
    enemy[:, :, :3] = friendly[:, :, :3] + raw[:, None, 9:12]
    # Aircraft relative velocity uses /600; missile relative velocity uses /1200.
    enemy[:, :, 3:6] = friendly[:, :, 3:6] + .5 * raw[:, None, 12:15]
    enemy[:, :, 6] = np.linalg.norm(enemy[:, :, :3] * [50000, 50000, 10000], axis=-1) / 50000
    enemy[~present] = 0
    return enemy.reshape(-1, len(ENEMY_MISSILE_FEATURES)).astype(np.float32)


def discover_corpus(root, source=None):
    """Select the latest completed notebook-11 package, or use an explicit path.

    Tracked packages use their last manifest commit time (checkout mtimes are
    misleading). Local unpublished packages use manifest mtime. Ties prefer
    more completed flights, then the lexical path. Partial runs are excluded.
    """
    root = Path(root).resolve()
    if source is not None:
        path = Path(source).expanduser()
        path = path if path.is_absolute() else root / path
        candidates = [path / "manifest.json"]
    else:
        candidates = []
        for parent in (root / "shared/corpora", root / "artifacts"):
            if parent.exists():
                candidates.extend(parent.glob("nn*/*/manifest.json"))
                candidates.extend(parent.glob("nn*/manifest.json"))
    ranked = []
    for path in set(candidates):
        if not path.is_file():
            continue
        manifest = read_json(path)
        if manifest.get("name") != "nn-duels-v1" or not manifest.get("complete"):
            continue
        if manifest.get("physical_flights") != manifest.get("requested_flights"):
            raise ValueError(f"Inconsistent completed flight count: {path}")
        stamp = path.stat().st_mtime
        try:
            relative = path.resolve().relative_to(root).as_posix()
            result = subprocess.run(["git", "log", "-1", "--format=%ct", "--", relative],
                                    cwd=root, capture_output=True, text=True, check=False)
            if result.returncode == 0 and result.stdout.strip():
                stamp = int(result.stdout.strip())
        except (OSError, ValueError):
            pass
        ranked.append((stamp, int(manifest["physical_flights"]), str(path), path.parent))
    if not ranked:
        raise ValueError("No completed notebook-11 NN duel corpus found. Set SOURCE to its "
                         "packaged directory containing manifest.json; partial runs are excluded.")
    return max(ranked)[-1].resolve()


def download_corpus(root, source=None, git_executable=None):
    """Pull only the selected package's data, excluding simulator/model archives."""
    root = Path(root).resolve()
    source = discover_corpus(root, source)
    try:
        prefix = source.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("External corpus: download its Parquet files before preparing it") from error
    # Use a few package-scoped patterns, rather than hundreds of exact shard
    # filenames, to remain below Windows' CreateProcess command-length limit.
    includes = [f"{prefix}/{name}" for name in ("config.json", "schema.json", "episodes.parquet")]
    includes += [f"{prefix}/{side}/{part}/*.parquet" for side in SIDES
                 for part in ("observations", "tracks", "target_decisions")]
    download_paired_corpora(root, includes, git_executable)
    return source


def _source_index(root, source):
    source = discover_corpus(root, source)
    manifest = read_json(source / "manifest.json")
    for name in ("config.json", "schema.json", "episodes.parquet"):
        verify_source(source, manifest, name)
    config, schema = read_json(source / "config.json"), read_json(source / "schema.json")
    if (config["sample_interval_s"] != DT or config["decision_interval_s"] != 1
            or schema["schema_version"] != "nn-duels-v1"
            or manifest["config_sha256"] != sha256(source / "config.json")):
        raise ValueError("Expected notebook-11 10 Hz observations and 1 Hz selected skills")
    episodes = pd.read_parquet(source / "episodes.parquet").sort_values(
        ["shard", "observer_id", "episode_id"]).reset_index(drop=True)
    if (len(episodes) != manifest["perspective_episodes"]
            or len(episodes) != 2 * manifest["physical_flights"]
            or episodes.duplicated(["episode_id", "observer_id"]).any()
            or episodes.episode_id.nunique() != manifest["physical_flights"]
            or set(episodes.observer_id) != set(SIDES)):
        raise ValueError("Invalid two-perspective episode index")
    for _, pair in episodes.groupby("episode_id", sort=False):
        if (len(pair) != 2 or set(pair.observer_id) != set(SIDES)
                or pair.scenario_group.nunique() != 1 or pair.shard.nunique() != 1
                or pair.frames.nunique() != 1
                or any(row.target_id != SIDES[1 - SIDES.index(row.observer_id)]
                       for row in pair.itertuples())):
            raise ValueError("Missing, misidentified or misgrouped paired aircraft perspective")
    cases = {case["episode_id"]: case for case in config["cases"]}
    if set(cases) != set(episodes.episode_id):
        raise ValueError("Source configuration and indexed physical flights differ")
    if any(cases[row.episode_id]["scenario_group"] != row.scenario_group
           for row in episodes.itertuples()):
        raise ValueError("Source scenario grouping differs from the collection configuration")
    return source, manifest, config, episodes


def _validate_decisions(decisions, episode_id, aircraft_id, frames, classes, complete=False):
    if (decisions.empty or set(decisions.episode_id) != {episode_id}
            or set(decisions.aircraft_id) != {aircraft_id}
            or decisions.step.duplicated().any()
            or not np.array_equal(decisions.step, np.rint(decisions.step))
            or (decisions.step < 0).any() or (decisions.step >= frames).any()
            or (decisions.step % DECISION_STEPS != 0).any()
            or not np.allclose(decisions.time_s, decisions.step * DT, atol=1e-7, rtol=0)):
        raise ValueError(f"Invalid decisions, aircraft role or timestamps for {episode_id}/{aircraft_id}")
    if not decisions.skill.isin(classes).all():
        raise ValueError("Unknown catalogue skill in decisions")
    if complete and not np.array_equal(decisions.step, np.arange(0, frames - 1, DECISION_STEPS)):
        raise ValueError("Observer decisions contain a gap; refusing to forward fill missing skills")


def current_own_skills(decisions, episode_id, observer_id, frames, classes):
    """Causally hold a selection for <1 second; never backfill or bridge gaps.

    The collector records exactly one selected skill, so these catalogue-wide
    multi-hot slots contain one 1 per nonterminal frame. A terminal frame at a
    new decision boundary has no new selection and remains all zero.
    """
    _validate_decisions(decisions, episode_id, observer_id, frames, classes, complete=True)
    result = np.zeros((frames, len(classes)), dtype=np.float32)
    labels = {name: i for i, name in enumerate(classes)}
    for decision in decisions.itertuples():
        result[int(decision.step):min(frames, int(decision.step) + DECISION_STEPS),
               labels[decision.skill]] = 1
    return result


def episode_examples(observations, tracks, target_decisions, own_decisions, classes,
                     observer_id, target_id, opposite_observations=None):
    """Return per-frame features, eligible origins and future opponent labels."""
    episode_id, frames = tracks.episode_id.iloc[0], len(tracks)
    validate_frames(tracks, episode_id, frames)
    validate_frames(observations, episode_id, frames)
    if not observations[KEYS].equals(tracks[KEYS]):
        raise ValueError("Observation/track keys differ")
    _validate_decisions(target_decisions, episode_id, target_id, frames, classes)
    own = current_own_skills(own_decisions, episode_id, observer_id, frames, classes)
    numeric = observation_features(observations[OBS_COLUMNS])
    if opposite_observations is not None:
        validate_frames(opposite_observations, episode_id, frames)
        if not observations[KEYS].equals(opposite_observations[KEYS]):
            raise ValueError("Opposite-view observation keys differ")
        enemy = reconstruct_enemy_missiles(observations[OBS_COLUMNS], opposite_observations[OBS_COLUMNS])
        numeric = np.concatenate([numeric, enemy], axis=1)
    features = np.concatenate([numeric, own], axis=1)
    live = (tracks.observer_alive & tracks.target_alive).to_numpy(dtype=bool)
    first = ((RAW_HISTORY_STEPS + STRIDE_STEPS - 1) // STRIDE_STEPS) * STRIDE_STEPS
    ends = np.arange(first, frames, STRIDE_STEPS)
    invalid = np.r_[0, np.cumsum(~live)]
    ends = ends[invalid[ends + 1] == invalid[ends - RAW_HISTORY_STEPS + 1]]
    labels = target_decisions.skill.map({name: i for i, name in enumerate(classes)})
    labels = pd.Series(labels.to_numpy(dtype=np.int64), index=target_decisions.step)
    future = ends[:, None] + np.rint(np.asarray(HORIZONS) / DT).astype(int)
    target = labels.reindex(future.ravel(), fill_value=-1).to_numpy().reshape(-1, len(HORIZONS))
    keep = (target >= 0).any(axis=1)
    return features, ends[keep], target[keep]


def _episodes(source, manifest, episodes, observations=True, enemy_missiles=False):
    """Read one episode row group at a time; never load the whole corpus/window tensor."""
    for shard, subset in episodes.groupby("shard", sort=True):
        files = {}
        for side in SIDES:
            parts = ("observations", "tracks", "target_decisions") if observations else (
                "tracks", "target_decisions")
            for part in parts:
                name = f"{side}/{part}/{shard}.parquet"
                verify_source(source, manifest, name)
                files[side, part] = pq.ParquetFile(source / name)
        # Both views are written in the same collection-case order. Confirm their
        # identities rather than assuming lexical IDs equal row-group positions.
        row_groups = {}
        for side in SIDES:
            track_file = files[side, "tracks"]
            if track_file.num_row_groups != len(subset[subset.observer_id.eq(side)]):
                raise ValueError("Expected one Parquet row group per perspective episode")
            for group in range(track_file.num_row_groups):
                ids = track_file.read_row_group(group, columns=["episode_id"]).column(0)
                if not len(ids):
                    raise ValueError("Empty episode row group")
                row_groups[side, ids[0].as_py()] = group
        for index, row in subset.iterrows():
            own_side = row.target_id  # Its target decisions describe this observer.
            group = row_groups[row.observer_id, row.episode_id]
            own_group = row_groups[own_side, row.episode_id]
            tracks = files[row.observer_id, "tracks"].read_row_group(group).to_pandas()
            target = files[row.observer_id, "target_decisions"].read_row_group(group).to_pandas()
            own = files[own_side, "target_decisions"].read_row_group(own_group).to_pandas()
            validate_frames(tracks, row.episode_id, int(row.frames))
            if observations:
                obs = files[row.observer_id, "observations"].read_row_group(group).to_pandas()
                opposite = (files[own_side, "observations"].read_row_group(own_group).to_pandas()
                            if enemy_missiles else None)
            else:
                obs = tracks[KEYS].copy()
                obs[OBS_COLUMNS] = np.zeros((len(tracks), len(OBS_COLUMNS)), dtype=np.float32)
                opposite = None
            yield index, row, episode_examples(obs, tracks, target, own, classes=manifest["classes"],
                                              observer_id=row.observer_id, target_id=row.target_id,
                                              opposite_observations=opposite)
        print(f"Read {shard}: {len(subset):,} perspective episodes", flush=True)


def analyze_corpus(root, source=None):
    """Survey exact eligible future labels before assigning any partition."""
    source, manifest, config, episodes = _source_index(root, source)
    classes = catalogue(root)
    counts = np.zeros((len(episodes), len(HORIZONS), len(classes)), dtype=np.int64)
    for index, _, (_, _, targets) in _episodes(source, {**manifest, "classes": classes},
                                              episodes, observations=False):
        for horizon in range(len(HORIZONS)):
            labels = targets[:, horizon]
            counts[index, horizon] = np.bincount(labels[labels >= 0], minlength=len(classes))
    return {"episodes": episodes, "counts": counts, "classes": classes,
            "cases": config["cases"], "source": str(source),
            "source_manifest_sha256": sha256(source / "manifest.json")}


def split_scenarios(analysis, seed=1729):
    """Scalable seeded 40/30/30 scenario split with range and skill balancing.

    Continuous initial ranges are divided into up to ten quantile strata. Within
    each stratum, fixed group quotas preserve range coverage. Bounded random
    pair-swap search balances each nonzero skill/horizon's episode presence and
    window counts with equal weight, without the quadratic all-pairs 09b search.
    """
    episodes, counts = analysis["episodes"], np.asarray(analysis["counts"])
    names = sorted(episodes.scenario_group.unique())
    if len(names) < 3:
        raise ValueError("At least three independent scenarios are required")
    group_index = {name: i for i, name in enumerate(names)}
    vectors = np.concatenate([(counts > 0).reshape(len(episodes), -1),
                              counts.reshape(len(episodes), -1)], axis=1).astype(float)
    matrix = np.zeros((len(names), vectors.shape[1]))
    np.add.at(matrix, episodes.scenario_group.map(group_index).to_numpy(), vectors)
    totals = matrix.sum(0)
    matrix = matrix[:, totals > 0] / totals[totals > 0]
    if not matrix.shape[1]:
        raise ValueError("No eligible forecast labels")
    ranges = {}
    for case in analysis["cases"]:
        initial = case["initial"]
        value = float(np.linalg.norm(np.asarray(initial["A01"]["position"])
                                     - np.asarray(initial["B01"]["position"])))
        name = case["scenario_group"]
        if name in ranges and not np.isclose(ranges[name], value):
            raise ValueError("A scenario group has conflicting initial ranges")
        ranges[name] = value
    distance = np.asarray([ranges[name] for name in names])
    # At least ten groups per stratum normally gives exact 4/3/3 quotas; small
    # fixture corpora retain a single stratum and three nonempty partitions.
    strata = pd.qcut(distance, q=min(10, max(1, len(names) // 10)),
                     labels=False, duplicates="drop")
    strata = np.nan_to_num(np.asarray(strata), nan=0).astype(int)
    # Repeated ranges can collapse quantile boundaries and leave a final bin
    # with one or two groups. Merge those bins into the nearest range stratum
    # so every retained stratum can contribute to all three partitions.
    while len(np.unique(strata)) > 1:
        labels, sizes = np.unique(strata, return_counts=True)
        sparse = labels[sizes < 3]
        if not len(sparse):
            break
        label = sparse[0]
        alternatives = labels[labels != label]
        center = distance[strata == label].mean()
        neighbor = min(alternatives,
                       key=lambda other: abs(distance[strata == other].mean() - center))
        strata[strata == label] = neighbor
    rng = np.random.default_rng(seed)
    fractions = np.asarray([.4, .3, .3])
    best = None
    for _ in range(4):
        assignment = np.empty(len(names), dtype=int)
        buckets = {}
        for stratum in np.unique(strata):
            ids = rng.permutation(np.flatnonzero(strata == stratum))
            ntrain, ntest = max(1, round(len(ids) * .4)), max(1, round(len(ids) * .3))
            if ntrain + ntest >= len(ids):
                ntrain, ntest = len(ids) - 2, 1
            for partition, part in enumerate(np.split(ids, [ntrain, ntrain + ntest])):
                assignment[part] = partition
            buckets[stratum] = ids
        residual = np.stack([matrix[assignment == i].sum(0) for i in range(3)]) - fractions[:, None]
        for _pass in range(8):
            improved = False
            for i in rng.permutation(len(names)):
                candidates = rng.choice(buckets[strata[i]], size=min(32, len(buckets[strata[i]])),
                                        replace=False)
                candidates = candidates[assignment[candidates] != assignment[i]]
                if not len(candidates):
                    continue
                delta = matrix[candidates] - matrix[i]
                gains = (2 * delta * (residual[assignment[i]] - residual[assignment[candidates]])
                         + 2 * delta**2).sum(1)
                selected = int(np.argmin(gains))
                if gains[selected] < -1e-12:
                    j = candidates[selected]
                    a, b = assignment[i], assignment[j]
                    residual[a] += delta[selected]
                    residual[b] -= delta[selected]
                    assignment[i], assignment[j] = b, a
                    improved = True
            if not improved:
                break
        score = float(np.square(residual).sum())
        if best is None or score < best[0]:
            best = score, assignment.copy()
    return {name: {"split": SPLITS[int(best[1][i])], "range_m": ranges[name],
                   "range_stratum": int(strata[i])} for i, name in enumerate(names)}


def nonoverlapping_train_mask(ends, episode_id, observer_id, seed=1729):
    """Keep six-second slots, assigning each slot to just one aircraft view.

    The fixed schedule never depends on labels, losses or row order. Missing
    origins leave gaps rather than moving another window into the reserved slot.
    Histories (including acceleration lookback) are disjoint; future labels are
    still exact +1/+2/+5 s decisions and may fall in a later input history.
    """
    import hashlib

    if observer_id not in SIDES:
        raise ValueError("Unknown observer aircraft")
    ends = np.asarray(ends)
    first = ((RAW_HISTORY_STEPS + STRIDE_STEPS - 1) // STRIDE_STEPS) * STRIDE_STEPS
    digest = hashlib.sha256(f"{seed}:{episode_id}".encode()).digest()
    starting_view = digest[0] % len(SIDES)
    slot = (ends - first) // NONOVERLAP_STRIDE_STEPS
    return ((ends >= first) & ((ends - first) % NONOVERLAP_STRIDE_STEPS == 0)
            & ((slot + starting_view) % len(SIDES) == SIDES.index(observer_id)))


def nonoverlap_audit(samples):
    """Verify input disjointness across views using physical-flight time, not offsets."""
    train = samples.loc[samples.split.eq("train")].sort_values(["episode_id", "step"])
    spacing = train.groupby("episode_id").step.diff()
    if spacing.lt(RAW_HISTORY_STEPS).any():
        raise ValueError("Overlapping training raw histories in the same physical flight")
    return {"training_windows": len(train),
            "minimum_origin_spacing_steps": int(spacing.min()) if spacing.notna().any() else None,
            "raw_history_steps": RAW_HISTORY_STEPS, "overlapping_pairs": 0}


def prepare_dataset(root, output, source=None, split_seed=1729, analysis=None,
                    nonoverlapping_train=False, enemy_missiles=False):
    """Write immutable frames, labels, grouped splits and training-only scaling.

    Optional thinning applies only to training, after the original split is fixed.
    Test/validation examples and scenario assignments match the dense preparation.
    """
    if not isinstance(nonoverlapping_train, bool):
        raise ValueError("nonoverlapping_train must be boolean")
    if not isinstance(enemy_missiles, bool) or (enemy_missiles and not nonoverlapping_train):
        raise ValueError("enemy_missiles must be boolean and requires non-overlapping training")
    root, output = Path(root).resolve(), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory or load the existing prepared dataset")
    analysis = analyze_corpus(root, source) if analysis is None else analysis
    source = discover_corpus(root, source if source is not None else analysis["source"])
    if analysis["source_manifest_sha256"] != sha256(source / "manifest.json"):
        raise ValueError("Source manifest changed; rerun the corpus analysis")
    source_manifest = read_json(source / "manifest.json")
    episodes, classes = analysis["episodes"].copy(), analysis["classes"]
    names = feature_names(classes, enemy_missiles)
    numeric_features = len(BASE_FEATURES) + (len(ENEMY_MISSILE_FEATURES) if enemy_missiles else 0)
    groups = split_scenarios(analysis, split_seed)
    episodes["split"] = episodes.scenario_group.map(lambda name: groups[name]["split"])
    output.mkdir(parents=True, exist_ok=True)
    features = np.lib.format.open_memmap(output / "features.npy", mode="w+", dtype="float32",
                                         shape=(int(episodes.frames.sum()), len(names)))
    total, squares, count = np.zeros(len(names)), np.zeros(len(names)), 0
    anchors, targets, metadata, offset = [], [], [], 0
    retained_counts = np.zeros_like(analysis["counts"])
    sampling_rows = []
    missile_rows = []
    for index, row, (frame, ends, target) in _episodes(
            source, {**source_manifest, "classes": classes}, episodes, enemy_missiles=enemy_missiles):
        features[offset:offset + len(frame)] = frame
        if enemy_missiles:
            slots = frame[:, len(BASE_FEATURES):numeric_features].reshape(-1, 4, 7)
            present = np.any(slots != 0, axis=-1)
            missile_rows.append({"episode_id": row.episode_id, "observer_id": row.observer_id,
                                 "split": row.split, "raw_frames": len(frame),
                                 "frames_with_enemy_missiles": int(present.any(1).sum()),
                                 **{f"slot_{s+1}_nonempty_frames": int(present[:, s].sum())
                                    for s in range(4)}})
        eligible = len(ends)
        if nonoverlapping_train and row.split == "train":
            keep = nonoverlapping_train_mask(ends, row.episode_id, row.observer_id, split_seed)
            ends, target = ends[keep], target[keep]
        for horizon in range(len(HORIZONS)):
            labels = target[:, horizon]
            retained_counts[index, horizon] = np.bincount(labels[labels >= 0], minlength=len(classes))
        sampling_rows.append({"episode_id": row.episode_id, "observer_id": row.observer_id,
                              "split": row.split, "eligible_windows": eligible,
                              "retained_windows": len(ends), "dropped_windows": eligible - len(ends)})
        if row.split == "train" and len(ends):
            used = np.zeros(len(frame), dtype=bool)
            for end in ends:
                used[end - HISTORY_STEPS + 1:end + 1] = True
            values = frame[used].astype(np.float64)
            total += values.sum(0)
            squares += np.square(values).sum(0)
            count += len(values)
        anchors.append(ends + offset)
        targets.append(target)
        metadata.extend({"episode_id": row.episode_id, "observer_id": row.observer_id,
                         "target_id": row.target_id, "pilot_id": row.pilot_id,
                         "scenario_group": row.scenario_group, "split": row.split,
                         "step": int(end), "time_s": float(end * DT)} for end in ends)
        offset += len(frame)
    features.flush()
    del features
    if not count:
        raise ValueError("No eligible training frames")
    mean = total / count
    std = np.maximum(np.sqrt(np.maximum(squares / count - mean**2, 0)), .01)
    mean[numeric_features:], std[numeric_features:] = 0, 1
    samples, target = pd.DataFrame(metadata), np.concatenate(targets)
    audit = nonoverlap_audit(samples) if nonoverlapping_train else None
    for split in SPLITS:
        if not ((target[samples.split.eq(split)] >= 0).sum(0) > 0).all():
            raise ValueError(f"No labels for one or more horizons in {split}")
    np.save(output / "anchors.npy", np.concatenate(anchors))
    np.save(output / "targets.npy", target)
    samples.to_parquet(output / "samples.parquet", index=False)
    episodes.to_csv(output / "split_assignments.csv", index=False)
    write_json(output / "scenario_splits.json", groups)
    write_json(output / "scaling.json", {"mean": mean.tolist(), "std": std.tolist(),
                                         "fitted_on": "unique train history frames only",
                                         "binary_skill_features": "unscaled", "frames": count})
    skill_support_table(analysis).to_csv(output / "skill_support_before_split.csv", index=False)
    skill_support_table({**analysis, "counts": retained_counts}, groups).to_csv(
        output / "skill_support_by_split.csv", index=False)
    if nonoverlapping_train:
        pd.DataFrame(sampling_rows).to_csv(output / "window_sampling_audit.csv", index=False)
        write_json(output / "nonoverlap_audit.json", audit)
    if enemy_missiles:
        pd.DataFrame(missile_rows).to_csv(output / "enemy_missile_audit.csv", index=False)
    try:
        source_name = source.relative_to(root).as_posix()
    except ValueError:
        source_name = str(source)
    manifest = {
        "data_contract": (ENEMY_MISSILE_DATA_CONTRACT if enemy_missiles else
                          NONOVERLAP_DATA_CONTRACT if nonoverlapping_train else DATA_CONTRACT),
        "classes": classes, "features": names,
        "history_steps": HISTORY_STEPS, "raw_history_steps": RAW_HISTORY_STEPS,
        "stride_steps": STRIDE_STEPS, "horizons_s": HORIZONS, "split_seed": split_seed,
        "split_strategy": "scenario-group range-quantile quota and bounded skill-horizon swaps",
        "split_fractions": dict(zip(SPLITS, (.4, .3, .3))),
        "source_manifests": {source_name: analysis["source_manifest_sha256"]},
        "source_physical_flights": source_manifest["physical_flights"],
        "source_perspective_episodes": source_manifest["perspective_episodes"],
        "source_code_sha256": sha256(__file__),
        "feature_helpers_sha256": sha256(Path(__file__).with_name("paired_skill_forecasting.py")),
        "catalogue_sha256": sha256(root / "bvr_sim_source/bvr_sim/agents/skill_manager.py"),
        "input_contract": "observer observations and backward acceleration plus current own skill",
        "target_contract": "opposing aircraft selected skill at exact +1/+2/+5 second timestamps",
        "own_skill_encoding": "catalogue-wide binary slots; current selected skill only",
        "own_skill_alignment": "opposite-view target decisions, causal hold for <1 second",
        "target": "observer's opponent selected skill at exact future timestamps; missing=-1",
        "counts": {s: {"groups": int(episodes.loc[episodes.split.eq(s), "scenario_group"].nunique()),
                       "episodes": int(episodes.split.eq(s).sum()),
                       "windows": int(samples.split.eq(s).sum()),
                       "labels_by_horizon": (target[samples.split.eq(s)] >= 0).sum(0).tolist()}
                   for s in SPLITS},
    }
    if nonoverlapping_train:
        # stride_steps above is the candidate/evaluation grid, not the training stride.
        manifest["window_sampling"] = dict(NONOVERLAP_POLICY)
    if enemy_missiles:
        manifest["enemy_missiles"] = dict(ENEMY_MISSILE_POLICY)
        manifest["own_skill_start_index"] = numeric_features
        manifest["input_contract"] += " and four same-time observer-relative enemy missile slots"
    manifest["files"] = {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}
    write_json(output / "manifest.json", manifest)
    return manifest


class NNDuelData(PairedData):
    """09b trainer-compatible loader that refuses another prepared-data contract."""

    data_contract = DATA_CONTRACT
    enemy_missiles = False
    numeric_features = len(BASE_FEATURES)

    def __init__(self, directory):
        manifest = read_json(Path(directory) / "manifest.json")
        if (manifest.get("data_contract") != self.data_contract
                or manifest.get("features") != feature_names(manifest["classes"], self.enemy_missiles)
                or manifest.get("history_steps") != HISTORY_STEPS
                or tuple(manifest.get("horizons_s", ())) != HORIZONS):
            raise ValueError(f"Expected own-skill-conditioned NN duel dataset: {self.data_contract}")
        super().__init__(directory)
        width = len(self.manifest["features"])
        if (self.features.ndim != 2 or self.features.shape[1] != width
                or self.mean.shape != (width,) or self.std.shape != (width,)
                or self.targets.shape != (len(self.samples), len(HORIZONS))
                or self.anchors.shape != (len(self.samples),)):
            raise ValueError("Prepared feature/label/scaling dimensions differ from the manifest")
        if (not np.array_equal(self.mean[self.numeric_features:], np.zeros(len(self.classes)))
                or not np.array_equal(self.std[self.numeric_features:], np.ones(len(self.classes)))):
            raise ValueError("Own skill binary slots must remain unscaled")
        assignments = pd.read_csv(self.directory / "split_assignments.csv")
        if assignments.groupby("episode_id").split.nunique().gt(1).any():
            raise ValueError("Two perspectives of a physical flight cross partitions")


class NonoverlappingNNDuelData(NNDuelData):
    """Notebook-14 dataset with verified disjoint raw training histories."""

    data_contract = NONOVERLAP_DATA_CONTRACT

    def __init__(self, directory):
        super().__init__(directory)
        if self.manifest.get("window_sampling") != NONOVERLAP_POLICY:
            raise ValueError("Expected the notebook-14 non-overlapping training-window policy")
        nonoverlap_audit(self.samples)
        train = self.samples.loc[self.samples.split.eq("train")]
        for (episode_id, observer_id), rows in train.groupby(["episode_id", "observer_id"]):
            if not nonoverlapping_train_mask(rows.step.to_numpy(), episode_id, observer_id,
                                            self.manifest["split_seed"]).all():
                raise ValueError("Training window violates the non-overlapping view schedule")


class EnemyMissileNNDuelData(NonoverlappingNNDuelData):
    """Notebook-17 dataset with four additional enemy missile slots per frame."""

    data_contract = ENEMY_MISSILE_DATA_CONTRACT
    enemy_missiles = True
    numeric_features = len(BASE_FEATURES) + len(ENEMY_MISSILE_FEATURES)

    def __init__(self, directory):
        super().__init__(directory)
        if (self.manifest.get("enemy_missiles") != ENEMY_MISSILE_POLICY
                or self.manifest.get("own_skill_start_index") != self.numeric_features):
            raise ValueError("Expected the notebook-17 enemy-missile feature contract")
