"""Trained-pilot-only skill forecasting for notebook 09b.

Keeps the paired experiment's data and training contract self-contained.
Inputs are B01 observation histories of the NN-trained A01 pilot; labels are A01's future selections.
"""

import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

DT = 0.1
HORIZONS = (1, 2, 5)
HISTORY_STEPS = 51
RAW_HISTORY_STEPS = 56
STRIDE_STEPS = 10
OBS_COLUMNS = [f"obs_{i:02d}" for i in range(51)]
KEYS = ["episode_id", "step", "time_s"]
SPLITS = ("train", "test", "validation")
HIERARCHICAL_ARCHITECTURES = ("hgt_hierarchical_intent", "hgt_hierarchical_intent_tcn")
INTENT_ARCHITECTURES = ("hgt_intent", *HIERARCHICAL_ARCHITECTURES)
HGT_ARCHITECTURES = ("hgt", *INTENT_ARCHITECTURES)
BASE_FEATURES = OBS_COLUMNS + [f"observed_target_acceleration_{a}" for a in "nwu"]
FEATURES = [f"B01_{f}" for f in BASE_FEATURES]
DATA_CONTRACT = "trained-pilot-only-skill-horizon-v2"
CORPUS = "shared/corpora/trained-pilot-episodes-v1"
DEFAULT_TRAINING = {
    "epochs": 100, "seed": 42, "batch_size": 512, "evaluation_batch_size": 1024,
    "learning_rate": 0.0003, "weight_decay": 0.01, "gradient_clip": 1.0,
    "width": 192, "heads": 6, "layers": 2, "dropout": 0.1,
    "early_stopping": False, "early_stopping_min_epochs": 50, "early_stopping_patience": 10,
    "precision": "bf16", "data_backend": "gpu_cache", "num_workers": 2,
    "prefetch_factor": 2, "cpu_threads": 4, "loader_timeout": 60,
    "compile_model": False, "compile_mode": "default",
    "l1_lambda": 0.0,
    "overtraining_shutdown": False, "overtraining_patience": 3,
    "overtraining_relative_increase": 0.05, "overtraining_min_gap": 0.1,
    "class_weighting": "none", "max_class_weight_ratio": 5.0,
    "evaluation_precision": "fp32", "evaluation_attention": "math",
    "epoch_streaming_metrics": False, "evaluation_frame_cache": False,
    "evaluation_frame_batch_size": 8192, "evaluation_cache_max_frames": 65536,
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_git_runtime(root, git_executable=None):
    """Return the resolved Git path and a child environment with working Git LFS."""
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Repository directory does not exist: {root}")
    git = None
    if git_executable:
        candidate = Path(git_executable).expanduser()
        git = str(candidate) if candidate.is_file() else shutil.which(str(git_executable))
        if not git:
            raise RuntimeError(f"GIT_EXECUTABLE does not identify an executable file: {git_executable}")
    else:
        git = shutil.which("git")
        if not git and os.name == "nt":
            candidates = [Path(os.environ.get(key, fallback)) / "Git" / "cmd" / "git.exe"
                          for key, fallback in (("ProgramFiles", r"C:\Program Files"),
                                                ("ProgramFiles(x86)", r"C:\Program Files (x86)"))]
            if os.environ.get("LOCALAPPDATA"):
                candidates.append(Path(os.environ["LOCALAPPDATA"]) / "Programs/Git/cmd/git.exe")
            git = next((str(p) for p in candidates if p.is_file()), None)
    if not git:
        raise RuntimeError(
            "Git executable not found by the notebook kernel. Install Git and Git LFS in "
            "the active environment (for Conda: conda install -c conda-forge git git-lfs), "
            "then restart Jupyter. Alternatively set GIT_EXECUTABLE to the full path, "
            r"for example r'C:\Program Files\Git\cmd\git.exe'. "
        )
    git = str(Path(git).resolve())
    # Some Jupyter kernels inherit a PATH that omits Git's sibling executables.
    environment = os.environ.copy()
    environment["PATH"] = str(Path(git).resolve().parent) + os.pathsep + environment.get("PATH", "")
    try:
        probe = subprocess.run([git, "lfs", "version"], cwd=root, env=environment,
                               capture_output=True, text=True, check=False)
    except OSError as error:
        raise RuntimeError(f"The notebook could not start Git at {git}: {error}") from error
    if probe.returncode:
        raise RuntimeError(
            f"Git was found at {git}, but Git LFS is unavailable. Install Git LFS "
            "(Conda: conda install -c conda-forge git-lfs), run git lfs install in "
            "the repository, and restart the notebook kernel. " + probe.stderr.strip()
        )
    return git, environment


def download_paired_corpora(root, includes, git_executable=None):
    """Download selected LFS files and return the Git executable used."""
    root = Path(root)
    git, environment = resolve_git_runtime(root, git_executable)
    print(f"Downloading corpus files using {git}", flush=True)
    subprocess.run([git, "-c", "filter.lfs.clean=git-lfs clean -- %f",
                    "-c", "filter.lfs.smudge=git-lfs smudge -- %f",
                    "-c", "filter.lfs.process=git-lfs filter-process", "-c", "filter.lfs.required=true",
                    "lfs", "pull", "--include=" + ",".join(includes), "--exclude="],
                   cwd=root, env=environment, check=True)
    return git


def catalogue(root):
    # Load the pure-Python catalogue without importing the native simulator.
    path = Path(root) / "bvr_sim_source/bvr_sim/agents/skill_manager.py"
    spec = importlib.util.spec_from_file_location("paired_skill_catalogue", path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return sorted(module.SkillManager().list_skills())


def scenario_splits(cases, episodes, counts, seed=1729):
    """Balance episode presence and window support for every skill/horizon.

    Whole scenario groups move together. Range-specific 40/30/30 group quotas
    are fixed; seeded pair swaps minimize squared deviations from those shares.
    Each nonzero skill/horizon has equal weight, regardless of its frequency.
    Exact balance can be impossible for rare skills or indivisible groups.
    """
    names = sorted(case["scenario_group"] for case in cases)
    if len(names) != len(set(names)) or set(episodes.scenario_group) != set(names):
        raise ValueError("Duplicate or unmatched scenario groups")
    counts = np.asarray(counts)
    if counts.shape[0] != len(episodes) or counts.ndim != 3 or (counts < 0).any():
        raise ValueError("Expected episode x horizon x skill counts")
    # Preserve episode representation, with window counts as an equal-weight objective.
    vectors = np.concatenate([(counts > 0).reshape(len(episodes), -1),
                              counts.reshape(len(episodes), -1)], axis=1).astype(float)
    group_index = {name: i for i, name in enumerate(names)}
    matrix = np.zeros((len(names), vectors.shape[1]))
    np.add.at(matrix, [group_index[g] for g in episodes.scenario_group], vectors)
    totals = matrix.sum(0)
    matrix = matrix[:, totals > 0] / totals[totals > 0]
    if not matrix.shape[1]:
        raise ValueError("No forecast labels to stratify")
    fractions = np.array([.4, .3, .3])
    ranges = {case["scenario_group"]: float(case["scenario"]["range_m"]) for case in cases}
    distance = np.array([ranges[g] for g in names])
    rng = np.random.default_rng(seed)
    best = None
    # Multiple starts avoid depending on one local optimum; deterministic for a seed.
    for _ in range(8):
        assignment = np.empty(len(names), dtype=int)
        for r in np.unique(distance):
            ids = rng.permutation(np.flatnonzero(distance == r))
            ntrain, ntest = round(len(ids) * .4), round(len(ids) * .3)
            if min(ntrain, ntest, len(ids) - ntrain - ntest) < 1:
                raise ValueError("Too few scenarios for three nonempty partitions per range")
            for k, subset in enumerate(np.split(ids, [ntrain, ntrain + ntest])):
                assignment[subset] = k
        residual = np.stack([matrix[assignment == k].sum(0) for k in range(3)]) - fractions[:, None]
        for _pass in range(30):
            improved = False
            for i in rng.permutation(len(names)):
                a = assignment[i]
                candidates = np.flatnonzero((distance == distance[i]) & (assignment != a))
                delta = matrix[candidates] - matrix[i]
                # Swapping i and j adds delta to a and subtracts it from j's split.
                gains = (2 * delta * (residual[a] - residual[assignment[candidates]])
                         + 2 * delta**2).sum(1)
                jpos = int(np.argmin(gains))
                if gains[jpos] < -1e-12:
                    j = candidates[jpos]
                    b = assignment[j]
                    residual[a] += delta[jpos]
                    residual[b] -= delta[jpos]
                    assignment[i], assignment[j] = b, a
                    improved = True
            if not improved:
                break
        score = float(np.square(residual).sum())
        if best is None or score < best[0]:
            best = score, assignment.copy()
    return {g: {"split": SPLITS[best[1][i]], "range_m": ranges[g]}
            for i, g in enumerate(names)}


def observation_features(raw):
    """Match notebook 09's three backward acceleration features per viewpoint."""
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 51 or not np.isfinite(raw).all():
        raise ValueError("Expected finite (frames, 51) observations")
    velocity = (raw[:, 3:6].astype(np.float64) + raw[:, 12:15]) * 600
    previous = np.maximum(0, np.arange(len(raw)) - 5)
    elapsed = np.maximum(1, np.arange(len(raw)) - previous) * DT
    acceleration = (velocity - velocity[previous]) / elapsed[:, None]
    return np.concatenate([raw, acceleration / 100], axis=1).astype(np.float32)


def validate_frames(frame, episode_id, frame_count):
    if (len(frame) != frame_count or set(frame.episode_id) != {episode_id}
            or not np.array_equal(frame.step, np.arange(frame_count))
            or not np.allclose(frame.time_s, frame.step * DT, atol=1e-7, rtol=0)):
        raise ValueError(f"Invalid frame cadence or episode: {episode_id}")


def forecast_targets(tracks_b, decisions_a, classes):
    """Eligibility and exact future labels shared by the survey and preparation."""
    n = len(tracks_b)
    if not n:
        raise ValueError("Empty episode")
    episode_id = tracks_b.episode_id.iloc[0]
    validate_frames(tracks_b, episode_id, n)
    live = (tracks_b.observer_alive & tracks_b.target_alive).to_numpy(dtype=bool)
    if (decisions_a.empty or set(decisions_a.episode_id) != {episode_id}
            or set(decisions_a.aircraft_id) != {"A01"}
            or decisions_a.step.duplicated().any()
            or not np.array_equal(decisions_a.step, np.rint(decisions_a.step))
            or (decisions_a.step < 0).any() or (decisions_a.step >= n).any()
            or not np.allclose(decisions_a.time_s, decisions_a.step * DT, atol=1e-7, rtol=0)):
        raise ValueError("Invalid A01 decision labels or timestamps")
    mapped = decisions_a.skill.map({name: i for i, name in enumerate(classes)})
    if mapped.isna().any():
        raise ValueError("Unknown catalogue label")
    labels = pd.Series(mapped.to_numpy(dtype=np.int64),
                       index=decisions_a.step.to_numpy(dtype=np.int64))
    # Same first origin (6 s) as notebook 09: 51 model frames and 5 lookback frames.
    first = ((RAW_HISTORY_STEPS + STRIDE_STEPS - 1) // STRIDE_STEPS) * STRIDE_STEPS
    ends = np.arange(first, n, STRIDE_STEPS)
    invalid = np.r_[0, np.cumsum(~live)]
    ends = ends[invalid[ends + 1] == invalid[ends - RAW_HISTORY_STEPS + 1]]
    future = ends[:, None] + np.rint(np.asarray(HORIZONS) / DT).astype(int)
    targets = labels.reindex(future.ravel(), fill_value=-1).to_numpy().reshape(-1, 3)
    keep = (targets >= 0).any(axis=1)
    return ends[keep], targets[keep]


def trained_examples(obs_b, tracks_b, decisions_a, classes):
    """Use only B01 observations of the learned A01 pilot from the trained corpus."""
    validate_frames(obs_b, tracks_b.episode_id.iloc[0], len(tracks_b))
    if not obs_b[KEYS].equals(tracks_b[KEYS]):
        raise ValueError("Observation/track keys differ")
    ends, targets = forecast_targets(tracks_b, decisions_a, classes)
    return observation_features(obs_b[OBS_COLUMNS]), ends, targets


def analyze_trained_corpus(root):
    """Count eligible forecast labels before assigning any train/test/validation split."""
    root = Path(root)
    corpus = root / CORPUS
    config, manifest = read_json(corpus / "config.json"), read_json(corpus / "manifest.json")
    if (not manifest["complete"] or config["observer_id"] != "B01"
            or config["target_id"] != "A01" or config["target_policy"] != "trained_pilot"
            or config["sample_interval_s"] != DT):
        raise ValueError("Expected a complete trained-pilot B01/A01 10 Hz corpus")
    for relative in ("config.json", "schema.json", "episodes.parquet"):
        verify_source(corpus, manifest, relative)
    episodes = pd.read_parquet(corpus / "episodes.parquet").sort_values(
        ["pilot_id", "episode_id"]).reset_index(drop=True)
    if episodes.episode_id.duplicated().any() or len(episodes) != manifest["episodes"]:
        raise ValueError("Invalid episode index")
    classes = catalogue(root)
    counts = np.zeros((len(episodes), len(HORIZONS), len(classes)), dtype=np.int64)
    for pilot, subset in episodes.groupby("pilot_id", sort=True):
        paths = [f"tracks/{pilot}.parquet", f"target_decisions/{pilot}.parquet"]
        for relative in paths:
            verify_source(corpus, manifest, relative)
        files = [pq.ParquetFile(corpus / relative) for relative in paths]
        if any(f.num_row_groups != len(subset) for f in files):
            raise ValueError("Expected one Parquet row group per episode")
        for row_group, (index, episode) in enumerate(subset.iterrows()):
            tracks, decisions = [f.read_row_group(row_group).to_pandas() for f in files]
            validate_frames(tracks, episode.episode_id, episode.frames)
            _, target = forecast_targets(tracks, decisions, classes)
            for h in range(len(HORIZONS)):
                valid = target[:, h]
                counts[index, h] = np.bincount(valid[valid >= 0], minlength=len(classes))
    return {"episodes": episodes, "counts": counts, "classes": classes,
            "cases": config["cases"], "source_manifest_sha256": sha256(corpus / "manifest.json")}


def skill_support_table(analysis, groups=None):
    """Include zero-support catalogue skills and count each episode once per label."""
    episodes, counts = analysis["episodes"], analysis["counts"]
    partitions = {"all": np.ones(len(episodes), dtype=bool)} if groups is None else {
        s: episodes.scenario_group.map(lambda g: groups[g]["split"]).eq(s).to_numpy()
        for s in SPLITS}
    rows = []
    for split, mask in partitions.items():
        presence = (counts[mask] > 0).sum(0)
        windows = counts[mask].sum(0)
        group_count = (counts[mask] > 0).reshape(int(mask.sum()), -1)
        group_count = pd.DataFrame(group_count, index=episodes.loc[mask, "scenario_group"])
        group_count = group_count.groupby(level=0).max().sum().to_numpy().reshape(presence.shape)
        for h, horizon in enumerate(HORIZONS):
            for c, skill in enumerate(analysis["classes"]):
                rows.append({"split": split, "horizon_s": horizon, "skill": skill,
                             "episodes": int(presence[h, c]), "windows": int(windows[h, c]),
                             "scenario_groups": int(group_count[h, c])})
    return pd.DataFrame(rows)


def verify_source(root, manifest, relative):
    path = root / relative
    if not path.exists() or sha256(path) != manifest["files"][relative]["sha256"]:
        raise ValueError(f"Missing, pointer-only or changed corpus file: {path}. Run selective LFS pull.")


def prepare_dataset(root, output, split_seed=1729, analysis=None):
    """Stratify whole scenarios; fit scaling on training histories only."""
    root, output = Path(root), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty prepared-data directory or load the existing dataset")
    corpus_b = root / CORPUS
    mb = read_json(corpus_b / "manifest.json")
    analysis = analyze_trained_corpus(root) if analysis is None else analysis
    if analysis["source_manifest_sha256"] != sha256(corpus_b / "manifest.json"):
        raise ValueError("Survey provenance changed; rerun the analysis cell")
    eb, classes = analysis["episodes"].copy(), analysis["classes"]
    identity = ["episode_id", "pilot_id", "scenario_group", "seed", "checkpoint_sha256", "frames"]
    groups = scenario_splits(analysis["cases"], eb, analysis["counts"], split_seed)
    eb["split"] = eb.scenario_group.map(lambda group: groups[group]["split"])
    output.mkdir(parents=True, exist_ok=True)
    features = np.lib.format.open_memmap(output / "features.npy", mode="w+", dtype="float32",
                                         shape=(int(eb.frames.sum()), len(FEATURES)))
    total, squares, count = np.zeros(len(FEATURES)), np.zeros(len(FEATURES)), 0
    anchors, all_targets, metadata, offset = [], [], [], 0
    for pilot, episodes in eb.groupby("pilot_id", sort=True):
        sources = [(corpus_b, mb, f"observations/{pilot}.parquet"),
                   (corpus_b, mb, f"tracks/{pilot}.parquet"),
                   (corpus_b, mb, f"target_decisions/{pilot}.parquet")]
        for p, m, relative in sources:
            verify_source(p, m, relative)
        files = [pq.ParquetFile(p / relative) for p, _, relative in sources]
        if any(f.num_row_groups != len(episodes) for f in files):
            raise ValueError("Expected one Parquet row group per episode")
        for row_group, episode in enumerate(episodes.itertuples()):
            columns = [KEYS + OBS_COLUMNS,
                       KEYS + ["observer_alive", "target_alive"],
                       KEYS + ["aircraft_id", "skill"]]
            frames = [f.read_row_group(row_group, columns=c).to_pandas()
                      for f, c in zip(files, columns, strict=True)]
            if (frames[0].episode_id.iloc[0] != episode.episode_id
                    or len(frames[0]) != episode.frames):
                raise ValueError("Row-group ordering differs from episode index")
            frame, ends, target = trained_examples(*frames, classes)
            features[offset:offset + len(frame)] = frame
            if episode.split == "train" and len(ends):
                used = np.zeros(len(frame), dtype=bool)
                for end in ends:
                    used[end - HISTORY_STEPS + 1:end + 1] = True
                values = frame[used].astype(np.float64)
                total += values.sum(0)
                squares += np.square(values).sum(0)
                count += len(values)
            anchors.append(ends + offset)
            all_targets.append(target)
            metadata.extend({"episode_id": episode.episode_id, "pilot_id": pilot,
                             "scenario_group": episode.scenario_group, "split": episode.split,
                             "step": int(end), "time_s": float(end * DT)} for end in ends)
            offset += len(frame)
        print(f"Prepared {pilot}: {offset:,} trained-pilot frames", flush=True)
    features.flush()
    del features
    if not count:
        raise ValueError("No training frames")
    mean = total / count
    std = np.maximum(np.sqrt(np.maximum(squares / count - mean**2, 0)), 0.01)
    samples, targets = pd.DataFrame(metadata), np.concatenate(all_targets)
    for split in SPLITS:
        if not ((targets[samples.split.eq(split)] >= 0).sum(0) > 0).all():
            raise ValueError(f"No valid labels for a horizon in {split}")
    np.save(output / "anchors.npy", np.concatenate(anchors))
    np.save(output / "targets.npy", targets)
    samples.to_parquet(output / "samples.parquet", index=False)
    eb[identity + ["split"]].to_csv(output / "split_assignments.csv", index=False)
    write_json(output / "scenario_splits.json", groups)
    write_json(output / "scaling.json", {"mean": mean.tolist(), "std": std.tolist(),
                                         "fitted_on": "train only", "frames": count})
    skill_support_table(analysis).to_csv(output / "skill_support_before_split.csv", index=False)
    skill_support_table(analysis, groups).to_csv(output / "skill_support_by_split.csv", index=False)
    manifest = {"data_contract": DATA_CONTRACT,
                "split_strategy": "scenario-group skill-horizon episode-and-window stratification",
                "classes": classes, "features": FEATURES, "history_steps": HISTORY_STEPS,
                "raw_history_steps": RAW_HISTORY_STEPS, "stride_steps": STRIDE_STEPS,
                "horizons_s": HORIZONS, "split_seed": split_seed,
                "split_fractions": {"train": 0.4, "test": 0.3, "validation": 0.3},
                "source_manifests": {p.relative_to(root).as_posix(): sha256(p / "manifest.json")
                                     for p in (corpus_b,)},
                "source_code_sha256": sha256(__file__),
                "catalogue_sha256": sha256(root / "bvr_sim_source/bvr_sim/agents/skill_manager.py"),
                "counts": {s: {"groups": int(eb.loc[eb.split.eq(s), "scenario_group"].nunique()),
                               "episodes": int(eb.split.eq(s).sum()),
                               "windows": int(samples.split.eq(s).sum()),
                               "labels_by_horizon": (targets[samples.split.eq(s)] >= 0).sum(0).tolist()}
                           for s in SPLITS}}
    manifest["files"] = {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}
    write_json(output / "manifest.json", manifest)
    return manifest


class PairedData:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = read_json(self.directory / "manifest.json")
        for name, digest in self.manifest["files"].items():
            if sha256(self.directory / name) != digest:
                raise ValueError(f"Prepared data changed: {name}")
        self.classes = self.manifest["classes"]
        self.features = np.load(self.directory / "features.npy", mmap_mode="r")
        self.anchors = np.load(self.directory / "anchors.npy")
        self.targets = np.load(self.directory / "targets.npy")
        self.samples = pd.read_parquet(self.directory / "samples.parquet")
        scaling = read_json(self.directory / "scaling.json")
        self.mean = np.asarray(scaling["mean"], dtype=np.float32)
        self.std = np.asarray(scaling["std"], dtype=np.float32)
        if (self.mean.shape != (self.features.shape[1],) or self.std.shape != self.mean.shape
                or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all()
                or np.any(self.std <= 0)):
            raise ValueError("Prepared scaling must have finite means and positive finite standard deviations")
        self.indices = {s: np.flatnonzero(self.samples.split.eq(s)) for s in SPLITS}
        for a in SPLITS:
            for b in SPLITS:
                if a != b and set(self.samples.loc[self.samples.split.eq(a), "scenario_group"]) & set(
                        self.samples.loc[self.samples.split.eq(b), "scenario_group"]):
                    raise ValueError("Scenario leakage across partitions")

    def batch(self, ids, device):
        steps = self.anchors[ids, None] + np.arange(1 - HISTORY_STEPS, 1)
        windows = (self.features[steps] - self.mean) / self.std
        return (torch.from_numpy(windows).to(device),
                torch.from_numpy(self.targets[ids]).long().to(device))


def resolve_precision(device, requested):
    """Keep weights/optimizer in FP32; use CUDA BF16 only on supported hardware."""
    if requested not in {"bf16", "fp32"}:
        raise ValueError("precision must be bf16 or fp32")
    if requested == "bf16" and torch.device(device).type == "cuda":
        if torch.cuda.is_bf16_supported():
            return "bf16"
    return "fp32"


def autocast_context(device, precision):
    return (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if precision == "bf16" and torch.device(device).type == "cuda" else nullcontext())


@contextmanager
def evaluation_context(device, attention="auto"):
    """Select math attention explicitly, including disabling the fused MHA path.

    CUDA training with dropout and evaluation without dropout can select different
    kernels. A separate FP32/math evaluation policy avoids that mixed-precision
    inference path without changing training precision or silently retrying batches.
    """
    if attention not in {"auto", "math"}:
        raise ValueError("evaluation_attention must be auto or math")
    if attention == "auto":
        yield
        return
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            context = sdpa_kernel(SDPBackend.MATH)
        except ImportError:  # PyTorch 2.1 compatibility.
            context = torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=True, enable_mem_efficient=False)
        with context:
            yield
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


class NonfiniteEvaluationError(ValueError):
    """A numerical failure with enough example identity to investigate it."""

    def __init__(self, details):
        self.details = details
        super().__init__(
            f"Nonfinite evaluation {details['stage']} in split={details['split']}, "
            f"batch={details['batch_index']}, sample_indices={details['sample_indices']}. "
            "No examples were dropped or replaced. See nonfinite_evaluation.json in the run directory.")


def _check_evaluation_tensor(tensor, stage, split, batch_index, sample_ids, data, precision, attention):
    finite = torch.isfinite(tensor)
    if bool(finite.all()):
        return
    bad = (~finite).nonzero(as_tuple=False)
    failing_rows = bad[:, 0].unique().cpu().tolist()
    ids = np.asarray(sample_ids, dtype=np.int64)
    selected = [int(ids[i]) for i in failing_rows[:16]]
    values = tensor[finite]
    details = {"stage": stage, "split": split, "batch_index": batch_index,
               "sample_indices": selected, "affected_samples": len(failing_rows),
               "nonfinite_count": int((~finite).sum().item()),
               "shape": list(tensor.shape), "dtype": str(tensor.dtype),
               "device": str(tensor.device), "evaluation_precision": precision,
               "evaluation_attention": attention,
               "finite_min": float(values.min().item()) if values.numel() else None,
               "finite_max": float(values.max().item()) if values.numel() else None}
    if stage == "inputs":
        columns = bad[:, -1].unique().cpu().tolist()
        names = getattr(data, "manifest", {}).get("features", [])
        details["feature_indices"] = columns
        details["feature_names"] = [names[i] for i in columns if i < len(names)]
    if hasattr(data, "samples"):
        columns = [c for c in ("episode_id", "observer_id", "scenario_group", "step", "time_s")
                   if c in data.samples.columns]
        details["samples"] = json.loads(data.samples.iloc[selected][columns].to_json(orient="records"))
    raise NonfiniteEvaluationError(details)


def _check_evaluation_tensors(tensors, split, batch_index, sample_ids, data, precision, attention,
                              diagnose=None):
    """Transfer stage flags together; retain detailed stage/sample diagnostics on failure."""
    flags = torch.stack([torch.isfinite(tensor).all() for _, tensor in tensors]).cpu().tolist()
    for (stage, tensor), finite in zip(tensors, flags):
        if not finite:
            if diagnose is not None:
                diagnose(tensor, stage)
            _check_evaluation_tensor(tensor, stage, split, batch_index, sample_ids, data, precision, attention)


class TensorFrameCache:
    """Normalize unique frames once, in chunks, without changing the source array.

    CUDA is used by the training pipeline. CPU construction is also supported for
    numeric checks of exactly the same normalization and indexing implementation.
    """
    def __init__(self, data, device, chunk_frames=65536):
        self.device = torch.device(device)
        self.features = torch.empty(data.features.shape, dtype=torch.float32, device=self.device)
        mean = torch.as_tensor(data.mean, device=self.device)
        std = torch.as_tensor(data.std, device=self.device)
        for start in range(0, len(data.features), chunk_frames):
            stop = min(start + chunk_frames, len(data.features))
            # Only a small writable CPU chunk is staged; never copy the whole map on CPU.
            chunk = torch.from_numpy(np.array(data.features[start:stop], dtype=np.float32, copy=True))
            self.features[start:stop].copy_(chunk)
            self.features[start:stop].sub_(mean).div_(std)
        self.anchors = torch.as_tensor(data.anchors, device=self.device)
        self.targets = torch.as_tensor(data.targets, dtype=torch.long, device=self.device)
        self.offsets = torch.arange(1 - HISTORY_STEPS, 1, device=self.device)

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in
                   (self.features, self.anchors, self.targets, self.offsets))

    def batches(self, ids, batch_size):
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        for batch_ids in ids.split(batch_size):
            steps = self.anchors[batch_ids, None] + self.offsets
            yield self.features[steps], self.targets[batch_ids]


class WindowDataset(Dataset):
    """Spawn-safe vectorized CPU batches; each worker opens its own memory maps."""
    def __init__(self, directory, mean, std, count):
        self.directory = str(directory)
        self.mean, self.std, self.count = mean, std, count
        self._features = self._anchors = self._targets = None

    def __len__(self):
        return self.count

    def __getstate__(self):
        state = self.__dict__.copy()
        # Never pickle a populated memory map into a Windows/spawn worker.
        state.update(_features=None, _anchors=None, _targets=None)
        return state

    def _open(self):
        if self._features is None:
            root = Path(self.directory)
            self._features = np.load(root / "features.npy", mmap_mode="r")
            self._anchors = np.load(root / "anchors.npy", mmap_mode="r")
            self._targets = np.load(root / "targets.npy", mmap_mode="r")

    def __getitems__(self, ids):
        self._open()
        ids = np.asarray(ids, dtype=np.int64)
        steps = self._anchors[ids, None] + np.arange(1 - HISTORY_STEPS, 1)
        windows = (self._features[steps] - self.mean) / self.std
        return torch.from_numpy(windows), torch.from_numpy(np.array(self._targets[ids], copy=True)).long()

    def __getitem__(self, index):
        x, y = self.__getitems__([index])
        return x[0], y[0]


class OrderedBatches(Sampler):
    """One reusable loader/pool handles shuffled training and ordered evaluation."""
    def set_order(self, ids, batch_size):
        self.ids, self.batch_size = np.asarray(ids), batch_size

    def __iter__(self):
        for start in range(0, len(self.ids), self.batch_size):
            yield self.ids[start:start + self.batch_size].tolist()

    def __len__(self):
        return (len(self.ids) + self.batch_size - 1) // self.batch_size


def identity_batch(batch):
    return batch


def initialize_loader_worker(worker_id):
    torch.set_num_threads(1)


def prefetch_to_cuda(batches, device):
    """Double-buffer pinned batches on a copy stream, overlapping GPU execution."""
    stream = torch.cuda.Stream(device=device)
    source = iter(batches)

    def preload():
        host_batch = next(source, None)
        if host_batch is None:
            return None
        with torch.cuda.stream(stream):
            return tuple(t.to(device, non_blocking=True) for t in host_batch)

    pending = preload()
    while pending is not None:
        current = torch.cuda.current_stream(device)
        current.wait_stream(stream)
        batch = pending
        for tensor in batch:
            tensor.record_stream(current)
        pending = preload()
        yield batch


class BatchPipeline:
    def __init__(self, data, device, config):
        self.data, self.device = data, torch.device(device)
        self.requested_backend = config["data_backend"]
        if self.requested_backend not in {"gpu_cache", "prefetch", "direct"}:
            raise ValueError("data_backend must be gpu_cache, prefetch or direct")
        self.backend = self.requested_backend
        self.cache = self.loader = None
        if self.backend == "gpu_cache" and self.device.type != "cuda":
            self.backend = "direct"
        if self.backend == "gpu_cache":
            try:
                self.cache = TensorFrameCache(data, self.device)
            except torch.cuda.OutOfMemoryError as error:
                raise RuntimeError("GPU frame cache did not fit. Select data_backend='prefetch' "
                                   "or 'direct', or free GPU memory.") from error
        elif self.backend == "prefetch":
            self.sampler = OrderedBatches()
            self.sampler.set_order([], 1)
            dataset = WindowDataset(getattr(data, "frame_directory", data.directory),
                                    data.mean, data.std, len(data.targets))
            workers = config["num_workers"]
            options = {"num_workers": workers, "pin_memory": self.device.type == "cuda",
                       "persistent_workers": workers > 0,
                       "timeout": config["loader_timeout"] if workers else 0,
                       "collate_fn": identity_batch, "worker_init_fn": initialize_loader_worker,
                       "generator": torch.Generator().manual_seed(config["seed"])}
            if workers:
                options.update(prefetch_factor=config["prefetch_factor"],
                               multiprocessing_context="spawn")
            self.loader = DataLoader(dataset, batch_sampler=self.sampler, **options)

    @property
    def cache_bytes(self):
        return self.cache.nbytes if self.cache is not None else 0

    def batches(self, ids, batch_size):
        if self.cache is not None:
            yield from self.cache.batches(ids, batch_size)
        elif self.loader is not None:
            self.sampler.set_order(ids, batch_size)
            if self.device.type == "cuda":
                yield from prefetch_to_cuda(self.loader, self.device)
            else:
                yield from self.loader
        else:
            for start in range(0, len(ids), batch_size):
                yield self.data.batch(ids[start:start + batch_size], self.device)

    def close(self):
        if self.loader is not None:
            iterator = getattr(self.loader, "_iterator", None)
            if iterator is not None:
                iterator._shutdown_workers()
        self.loader = self.cache = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class PairedTransformer(nn.Module):
    def __init__(self, num_classes=33, width=192, heads=6, layers=2, dropout=0.1,
                 input_features=len(FEATURES)):
        super().__init__()
        if width % heads:
            raise ValueError("Width must be divisible by attention heads")
        if not isinstance(input_features, int) or isinstance(input_features, bool) or input_features < 1:
            raise ValueError("input_features must be a positive integer")
        self.num_classes = num_classes
        self.input_features = input_features
        self.projection = nn.Linear(input_features, width)
        self.position = nn.Parameter(torch.randn(1, HISTORY_STEPS, width) * 0.02)
        block = nn.TransformerEncoderLayer(width, heads, width * 4, dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        for layer in self.encoder.layers:
            for parameter in layer.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(width, 3 * num_classes))

    def forward(self, windows):
        if windows.ndim != 3 or tuple(windows.shape[1:]) != (HISTORY_STEPS, self.input_features):
            raise ValueError(f"Expected (batch, {HISTORY_STEPS}, {self.input_features}) feature histories")
        encoded = self.encoder(self.projection(windows) + self.position)
        # Bidirectional attention is within the past-only window; no future tokens exist.
        return self.head(self.norm(encoded[:, -1])).reshape(-1, 3, self.num_classes)


def model_configuration(config, input_features, skill_hierarchy=None):
    """Keep historical Transformer checkpoint configuration compatible."""
    name = config.get("architecture", "transformer")
    if name not in ("transformer", "missile_set", "typed_graph", *HGT_ARCHITECTURES):
        raise ValueError(f"Unknown forecasting architecture: {name}")
    temporal_tcn = name == "hgt_hierarchical_intent_tcn"
    keys = ("width", "dropout") if temporal_tcn else ("width", "heads", "layers", "dropout")
    result = {k: config[k] for k in keys}
    if temporal_tcn:
        result.update(tcn_kernel_size=config.get("tcn_kernel_size", 3),
                      tcn_dilations=config.get("tcn_dilations", [1, 2, 4, 8]))
    result["input_features"] = input_features
    if name != "transformer":
        result.update(architecture=name, graph_width=config.get("graph_width", 64),
                      graph_layers=config.get("graph_layers", 2))
        if config.get("observer_acceleration", False):
            result["observer_acceleration"] = True
        if config.get("target_relative_missiles", False):
            result["target_relative_missiles"] = True
        if config.get("target_relative_missile_velocities", False):
            result["target_relative_missile_velocities"] = True
        if name in HGT_ARCHITECTURES:
            result.update(graph_heads=config.get("graph_heads", 4),
                          missile_distance_floor_m=config.get("missile_distance_floor_m", 1000.0),
                          missile_distance_power=config.get("missile_distance_power", 1.0))
            version = config.get("hgt_version", 1 if name == "hgt" else 2)
            if version != 1:
                result.update(hgt_version=version,
                              threat_cpa_horizon_s=config.get("threat_cpa_horizon_s", 30.0),
                              threat_time_scale_s=config.get("threat_time_scale_s", 5.0),
                              threat_prior_strength=config.get("threat_prior_strength", 1.0))
            if name in INTENT_ARCHITECTURES:
                result["num_intents"] = 3
                result["hgt_version"] = version
            if name in HIERARCHICAL_ARCHITECTURES:
                hierarchy = skill_hierarchy if skill_hierarchy is not None else config.get("skill_hierarchy")
                if hierarchy is None:
                    raise ValueError("Hierarchical model requires notebook-26 family/direction mapping")
                result["skill_hierarchy"] = hierarchy
    return result


def make_model(num_classes, model_config, mean=None, std=None):
    """One constructor for training, benchmarks, checkpoint reload and export."""
    settings = dict(model_config)
    name = settings.pop("architecture", "transformer")
    if name == "transformer":
        return PairedTransformer(num_classes=num_classes, **settings)
    from .enemy_missile_models import (
        HGTHierarchicalIntentTCN, HGTHierarchicalIntentTransformer, HGTIntentTransformer, HGTTransformer,
        MissileSetTransformer, TypedGraphTransformer,
    )
    constructors = {"missile_set": MissileSetTransformer, "typed_graph": TypedGraphTransformer,
                    "hgt": HGTTransformer, "hgt_intent": HGTIntentTransformer,
                    "hgt_hierarchical_intent": HGTHierarchicalIntentTransformer,
                    "hgt_hierarchical_intent_tcn": HGTHierarchicalIntentTCN}
    if name not in constructors:
        raise ValueError(f"Unknown forecasting architecture: {name}")
    return constructors[name](num_classes=num_classes, mean=mean, std=std, **settings)


def fit_class_weighting(data, config):
    """Fit per-horizon weights using only retained training targets.

    Cap sqrt(max_count / count) before normalising its training-frequency
    weighted mean to one. Thus the rare/common weight ratio is bounded and
    the expected data-loss scale stays comparable with unweighted CE.
    """
    mode, cap = config["class_weighting"], config["max_class_weight_ratio"]
    if mode not in ("none", "sqrt_inverse", "family_sqrt_inverse"):
        raise ValueError("class_weighting must be none, sqrt_inverse or family_sqrt_inverse")
    if (isinstance(cap, bool) or not isinstance(cap, (int, float))
            or not np.isfinite(cap) or cap < 1):
        raise ValueError("max_class_weight_ratio must be finite and at least one")
    if mode == "none":
        return None
    hierarchy = None
    if mode == "family_sqrt_inverse":
        from .skill_intent_forecasting import hierarchy_policy
        hierarchy = getattr(data, "manifest", {}).get("skill_hierarchy")
        if hierarchy != hierarchy_policy(data.classes):
            raise ValueError("Family weighting requires the frozen notebook-26 hierarchy")
    targets = np.asarray(data.targets[data.indices["train"]])
    counts, weights = [], []
    family_counts, family_weights = [], []
    for h in range(len(HORIZONS)):
        labels = targets[:, h]
        count = np.bincount(labels[labels >= 0], minlength=len(data.classes))
        balancing_counts = (np.bincount(hierarchy["skill_to_family"], weights=count,
                                        minlength=len(hierarchy["families"])).astype(np.int64)
                            if hierarchy is not None else count)
        present = balancing_counts > 0
        if not present.any():
            raise ValueError(f"No retained training labels at +{HORIZONS[h]}s")
        weight = np.zeros(len(balancing_counts), dtype=np.float64)
        weight[present] = np.minimum(cap, np.sqrt(balancing_counts.max() / balancing_counts[present]))
        weight /= np.dot(balancing_counts, weight) / balancing_counts.sum()
        if hierarchy is not None:
            family_counts.append(balancing_counts.tolist())
            family_weights.append(weight.tolist())
            weight = weight[np.asarray(hierarchy["skill_to_family"])]
        counts.append(count.tolist())
        weights.append(weight.tolist())
    state = {"mode": mode, "max_class_weight_ratio": cap, "classes": list(data.classes),
            "horizons_s": list(HORIZONS), "counts": counts, "weights": weights,
            "fitted_on": "retained training labels only; missing=-1 excluded",
            "normalisation": "training-frequency weighted mean one per horizon",
            "zero_support": "zero positive-target weight; output classes retained"}
    if hierarchy is not None:
        state.update(weighting_level="skill_family", families=hierarchy["families"],
                     skill_to_family=hierarchy["skill_to_family"], family_counts=family_counts,
                     family_weights=family_weights,
                     normalisation="training-frequency weighted mean one across families; both sides share family weight",
                     zero_support="unseen families have zero weight; an unseen side retains its seen family's weight")
    return state


def class_weight_table(state):
    """Human-readable audit for the exact weights saved in each run."""
    table = pd.DataFrame([
        {"horizon_s": horizon, "skill": skill, "training_support": state["counts"][h][c],
         "training_frequency": state["counts"][h][c] / sum(state["counts"][h]),
         "weight": state["weights"][h][c]}
        for h, horizon in enumerate(state["horizons_s"]) for c, skill in enumerate(state["classes"])])
    if state.get("weighting_level") == "skill_family":
        table["family"] = [state["families"][f] for _ in HORIZONS for f in state["skill_to_family"]]
    return table


def fit_handedness_weighting(data, config):
    """Balance L/R inside each mirrored family and horizon, using retained TRAIN labels only."""
    mode = config.get("handedness_weighting", "none")
    cap = config.get("max_handedness_weight_ratio", 5.0)
    if mode not in ("none", "sqrt_inverse", "inverse"):
        raise ValueError("handedness_weighting must be none, sqrt_inverse or inverse")
    if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not np.isfinite(cap) or cap < 1:
        raise ValueError("max_handedness_weight_ratio must be finite and at least one")
    if mode == "none":
        return None
    from .skill_intent_forecasting import hierarchy_policy
    hierarchy = getattr(data, "manifest", {}).get("skill_hierarchy")
    if hierarchy != hierarchy_policy(data.classes):
        raise ValueError("Handedness weighting requires the frozen notebook-26 hierarchy")
    targets = np.asarray(data.targets[data.indices["train"]])
    counts = np.zeros((len(HORIZONS), len(hierarchy["paired_families"]), 2), dtype=np.int64)
    for h in range(len(HORIZONS)):
        labels = targets[:, h]
        leaf_counts = np.bincount(labels[labels >= 0], minlength=len(data.classes))
        for c, (p, d) in enumerate(zip(hierarchy["skill_to_pair"], hierarchy["skill_to_direction"])):
            if p >= 0:
                counts[h, p, d] = leaf_counts[c]
    weights = np.zeros_like(counts, dtype=np.float64)
    power = .5 if mode == "sqrt_inverse" else 1.0
    for h in range(len(HORIZONS)):
        for p in range(len(hierarchy["paired_families"])):
            support = counts[h, p]
            seen = support > 0
            if seen.any():
                raw = np.zeros(2, dtype=np.float64)
                raw[seen] = np.minimum(cap, (support.max() / support[seen]) ** power)
                weights[h, p] = raw / (np.dot(support, raw) / support.sum())
    return {"mode": mode, "max_handedness_weight_ratio": float(cap),
            "horizons_s": list(HORIZONS), "paired_families": hierarchy["paired_families"],
            "directions": hierarchy["directions"], "counts": counts.tolist(), "weights": weights.tolist(),
            "fitted_on": "retained training labels only; missing and singleton labels excluded",
            "normalisation": "training-frequency weighted mean one inside each pair/horizon",
            "application": "multiply only handedness loss/penalty, after family-frequency weighting",
            "zero_support": "unseen sides have zero target weight; single seen side has weight one; empty pairs have zero weights"}


def handedness_weight_table(state):
    columns = ["horizon_s", "family", "direction", "training_support", "conditional_frequency", "weight"]
    rows = []
    for h, horizon in enumerate(HORIZONS):
        for p, family in enumerate(state["paired_families"]):
            total = sum(state["counts"][h][p])
            for d, direction in enumerate(state["directions"]):
                count = state["counts"][h][p][d]
                rows.append(dict(zip(columns, (horizon, family, direction, count,
                                               count / total if total else 0., state["weights"][h][p][d]))))
    return pd.DataFrame(rows, columns=columns)


def handedness_weight_tensor(state, device):
    return (None if state is None else torch.tensor(state["weights"], dtype=torch.float32, device=device).reshape(
        len(HORIZONS), len(state["paired_families"]), 2))


def masked_loss(logits, targets, class_weights=None):
    # Same equal-horizon objective, without three Python/GPU mask synchronizations.
    losses = nn.functional.cross_entropy(logits.float().transpose(1, 2), targets,
                                         ignore_index=-1, reduction="none")
    if class_weights is not None:
        if tuple(class_weights.shape) != tuple(logits.shape[1:]):
            raise ValueError("Expected per-horizon class weights with shape (3, classes)")
        losses = losses * class_weights.gather(1, targets.clamp_min(0).T).T
    counts = (targets >= 0).sum(0)
    active = counts > 0
    per_horizon = losses.sum(0) / counts.clamp_min(1)
    return (per_horizon * active).sum() / active.sum().clamp_min(1)


def masked_intent_loss(skill_logits, intent_logits, targets, mapping, intent_loss_weight, class_weights=None):
    """Equal-horizon masked skill CE plus an unweighted auxiliary intent CE."""
    intent_targets = torch.where(targets >= 0, mapping[targets.clamp_min(0)], -1)
    skill_loss = masked_loss(skill_logits, targets, class_weights)
    intent_loss = masked_loss(intent_logits, intent_targets)
    return skill_loss + intent_loss_weight * intent_loss


def hierarchical_target_losses(family_logits, direction_logits, targets, mapping):
    """Per-example family and direction CE, using the TRUE family; missing/nonpaired direction=-1."""
    valid = targets >= 0
    family_y = torch.where(valid, mapping["family"][targets.clamp_min(0)], -1)
    pair = mapping["pair"][targets.clamp_min(0)]
    direction_y = torch.where(valid & (pair >= 0), mapping["direction"][targets.clamp_min(0)], -1)
    family_ce = nn.functional.cross_entropy(family_logits.float().transpose(1, 2), family_y,
                                            ignore_index=-1, reduction="none")
    if direction_logits.shape[2]:
        selected = direction_logits.gather(2, pair.clamp_min(0)[:, :, None, None].expand(-1, -1, 1, 2)).squeeze(2)
        direction_ce = nn.functional.cross_entropy(selected.float().transpose(1, 2), direction_y,
                                                   ignore_index=-1, reduction="none")
    else:
        selected = family_logits.new_zeros((*targets.shape, 2))
        direction_ce = torch.zeros_like(family_ce)
    return family_ce, direction_ce, family_y, direction_y, selected


def hierarchical_loss_settings(config):
    mode = config.get("hierarchical_loss", "joint_nll")
    if mode not in ("joint_nll", "partial_credit"):
        raise ValueError("hierarchical_loss must be joint_nll or partial_credit")
    result = {"hierarchical_loss": mode}
    for name, default in (("family_reward", 1.0), ("handedness_reward", 1.0)):
        value = config.get(name, default)
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or value < 0 or (name == "family_reward" and value == 0)):
            raise ValueError(f"{name} must be finite and {'positive' if name == 'family_reward' else 'nonnegative'}")
        result[name] = float(value)
    return result


def hierarchical_example_loss(family_ce, direction_ce, paired, hierarchical_loss="joint_nll",
                              family_reward=1.0, handedness_reward=1.0, handedness_weight=None):
    """Partial credit C=P(family)*(a+b*P(side|family))/(a+b), or joint NLL."""
    if hierarchical_loss == "joint_nll":
        return family_ce + (direction_ce * handedness_weight if handedness_weight is not None else direction_ce)
    if handedness_reward == 0:
        return family_ce
    log_family, log_direction = math.log(family_reward), math.log(handedness_reward)
    normalizer = float(np.logaddexp(log_family, log_direction))
    # Stable even when the true-side probability underflows or is effectively zero.
    log_credit_given_family = torch.logaddexp(direction_ce.new_tensor(log_family), log_direction - direction_ce)
    side_penalty = (normalizer - log_credit_given_family).clamp_min(0)
    if handedness_weight is not None:
        side_penalty = side_penalty * handedness_weight
    return family_ce + torch.where(paired, side_penalty, 0)


def target_handedness_weights(targets, mapping, handedness_weights):
    if handedness_weights is None:
        return None
    if (handedness_weights.ndim != 3 or handedness_weights.shape[0] != len(HORIZONS)
            or handedness_weights.shape[-1] != 2):
        raise ValueError("Expected handedness weights in (horizons, paired families, 2) order")
    if handedness_weights.shape[1] == 0:
        return torch.ones_like(targets, dtype=torch.float32)
    pair = mapping["pair"][targets.clamp_min(0)]
    side = mapping["direction"][targets.clamp_min(0)]
    selected = handedness_weights[torch.arange(len(HORIZONS), device=targets.device)[None],
                                  pair.clamp_min(0), side.clamp_min(0)]
    return torch.where((targets >= 0) & (pair >= 0), selected, 1.)


def masked_hierarchical_skill_loss(family_logits, direction_logits, targets, mapping, class_weights=None,
                                    hierarchical_loss="joint_nll", family_reward=1.0, handedness_reward=1.0,
                                    handedness_weights=None):
    settings = hierarchical_loss_settings({"hierarchical_loss": hierarchical_loss,
                                            "family_reward": family_reward, "handedness_reward": handedness_reward})
    family_ce, direction_ce, _, direction_y, _ = hierarchical_target_losses(family_logits, direction_logits, targets, mapping)
    hand_weight = target_handedness_weights(targets, mapping, handedness_weights)
    combined = hierarchical_example_loss(family_ce, direction_ce, direction_y >= 0,
                                         handedness_weight=hand_weight, **settings)
    if class_weights is not None:
        combined = combined * class_weights.gather(1, targets.clamp_min(0).T).T
    counts = (targets >= 0).sum(0)
    represented = counts > 0
    # Normalize both terms over ALL valid skill labels, not just the directional subset.
    return (combined.sum(0) / counts.clamp_min(1) * represented).sum() / represented.sum().clamp_min(1)


def hierarchy_mapping_tensors(policy, device):
    return {name: torch.tensor(policy[f"skill_to_{name}"], dtype=torch.long, device=device)
            for name in ("family", "pair", "direction")}


def metric_tables(targets, probabilities, classes, split, losses=None, allow_empty=False):
    """Per-true-class NLL and F1. Unsupported classes have NaN NLL and zero F1."""
    summary, per_class, matrices = [], [], []
    c = len(classes)
    for h, horizon in enumerate(HORIZONS):
        valid = targets[:, h] >= 0
        y, p = targets[valid, h], probabilities[valid, h]
        if not len(y):
            if allow_empty:
                summary.append({"split": split, "horizon_s": horizon, "count": 0,
                                "nll": np.nan, "accuracy": np.nan, "macro_f1": 0.,
                                "supported_macro_f1": np.nan, "weighted_f1": np.nan})
                per_class.extend({"split": split, "horizon_s": horizon, "skill": name,
                                  "support": 0, "class_nll": np.nan, "precision": 0., "recall": 0., "f1": 0.}
                                 for name in classes)
                matrices.append(np.zeros((c, c), dtype=np.int64))
                continue
            raise ValueError(f"No labels for {split}, +{horizon}s")
        pred = p.argmax(-1)
        nll = (-np.log(np.maximum(p[np.arange(len(y)), y], 1e-30)) if losses is None
               else losses[valid, h])
        cm = np.bincount(y * c + pred, minlength=c*c).reshape(c, c)
        support, predicted, tp = cm.sum(1), cm.sum(0), cm.diagonal()
        precision = np.divide(tp, predicted, out=np.zeros(c, float), where=predicted > 0)
        recall = np.divide(tp, support, out=np.zeros(c, float), where=support > 0)
        f1 = np.divide(2*tp, support+predicted, out=np.zeros(c, float), where=(support+predicted) > 0)
        sums = np.bincount(y, weights=nll, minlength=c)
        cnll = np.divide(sums, support, out=np.full(c, np.nan), where=support > 0)
        summary.append({"split": split, "horizon_s": horizon, "count": len(y),
                        "nll": float(nll.mean(dtype=np.float64)), "accuracy": float(tp.sum()/len(y)),
                        "macro_f1": float(f1.mean()),
                        "supported_macro_f1": float(f1[support > 0].mean()),
                        "weighted_f1": float(np.dot(f1, support)/len(y))})
        per_class.extend({"split": split, "horizon_s": horizon, "skill": name,
                          "support": int(support[k]), "class_nll": float(cnll[k]),
                          "precision": precision[k], "recall": recall[k], "f1": f1[k]}
                         for k, name in enumerate(classes))
        matrices.append(cm)
    return pd.DataFrame(summary), pd.DataFrame(per_class), np.stack(matrices)


@torch.inference_mode()
def _evaluation_batch(model, x, y, data, sample_ids, device, precision, split, batch_index,
                       attention, hierarchy_mapping=None, frame_encodings=False):
    """Common tensor computations for full-prediction and streaming evaluators."""
    input_stage = "frame_encodings" if frame_encodings else "inputs"
    _check_evaluation_tensors([(input_stage, x)], split, batch_index, sample_ids, data, precision, attention)
    checks = []
    forward = model.forward_from_frame_encodings if frame_encodings else model
    multitask = getattr(model, "num_intents", 0) == 3
    with autocast_context(device, precision):
        if hierarchy_mapping is not None:
            output = forward(x, return_hierarchy=True)
            logits, intent_logits = output["skill_log_probs"], output["intent_logits"]
        elif multitask:
            logits, intent_logits = forward(x, return_intent=True)
        else:
            logits = forward(x)
    logits = logits.float()
    ce = nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten(),
                                     ignore_index=-1, reduction="none").reshape(-1, len(HORIZONS))
    probability = logits.softmax(-1)
    checks.extend((("logits", logits), ("loss", ce), ("probabilities", probability)))
    tasks = {"skill": {"probabilities": probability, "losses": ce, "targets": y}}
    if hierarchy_mapping is not None:
        family, direction = output["family_logits"], output["direction_logits"]
        family_ce, direction_ce, family_y, direction_y, selected = hierarchical_target_losses(
            family, direction, y, hierarchy_mapping)
        checks.extend((("family_logits", family), ("direction_logits", direction),
                       ("family_loss", family_ce), ("direction_loss", direction_ce)))
        tasks["family"] = {"probabilities": family.float().softmax(-1), "losses": family_ce, "targets": family_y}
        tasks["direction"] = {"probabilities": selected.float().softmax(-1),
                               "losses": direction_ce, "targets": direction_y}
    if multitask:
        intent_logits = intent_logits.float()
        intent_y = torch.as_tensor(data.intent_targets[sample_ids], dtype=torch.long, device=device)
        intent_ce = nn.functional.cross_entropy(intent_logits.flatten(0, 1), intent_y.flatten(),
                                                ignore_index=-1, reduction="none").reshape(-1, len(HORIZONS))
        intent_p = intent_logits.softmax(-1)
        checks.extend((("intent_logits", intent_logits), ("intent_loss", intent_ce), ("intent_probabilities", intent_p)))
        tasks["intent"] = {"probabilities": intent_p, "losses": intent_ce, "targets": intent_y}
    _check_evaluation_tensors(checks, split, batch_index, sample_ids, data, precision, attention)
    return tasks


@torch.inference_mode()
def evaluate(model, data, split, device, batch_size=1024, precision="fp32", pipeline=None,
             class_weights=None, evaluation_precision=None, evaluation_attention="auto", intent_loss_weight=0.0,
             hierarchical_loss="joint_nll", family_reward=1.0, handedness_reward=1.0, handedness_weights=None,
             streaming=False, frame_cache=False, frame_batch_size=8192, cache_max_frames=65536):
    from .forecasting_evaluation import evaluation_batches, evaluate_streaming

    if evaluation_precision is not None:
        precision = resolve_precision(device, evaluation_precision)
    model.eval()
    if streaming:
        return evaluate_streaming(model, data, split, device, batch_size, precision, pipeline,
            class_weights, evaluation_attention, intent_loss_weight, hierarchical_loss, family_reward,
            handedness_reward, handedness_weights, frame_cache, frame_batch_size, cache_max_frames)
    ids = data.indices[split]
    probabilities, losses = [], []
    multitask = getattr(model, "num_intents", 0) == 3
    intent_probabilities, intent_losses = [], []
    hierarchical = hasattr(model, "num_families")
    family_probabilities, family_losses, direction_probabilities, direction_losses = [], [], [], []
    if hierarchical:
        hierarchy_mapping = {"family": model.skill_family_index,
                             "pair": torch.where(model.directional_skills, model.skill_pair_index, -1),
                             "direction": model.skill_direction_index}
    batches = evaluation_batches(model, data, split, device, batch_size, precision, pipeline,
                                  evaluation_attention, frame_cache, frame_batch_size, cache_max_frames)
    with evaluation_context(device, evaluation_attention):
        for batch_index, (sample_ids, x, y, encoded) in enumerate(batches):
            tasks = _evaluation_batch(model, x, y, data, sample_ids, device, precision, split,
                                       batch_index, evaluation_attention, hierarchy_mapping if hierarchical else None, encoded)
            losses.append(tasks["skill"]["losses"].cpu().numpy())
            probabilities.append(tasks["skill"]["probabilities"].cpu().numpy())
            if hierarchical:
                family_probabilities.append(tasks["family"]["probabilities"].cpu().numpy())
                direction_probabilities.append(tasks["direction"]["probabilities"].cpu().numpy())
                family_losses.append(tasks["family"]["losses"].cpu().numpy())
                direction_losses.append(tasks["direction"]["losses"].cpu().numpy())
            if multitask:
                intent_losses.append(tasks["intent"]["losses"].cpu().numpy())
                intent_probabilities.append(tasks["intent"]["probabilities"].cpu().numpy())
    p, loss = np.concatenate(probabilities), np.concatenate(losses)
    targets = data.targets[ids]
    summary, scores, cm = metric_tables(targets, p, data.classes, split, loss)
    # Summary NLL and all selection/reporting metrics remain unweighted. Only
    # this separate objective diagnostic uses the fixed training class weights.
    data_loss = float(summary.nll.mean())
    if class_weights is not None:
        weighted = loss.astype(np.float64) * np.asarray(class_weights)[np.arange(len(HORIZONS)), np.maximum(targets, 0)]
        data_loss = float((weighted.sum(0) / (targets >= 0).sum(0)).mean())
    result = {"summary": summary, "per_class": scores, "confusion": cm,
              "probabilities": p, "targets": targets, "sample_indices": ids, "data_loss": data_loss}
    if multitask:
        intent_p, intent_ce = np.concatenate(intent_probabilities), np.concatenate(intent_losses)
        intent_y = data.intent_targets[ids]
        intent_summary, intent_scores, intent_cm = metric_tables(intent_y, intent_p, data.intent_classes, split, intent_ce)
        result["intent"] = {"summary": intent_summary, "per_class": intent_scores, "confusion": intent_cm,
                            "probabilities": intent_p, "targets": intent_y, "sample_indices": ids, "losses": intent_ce}
        result["skill_data_loss"] = data_loss
        result["data_loss"] += intent_loss_weight * float(intent_summary.nll.mean())
    if hierarchical:
        for name, prob, ce, labels, classes in (
            ("family", family_probabilities, family_losses, data.family_targets, data.family_classes),
            ("direction", direction_probabilities, direction_losses, data.direction_targets, data.direction_classes),
        ):
            prob, ce, labels = np.concatenate(prob), np.concatenate(ce), labels[ids]
            table, per_class, matrix = metric_tables(labels, prob, classes, split, ce, allow_empty=name == "direction")
            result[name] = {"summary": table, "per_class": per_class, "confusion": matrix,
                            "probabilities": prob, "targets": labels, "sample_indices": ids, "losses": ce}
        settings = hierarchical_loss_settings({"hierarchical_loss": hierarchical_loss,
                                                "family_reward": family_reward, "handedness_reward": handedness_reward})
        family_ce, direction_ce = np.concatenate(family_losses), np.concatenate(direction_losses)
        paired = data.direction_targets[ids] >= 0
        hand_weight = target_handedness_weights(torch.from_numpy(targets),
            {key: value.detach().cpu() for key, value in hierarchy_mapping.items()},
            handedness_weights.detach().cpu() if handedness_weights is not None else None)
        objective = hierarchical_example_loss(torch.from_numpy(family_ce), torch.from_numpy(direction_ce),
                                               torch.from_numpy(paired), handedness_weight=hand_weight, **settings).numpy()
        weighted = objective.astype(np.float64)
        weighted_family = family_ce.astype(np.float64)
        if class_weights is not None:
            outer_weights = np.asarray(class_weights)[np.arange(len(HORIZONS)), np.maximum(targets, 0)]
            weighted *= outer_weights
            weighted_family *= outer_weights
        skill_data_loss = float((weighted.sum(0) / (targets >= 0).sum(0)).mean())
        previous = result.get("skill_data_loss", result["data_loss"])
        result["joint_skill_data_loss"] = previous
        result["skill_data_loss"] = skill_data_loss
        result["family_data_loss"] = float((weighted_family.sum(0) / (targets >= 0).sum(0)).mean())
        result["handedness_data_loss"] = skill_data_loss - result["family_data_loss"]
        result["data_loss"] += skill_data_loss - previous
        # Rewards are normalized to [0,1]. Handedness adds credit only when the
        # predicted family is correct; singleton families have no handedness bonus.
        log_a = math.log(family_reward)
        log_b = math.log(handedness_reward) if handedness_reward else -float("inf")
        log_total = float(np.logaddexp(log_a, log_b))
        a, b = math.exp(log_a - log_total), math.exp(log_b - log_total)
        family_correct = np.concatenate(family_probabilities).argmax(-1) == data.family_targets[ids]
        side_correct = np.concatenate(direction_probabilities).argmax(-1) == data.direction_targets[ids]
        credit = np.where(paired, family_correct * (a + b * side_correct), family_correct)
        expected = np.where(paired, a * np.exp(-family_ce) + b * np.exp(-family_ce - direction_ce),
                            np.exp(-family_ce))
        credit_loss = hierarchical_example_loss(torch.from_numpy(family_ce), torch.from_numpy(direction_ce),
            torch.from_numpy(paired), hierarchical_loss="partial_credit", family_reward=family_reward,
            handedness_reward=handedness_reward).numpy()
        reward_rows = []
        for h, horizon in enumerate(HORIZONS):
            valid = targets[:, h] >= 0
            reward_rows.append({"split": split, "horizon_s": horizon, "count": int(valid.sum()),
                                "family_accuracy": float(family_correct[valid, h].mean()),
                                "hierarchical_credit": float(credit[valid, h].mean()),
                                "expected_hierarchical_credit": float(expected[valid, h].mean()),
                                "hierarchical_credit_loss": float(credit_loss[valid, h].mean())})
        result["hierarchical_reward"] = pd.DataFrame(reward_rows)
    return result


def atomic_checkpoint(path, payload):
    temporary = Path(path).with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def execution_model(model, config):
    """Compile only the network, while saving portable unwrapped checkpoints."""
    if config["compile_mode"] not in {"default", "reduce-overhead", "max-autotune"}:
        raise ValueError("Unknown torch.compile mode")
    return (torch.compile(model, mode=config["compile_mode"], dynamic=False)
            if config["compile_model"] else model)


def l1_penalty(model):
    """Sum absolute trainable matrix/tensor weights, including position embeddings.

    Biases and LayerNorm scales are excluded. Accumulate in FP32 under BF16 too.
    """
    return sum(p.float().abs().sum() for p in model.parameters()
               if p.requires_grad and p.ndim > 1)


def fit_batch(model, optimizer, x, y, device, precision, gradient_clip, l1_lambda=0.0,
              class_weights=None, intent_mapping=None, intent_loss_weight=0.0, hierarchy_mapping=None,
              hierarchical_loss="joint_nll", family_reward=1.0, handedness_reward=1.0, handedness_weights=None):
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(device, precision):
        if hierarchy_mapping is not None:
            hierarchy = model(x, return_hierarchy=True)
            logits, intent_logits = hierarchy["skill_log_probs"], hierarchy["intent_logits"]
        elif intent_mapping is not None:
            logits, intent_logits = model(x, return_intent=True)
        else:
            logits = model(x)
    if hierarchy_mapping is not None:
        loss = masked_hierarchical_skill_loss(hierarchy["family_logits"], hierarchy["direction_logits"],
                                              y, hierarchy_mapping, class_weights, hierarchical_loss=hierarchical_loss,
                                              family_reward=family_reward, handedness_reward=handedness_reward,
                                              handedness_weights=handedness_weights)
        intent_targets = torch.where(y >= 0, intent_mapping[y.clamp_min(0)], -1)
        loss = loss + intent_loss_weight * masked_loss(intent_logits, intent_targets)
    else:
        loss = (masked_intent_loss(logits, intent_logits, y, intent_mapping, intent_loss_weight, class_weights)
                if intent_mapping is not None else masked_loss(logits, y, class_weights))
    if l1_lambda:
        loss = loss + l1_lambda * l1_penalty(model)
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite training loss")
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
    optimizer.step()
    return loss.detach()


def learning_rate_schedule_settings(config):
    """Validate the optional update-based schedule without changing historical defaults."""
    name = config.get("lr_schedule", "constant")
    if name not in ("constant", "warmup_cosine"):
        raise ValueError("lr_schedule must be constant or warmup_cosine")
    settings = {"name": name}
    for key, default, lower, upper in (("warmup_fraction", .05, 0, 1),
                                        ("warmup_start_ratio", .1, 0, 1),
                                        ("min_lr_ratio", .05, 0, 1)):
        value = config.get(key, default)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not lower <= value <= upper
                or (key == "warmup_fraction" and value == 1)):
            raise ValueError(f"Invalid {key} for learning-rate schedule")
        settings[key] = float(value)
    if (isinstance(config["learning_rate"], bool) or not isinstance(config["learning_rate"], (int, float))
            or not math.isfinite(config["learning_rate"]) or config["learning_rate"] <= 0):
        raise ValueError("learning_rate must be finite and positive")
    return settings


def warmup_cosine_multiplier(update, total_updates, warmup_updates, start_ratio, min_ratio):
    """Multiplier for the next optimizer update; endpoints refer to actual used rates."""
    update = min(update, total_updates - 1)
    if total_updates == 1:
        return 1.0
    if update < warmup_updates:
        return (start_ratio + (1 - start_ratio) * update / (warmup_updates - 1)
                if warmup_updates > 1 else 1.0)
    decay_updates = total_updates - warmup_updates
    progress = ((update - warmup_updates) / (decay_updates - 1)
                if decay_updates > 1 else 1.0)
    return min_ratio + .5 * (1 - min_ratio) * (1 + math.cos(math.pi * progress))


def train_model(data, output, config=None, callback=None, device=None, resume=False, tracking=None):
    """Fit with optional NLL patience and sustained-overtraining shutdown.

    Both stopping rules use unregularised evaluation NLL and independent minimum
    epoch settings. Every strict improvement in test NLL receives a checkpoint,
    including during learning-rate warm-up and before either stopping minimum.
    """
    from .paired_skill_tracking import RunTracker

    config = {**DEFAULT_TRAINING, **(config or {})}
    if config.get("architecture") == "hgt_hierarchical_intent_tcn":
        # Transformer-only defaults must not reappear in the saved TCN recipe.
        config.pop("heads", None)
        config.pop("layers", None)
    multitask = config.get("architecture") in INTENT_ARCHITECTURES
    hierarchical = config.get("architecture") in HIERARCHICAL_ARCHITECTURES
    hierarchy_settings = hierarchical_loss_settings(config) if hierarchical else {}
    if multitask:
        config.setdefault("intent_loss_weight", .25)
    intent_loss_weight = config.get("intent_loss_weight", 0.0)
    if (isinstance(intent_loss_weight, bool) or not isinstance(intent_loss_weight, (int, float))
            or not math.isfinite(intent_loss_weight) or intent_loss_weight < 0
            or (not multitask and intent_loss_weight != 0)):
        raise ValueError("intent_loss_weight must be finite/nonnegative and requires an intent architecture")
    if config.get("handedness_weighting", "none") != "none" and not hierarchical:
        raise ValueError("Handedness weighting requires a hierarchical intent architecture")
    schedule_settings = learning_rate_schedule_settings(config)
    if config["evaluation_precision"] not in {"fp32", "bf16"}:
        raise ValueError("evaluation_precision must be fp32 or bf16")
    if config["evaluation_attention"] not in {"auto", "math"}:
        raise ValueError("evaluation_attention must be auto or math")
    if config["epochs"] < 1:
        raise ValueError("Epoch count must be positive")
    for key in ("batch_size", "evaluation_batch_size", "prefetch_factor", "cpu_threads", "loader_timeout",
                 "evaluation_frame_batch_size", "evaluation_cache_max_frames"):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if (not isinstance(config["epoch_streaming_metrics"], bool)
            or not isinstance(config["evaluation_frame_cache"], bool)
            or config["evaluation_cache_max_frames"] < HISTORY_STEPS):
        raise ValueError("Evaluation switches must be boolean and the frame cache cap must be at least 51")
    if config["evaluation_frame_cache"] and config.get("architecture", "transformer") == "transformer":
        raise ValueError("Frame caching requires a frame-local missile architecture")
    if (not isinstance(config["num_workers"], int) or config["num_workers"] < 0
            or not isinstance(config["compile_model"], bool)):
        raise ValueError("num_workers must be nonnegative and compile_model must be boolean")
    minimum, patience = config["early_stopping_min_epochs"], config["early_stopping_patience"]
    overtraining_minimum = config.get("overtraining_min_epochs", 0)
    if (isinstance(overtraining_minimum, bool) or not isinstance(overtraining_minimum, int)
            or overtraining_minimum < 0):
        raise ValueError("overtraining_min_epochs must be a nonnegative integer")
    if (not isinstance(config["early_stopping"], bool)
            or not isinstance(minimum, int) or minimum < 0
            or not isinstance(patience, int) or patience < 1):
        raise ValueError("Early stopping requires a boolean switch, nonnegative minimum and positive patience")
    for key in ("l1_lambda", "overtraining_relative_increase", "overtraining_min_gap"):
        if (isinstance(config[key], bool) or not isinstance(config[key], (int, float))
                or not np.isfinite(config[key]) or config[key] < 0):
            raise ValueError(f"{key} must be finite and nonnegative")
    if (not isinstance(config["overtraining_shutdown"], bool)
            or isinstance(config["overtraining_patience"], bool)
            or not isinstance(config["overtraining_patience"], int)
            or config["overtraining_patience"] < 1):
        raise ValueError("Overtraining shutdown requires a boolean switch and positive integer patience")
    handedness_weighting = fit_handedness_weighting(data, config)
    class_weighting = fit_class_weighting(data, config)
    output = Path(output)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    weights = (None if class_weighting is None else
               torch.tensor(class_weighting["weights"], dtype=torch.float32, device=device))
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
    cpu_threads = min(config["cpu_threads"], os.cpu_count() or 1)
    torch.set_num_threads(cpu_threads)
    precision = resolve_precision(device, config["precision"])
    evaluation_precision = resolve_precision(device, config["evaluation_precision"])
    data_manifest = getattr(data, "manifest", {})
    feature_names = list(data_manifest.get("features", FEATURES))
    architecture = model_configuration(config, len(feature_names), data_manifest.get("skill_hierarchy"))
    architecture_name = config.get("architecture", "transformer")
    if architecture_name != "transformer":
        from .nn_duel_skill_forecasting import (
            ENEMY_MISSILE_DATA_CONTRACT, OBSERVER_ACCELERATION_DATA_CONTRACT,
            TARGET_RELATIVE_MISSILE_DATA_CONTRACT,
            feature_names as enemy_names,
        )
        observer_acceleration = config.get("observer_acceleration", False)
        target_relative_missiles = config.get("target_relative_missiles", False)
        target_relative_missile_velocities = config.get("target_relative_missile_velocities", False)
        expected_contract = (TARGET_RELATIVE_MISSILE_DATA_CONTRACT if target_relative_missiles else
                             OBSERVER_ACCELERATION_DATA_CONTRACT if observer_acceleration
                             else ENEMY_MISSILE_DATA_CONTRACT)
        if multitask:
            from .skill_intent_forecasting import INTENT_DATA_CONTRACT, hierarchy_policy, intent_policy
            expected_contract = INTENT_DATA_CONTRACT
            if (data_manifest.get("intent_supervision") != intent_policy(data.classes)
                    or not hasattr(data, "intent_targets") or not target_relative_missiles):
                raise ValueError("Intent training requires the verified notebook-26 intent dataset and taxonomy")
            if hierarchical and (data_manifest.get("skill_hierarchy") != hierarchy_policy(data.classes)
                                 or not hasattr(data, "family_targets") or not hasattr(data, "direction_targets")):
                raise ValueError("Hierarchical training requires notebook-26 v2 family/direction supervision")
        if (not isinstance(observer_acceleration, bool)
                or not isinstance(target_relative_missiles, bool)
                or not isinstance(target_relative_missile_velocities, bool)
                or target_relative_missile_velocities != target_relative_missiles
                or (target_relative_missiles and not observer_acceleration)
                or data_manifest.get("data_contract") != expected_contract
                or feature_names != enemy_names(data.classes, enemy_missiles=True,
                                                observer_acceleration=observer_acceleration)):
            raise ValueError("Missile models require the matching notebook 17 or 22 enemy-missile "
                             "or notebook 24 target-relative dataset contract and matching "
                             "observer_acceleration/target_relative_missiles/"
                             "target_relative_missile_velocities settings")
    base_model = make_model(len(data.classes), architecture, data.mean, data.std).to(device)
    intent_options = ({"intent_mapping": torch.tensor(
        data_manifest["intent_supervision"]["skill_to_intent"], dtype=torch.long, device=device),
                       "intent_loss_weight": intent_loss_weight} if multitask else {})
    if hierarchical:
        intent_options["hierarchy_mapping"] = hierarchy_mapping_tensors(data_manifest["skill_hierarchy"], device)
        intent_options.update(hierarchy_settings)
        intent_options["handedness_weights"] = handedness_weight_tensor(handedness_weighting, device)
    model = execution_model(base_model, config)
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    steps_per_epoch = math.ceil(len(data.indices["train"]) / config["batch_size"])
    if steps_per_epoch < 1:
        raise ValueError("No training batches")
    total_updates = config["epochs"] * steps_per_epoch
    warmup_updates = (min(total_updates - 1, max(1, round(total_updates * schedule_settings["warmup_fraction"])))
                      if schedule_settings["warmup_fraction"] > 0 else 0)
    scheduler = None
    if schedule_settings["name"] == "warmup_cosine":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda update: warmup_cosine_multiplier(
            update, total_updates, warmup_updates, schedule_settings["warmup_start_ratio"],
            schedule_settings["min_lr_ratio"]))
    schedule_metadata = {**schedule_settings, "steps_per_epoch": steps_per_epoch,
                         "total_updates": total_updates, "warmup_updates": warmup_updates}
    optimizer_steps = 0
    rng = np.random.default_rng(config["seed"])
    manifest_hash = sha256(data.directory / "manifest.json")
    run = {"training": config, "model": architecture, "classes": data.classes,
           "architecture": architecture_name,
           "parameter_count": sum(p.numel() for p in base_model.parameters()),
           "horizons_s": list(HORIZONS), "history_steps": HISTORY_STEPS,
           "architecture_source_sha256": (sha256(Path(__file__).with_name("enemy_missile_models.py"))
                                          if architecture_name != "transformer" else None),
           "data_contract": data_manifest.get("data_contract", DATA_CONTRACT),
           "feature_names": feature_names,
           "input_contract": data_manifest.get("input_contract"),
           "target_contract": data_manifest.get("target_contract"),
           "data_manifest_sha256": manifest_hash, "source_sha256": sha256(__file__),
           "tracking_source_sha256": sha256(Path(__file__).with_name("paired_skill_tracking.py")),
           "evaluation_source_sha256": sha256(Path(__file__).with_name("forecasting_evaluation.py")),
           "torch_version": str(torch.__version__), "numpy_version": np.__version__,
           "device": str(device), "selection": "Strict improvement of mean test NLL across horizons",
           "validation_policy": "Evaluated only after checkpoint selection",
           "effective_precision": precision, "effective_cpu_threads": cpu_threads,
           "effective_evaluation_precision": evaluation_precision,
           "evaluation_attention": config["evaluation_attention"],
           "class_weighting": class_weighting, "handedness_weighting": handedness_weighting,
           "learning_rate_schedule": schedule_metadata}
    if multitask:
        run["intent_supervision"] = data_manifest["intent_supervision"]
        run["intent_source_sha256"] = sha256(Path(__file__).with_name("skill_intent_forecasting.py"))
        run["training_objective"] = "weighted skill CE + intent_loss_weight * unweighted intent CE + L1"
    if hierarchical:
        run["skill_hierarchy"] = data_manifest["skill_hierarchy"]
        run["hierarchical_objective"] = hierarchy_settings
        run["training_objective"] = (
            "leaf-weighted negative log normalized (family reward + correct-handedness bonus) + intent CE + L1"
            if hierarchy_settings["hierarchical_loss"] == "partial_credit" else
            "leaf-weighted (family CE + paired-direction CE) + intent_loss_weight * intent CE + L1")
    history, best, best_epoch, first = [], float("inf"), 0, 1
    epochs_without_improvement, stopped_early = 0, False
    overtraining_epochs, train_nll_at_best, stop_reason = 0, None, None
    if resume:
        saved_run = read_json(output / "run.json")
        if any(saved_run[k] != run[k] for k in ("training", "model", "classes",
                                               "data_manifest_sha256", "source_sha256", "tracking_source_sha256",
                                               "class_weighting")) or (
                saved_run.get("architecture_source_sha256") != run["architecture_source_sha256"]
                or saved_run.get("intent_source_sha256") != run.get("intent_source_sha256")
                or saved_run.get("evaluation_source_sha256") != run["evaluation_source_sha256"]
                or saved_run.get("handedness_weighting") != handedness_weighting):
            raise ValueError("Resume requires the same configuration, data and source")
        saved = torch.load(output / "last.pt", map_location=device, weights_only=False)
        if saved.get("evaluation_source_sha256") != run["evaluation_source_sha256"]:
            raise ValueError("Resume requires the original evaluation source")
        if saved.get("handedness_weighting") != handedness_weighting:
            raise ValueError("Resume requires the original training-fitted handedness weights")
        base_model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        optimizer_steps = saved.get("optimizer_steps", saved["epoch"] * steps_per_epoch)
        if scheduler is not None:
            if (saved.get("scheduler_state") is None
                    or saved.get("learning_rate_schedule") != schedule_metadata
                    or optimizer_steps != saved["epoch"] * steps_per_epoch
                    or saved["scheduler_state"].get("last_epoch") != optimizer_steps):
                raise ValueError("Resume requires a matching scheduler and optimizer update count")
            scheduler.load_state_dict(saved["scheduler_state"])
        rng.bit_generator.state = saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device.type == "cuda" and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
        history, best, best_epoch = saved["history"], saved["best_loss"], saved["best_epoch"]
        epochs_without_improvement = saved.get("epochs_without_improvement", 0)
        stopped_early = saved.get("early_stopped", False)
        overtraining_epochs = saved.get("overtraining_epochs", 0)
        train_nll_at_best = saved.get("train_nll_at_best")
        stop_reason = saved.get("stop_reason")
        first = config["epochs"] + 1 if stopped_early else saved["epoch"] + 1
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Run output exists; resume or choose a fresh run directory")
        (output / "checkpoints").mkdir(parents=True)
        (output / "epochs").mkdir()
        write_json(output / "run.json", run)
        if class_weighting is not None:
            write_json(output / "class_weights.json", class_weighting)
            class_weight_table(class_weighting).to_csv(output / "class_weights.csv", index=False)
        if handedness_weighting is not None:
            write_json(output / "handedness_weights.json", handedness_weighting)
            handedness_weight_table(handedness_weighting).to_csv(output / "handedness_weights.csv", index=False)
        shutil.copy2(__file__, output / "training_source.py")
        shutil.copy2(Path(__file__).with_name("forecasting_evaluation.py"), output / "evaluation_source.py")
        if architecture_name != "transformer":
            shutil.copy2(Path(__file__).with_name("enemy_missile_models.py"),
                         output / "architecture_source.py")
        if multitask:
            shutil.copy2(Path(__file__).with_name("skill_intent_forecasting.py"), output / "intent_source.py")
            write_json(output / "intent_policy.json", data_manifest["intent_supervision"])
        if hierarchical:
            write_json(output / "skill_hierarchy.json", data_manifest["skill_hierarchy"])
    with RunTracker(output, tracking) as tracker:
        tracker.begin(run, data)
        tracker.backfill()
        with BatchPipeline(data, device, config) as pipeline:
            run.update(effective_data_backend=pipeline.backend, frame_cache_bytes=pipeline.cache_bytes)
            write_json(output / "run.json", run)
            tracker.runtime(run)
            print(f"Runtime: {device}, {precision}, {pipeline.backend}, "
                  f"batch {config['batch_size']}, evaluation batch {config['evaluation_batch_size']}, "
                  f"frame cache {pipeline.cache_bytes / 2**30:.3f} GiB", flush=True)
            print(f"Evaluation: {evaluation_precision}, {config['evaluation_attention']} attention", flush=True)
            print("Learning-rate schedule:", schedule_metadata, flush=True)
            for epoch in range(first, config["epochs"] + 1):
                started = time.perf_counter()
                model.train()
                learning_rate_start = optimizer.param_groups[0]["lr"]
                order = rng.permutation(data.indices["train"])
                for x, y in pipeline.batches(order, config["batch_size"]):
                    learning_rate_used = optimizer.param_groups[0]["lr"]
                    fit_batch(model, optimizer, x, y, device, precision, config["gradient_clip"],
                              config["l1_lambda"], weights, **intent_options)
                    optimizer_steps += 1
                    if scheduler is not None:
                        scheduler.step()
                try:
                    results = [evaluate(model, data, s, device, config["evaluation_batch_size"],
                                        precision=evaluation_precision, pipeline=pipeline,
                                        evaluation_attention=config["evaluation_attention"],
                                        streaming=config["epoch_streaming_metrics"],
                                        frame_cache=config["evaluation_frame_cache"],
                                        frame_batch_size=config["evaluation_frame_batch_size"],
                                        cache_max_frames=config["evaluation_cache_max_frames"],
                                        **({"intent_loss_weight": intent_loss_weight} if multitask else {}),
                                        **hierarchy_settings,
                                        **({"handedness_weights": intent_options["handedness_weights"]} if hierarchical else {}),
                                        class_weights=class_weighting["weights"]
                                        if class_weighting is not None and s == "train" else None)
                               for s in ("train", "test")]
                except NonfiniteEvaluationError as error:
                    error.details.update(epoch=epoch, training_precision=precision,
                        nonfinite_parameters=[name for name, parameter in base_model.named_parameters()
                                              if not bool(torch.isfinite(parameter).all())])
                    write_json(output / "nonfinite_evaluation.json", error.details)
                    raise
                summary = pd.concat([r["summary"] for r in results], ignore_index=True)
                scores = pd.concat([r["per_class"] for r in results], ignore_index=True)
                train_nll, test_nll = [float(r["summary"].nll.mean()) for r in results]
                train_data_loss = float(results[0].get("data_loss", train_nll))
                if not np.isfinite([train_nll, test_nll]).all():
                    raise ValueError("Nonfinite evaluation loss")
                improved = test_nll < best
                if improved:
                    best, best_epoch = test_nll, epoch
                    train_nll_at_best = train_nll
                if config["early_stopping"] and epoch > minimum:
                    epochs_without_improvement = 0 if improved else epochs_without_improvement + 1
                else:
                    epochs_without_improvement = 0
                generalization_gap = test_nll - train_nll
                overtraining_detected = bool(
                    config["overtraining_shutdown"] and epoch > overtraining_minimum and not improved
                    and train_nll_at_best is not None and train_nll < train_nll_at_best
                    and test_nll > best * (1 + config["overtraining_relative_increase"])
                    and generalization_gap > config["overtraining_min_gap"])
                overtraining_epochs = overtraining_epochs + 1 if overtraining_detected else 0
                stop_reason = (
                    "overtraining" if overtraining_epochs >= config["overtraining_patience"] else
                    "patience" if config["early_stopping"] and epochs_without_improvement >= patience
                    else None)
                stopped_early = stop_reason is not None
                with torch.no_grad():
                    penalty = (float(config["l1_lambda"] * l1_penalty(base_model))
                               if config["l1_lambda"] else 0.0)
                row = {"epoch": epoch, "train_nll": train_nll, "test_nll": test_nll,
                       "learning_rate_start": learning_rate_start, "learning_rate": learning_rate_used,
                       "next_learning_rate": optimizer.param_groups[0]["lr"], "optimizer_steps": optimizer_steps,
                       "train_data_loss": train_data_loss,
                       "train_l1_penalty": penalty, "train_objective": train_data_loss + penalty,
                       "generalization_gap": generalization_gap,
                       "overtraining_detected": overtraining_detected,
                       "overtraining_epochs": overtraining_epochs, "stop_reason": stop_reason,
                       "train_macro_f1": float(results[0]["summary"].macro_f1.mean()),
                       "test_macro_f1": float(results[1]["summary"].macro_f1.mean()),
                       "improved": improved, "epochs_without_improvement": epochs_without_improvement,
                       "early_stopped": stopped_early, "seconds": time.perf_counter() - started}
                if multitask:
                    row.update(train_skill_data_loss=results[0]["skill_data_loss"],
                               train_intent_nll=float(results[0]["intent"]["summary"].nll.mean()),
                               test_intent_nll=float(results[1]["intent"]["summary"].nll.mean()),
                               train_intent_accuracy=float(results[0]["intent"]["summary"].accuracy.mean()),
                               test_intent_accuracy=float(results[1]["intent"]["summary"].accuracy.mean()))
                if hierarchical:
                    row.update(train_family_data_loss=results[0]["family_data_loss"],
                               train_handedness_data_loss=results[0]["handedness_data_loss"],
                               train_family_nll=float(results[0]["family"]["summary"].nll.mean()),
                               test_family_nll=float(results[1]["family"]["summary"].nll.mean()),
                               train_direction_nll=float(results[0]["direction"]["summary"].nll.mean()),
                               test_direction_nll=float(results[1]["direction"]["summary"].nll.mean()),
                               train_hierarchical_credit=float(results[0]["hierarchical_reward"].hierarchical_credit.mean()),
                               test_hierarchical_credit=float(results[1]["hierarchical_reward"].hierarchical_credit.mean()),
                               train_expected_hierarchical_credit=float(results[0]["hierarchical_reward"].expected_hierarchical_credit.mean()),
                               test_expected_hierarchical_credit=float(results[1]["hierarchical_reward"].expected_hierarchical_credit.mean()))
                history.append(row)
                scores.to_csv(output / "epochs" / f"epoch_{epoch:03d}_per_class.csv", index=False)
                summary.to_csv(output / "epochs" / f"epoch_{epoch:03d}_summary.csv", index=False)
                pd.DataFrame(history).to_csv(output / "history.csv", index=False)
                payload = {"epoch": epoch, "model_state": base_model.state_dict(),
                           "optimizer_state": optimizer.state_dict(), "model_config": architecture,
                           "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                           "optimizer_steps": optimizer_steps, "learning_rate_schedule": schedule_metadata,
                           "classes": data.classes, "mean": data.mean, "std": data.std,
                           "history_steps": HISTORY_STEPS, "horizons_s": HORIZONS,
                           "feature_names": feature_names,
                           "data_contract": run["data_contract"],
                           "class_weighting": class_weighting,
                           "evaluation_source_sha256": run["evaluation_source_sha256"],
                           "handedness_weighting": handedness_weighting,
                           "input_contract": run["input_contract"],
                           "target_contract": run["target_contract"],
                           "best_loss": best, "best_epoch": best_epoch,
                           "test_nll": test_nll, "history": history, "numpy_rng": rng.bit_generator.state,
                           "epochs_without_improvement": epochs_without_improvement,
                           "early_stopped": stopped_early,
                           "overtraining_epochs": overtraining_epochs,
                           "train_nll_at_best": train_nll_at_best, "stop_reason": stop_reason,
                           "effective_precision": precision,
                           "effective_evaluation_precision": evaluation_precision,
                           "evaluation_attention": config["evaluation_attention"],
                           "torch_rng": torch.get_rng_state(), "data_manifest_sha256": manifest_hash,
                           "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}
                if multitask:
                    payload["intent_supervision"] = data_manifest["intent_supervision"]
                if hierarchical:
                    payload["skill_hierarchy"] = data_manifest["skill_hierarchy"]
                    payload["hierarchical_objective"] = hierarchy_settings
                if improved:
                    path = output / "checkpoints" / f"epoch_{epoch:03d}.pt"
                    atomic_checkpoint(path, payload)
                    best_temporary = output / "best.tmp"
                    shutil.copy2(path, best_temporary)
                    best_temporary.replace(output / "best.pt")
                # Advance the resume point only after its selected checkpoint is durable.
                atomic_checkpoint(output / "last.pt", payload)
                tracker.epoch(row, summary, scores)
                if callback:
                    callback(row, summary, scores)
                else:
                    print(row, flush=True)
                if stopped_early:
                    print(f"Early stopping after epoch {epoch}: {stop_reason}; "
                          f"stale epochs={epochs_without_improvement}, "
                          f"consecutive overtraining epochs={overtraining_epochs}. "
                          f"Best checkpoint: epoch {best_epoch}.", flush=True)
                    break
        write_json(output / "training_status.json", {
            "status": "early_stopped" if stopped_early else "completed",
            "completed_epochs": len(history), "max_epochs": config["epochs"],
            "early_stopping_enabled": config["early_stopping"],
            "early_stopping_min_epochs": minimum, "early_stopping_patience": patience,
            "epochs_without_improvement": epochs_without_improvement,
            "stop_reason": stop_reason, "overtraining_epochs": overtraining_epochs,
            "overtraining_min_epochs": overtraining_minimum,
            "best_epoch": best_epoch, "best_test_nll": best,
        })
        tracker.finish_training()
    return pd.DataFrame(history)


def final_evaluation(data, output, device=None, tracking=None):
    """Evaluate the test-selected best checkpoint on all three partitions."""
    from .paired_skill_tracking import RunTracker

    output = Path(output)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    saved = torch.load(output / "best.pt", map_location=device, weights_only=False)
    if saved["data_manifest_sha256"] != sha256(data.directory / "manifest.json"):
        raise ValueError("Checkpoint belongs to different prepared data")
    model = make_model(len(data.classes), saved["model_config"], saved["mean"], saved["std"]).to(device)
    model.load_state_dict(saved["model_state"])
    run = read_json(output / "run.json")
    config = {**DEFAULT_TRAINING, **run["training"]}
    if saved.get("handedness_weighting") != run.get("handedness_weighting"):
        raise ValueError("Checkpoint handedness weights differ from run metadata")
    if saved.get("evaluation_source_sha256") != run.get("evaluation_source_sha256"):
        raise ValueError("Checkpoint evaluation source differs from run metadata")
    precision = resolve_precision(device, saved.get("effective_precision",
                                                    run.get("effective_precision", "fp32")))
    # Old checkpoints retain their original evaluation policy when explicitly
    # reloaded; new runs record evaluation independently of training autocast.
    evaluation_precision = resolve_precision(device, saved.get("effective_evaluation_precision",
        run["training"].get("evaluation_precision", precision)))
    evaluation_attention = saved.get("evaluation_attention",
                                    run["training"].get("evaluation_attention", "auto"))
    model = execution_model(model, config)
    with RunTracker(output, tracking) as tracker:
        tracker.begin(run, data)
        tracker.backfill()
        if tracker.enabled and tracker.previous is None:
            tracker.finish_training()
        with BatchPipeline(data, device, config) as pipeline:
            try:
                results = {s: evaluate(model, data, s, device, config["evaluation_batch_size"],
                                       precision=evaluation_precision, pipeline=pipeline,
                                       evaluation_attention=evaluation_attention,
                                       frame_cache=config["evaluation_frame_cache"],
                                       frame_batch_size=config["evaluation_frame_batch_size"],
                                       cache_max_frames=config["evaluation_cache_max_frames"],
                                       **({"intent_loss_weight": config["intent_loss_weight"]}
                                          if config.get("architecture") in INTENT_ARCHITECTURES else {}),
                                       **(hierarchical_loss_settings(config) if config.get("architecture") in HIERARCHICAL_ARCHITECTURES
                                          else {}),
                                       **({"handedness_weights": handedness_weight_tensor(saved.get("handedness_weighting"), device)}
                                          if config.get("architecture") in HIERARCHICAL_ARCHITECTURES else {})) for s in SPLITS}
            except NonfiniteEvaluationError as error:
                error.details.update(epoch=saved["epoch"], phase="final_evaluation")
                write_json(output / "nonfinite_evaluation.json", error.details)
                raise
        summary = pd.concat([r["summary"] for r in results.values()], ignore_index=True)
        scores = pd.concat([r["per_class"] for r in results.values()], ignore_index=True)
        summary.to_csv(output / "final_metrics.csv", index=False)
        scores.to_csv(output / "final_per_class.csv", index=False)
        for split, result in results.items():
            np.savez_compressed(output / f"final_{split}.npz", **{
                k: result[k] for k in ("probabilities", "targets", "sample_indices", "confusion")})
        if config.get("architecture") in INTENT_ARCHITECTURES:
            intent_summary = pd.concat([r["intent"]["summary"] for r in results.values()], ignore_index=True)
            intent_scores = pd.concat([r["intent"]["per_class"] for r in results.values()], ignore_index=True)
            intent_summary.to_csv(output / "final_intent_metrics.csv", index=False)
            intent_scores.to_csv(output / "final_intent_per_class.csv", index=False)
            for split, result in results.items():
                np.savez_compressed(output / f"final_intent_{split}.npz", **{
                    k: result["intent"][k] for k in ("probabilities", "targets", "sample_indices", "confusion", "losses")})
            tracker.metrics({f"final/intent/{row.split}/h{int(row.horizon_s)}/{key}": float(getattr(row, key))
                             for row in intent_summary.itertuples() for key in ("nll", "accuracy", "macro_f1")}, saved["epoch"])
        if config.get("architecture") in HIERARCHICAL_ARCHITECTURES:
            credit_metrics = pd.concat([r["hierarchical_reward"] for r in results.values()], ignore_index=True)
            credit_metrics.to_csv(output / "final_hierarchical_credit_metrics.csv", index=False)
            tracker.metrics({f"final/{row.split}/h{int(row.horizon_s)}/{key}": float(getattr(row, key))
                             for row in credit_metrics.itertuples()
                             for key in ("hierarchical_credit", "expected_hierarchical_credit")}, saved["epoch"])
            for name in ("family", "direction"):
                tables = pd.concat([r[name]["summary"] for r in results.values()], ignore_index=True)
                per_class = pd.concat([r[name]["per_class"] for r in results.values()], ignore_index=True)
                if name == "direction":
                    tables["conditioning"] = "true mirrored family; singletons/missing masked"
                    per_class["conditioning"] = "true mirrored family; singletons/missing masked"
                tables.to_csv(output / f"final_{name}_metrics.csv", index=False)
                per_class.to_csv(output / f"final_{name}_per_class.csv", index=False)
                for split, result in results.items():
                    np.savez_compressed(output / f"final_{name}_{split}.npz", **{
                        k: result[name][k] for k in ("probabilities", "targets", "sample_indices", "confusion", "losses")})
                tracker.metrics({f"final/{name}/{row.split}/h{int(row.horizon_s)}/{key}": float(getattr(row, key))
                                 for row in tables.itertuples() for key in ("nll", "accuracy", "macro_f1")}, saved["epoch"])
        write_json(output / "selection.json", {"epoch": saved["epoch"], "test_nll": saved["test_nll"],
                                               "validation_used_for_selection": False})
        tracker.final(saved["epoch"], summary, scores)
    return saved["epoch"], summary, scores, results


def benchmark_batches(data, config=None, device=None, batch_sizes=(512, 1024),
                      warmup=2, repeats=5, compare_compilation=False):
    """Measure disposable training-only models; preserve RNGs and never select a checkpoint.

    The table includes gathering, transfers, forward/backward and AdamW. Cold-start
    time is separate so compilation/cache preparation cannot masquerade as steady-state gain.
    Validation and test labels are never accessed, and trained run files are not written.
    """
    config = {**DEFAULT_TRAINING, **(config or {})}
    if warmup < 1 or repeats < 1 or not batch_sizes or any(b < 1 for b in batch_sizes):
        raise ValueError("Use positive batch sizes and at least one warmup/repetition")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    intent_options = ({"intent_mapping": torch.tensor(data.manifest["intent_supervision"]["skill_to_intent"],
                                                      dtype=torch.long, device=device),
                       "intent_loss_weight": config.get("intent_loss_weight", .25)}
                      if config.get("architecture") in INTENT_ARCHITECTURES else {})
    if config.get("architecture") in HIERARCHICAL_ARCHITECTURES:
        intent_options["hierarchy_mapping"] = hierarchy_mapping_tensors(data.manifest["skill_hierarchy"], device)
        intent_options.update(hierarchical_loss_settings(config))
        intent_options["handedness_weights"] = handedness_weight_tensor(fit_handedness_weighting(data, config), device)
    class_weighting = fit_class_weighting(data, config)
    weights = (None if class_weighting is None else
               torch.tensor(class_weighting["weights"], dtype=torch.float32, device=device))
    precision = resolve_precision(device, config["precision"])
    fork_devices = [device.index if device.index is not None else torch.cuda.current_device()] \
        if device.type == "cuda" else []
    rows = []
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(min(config["cpu_threads"], os.cpu_count() or 1))
    try:
        with torch.random.fork_rng(devices=fork_devices), BatchPipeline(data, device, config) as pipeline:
            rng = np.random.default_rng(config["seed"])
            compilation_options = (False, True) if compare_compilation else (config["compile_model"],)
            for compiled in compilation_options:
                for batch_size in batch_sizes:
                    torch.random.default_generator.manual_seed(config["seed"])
                    if device.type == "cuda":
                        with torch.cuda.device(device):
                            torch.cuda.manual_seed(config["seed"])
                    base = make_model(len(data.classes), model_configuration(config,
                        len(getattr(data, "manifest", {}).get("features", FEATURES)),
                        getattr(data, "manifest", {}).get("skill_hierarchy")),
                        data.mean, data.std).to(device)
                    model = execution_model(base, {**config, "compile_model": compiled})
                    optimizer = torch.optim.AdamW(base.parameters(), lr=config["learning_rate"],
                                                  weight_decay=config["weight_decay"])
                    ids = rng.choice(data.indices["train"], size=batch_size,
                                     replace=batch_size > len(data.indices["train"]))
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                        torch.cuda.reset_peak_memory_stats(device)
                    cold_started = time.perf_counter()
                    try:
                        durations = []
                        model.train()
                        for repeat in range(warmup + repeats):
                            if device.type == "cuda":
                                torch.cuda.synchronize(device)
                            started = time.perf_counter()
                            x, y = next(iter(pipeline.batches(ids, batch_size)))
                            fit_batch(model, optimizer, x, y, device, precision, config["gradient_clip"],
                                      config["l1_lambda"], weights, **intent_options)
                            if device.type == "cuda":
                                torch.cuda.synchronize(device)
                            elapsed = time.perf_counter() - started
                            del x, y
                            if repeat == warmup - 1:
                                cold_seconds = time.perf_counter() - cold_started
                            if repeat >= warmup:
                                durations.append(elapsed)
                        seconds = float(np.median(durations))
                        rows.append({"batch_size": batch_size, "compiled": compiled, "status": "ok",
                                     "precision": precision, "backend": pipeline.backend,
                                     "frame_cache_GiB": pipeline.cache_bytes / 2**30,
                                     "warmup_seconds": cold_seconds, "median_step_seconds": seconds,
                                     "windows_per_second": batch_size / seconds,
                                     "peak_allocated_MiB": torch.cuda.max_memory_allocated(device) / 2**20
                                     if device.type == "cuda" else None,
                                     "peak_reserved_MiB": torch.cuda.max_memory_reserved(device) / 2**20
                                     if device.type == "cuda" else None})
                    except torch.cuda.OutOfMemoryError:
                        rows.append({"batch_size": batch_size, "compiled": compiled,
                                     "status": "out_of_memory"})
                    finally:
                        del optimizer, model, base
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
    finally:
        torch.set_num_threads(previous_threads)
    return pd.DataFrame(rows)
