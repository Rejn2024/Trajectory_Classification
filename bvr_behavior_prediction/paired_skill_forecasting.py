"""Paired A01/B01 skill forecasting for notebook 09b.

Keeps the paired experiment's data and training contract self-contained.
Inputs are synchronized observation histories; labels are A01's future selections.
"""

import hashlib
import importlib.util
import json
import os
import random
import shutil
import subprocess
import time
from contextlib import nullcontext
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
BASE_FEATURES = OBS_COLUMNS + [f"observed_target_acceleration_{a}" for a in "nwu"]
FEATURES = [f"{aircraft}_{f}" for aircraft in ("A01", "B01") for f in BASE_FEATURES]
DEFAULT_TRAINING = {
    "epochs": 100, "seed": 42, "batch_size": 512, "evaluation_batch_size": 1024,
    "learning_rate": 0.0003, "weight_decay": 0.01, "gradient_clip": 1.0,
    "width": 192, "heads": 6, "layers": 2, "dropout": 0.1,
    "early_stopping": False, "early_stopping_min_epochs": 50, "early_stopping_patience": 10,
    "precision": "bf16", "data_backend": "gpu_cache", "num_workers": 2,
    "prefetch_factor": 2, "cpu_threads": 4, "loader_timeout": 60,
    "compile_model": False, "compile_mode": "default",
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


def download_paired_corpora(root, includes, git_executable=None):
    """Find Git in the kernel environment, including standard Windows installs."""
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
            "If the corpus files are already downloaded, set DOWNLOAD_LFS=False."
        )
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
    print(f"Downloading corpus files using {git}", flush=True)
    subprocess.run([git, "-c", "filter.lfs.clean=git-lfs clean -- %f",
                    "-c", "filter.lfs.smudge=git-lfs smudge -- %f",
                    "-c", "filter.lfs.process=git-lfs filter-process", "-c", "filter.lfs.required=true",
                    "lfs", "pull", "--include=" + ",".join(includes), "--exclude="],
                   cwd=root, env=environment, check=True)


def catalogue(root):
    # Load the pure-Python catalogue without importing the native simulator.
    path = Path(root) / "bvr_sim_source/bvr_sim/agents/skill_manager.py"
    spec = importlib.util.spec_from_file_location("paired_skill_catalogue", path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return sorted(module.SkillManager().list_skills())


def scenario_splits(cases, seed=1729):
    """40/30/30 by scenario, stratified by requested range, never by window."""
    rng = np.random.default_rng(seed)
    by_range, result = {}, {}
    for case in cases:
        by_range.setdefault(case["scenario"]["range_m"], []).append(case["scenario_group"])
    for distance, groups in sorted(by_range.items()):
        groups = np.asarray(sorted(groups))
        rng.shuffle(groups)
        train_end = round(len(groups) * 0.4)
        test_end = train_end + round(len(groups) * 0.3)
        if not 0 < train_end < test_end < len(groups):
            raise ValueError("Too few scenarios for three nonempty partitions")
        for split, subset in zip(SPLITS, np.split(groups, [train_end, test_end]), strict=True):
            for group in subset:
                if str(group) in result:
                    raise ValueError("Duplicate scenario group")
                result[str(group)] = {"split": split, "range_m": float(distance)}
    return result


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


def paired_examples(obs_a, obs_b, tracks_b, decisions_a, classes):
    """Exact time pairing, causal inputs, independent masks at +1/+2/+5 seconds."""
    n = len(obs_b)
    if not n:
        raise ValueError("Empty episode")
    episode_id = obs_b.episode_id.iloc[0]
    for frame in (obs_a, obs_b, tracks_b):
        validate_frames(frame, episode_id, n)
        if not frame[KEYS].equals(obs_b[KEYS]):
            raise ValueError("A01/B01 observation/track keys differ")
    a, b = (o[OBS_COLUMNS].to_numpy(dtype=np.float32) for o in (obs_a, obs_b))
    live = (tracks_b.observer_alive & tracks_b.target_alive).to_numpy(dtype=bool)
    # The corpora were rerun separately: reject divergent trajectories, not just bad keys.
    for own, other in ((a, b), (b, a)):
        own_pos = own[:, :3].astype(float) * [50000, 50000, 10000]
        from_other = (other[:, :3].astype(float) + other[:, 9:12]) * [50000, 50000, 10000]
        own_vel = own[:, 3:6].astype(float) * 600
        other_vel = (other[:, 3:6].astype(float) + other[:, 12:15]) * 600
        if (not np.allclose(own_pos[live], from_other[live], atol=0.1, rtol=0)
                or not np.allclose(own_vel[live], other_vel[live], atol=0.01, rtol=0)):
            raise ValueError("Paired corpora contain divergent aircraft trajectories")
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
    features = np.concatenate([observation_features(a), observation_features(b)], axis=1)
    return features, ends[keep], targets[keep]


def verify_source(root, manifest, relative):
    path = root / relative
    if not path.exists() or sha256(path) != manifest["files"][relative]["sha256"]:
        raise ValueError(f"Missing, pointer-only or changed corpus file: {path}. Run selective LFS pull.")


def prepare_dataset(root, output, split_seed=1729):
    """Store frames once in a memory map; fit scaling on training histories only."""
    root, output = Path(root), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty prepared-data directory or load the existing dataset")
    corpus_a = root / "shared/corpora/tactical-episodes-v1"
    corpus_b = root / "shared/corpora/trained-pilot-episodes-v1"
    ca, cb = [read_json(p / "config.json") for p in (corpus_a, corpus_b)]
    ma, mb = [read_json(p / "manifest.json") for p in (corpus_a, corpus_b)]
    if (not ma["complete"] or not mb["complete"] or ca["cases"] != cb["cases"]
            or ca["pilots"] != cb["pilots"] or ca["observer_id"] != "A01"
            or cb["observer_id"] != "B01" or ca["target_id"] != "B01"
            or cb["target_id"] != "A01" or ca["sample_interval_s"] != DT
            or cb["sample_interval_s"] != DT):
        raise ValueError("Expected complete matched A01/B01 10 Hz corpora")
    for corpus, manifest in ((corpus_a, ma), (corpus_b, mb)):
        for relative in ("config.json", "schema.json", "episodes.parquet"):
            verify_source(corpus, manifest, relative)
    ea, eb = [pd.read_parquet(p / "episodes.parquet").sort_values(
        ["pilot_id", "episode_id"]).reset_index(drop=True) for p in (corpus_a, corpus_b)]
    identity = ["episode_id", "pilot_id", "scenario_group", "seed", "checkpoint_sha256", "frames"]
    if (not ea[identity].equals(eb[identity]) or ea.episode_id.duplicated().any()
            or len(ea) != ma["episodes"] or len(eb) != mb["episodes"]):
        raise ValueError("Episode identities, policies, scenarios or lengths differ")
    groups, classes = scenario_splits(cb["cases"], split_seed), catalogue(root)
    eb["split"] = eb.scenario_group.map(lambda group: groups[group]["split"])
    output.mkdir(parents=True, exist_ok=True)
    features = np.lib.format.open_memmap(output / "features.npy", mode="w+", dtype="float32",
                                         shape=(int(eb.frames.sum()), len(FEATURES)))
    total, squares, count = np.zeros(len(FEATURES)), np.zeros(len(FEATURES)), 0
    anchors, all_targets, metadata, offset = [], [], [], 0
    for pilot, episodes in eb.groupby("pilot_id", sort=True):
        sources = [(corpus_a, ma, f"observations/{pilot}.parquet"),
                   (corpus_b, mb, f"observations/{pilot}.parquet"),
                   (corpus_b, mb, f"tracks/{pilot}.parquet"),
                   (corpus_b, mb, f"target_decisions/{pilot}.parquet")]
        for p, m, relative in sources:
            verify_source(p, m, relative)
        files = [pq.ParquetFile(p / relative) for p, _, relative in sources]
        if any(f.num_row_groups != len(episodes) for f in files):
            raise ValueError("Expected one Parquet row group per episode")
        for row_group, episode in enumerate(episodes.itertuples()):
            columns = [KEYS + OBS_COLUMNS, KEYS + OBS_COLUMNS,
                       KEYS + ["observer_alive", "target_alive"],
                       KEYS + ["aircraft_id", "skill"]]
            frames = [f.read_row_group(row_group, columns=c).to_pandas()
                      for f, c in zip(files, columns, strict=True)]
            if (frames[1].episode_id.iloc[0] != episode.episode_id
                    or len(frames[1]) != episode.frames):
                raise ValueError("Row-group ordering differs from episode index")
            frame, ends, target = paired_examples(*frames, classes)
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
        print(f"Prepared {pilot}: {offset:,} paired frames", flush=True)
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
    manifest = {"classes": classes, "features": FEATURES, "history_steps": HISTORY_STEPS,
                "raw_history_steps": RAW_HISTORY_STEPS, "stride_steps": STRIDE_STEPS,
                "horizons_s": HORIZONS, "split_seed": split_seed,
                "split_fractions": {"train": 0.4, "test": 0.3, "validation": 0.3},
                "source_manifests": {str(p.relative_to(root)): sha256(p / "manifest.json")
                                     for p in (corpus_a, corpus_b)},
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
            dataset = WindowDataset(data.directory, data.mean, data.std, len(data.targets))
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
    def __init__(self, num_classes=33, width=192, heads=6, layers=2, dropout=0.1):
        super().__init__()
        if width % heads:
            raise ValueError("Width must be divisible by attention heads")
        self.num_classes = num_classes
        self.projection = nn.Linear(len(FEATURES), width)
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
        if windows.ndim != 3 or tuple(windows.shape[1:]) != (HISTORY_STEPS, len(FEATURES)):
            raise ValueError("Expected (batch, 51, 108) paired histories")
        encoded = self.encoder(self.projection(windows) + self.position)
        # Bidirectional attention is within the past-only window; no future tokens exist.
        return self.head(self.norm(encoded[:, -1])).reshape(-1, 3, self.num_classes)


def masked_loss(logits, targets):
    # Same equal-horizon objective, without three Python/GPU mask synchronizations.
    losses = nn.functional.cross_entropy(logits.float().transpose(1, 2), targets,
                                         ignore_index=-1, reduction="none")
    counts = (targets >= 0).sum(0)
    active = counts > 0
    per_horizon = losses.sum(0) / counts.clamp_min(1)
    return (per_horizon * active).sum() / active.sum().clamp_min(1)


def metric_tables(targets, probabilities, classes, split, losses=None):
    """Per-true-class NLL and F1. Unsupported classes have NaN NLL and zero F1."""
    summary, per_class, matrices = [], [], []
    c = len(classes)
    for h, horizon in enumerate(HORIZONS):
        valid = targets[:, h] >= 0
        y, p = targets[valid, h], probabilities[valid, h]
        if not len(y):
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
                        "nll": float(nll.mean()), "accuracy": float(tp.sum()/len(y)),
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
def evaluate(model, data, split, device, batch_size=1024, precision="fp32", pipeline=None):
    model.eval()
    ids = data.indices[split]
    probabilities, losses = [], []
    batches = (pipeline.batches(ids, batch_size) if pipeline is not None else
               (data.batch(ids[start:start + batch_size], device)
                for start in range(0, len(ids), batch_size)))
    for x, y in batches:
        with autocast_context(device, precision):
            logits = model(x)
        logits = logits.float()  # Probabilities, CE and reporting always use FP32.
        ce = nn.functional.cross_entropy(logits.reshape(-1, model.num_classes),
                                         y.reshape(-1), ignore_index=-1, reduction="none")
        losses.append(ce.reshape(-1, 3).cpu().numpy())
        probabilities.append(logits.softmax(-1).cpu().numpy())
    p, loss = np.concatenate(probabilities), np.concatenate(losses)
    summary, scores, cm = metric_tables(data.targets[ids], p, data.classes, split, loss)
    return {"summary": summary, "per_class": scores, "confusion": cm,
            "probabilities": p, "targets": data.targets[ids], "sample_indices": ids}


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


def fit_batch(model, optimizer, x, y, device, precision, gradient_clip):
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(device, precision):
        logits = model(x)
    loss = masked_loss(logits, y)
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite training loss")
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
    optimizer.step()
    return loss.detach()


def train_model(data, output, config=None, callback=None, device=None, resume=False, tracking=None):
    """Train up to the epoch limit, optionally stopping on post-warmup test-NLL patience.

    With minimum 50 and patience 10, counting starts at epoch 51 and the earliest
    stop is epoch 60. Every strict improvement over the all-time test minimum
    still receives a checkpoint, including during the first 50 epochs.
    """
    from .paired_skill_tracking import RunTracker

    config = {**DEFAULT_TRAINING, **(config or {})}
    if config["epochs"] < 1:
        raise ValueError("Epoch count must be positive")
    for key in ("batch_size", "evaluation_batch_size", "prefetch_factor", "cpu_threads", "loader_timeout"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if (not isinstance(config["num_workers"], int) or config["num_workers"] < 0
            or not isinstance(config["compile_model"], bool)):
        raise ValueError("num_workers must be nonnegative and compile_model must be boolean")
    minimum, patience = config["early_stopping_min_epochs"], config["early_stopping_patience"]
    if (not isinstance(config["early_stopping"], bool)
            or not isinstance(minimum, int) or minimum < 0
            or not isinstance(patience, int) or patience < 1):
        raise ValueError("Early stopping requires a boolean switch, nonnegative minimum and positive patience")
    output = Path(output)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
    cpu_threads = min(config["cpu_threads"], os.cpu_count() or 1)
    torch.set_num_threads(cpu_threads)
    precision = resolve_precision(device, config["precision"])
    architecture = {k: config[k] for k in ("width", "heads", "layers", "dropout")}
    base_model = PairedTransformer(num_classes=len(data.classes), **architecture).to(device)
    model = execution_model(base_model, config)
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    rng = np.random.default_rng(config["seed"])
    manifest_hash = sha256(data.directory / "manifest.json")
    run = {"training": config, "model": architecture, "classes": data.classes,
           "data_manifest_sha256": manifest_hash, "source_sha256": sha256(__file__),
           "tracking_source_sha256": sha256(Path(__file__).with_name("paired_skill_tracking.py")),
           "torch_version": str(torch.__version__), "numpy_version": np.__version__,
           "device": str(device), "selection": "Strict improvement of mean test NLL across horizons",
           "validation_policy": "Evaluated only after checkpoint selection",
           "effective_precision": precision, "effective_cpu_threads": cpu_threads}
    history, best, best_epoch, first = [], float("inf"), 0, 1
    epochs_without_improvement, stopped_early = 0, False
    if resume:
        saved_run = read_json(output / "run.json")
        if any(saved_run[k] != run[k] for k in ("training", "model", "classes",
                                               "data_manifest_sha256", "source_sha256", "tracking_source_sha256")):
            raise ValueError("Resume requires the same configuration, data and source")
        saved = torch.load(output / "last.pt", map_location=device, weights_only=False)
        base_model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        rng.bit_generator.state = saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device.type == "cuda" and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
        history, best, best_epoch = saved["history"], saved["best_loss"], saved["best_epoch"]
        epochs_without_improvement = saved.get("epochs_without_improvement", 0)
        stopped_early = saved.get("early_stopped", False)
        first = config["epochs"] + 1 if stopped_early else saved["epoch"] + 1
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Run output exists; resume or choose a fresh run directory")
        (output / "checkpoints").mkdir(parents=True)
        (output / "epochs").mkdir()
        write_json(output / "run.json", run)
        shutil.copy2(__file__, output / "training_source.py")
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
            for epoch in range(first, config["epochs"] + 1):
                started = time.perf_counter()
                model.train()
                order = rng.permutation(data.indices["train"])
                for x, y in pipeline.batches(order, config["batch_size"]):
                    fit_batch(model, optimizer, x, y, device, precision, config["gradient_clip"])
                results = [evaluate(model, data, s, device, config["evaluation_batch_size"],
                                    precision=precision, pipeline=pipeline)
                           for s in ("train", "test")]
                summary = pd.concat([r["summary"] for r in results], ignore_index=True)
                scores = pd.concat([r["per_class"] for r in results], ignore_index=True)
                train_nll, test_nll = [float(r["summary"].nll.mean()) for r in results]
                if not np.isfinite([train_nll, test_nll]).all():
                    raise ValueError("Nonfinite evaluation loss")
                improved = test_nll < best
                if improved:
                    best, best_epoch = test_nll, epoch
                if config["early_stopping"] and epoch > minimum:
                    epochs_without_improvement = 0 if improved else epochs_without_improvement + 1
                else:
                    epochs_without_improvement = 0
                stopped_early = config["early_stopping"] and epochs_without_improvement >= patience
                row = {"epoch": epoch, "train_nll": train_nll, "test_nll": test_nll,
                       "train_macro_f1": float(results[0]["summary"].macro_f1.mean()),
                       "test_macro_f1": float(results[1]["summary"].macro_f1.mean()),
                       "improved": improved, "epochs_without_improvement": epochs_without_improvement,
                       "early_stopped": stopped_early, "seconds": time.perf_counter() - started}
                history.append(row)
                scores.to_csv(output / "epochs" / f"epoch_{epoch:03d}_per_class.csv", index=False)
                summary.to_csv(output / "epochs" / f"epoch_{epoch:03d}_summary.csv", index=False)
                pd.DataFrame(history).to_csv(output / "history.csv", index=False)
                payload = {"epoch": epoch, "model_state": base_model.state_dict(),
                           "optimizer_state": optimizer.state_dict(), "model_config": architecture,
                           "classes": data.classes, "mean": data.mean, "std": data.std,
                           "history_steps": HISTORY_STEPS, "horizons_s": HORIZONS,
                           "feature_names": FEATURES, "best_loss": best, "best_epoch": best_epoch,
                           "test_nll": test_nll, "history": history, "numpy_rng": rng.bit_generator.state,
                           "epochs_without_improvement": epochs_without_improvement,
                           "early_stopped": stopped_early,
                           "effective_precision": precision,
                           "torch_rng": torch.get_rng_state(), "data_manifest_sha256": manifest_hash,
                           "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}
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
                    print(f"Early stopping after epoch {epoch}: {patience} consecutive epochs "
                          f"without improved test loss after the first {minimum} epochs. "
                          f"Best checkpoint: epoch {best_epoch}.", flush=True)
                    break
        write_json(output / "training_status.json", {
            "status": "early_stopped" if stopped_early else "completed",
            "completed_epochs": len(history), "max_epochs": config["epochs"],
            "early_stopping_enabled": config["early_stopping"],
            "early_stopping_min_epochs": minimum, "early_stopping_patience": patience,
            "epochs_without_improvement": epochs_without_improvement,
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
    model = PairedTransformer(num_classes=len(data.classes), **saved["model_config"]).to(device)
    model.load_state_dict(saved["model_state"])
    run = read_json(output / "run.json")
    config = {**DEFAULT_TRAINING, **run["training"]}
    precision = resolve_precision(device, saved.get("effective_precision",
                                                    run.get("effective_precision", "fp32")))
    model = execution_model(model, config)
    with RunTracker(output, tracking) as tracker:
        tracker.begin(run, data)
        tracker.backfill()
        if tracker.enabled and tracker.previous is None:
            tracker.finish_training()
        with BatchPipeline(data, device, config) as pipeline:
            results = {s: evaluate(model, data, s, device, config["evaluation_batch_size"],
                                   precision=precision, pipeline=pipeline) for s in SPLITS}
        summary = pd.concat([r["summary"] for r in results.values()], ignore_index=True)
        scores = pd.concat([r["per_class"] for r in results.values()], ignore_index=True)
        summary.to_csv(output / "final_metrics.csv", index=False)
        scores.to_csv(output / "final_per_class.csv", index=False)
        for split, result in results.items():
            np.savez_compressed(output / f"final_{split}.npz", **{
                k: result[k] for k in ("probabilities", "targets", "sample_indices", "confusion")})
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
                    base = PairedTransformer(num_classes=len(data.classes), **{
                        k: config[k] for k in ("width", "heads", "layers", "dropout")}).to(device)
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
                            fit_batch(model, optimizer, x, y, device, precision, config["gradient_clip"])
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
