"""Checks for pairing, causal labels, grouped splits and test-based checkpoints."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from bvr_behavior_prediction import paired_skill_forecasting as paired


def episode_frames(n=141, episode_id="p_case"):
    keys = pd.DataFrame({"episode_id": episode_id, "step": np.arange(n),
                         "time_s": np.arange(n) * .1})
    a, b = np.zeros((n, 51), np.float32), np.zeros((n, 51), np.float32)
    a[:, 0] = np.arange(n) * .0001
    b[:, 0] = .2 - np.arange(n) * .0001
    a[:, 3], b[:, 3] = .1, -.1
    a[:, 9:12], b[:, 9:12] = b[:, :3] - a[:, :3], a[:, :3] - b[:, :3]
    a[:, 12:15], b[:, 12:15] = b[:, 3:6] - a[:, 3:6], a[:, 3:6] - b[:, 3:6]
    oa = pd.concat([keys, pd.DataFrame(a, columns=paired.OBS_COLUMNS)], axis=1)
    ob = pd.concat([keys, pd.DataFrame(b, columns=paired.OBS_COLUMNS)], axis=1)
    tracks = keys.assign(observer_alive=True, target_alive=True)
    decisions = keys.iloc[::10].copy().assign(aircraft_id="A01")
    decisions["skill"] = np.where(decisions.step % 20 == 0, "alpha", "beta")
    return oa, ob, tracks, decisions


def test_exact_horizons_masks_and_causal_histories():
    frames = list(episode_frames())
    features, ends, targets = paired.trained_examples(*frames[1:], ["alpha", "beta"])
    assert features.shape == (141, 54)
    assert ends[0] == 60
    assert targets[0].tolist() == [1, 0, 1]  # Exact decisions at ticks 70, 80 and 110.
    # Remove the +2 s decision: never substitute the previous or next decision.
    frames[3] = frames[3].loc[frames[3].step.ne(80)]
    _, _, targets = paired.trained_examples(*frames[1:], ["alpha", "beta"])
    assert targets[0].tolist() == [1, -1, 1]
    assert targets[-1, 2] == -1  # Never extend into an unrecorded future.
    # Modify future-only non-geometric observations and prove past features unchanged.
    for frame in frames[:2]:
        frame.loc[frame.step.gt(60), "obs_19"] = 123
    modified, _, _ = paired.trained_examples(*frames[1:], ["alpha", "beta"])
    np.testing.assert_array_equal(features[10:61], modified[10:61])
    frames[2].loc[20, "target_alive"] = False
    _, ends, _ = paired.trained_examples(*frames[1:], ["alpha", "beta"])
    assert 60 not in ends  # Includes the extra acceleration lookback in the alive check.


def test_bad_pairing_and_bad_labels_fail():
    frames = list(episode_frames())
    frames[1].loc[10, "step"] = 99
    with pytest.raises(ValueError, match="cadence"):
        paired.trained_examples(*frames[1:], ["alpha", "beta"])
    frames = list(episode_frames())
    frames[1].loc[10, "time_s"] += .01
    with pytest.raises(ValueError, match="cadence"):
        paired.trained_examples(*frames[1:], ["alpha", "beta"])
    frames = list(episode_frames())
    frames[3] = pd.concat([frames[3], frames[3].iloc[:1]])
    with pytest.raises(ValueError, match="decision"):
        paired.trained_examples(*frames[1:], ["alpha", "beta"])


def test_grouped_split_has_exact_requested_ratios():
    cases = [{"scenario_group": f"case_{r}_{i:03d}", "scenario": {"range_m": r}}
             for r in [18000, 30000, 45000] for i in range(100)]
    episodes = pd.DataFrame({"scenario_group": [c["scenario_group"] for c in cases]})
    counts = np.ones((len(cases), 3, 2), dtype=int)
    groups = paired.scenario_splits(cases, episodes, counts)
    assert groups == paired.scenario_splits(cases, episodes, counts)
    for distance in [18000, 30000, 45000]:
        counts = pd.Series([g["split"] for g in groups.values() if g["range_m"] == distance]).value_counts()
        assert counts.to_dict() == {"train": 40, "test": 30, "validation": 30}


def test_metrics_class_nll_f1_and_missing_labels():
    truth = np.array([[0, 0, -1], [0, 1, 1], [1, 1, 0]])
    probability = np.array([[[.8, .2]]*3, [[.4, .6]]*3, [[.3, .7]]*3])
    summary, scores, cm = paired.metric_tables(truth, probability, ["a", "b"], "test")
    assert cm[0].tolist() == [[1, 1], [0, 1]]
    row = scores.query("horizon_s == 1 and skill == 'a'").iloc[0]
    assert row.class_nll == pytest.approx(-np.log([.8, .4]).mean())
    assert row.f1 == pytest.approx(2/3)
    assert summary.query("horizon_s == 5").iloc[0]["count"] == 2


def test_model_shape_and_masked_gradients():
    model = paired.PairedTransformer(num_classes=3, width=12, heads=6, layers=1)
    assert model.encoder.layers[0].self_attn.num_heads == 6
    assert model(torch.zeros(2, 51, len(paired.FEATURES))).shape == (2, 3, 3)
    logits = torch.randn(2, 3, 3, requires_grad=True)
    labels = torch.tensor([[0, -1, 2], [1, 0, -1]])
    paired.masked_loss(logits, labels).backward()
    assert logits.grad[0, 1].abs().sum() == 0
    assert logits.grad[1, 2].abs().sum() == 0


def synthetic_corpora(tmp_path):
    root = tmp_path / "repo"
    manager = root / "bvr_sim_source/bvr_sim/agents/skill_manager.py"
    manager.parent.mkdir(parents=True)
    manager.write_text("class SkillManager:\n    def list_skills(self): return ['alpha', 'beta']\n")
    cases = [{"scenario_group": f"case_{i:04d}", "scenario": {"range_m": 18000}}
             for i in range(10)]
    metadata = []
    for i in range(10):
        metadata.append({"episode_id": f"p_case_{i:04d}", "pilot_id": "pilot_001",
                         "scenario_group": f"case_{i:04d}", "seed": i,
                         "checkpoint_sha256": "fixed", "frames": 141})
    for name, observer, target in [("trained-pilot-episodes-v1", "B01", "A01")]:
        corpus = root / "shared/corpora" / name
        corpus.mkdir(parents=True)
        paired.write_json(corpus / "config.json", {"cases": cases, "pilots": ["pilot_001"],
                          "observer_id": observer, "target_id": target, "target_policy": "trained_pilot", "sample_interval_s": .1})
        paired.write_json(corpus / "schema.json", {})
        pd.DataFrame(metadata).to_parquet(corpus / "episodes.parquet", index=False)
        parts = ["observations"] if observer == "A01" else ["observations", "tracks", "target_decisions"]
        for part in parts:
            directory = corpus / part
            directory.mkdir()
            writer = None
            try:
                for episode in metadata:
                    frames = episode_frames(episode_id=episode["episode_id"])
                    f = frames[0 if observer == "A01" else 1] if part == "observations" else frames[
                        2 if part == "tracks" else 3]
                    table = pa.Table.from_pandas(f, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(directory / "pilot_001.parquet", table.schema)
                    writer.write_table(table)
            finally:
                if writer:
                    writer.close()
        records = {p.relative_to(corpus).as_posix(): {"sha256": paired.sha256(p)}
                   for p in corpus.rglob("*") if p.is_file()}
        paired.write_json(corpus / "manifest.json", {"complete": True, "episodes": 10, "files": records})
    return root


def test_preparation_train_only_scaling_and_end_to_end(tmp_path, monkeypatch):
    root = synthetic_corpora(tmp_path)
    destination = tmp_path / "prepared"
    manifest = paired.prepare_dataset(root, destination)
    assert {s: c["groups"] for s, c in manifest["counts"].items()} == {
        "train": 4, "test": 3, "validation": 3}
    data = paired.PairedData(destination)
    # Independently compute the union of training history frames, not held-out frames.
    ids = data.indices["train"]
    steps = np.unique((data.anchors[ids, None] + np.arange(-50, 1)).ravel())
    np.testing.assert_allclose(data.mean, data.features[steps].astype(float).mean(0), atol=1e-7)
    # Use a small real Transformer and a small set of windows; run all 100 epochs.
    for split in paired.SPLITS:
        data.indices[split] = data.indices[split][:3]
    evaluated, events = [], []
    original = paired.evaluate

    def tracked_evaluation(model, dataset, split, *args, **kwargs):
        evaluated.append(split)
        return original(model, dataset, split, *args, **kwargs)

    monkeypatch.setattr(paired, "evaluate", tracked_evaluation)
    config = {"epochs": 100, "width": 12, "heads": 6, "layers": 1,
              "batch_size": 3, "evaluation_batch_size": 3, "dropout": 0.0}
    output = tmp_path / "run"
    history = paired.train_model(data, output, config, callback=lambda *x: events.append(x), device="cpu")
    assert len(history) == len(events) == 100
    assert evaluated == ["train", "test"] * 100
    expected = history.loc[history.improved, "epoch"].tolist()
    actual = sorted(int(p.stem.split('_')[1]) for p in (output / "checkpoints").glob("*.pt"))
    assert expected == actual
    previous_min = float("inf")
    for row in history.itertuples():
        assert row.improved == (row.test_nll < previous_min)
        previous_min = min(previous_min, row.test_nll)
    best_epoch, metrics, scores, results = paired.final_evaluation(data, output, "cpu")
    assert best_epoch == int(history.loc[history.test_nll.idxmin(), "epoch"])
    assert evaluated[-3:] == ["train", "test", "validation"]
    assert set(metrics.split) == set(paired.SPLITS)
    assert set(scores.split) == set(paired.SPLITS)
    assert results["validation"]["confusion"].shape == (3, 2, 2)
    resumed = paired.train_model(data, output, config, device="cpu", resume=True)
    pd.testing.assert_frame_equal(history, resumed)


def test_notebook_schema_and_code_compile():
    import nbformat
    path = Path(__file__).parents[1] / "notebooks/09b_paired_skill_forecasting.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            # %%time is an IPython cell magic; validate its Python body as well.
            source = cell.source
            if source.startswith("%%time\n"):
                source = source.partition("\n")[2]
            compile(source, str(path), "exec")


def controlled_training(tmp_path, monkeypatch, loss_at_epoch, train_loss_at_epoch=None):
    """Drive the real fitting/checkpoint loop with a controlled test-loss sequence."""
    from types import SimpleNamespace

    directory = tmp_path / "data"
    directory.mkdir()
    paired.write_json(directory / "manifest.json", {"fixture": "controlled test loss"})
    data = SimpleNamespace(directory=directory, classes=["a", "b"],
                           indices={"train": np.array([0])},
                           mean=np.zeros(len(paired.FEATURES), np.float32), std=np.ones(len(paired.FEATURES), np.float32))
    data.batch = lambda ids, device: (torch.zeros(len(ids), 51, len(paired.FEATURES), device=device),
                                    torch.zeros(len(ids), 3, dtype=torch.long, device=device))
    calls = []

    def evaluate(model, data, split, device, batch_size, **kwargs):
        if split == "test":
            calls.append(split)
        value = (loss_at_epoch(len(calls)) if split == "test" else
                 train_loss_at_epoch(len(calls) + 1) if train_loss_at_epoch else 1.0)
        return {"summary": pd.DataFrame({"nll": [value] * 3, "macro_f1": [0.0] * 3}),
                "per_class": pd.DataFrame({"f1": [0.0]})}

    monkeypatch.setattr(paired, "evaluate", evaluate)
    return data, calls


@pytest.mark.parametrize("enabled,sequence,expected_epochs,expected_best", [
    (True, "flat", 60, 1),
    (True, "reset", 65, 55),
    (True, "improving", 100, 100),
    (False, "flat", 100, 1),
])
def test_optional_early_stopping(tmp_path, monkeypatch, enabled, sequence,
                                 expected_epochs, expected_best):
    def loss_at_epoch(epoch):
        if sequence == "improving":
            return 1 / (epoch + 1)
        return 0.9 if sequence == "reset" and epoch >= 55 else 1.0

    data, calls = controlled_training(tmp_path, monkeypatch, loss_at_epoch)
    output = tmp_path / "run"
    config = {"width": 6, "heads": 6, "layers": 1, "dropout": 0.0,
              "early_stopping": enabled}
    history = paired.train_model(data, output, config, callback=lambda *args: None, device="cpu")
    assert len(history) == len(calls) == expected_epochs
    assert history.iloc[-1].early_stopped == (expected_epochs < 100)
    assert not history.loc[history.epoch.le(50), "epochs_without_improvement"].any()
    status = paired.read_json(output / "training_status.json")
    assert status["status"] == ("early_stopped" if expected_epochs < 100 else "completed")
    assert status["best_epoch"] == expected_best
    best = torch.load(output / "best.pt", weights_only=False)
    assert best["epoch"] == expected_best
    # Completed or stopped runs are idempotent when resumed.
    resumed = paired.train_model(data, output, config, device="cpu", resume=True)
    pd.testing.assert_frame_equal(history, resumed)
    assert len(calls) == expected_epochs
    if expected_epochs < 100:
        # The notebook must permit its final diagnostics after a legitimate early stop.
        notebook = json.loads((Path(__file__).parents[1] /
                              "notebooks/09b_paired_skill_forecasting.ipynb").read_text())
        source = next("".join(c["source"]) for c in notebook["cells"]
                      if c["cell_type"] == "code" and "best_epoch, final_metrics" in "".join(c["source"]))
        source = source.split("best_epoch, final_metrics")[0]
        if source.startswith("%%time\n"):
            source = source.partition("\n")[2]
        exec(compile(source, "final-evaluation-guard", "exec"), {
            "read_json": paired.read_json, "RUN": output, "history": history,
            "CONFIG": {**paired.DEFAULT_TRAINING, **config}, "display": lambda *args: None,
            "Markdown": str,
        })


def test_early_stopping_counter_survives_interruption(tmp_path, monkeypatch):
    data, calls = controlled_training(tmp_path, monkeypatch, lambda epoch: 1.0)
    output = tmp_path / "run"
    config = {"width": 6, "heads": 6, "layers": 1, "dropout": 0.0,
              "early_stopping": True}

    def interrupt(row, *args):
        if row["epoch"] == 57:
            raise RuntimeError("Simulated notebook interruption")

    with pytest.raises(RuntimeError, match="Simulated"):
        paired.train_model(data, output, config, callback=interrupt, device="cpu")
    last = torch.load(output / "last.pt", weights_only=False)
    assert last["epoch"] == 57 and last["epochs_without_improvement"] == 7
    history = paired.train_model(data, output, config, device="cpu", resume=True,
                                 callback=lambda *args: None)
    assert len(history) == len(calls) == 60
    assert history.iloc[-1].early_stopped


def test_l1_changes_weight_gradients_without_penalising_biases_or_layernorm():
    import copy

    torch.manual_seed(42)
    plain = paired.PairedTransformer(num_classes=2, width=6, heads=6, layers=1, dropout=0.0)
    regularised = copy.deepcopy(plain)
    x = torch.randn(2, 51, len(paired.FEATURES))
    y = torch.tensor([[0, -1, 1], [1, 0, -1]])
    losses = []
    for model, coefficient in ((plain, 0.0), (regularised, .01)):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        losses.append(paired.fit_batch(model, optimizer, x, y, "cpu", "fp32", 1e6, coefficient))
    expected = .01 * sum(p.abs().sum() for p in plain.parameters() if p.ndim > 1)
    torch.testing.assert_close(losses[1] - losses[0], expected)
    for p, q in zip(plain.parameters(), regularised.parameters()):
        extra = .01 * p.sign() if p.ndim > 1 else torch.zeros_like(p)
        torch.testing.assert_close(q.grad - p.grad, extra, atol=1e-6, rtol=1e-5)


def test_no_minimum_epoch_delay(tmp_path, monkeypatch):
    data, calls = controlled_training(tmp_path, monkeypatch, lambda epoch: 1.0)
    config = {"width": 6, "heads": 6, "layers": 1, "early_stopping": True,
              "early_stopping_min_epochs": 0, "early_stopping_patience": 10}
    history = paired.train_model(data, tmp_path / "run", config, device="cpu")
    assert len(history) == len(calls) == 11
    assert history.iloc[1].epochs_without_improvement == 1
    assert history.iloc[-1].stop_reason == "patience"


@pytest.mark.parametrize("test_losses,train_losses,enabled,expected_epochs", [
    ([1, 1.1, 1.2, 1.3, 1.4, 1.5], [1, .9, .8, .7, .6, .5], True, 4),
    # A transient spike or a recovery must not accumulate a shutdown streak.
    ([1, 1.2, 1, 1.2, 1.3, 1.4], [1, .8, .7, .6, .5, .4], True, 6),
    # A new best resets the counter and anchors the new training-loss reference.
    ([1, 1.2, .9, 1, 1.1, 1.2], [1, .8, .7, .6, .5, .4], True, 6),
    # Rising train and test loss is not the improving-train overtraining signal.
    ([1, 1.1, 1.2, 1.3, 1.4, 1.5], [1, 1, 1, 1, 1, 1], True, 6),
    # Neither a small relative deterioration nor a small absolute gap triggers it.
    ([1, 1.01, 1.02, 1.03, 1.04, 1.05], [1, .5, .5, .5, .5, .5], True, 6),
    ([.1, .11, .11, .11, .11, .11], [.05, .04, .04, .04, .04, .04], True, 6),
    ([1, 1.1, 1.2, 1.3, 1.4, 1.5], [1, .9, .8, .7, .6, .5], False, 6),
])
def test_overtraining_shutdown(tmp_path, monkeypatch, test_losses, train_losses,
                              enabled, expected_epochs):
    data, calls = controlled_training(tmp_path, monkeypatch,
        lambda epoch: test_losses[epoch - 1], lambda epoch: train_losses[epoch - 1])
    output = tmp_path / "run"
    # Guard works independently of ordinary early stopping and its warmup.
    config = {"epochs": 6, "width": 6, "heads": 6, "layers": 1,
              "early_stopping": False, "early_stopping_min_epochs": 50,
              "overtraining_shutdown": enabled}
    history = paired.train_model(data, output, config, device="cpu")
    assert len(history) == len(calls) == expected_epochs
    stopped = bool(history.iloc[-1].overtraining_detected and
                   history.iloc[-1].overtraining_epochs == 3)
    status = paired.read_json(output / "training_status.json")
    assert status["stop_reason"] == ("overtraining" if stopped else None)
    assert status["status"] == ("early_stopped" if stopped else "completed")
    best = torch.load(output / "best.pt", weights_only=False)
    assert best["epoch"] == int(np.argmin(test_losses[:expected_epochs])) + 1
    resumed = paired.train_model(data, output, config, device="cpu", resume=True)
    pd.testing.assert_frame_equal(history, resumed)
    assert len(calls) == expected_epochs


def test_overtraining_shutdown_survives_interruption(tmp_path, monkeypatch):
    data, calls = controlled_training(tmp_path, monkeypatch,
                                     lambda epoch: 1 + .1 * epoch, lambda epoch: 1 / epoch)
    output = tmp_path / "run"
    config = {"epochs": 10, "width": 6, "heads": 6, "layers": 1,
              "overtraining_shutdown": True}

    def interrupt(row, *args):
        if row["epoch"] == 3:
            raise RuntimeError("Simulated interruption")

    with pytest.raises(RuntimeError, match="Simulated"):
        paired.train_model(data, output, config, callback=interrupt, device="cpu")
    saved = torch.load(output / "last.pt", weights_only=False)
    assert saved["overtraining_epochs"] == 2
    assert saved["train_nll_at_best"] == 1.0
    history = paired.train_model(data, output, config, device="cpu", resume=True)
    assert len(history) == len(calls) == 4
    assert history.iloc[-1].stop_reason == "overtraining"


@pytest.mark.parametrize("setting,value", [
    ("l1_lambda", -1), ("l1_lambda", float("nan")), ("l1_lambda", float("inf")),
    ("overtraining_patience", 0), ("overtraining_patience", True),
    ("overtraining_min_gap", -1), ("overtraining_relative_increase", float("nan")),
    ("overtraining_shutdown", 1),
])
def test_invalid_overtraining_settings_fail_before_fitting(tmp_path, setting, value):
    with pytest.raises(ValueError):
        paired.train_model(None, tmp_path / "unused", {setting: value}, device="cpu")
    assert not (tmp_path / "unused").exists()


@pytest.fixture
def prepared_pair(tmp_path):
    root = synthetic_corpora(tmp_path)
    output = tmp_path / "prepared"
    paired.prepare_dataset(root, output)
    return paired.PairedData(output)


def test_normalized_tensor_cache_matches_direct_windows(prepared_pair):
    data = prepared_pair
    original = np.array(data.features)
    cache = paired.TensorFrameCache(data, "cpu", chunk_frames=17)
    ids = data.indices["train"][::-1].copy()
    batches = list(cache.batches(ids, 5))
    x, y = data.batch(ids, "cpu")
    torch.testing.assert_close(torch.cat([b[0] for b in batches]), x, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(torch.cat([b[1] for b in batches]), y)
    assert len(batches[-1][0]) == len(ids) % 5
    np.testing.assert_array_equal(data.features, original)
    assert cache.features.dtype == torch.float32


@pytest.mark.parametrize("workers", [0, 2])
def test_prefetch_preserves_order_masks_and_reuses_pool(prepared_pair, workers):
    config = {**paired.DEFAULT_TRAINING, "data_backend": "prefetch", "num_workers": workers}
    data = prepared_pair
    pids, processes = None, []
    with paired.BatchPipeline(data, "cpu", config) as pipeline:
        for split in ("train", "test", "train"):
            ids = data.indices[split][::-1].copy()
            actual = list(pipeline.batches(ids, 7))
            expected = data.batch(ids, "cpu")
            torch.testing.assert_close(torch.cat([b[0] for b in actual]), expected[0])
            torch.testing.assert_close(torch.cat([b[1] for b in actual]), expected[1])
            if workers:
                processes = pipeline.loader._iterator._workers
                current_pids = [w.pid for w in processes]
                assert pids is None or current_pids == pids
                pids = current_pids
    assert all(not w.is_alive() for w in processes)


def test_worker_pickle_excludes_open_memory_maps(prepared_pair):
    import pickle
    data = prepared_pair
    dataset = paired.WindowDataset(data.directory, data.mean, data.std, len(data.targets))
    expected = dataset.__getitems__([0, 1])
    serialized = pickle.dumps(dataset)
    assert len(serialized) < 10000  # Only paths, scaling and small metadata are sent.
    restored = pickle.loads(serialized)
    assert restored._features is None
    actual = restored.__getitems__([0, 1])
    torch.testing.assert_close(expected[0], actual[0])
    torch.testing.assert_close(expected[1], actual[1])


def test_precision_selection_falls_back_on_cpu(monkeypatch):
    assert paired.resolve_precision("cpu", "bf16") == "fp32"
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert paired.resolve_precision("cuda", "bf16") == "bf16"
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert paired.resolve_precision("cuda", "bf16") == "fp32"
    with pytest.raises(ValueError, match="precision"):
        paired.resolve_precision("cpu", "invalid")


@pytest.mark.parametrize("all_missing", [False, True])
def test_vectorized_loss_preserves_equal_horizon_objective(all_missing):
    generator = torch.Generator().manual_seed(18)
    logits = torch.randn(8, 3, 4, generator=generator, requires_grad=True)
    targets = torch.randint(-1, 4, (8, 3), generator=generator)
    targets[:, 1] = -1
    if all_missing:
        targets[:] = -1
    losses = [torch.nn.functional.cross_entropy(logits[targets[:, h] >= 0, h],
                                                targets[targets[:, h] >= 0, h])
              for h in range(3) if (targets[:, h] >= 0).any()]
    reference = torch.stack(losses).mean() if losses else logits.sum() * 0
    actual = paired.masked_loss(logits, targets)
    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(torch.autograd.grad(actual, logits, retain_graph=True)[0],
                               torch.autograd.grad(reference, logits)[0])


class CompileWrapper(torch.nn.Module):
    """Simulate compile's state-key prefix without requiring a local compiler."""
    def __init__(self, model):
        super().__init__()
        self._orig_mod = model

    @property
    def num_classes(self):
        return self._orig_mod.num_classes

    def forward(self, x):
        return self._orig_mod(x)


def test_compiled_checkpoints_are_portable(prepared_pair, tmp_path, monkeypatch):
    calls = []

    def compile_model(model, **kwargs):
        calls.append(kwargs)
        return CompileWrapper(model)

    monkeypatch.setattr(torch, "compile", compile_model)
    data = prepared_pair
    for split in paired.SPLITS:
        data.indices[split] = data.indices[split][:3]
    config = {"epochs": 2, "width": 12, "heads": 6, "layers": 1,
              "dropout": 0.0, "compile_model": True}
    output = tmp_path / "compiled_run"
    paired.train_model(data, output, config, callback=lambda *args: None, device="cpu")
    saved = torch.load(output / "best.pt", weights_only=False)
    assert all(not name.startswith("_orig_mod") for name in saved["model_state"])
    assert all(t.dtype == torch.float32 for t in saved["model_state"].values())
    epoch, metrics, _, _ = paired.final_evaluation(data, output, "cpu")
    assert epoch in [1, 2] and len(metrics) == 9
    assert len(calls) == 2


def test_training_only_benchmark_restores_rng_and_threads(prepared_pair):
    data = prepared_pair
    data.indices = {"train": data.indices["train"]}
    state = torch.get_rng_state().clone()
    numpy_state = np.random.get_state()
    threads = torch.get_num_threads()
    config = {"width": 12, "heads": 6, "layers": 1, "dropout": 0.0}
    result = paired.benchmark_batches(data, config, "cpu", batch_sizes=(2, 4), warmup=1, repeats=2)
    assert result.status.tolist() == ["ok", "ok"]
    assert (result.median_step_seconds > 0).all()
    assert (result.windows_per_second > 0).all()
    assert result.precision.tolist() == ["fp32", "fp32"]
    assert result.backend.tolist() == ["direct", "direct"]
    assert torch.equal(torch.get_rng_state(), state)
    assert torch.get_num_threads() == threads
    after = np.random.get_state()
    assert after[0] == numpy_state[0] and after[2:] == numpy_state[2:]
    np.testing.assert_array_equal(after[1], numpy_state[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_cuda_cache_and_bf16_training(prepared_pair):
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 unavailable")
    data = prepared_pair
    config = {**paired.DEFAULT_TRAINING, "width": 12, "layers": 1}
    model = paired.PairedTransformer(num_classes=len(data.classes), width=12, heads=6, layers=1).cuda()
    optimizer = torch.optim.AdamW(model.parameters())
    with paired.BatchPipeline(data, "cuda", config) as pipeline:
        ids = data.indices["train"][:3]
        x, y = next(iter(pipeline.batches(ids, 3)))
        expected, expected_y = data.batch(ids, "cuda")
        torch.testing.assert_close(x, expected)
        torch.testing.assert_close(y, expected_y)
        loss = paired.fit_batch(model, optimizer, x, y, "cuda", "bf16", 1.0)
        assert torch.isfinite(loss) and loss.dtype == torch.float32
        assert all(p.dtype == torch.float32 for p in model.parameters())
        for split in paired.SPLITS:
            data.indices[split] = data.indices[split][:3]
        result = paired.evaluate(model, data, "test", "cuda", precision="bf16", pipeline=pipeline)
        assert result["probabilities"].dtype == np.float32
        assert np.isfinite(result["summary"].nll).all()


def test_mlflow_runs_models_and_comparison(prepared_pair, tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    import mlflow.pytorch
    from mlflow.tracking import MlflowClient
    from bvr_behavior_prediction.paired_skill_tracking import compare_runs, log_run_artifacts

    data = prepared_pair
    for split in paired.SPLITS:
        data.indices[split] = data.indices[split][:6]
    settings = {"enabled": True, "tracking_uri": f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
                "experiment": "paired-comparison", "artifact_root": (tmp_path / "mlflow-artifacts").as_uri()}
    previous_uri, previous_run = mlflow.get_tracking_uri(), mlflow.active_run()
    config = {"epochs": 2, "width": 12, "heads": 6, "layers": 1,
              "batch_size": 3, "evaluation_batch_size": 4, "dropout": 0.0}
    first = tmp_path / "first_run"
    history = paired.train_model(data, first, config, callback=lambda *args: None,
                                 device="cpu", tracking=settings)
    link = paired.read_json(first / "mlflow.json")
    client = MlflowClient(tracking_uri=settings["tracking_uri"])
    tracked = client.get_run(link["run_id"])
    assert tracked.info.status == "FINISHED"
    assert tracked.data.params["training.batch_size"] == "3"
    assert tracked.data.tags["runtime.effective_precision"] == "fp32"
    assert tracked.data.metrics["best/test_nll"] == pytest.approx(history.test_nll.min())
    assert not any("validation" in key for key in tracked.data.metrics)
    assert len(client.get_metric_history(link["run_id"], "epoch/test/nll")) == 2
    assert "epoch/train/1s/class_00_alpha/f1" in tracked.data.metrics
    epoch, final, _, _ = paired.final_evaluation(data, first, "cpu")
    tracked = client.get_run(link["run_id"])
    assert tracked.data.metrics["final/validation/nll"] == pytest.approx(
        final.loc[final.split.eq("validation"), "nll"].mean())
    assert tracked.data.metrics["final/selected_epoch"] == epoch
    diagnostics = first / "diagnostics"
    diagnostics.mkdir()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(history.epoch, history.test_nll)
    fig.savefig(diagnostics / "loss.png")
    plt.close(fig)
    log_run_artifacts(first)
    assert any(a.path == "diagnostics/loss.png" for a in client.list_artifacts(link["run_id"], "diagnostics"))
    model_path = client.download_artifacts(link["run_id"], "model")
    # This test loads only its own just-produced artifact. Torch 2.5 cannot perform
    # the restricted PT2 verification required by newer MLflow; the user's 2.11 can.
    with monkeypatch.context() as patch:
        patch.setenv("MLFLOW_ALLOW_MODEL_CODE_EXECUTION", "true")
        loaded = mlflow.pytorch.load_model(model_path, map_location="cpu")
    checkpoint = torch.load(first / "best.pt", weights_only=False)
    expected = paired.PairedTransformer(num_classes=2, **checkpoint["model_config"])
    expected.load_state_dict(checkpoint["model_state"])
    expected.eval()
    x, _ = data.batch(data.indices["train"], "cpu")
    with torch.inference_mode():
        torch.testing.assert_close(loaded(x), expected(x))
        torch.testing.assert_close(loaded(x[:1]), expected(x[:1]))
    schema = paired.read_json(Path(model_path) / "paired_input_schema.json")
    assert schema["classes"] == data.classes and schema["horizons_s"] == [1, 2, 5]
    resumed = paired.train_model(data, first, config, device="cpu", resume=True)
    pd.testing.assert_frame_equal(history, resumed)
    assert paired.read_json(first / "mlflow.json")["run_id"] == link["run_id"]
    assert len(client.get_metric_history(link["run_id"], "epoch/test/nll")) == 2
    second = tmp_path / "second_run"
    paired.train_model(data, second, {**config, "batch_size": 6}, device="cpu",
                       callback=lambda *args: None, tracking={**settings, "log_model": False})
    comparison = compare_runs(settings, paired.sha256(data.directory / "manifest.json"))
    assert len(comparison) == 2 and set(comparison.batch_size) == {"3", "6"}
    assert link["model_uri"] == f"runs:/{link['run_id']}/model"
    assert mlflow.get_tracking_uri() == previous_uri and mlflow.active_run() is previous_run


def test_mlflow_write_failure_resumes_same_run_and_backfills(prepared_pair, tmp_path, monkeypatch):
    pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient
    data = prepared_pair
    for split in paired.SPLITS:
        data.indices[split] = data.indices[split][:3]
    settings = {"enabled": True, "tracking_uri": f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
                "experiment": "paired-resume", "artifact_root": (tmp_path / "artifacts").as_uri(),
                "log_model": False}
    output = tmp_path / "run"
    original = MlflowClient.log_batch

    def fail_epoch_write(self, run_id, metrics=(), **kwargs):
        if any(metric.key == "epoch/test/nll" for metric in metrics):
            raise RuntimeError("Simulated MLflow interruption")
        return original(self, run_id, metrics=metrics, **kwargs)

    config = {"epochs": 2, "width": 12, "heads": 6, "layers": 1, "dropout": 0.0}
    with monkeypatch.context() as patch:
        patch.setattr(MlflowClient, "log_batch", fail_epoch_write)
        with pytest.raises(RuntimeError, match="Simulated MLflow"):
            paired.train_model(data, output, config, device="cpu", tracking=settings)
    state = paired.read_json(output / "mlflow.json")
    client = MlflowClient(tracking_uri=settings["tracking_uri"])
    assert client.get_run(state["run_id"]).info.status == "FAILED"
    assert (output / "last.pt").exists()
    assert state["last_logged_epoch"] == 0
    history = paired.train_model(data, output, config, device="cpu", resume=True)
    assert len(history) == 2
    assert paired.read_json(output / "mlflow.json")["run_id"] == state["run_id"]
    assert sorted(m.step for m in client.get_metric_history(state["run_id"], "epoch/test/nll")) == [1, 2]
    assert client.get_run(state["run_id"]).info.status == "FINISHED"


def test_download_reports_missing_git(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(paired.shutil, "which", lambda command: None)
    monkeypatch.setattr(paired, "os", SimpleNamespace(name="posix"))
    with pytest.raises(RuntimeError, match="Git executable not found.*Conda"):
        paired.download_paired_corpora(tmp_path, ["shared/**/*.parquet"])


def test_download_discovers_windows_git_and_uses_exact_path(tmp_path, monkeypatch):
    import os
    from types import SimpleNamespace
    git = tmp_path / "Program Files/Git/cmd/git.exe"
    git.parent.mkdir(parents=True)
    git.touch()
    calls = []
    monkeypatch.setattr(paired.shutil, "which", lambda command: None)
    monkeypatch.setattr(paired, "os", SimpleNamespace(
        name="nt", environ={"ProgramFiles": str(tmp_path / "Program Files")}, pathsep=os.pathsep))

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(paired.subprocess, "run", run)
    paired.download_paired_corpora(tmp_path, ["shared/test/*.parquet"])
    assert len(calls) == 2 and all(c[0][0] == str(git) for c in calls)
    assert calls[1][0][-2:] == ["--include=shared/test/*.parquet", "--exclude="]
    assert calls[1][1]["env"]["PATH"].startswith(str(git.parent))


def test_download_reports_missing_lfs(tmp_path, monkeypatch):
    from types import SimpleNamespace
    git = tmp_path / "git.exe"
    git.touch()
    monkeypatch.setattr(paired.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stderr="not a git command"))
    with pytest.raises(RuntimeError, match="Git LFS is unavailable"):
        paired.download_paired_corpora(tmp_path, ["shared/test/*.parquet"], git_executable=str(git))


def test_pre_split_episode_counts_match_prepared_labels(tmp_path):
    root = synthetic_corpora(tmp_path)
    assert not (root / "shared/corpora/tactical-episodes-v1").exists()
    analysis = paired.analyze_trained_corpus(root)
    assert "split" not in analysis["episodes"]
    table = paired.skill_support_table(analysis)
    # Repeated labels count once per episode; both skills occur at every horizon.
    assert table.episodes.tolist() == [10] * 6
    assert (table.windows > table.episodes).all()
    output = tmp_path / "prepared"
    manifest = paired.prepare_dataset(root, output, analysis=analysis)
    assert set(manifest["source_manifests"]) == {paired.CORPUS}
    assert manifest["data_contract"] == paired.DATA_CONTRACT
    data = paired.PairedData(output)
    assert data.features.shape[1] == 54
    expected = pd.read_csv(output / "skill_support_by_split.csv")
    for row in expected.itertuples():
        h = paired.HORIZONS.index(row.horizon_s)
        c = data.classes.index(row.skill)
        ids = data.indices[row.split]
        selected = data.samples.iloc[ids].loc[data.targets[ids, h] == c]
        assert selected.episode_id.nunique() == row.episodes
        assert len(selected) == row.windows


def test_multilabel_stratification_balances_each_horizon_and_rare_skills():
    cases = [{"scenario_group": f"g{i:03d}", "scenario": {"range_m": 18000}}
             for i in range(100)]
    # Two pilots share each scenario, including a skill present in just ten groups.
    episodes = pd.DataFrame({"scenario_group": np.repeat([c["scenario_group"] for c in cases], 2)})
    counts = np.zeros((200, 3, 4), dtype=int)
    for i in range(100):
        for h in range(3):
            counts[2*i:2*i+2, h, (i + h) % 2] = 20
            if (i + h) % 10 == 0:
                counts[2*i:2*i+2, h, 2] = 1
    # Fourth skill is globally absent and must remain visible with zero support.
    groups = paired.scenario_splits(cases, episodes, counts)
    analysis = {"episodes": episodes, "counts": counts, "classes": ["a", "b", "rare", "absent"]}
    table = paired.skill_support_table(analysis, groups)
    for h in paired.HORIZONS:
        for name, total in [("a", 100), ("b", 100), ("rare", 20)]:
            subset = table.loc[table.horizon_s.eq(h) & table.skill.eq(name)].set_index("split")
            for split, fraction in zip(paired.SPLITS, [.4, .3, .3]):
                assert abs(subset.loc[split, "episodes"] - total*fraction) <= 2
    assert table.loc[table.skill.eq("absent"), "episodes"].sum() == 0


def test_analysis_cell_precedes_split_and_downloads_only_trained_corpus():
    notebook = json.loads((Path(__file__).parents[1] /
                          "notebooks/09b_paired_skill_forecasting.ipynb").read_text())
    code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    survey = next(i for i, c in enumerate(code) if "analysis = analyze_trained_corpus" in c)
    prepare = next(i for i, c in enumerate(code) if "prepare_dataset(ROOT" in c)
    assert survey < prepare
    assert "tactical-episodes-v1" not in "".join(code)
    assert 'prepared_manifest.get("data_contract") == DATA_CONTRACT' in code[prepare]


def test_notebook_accepts_windows_source_path_but_rejects_other_corpora():
    notebook = json.loads((Path(__file__).parents[1] /
                          "notebooks/09b_paired_skill_forecasting.ipynb").read_text())
    source = next("".join(c["source"]) for c in notebook["cells"]
                  if c["cell_type"] == "code" and "source_corpora =" in "".join(c["source"]))
    check = source[source.index("source_corpora ="):source.index("data = PairedData")]
    for relative in (paired.CORPUS, paired.CORPUS.replace("/", "\\")):
        exec(check, {"prepared_manifest": {"source_manifests": {relative: "hash"}},
                     "CORPUS": paired.CORPUS})
    with pytest.raises(AssertionError, match="Unexpected raw corpus"):
        exec(check, {"prepared_manifest": {"source_manifests": {
            paired.CORPUS: "hash", "shared/corpora/tactical-episodes-v1": "hash"}},
                     "CORPUS": paired.CORPUS})
