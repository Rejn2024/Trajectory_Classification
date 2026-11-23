"""Relative position/velocity targets from the 91-feature notebook-32 corpus."""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from bvr_behavior_prediction.forecasting import relative_motion as motion
from bvr_behavior_prediction.forecasting import relative_motion_gnn as gnn
from scripts.tactical_corpus_schema import observation_feature_schema

ROOT = Path(__file__).resolve().parents[1]


def _physical_tracks(episode_id, frames=90):
    time_s = np.arange(frames) * motion.DT
    a_position = np.column_stack((1000 + 210 * time_s, 300 + 10 * time_s, 7000 + time_s))
    b_position = np.column_stack((9000 + 180 * time_s, -1200 + 25 * time_s, 7600 + 2 * time_s))
    a_velocity = np.tile([210.0, 10.0, 1.0], (frames, 1))
    b_velocity = np.tile([180.0, 25.0, 2.0], (frames, 1))
    return {
        "A01": (a_position, a_velocity),
        "B01": (b_position, b_velocity),
    }


def _perspective(episode_id, side, frames=90, marker=0.0):
    state = _physical_tracks(episode_id, frames)
    target_side = "B01" if side == "A01" else "A01"
    own_position, own_velocity = state[side]
    target_position, target_velocity = state[target_side]
    raw = np.zeros((frames, 91), dtype=np.float32)
    raw[:, :3] = own_position / [50000, 50000, 10000]
    raw[:, 3:6] = own_velocity / 600
    raw[:, 6] = own_position[:, 2] / 10000
    relative_position = target_position - own_position
    relative_velocity = target_velocity - own_velocity
    raw[:, 9:12] = relative_position / [50000, 50000, 10000]
    raw[:, 12:15] = relative_velocity / 600
    raw[:, 15] = np.linalg.norm(relative_position, axis=1) / 50000
    raw[:, 90] = marker
    observations = pd.DataFrame(raw, columns=motion.OBSERVATION_COLUMNS)
    tracks = pd.DataFrame(
        np.column_stack((own_position, own_velocity, target_position, target_velocity)),
        columns=motion.STATE_COLUMNS,
    )
    for frame in (observations, tracks):
        frame.insert(0, "time_s", np.arange(frames) * motion.DT)
        frame.insert(0, "step", np.arange(frames))
        frame.insert(0, "episode_id", episode_id)
    tracks["observer_alive"] = True
    tracks["target_alive"] = True
    return observations, tracks


def test_exact_future_relative_position_velocity_and_causal_inputs():
    observations, tracks = _perspective("episode", "A01")
    features, ends, target, residual, valid, baseline = motion.episode_examples(
        observations, tracks
    )
    assert features.shape == (90, 91)
    assert ends[0] == 20
    assert valid[0].tolist() == [True, True, True]
    future = ends[0] + motion.HORIZON_STEPS
    expected = tracks.loc[
        future, [f"target_{field}" for field in ("x", "y", "z", "vx", "vy", "vz")]
    ].to_numpy(copy=True)
    expected -= tracks.loc[
        future, [f"observer_{field}" for field in ("x", "y", "z", "vx", "vy", "vz")]
    ].to_numpy()
    np.testing.assert_allclose(target[0], expected, atol=1e-3)
    np.testing.assert_allclose(target[valid], (baseline + residual)[valid], atol=1e-3)

    changed = observations.copy()
    changed.loc[ends[0] + 1:, "obs_90"] = 999
    changed_features, changed_ends, changed_target, *_ = motion.episode_examples(changed, tracks)
    np.testing.assert_array_equal(changed_ends, ends)
    np.testing.assert_array_equal(changed_target, target)
    np.testing.assert_array_equal(
        changed_features[ends[0] - motion.HISTORY_STEPS + 1:ends[0] + 1],
        features[ends[0] - motion.HISTORY_STEPS + 1:ends[0] + 1],
    )


def _write_row_groups(path, tables):
    path.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(path, pa.Table.from_pandas(tables[0], preserve_index=False).schema) as writer:
        for frame in tables:
            table = pa.Table.from_pandas(frame, preserve_index=False)
            writer.write_table(table, row_group_size=len(table))


def _corpus(tmp_path):
    corpus = tmp_path / "corpus"
    cases, episode_rows = [], []
    per_side = {side: {"observations": [], "tracks": []} for side in motion.SIDES}
    for index in range(6):
        episode_id = f"flight_{index:02d}"
        group = f"group_{index:02d}"
        cases.append(
            {
                "episode_id": episode_id,
                "scenario_group": group,
                "initial": {
                    "A01": {"position": [0, 0, 7000]},
                    "B01": {"position": [10000 + index * 1000, 0, 7500]},
                },
            }
        )
        for side in motion.SIDES:
            observations, tracks = _perspective(episode_id, side, marker=float(index >= 4) * 100)
            per_side[side]["observations"].append(observations)
            per_side[side]["tracks"].append(tracks)
            target = "B01" if side == "A01" else "A01"
            episode_rows.append(
                {
                    "episode_id": episode_id,
                    "scenario_group": group,
                    "seed": index,
                    "observer_id": side,
                    "target_id": target,
                    "pilot_id": f"pilot_{index:03d}",
                    "frames": len(tracks),
                    "shard": "part-00000",
                }
            )
    config = {
        "sample_interval_s": motion.DT,
        "decision_interval_s": 1.0,
        "flights": len(cases),
        "cases": cases,
    }
    schema = {
        "schema_version": "nn-duels-varied-v1",
        "observation_columns": list(motion.OBSERVATION_COLUMNS),
        "features": observation_feature_schema(91),
    }
    motion.write_json(corpus / "config.json", config)
    motion.write_json(corpus / "schema.json", schema)
    pd.DataFrame(episode_rows).to_parquet(corpus / "episodes.parquet", index=False)
    for side in motion.SIDES:
        for part in ("observations", "tracks"):
            _write_row_groups(
                corpus / side / part / "part-00000.parquet", per_side[side][part]
            )
    files = {
        path.relative_to(corpus).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": motion.sha256(path),
        }
        for path in corpus.rglob("*")
        if path.is_file()
    }
    manifest = {
        "name": "nn-duels-varied-v1",
        "complete": True,
        "requested_flights": len(cases),
        "physical_flights": len(cases),
        "perspective_episodes": 2 * len(cases),
        "config_sha256": motion.sha256(corpus / "config.json"),
        "files": files,
    }
    motion.write_json(corpus / "manifest.json", manifest)
    return corpus


def test_prepare_grouped_91_feature_dataset_and_reload(tmp_path):
    corpus = _corpus(tmp_path)
    analysis = motion.analyze_corpus(corpus)
    output = tmp_path / "prepared"
    manifest = motion.prepare_dataset(ROOT, output, corpus, analysis=analysis)
    data = motion.RelativeMotionData(output)

    assert manifest["data_contract"] == motion.DATA_CONTRACT
    assert manifest["source_schema_version"] == "nn-duels-varied-v1"
    assert len(manifest["features"]) == data.features.shape[1] == 91
    assert manifest["history_steps"] == motion.HISTORY_STEPS == 20
    assert data.targets.shape[1:] == (3, 6)
    assert all(len(data.indices[split]) for split in motion.SPLITS)
    assignments = pd.read_csv(output / "split_assignments.csv")
    assert assignments.groupby("episode_id").observer_id.nunique().eq(2).all()
    assert assignments.groupby("episode_id").split.nunique().eq(1).all()
    assert assignments.groupby("scenario_group").split.nunique().eq(1).all()
    assert motion.nonoverlap_audit(data.samples)["verified"]

    for _, rows in data.samples.groupby(["episode_id", "step"]):
        if len(rows) == 2:
            left, right = rows.index
            np.testing.assert_allclose(data.targets[left], -data.targets[right], atol=1e-3)
    raw = data.history(data.indices["train"][:2], standardize=False)
    assert raw.shape == (2, 20, 91)


def test_notebook_33_is_clean_and_compiles():
    notebook = json.loads(
        (ROOT / "notebooks/33_prepare_relative_motion_forecasting.ipynb").read_text()
    )
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "BVR_RELATIVE_MOTION_CORPUS" in source
    assert "prepare_dataset" in source and "RelativeMotionData" in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["execution_count"] is None
            assert not cell["outputs"]


@pytest.mark.parametrize("architecture", [
    gnn.ARCHITECTURE, gnn.TRANSFORMER_ARCHITECTURE, gnn.HORIZON_TCN_ARCHITECTURE,
])
def test_hgt_regression_train_reload_and_final_metrics(tmp_path, architecture):
    torch.set_num_threads(1)
    corpus = _corpus(tmp_path)
    prepared = tmp_path / "prepared_gnn"
    motion.prepare_dataset(ROOT, prepared, corpus)
    data = motion.RelativeMotionData(prepared)
    config = {
        **gnn.default_training_config(architecture),
        "width": 12,
        "graph_width": 8,
        "graph_layers": 1,
        "graph_heads": 2,
        "dropout": 0.0,
        "epochs": 1,
        "batch_size": 8,
        "evaluation_batch_size": 8,
        "early_stopping": False,
        "precision": "fp32",
        "cpu_threads": 1,
    }
    if architecture == gnn.TRANSFORMER_ARCHITECTURE:
        config.update(transformer_layers=1, transformer_heads=2, transformer_ff_width=24)
    else:
        config["tcn_dilations"] = [1, 2, 4]
    model = gnn.make_model(config, data)
    ids = data.indices["train"][:2]
    inputs = torch.from_numpy(data.history(ids).astype(np.float32))
    prediction = model(inputs)
    assert prediction.shape == (len(ids), 3, 6)
    target = torch.from_numpy((data.residuals[ids] / data.target_scale).astype(np.float32))
    valid = torch.from_numpy(data.valid[ids])
    loss = gnn.masked_huber_loss(prediction, target, valid)
    loss.backward()
    if architecture == gnn.ARCHITECTURE:
        assert model.head[-1].weight.grad.abs().sum() > 0
    else:
        # An unlabeled horizon contributes no data gradient.
        for section, has_labels in zip(model.sections, valid.any(0)):
            assert bool(section.head[-1].weight.grad.abs().sum() > 0) == bool(has_labels)
    assert model.messages[0].relation_message.grad.abs().sum() > 0

    run = tmp_path / "run"
    history = gnn.train_model(data, run, config, device="cpu")
    assert len(history) == 1 and (run / "best.pt").is_file()
    reloaded, checkpoint = gnn.load_selected_model(data, run)
    assert checkpoint["model_contract"] == gnn.MODEL_CONTRACTS[architecture]
    repeated, _ = gnn.load_selected_model(data, run)
    with torch.inference_mode():
        torch.testing.assert_close(reloaded(inputs), repeated(inputs), rtol=0, atol=0)
    metrics = gnn.final_evaluation(data, run, batch_size=8)
    assert set(metrics.split) == set(motion.SPLITS)
    assert set(metrics.horizon_s) == set(motion.HORIZONS.tolist())
    assert np.isfinite(metrics.select_dtypes(include="number")).all().all()
    assert all((run / f"predictions_{split}.npz").is_file() for split in motion.SPLITS)


def test_hgt_transformer_sections_are_independent_and_causal():
    torch.set_num_threads(1)
    model = gnn.RelativeMotionHGTTransformer(
        width=8, graph_width=8, graph_heads=2, graph_layers=1,
        transformer_heads=2, transformer_ff_width=16, dropout=0.0,
    ).eval()
    windows = torch.randn(2, motion.HISTORY_STEPS, 91)
    with torch.inference_mode():
        frames = model.encode_frames(windows)
        changed = frames.clone()
        changed[:, -1] += torch.randn_like(changed[:, -1])
        for section in model.sections:
            original = section.encode(frames, model.causal_mask)
            updated = section.encode(changed, model.causal_mask)
            torch.testing.assert_close(original[:, :-1], updated[:, :-1])
            assert not torch.allclose(original[:, -1], updated[:, -1])

        prediction = model(windows)
        model.sections[1].head[-1].bias.add_(1)
        expected = prediction.clone()
        expected[:, 1] += 1
        torch.testing.assert_close(model(windows), expected)


def test_horizon_tcns_are_independent_and_causal():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = gnn.RelativeMotionHGTHorizonTCN(
        width=8, graph_width=8, graph_heads=2, graph_layers=1, dropout=0.0,
    ).eval()
    assert len(model.sections) == len(motion.HORIZONS) == 3
    windows = torch.randn(2, motion.HISTORY_STEPS, 91)
    with torch.inference_mode():
        frames = model.encode_frames(windows)
        changed = frames.clone()
        changed[:, -1] += torch.randn_like(changed[:, -1])
        for section in model.sections:
            assert section.temporal.receptive_field >= motion.HISTORY_STEPS
            original = section.temporal(frames)
            updated = section.temporal(changed)
            torch.testing.assert_close(original[:, :-1], updated[:, :-1])
            assert not torch.allclose(original[:, -1], updated[:, -1])

        prediction = model(windows)
        model.sections[1].temporal.blocks[-1].convolutions[-1].bias[0].add_(1)
        changed_prediction = model(windows)
        torch.testing.assert_close(prediction[:, (0, 2)], changed_prediction[:, (0, 2)])
        assert not torch.allclose(prediction[:, 1], changed_prediction[:, 1])


@pytest.mark.parametrize(("filename", "architecture_symbol"), [
    ("34_train_relative_motion_hgt_tcn.ipynb", "RelativeMotionHGTTCN"),
    ("35_train_relative_motion_hgt_transformer.ipynb", "TRANSFORMER_ARCHITECTURE"),
    ("36_train_relative_motion_hgt_horizon_tcn.ipynb", "HORIZON_TCN_ARCHITECTURE"),
])
def test_hgt_training_notebook_is_clean_and_compiles(filename, architecture_symbol):
    notebook = json.loads((ROOT / "notebooks" / filename).read_text())
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert architecture_symbol in source
    assert "train_model" in source and "final_evaluation" in source
    assert "BVR_RELATIVE_MOTION_DATA" in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["execution_count"] is None
            assert not cell["outputs"]
