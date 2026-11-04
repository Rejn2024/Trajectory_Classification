"""Causal label alignment, train-only priors, masked gradients and metrics."""

import numpy as np
import pandas as pd
import pytest
import torch

from bvr_behavior_prediction.skill_forecasting.data import decision_series, episode_examples
from bvr_behavior_prediction.skill_forecasting.evaluation import (
    classification_metrics,
    frequency_probabilities,
)
from bvr_behavior_prediction.skill_forecasting.models import (
    SkillForecaster,
    class_weights,
    masked_cross_entropy,
)


def episode():
    obs = np.zeros((121, 51), dtype=np.float32)
    obs[:, 15] = 1
    obs[:, 12] = np.arange(121) / 600
    track = pd.DataFrame({"observer_alive": [True] * 121, "target_alive": [True] * 121})
    decisions = pd.DataFrame({"episode_id": "example", "step": np.arange(0, 120, 10),
                              "time_s": np.arange(12, dtype=float), "skill": ["a", "b"] * 6, "aircraft_id": "A01"})
    return obs, track, decisions


def test_exact_decision_times_missing_horizons_and_no_forward_fill():
    obs, track, decisions = episode()
    _, ends, target, current = episode_examples(obs, track, decisions, ["a", "b"])
    assert ends.tolist() == [60, 70, 80, 90, 100]
    assert target[0].tolist() == [1, 0, 1]  # t=6 -> decisions at 7, 8, 11
    assert current[0] == 0
    assert target[-1].tolist() == [1, -1, -1]  # frame at 12 exists, selection does not
    missing = decisions.loc[decisions.step != 70]
    assert episode_examples(obs, track, missing, ["a", "b"])[2][0].tolist() == [-1, 0, 1]


def test_inputs_ignore_future_observations_and_all_private_labels():
    obs, track, decisions = episode()
    features, *_ = episode_examples(obs, track, decisions, ["a", "b"])
    changed = decisions.copy()
    changed["skill"] = "a"
    changed["parameters_json"] = "privileged data"
    np.testing.assert_array_equal(features, episode_examples(obs, track, changed, ["a", "b"])[0])
    later = obs.copy()
    later[61:] = 999
    np.testing.assert_array_equal(features[:61], episode_examples(later, track, decisions, ["a", "b"])[0][:61])
    # A dead extra acceleration-lookback frame invalidates the first history.
    track.loc[5, "target_alive"] = False
    assert episode_examples(obs, track, decisions, ["a", "b"])[1][0] == 70


@pytest.mark.parametrize("change", ["duplicate", "time", "aircraft", "class"])
def test_bad_decision_records_are_rejected(change):
    _, _, decisions = episode()
    if change == "duplicate":
        decisions = pd.concat([decisions, decisions.iloc[:1]])
    elif change == "time":
        decisions.loc[0, "time_s"] = 0.1
    elif change == "aircraft":
        decisions.loc[0, "aircraft_id"] = "B01"
    else:
        decisions.loc[0, "skill"] = "unknown"
    with pytest.raises(ValueError):
        decision_series(decisions, ["a", "b"], 121)


@pytest.mark.parametrize("architecture", ["gru", "transformer"])
def test_classifier_and_masked_loss(architecture):
    torch.set_num_threads(1)
    model = SkillForecaster(architecture=architecture, num_classes=4, history_steps=7, width=16, layers=1)
    logits = model(torch.randn(4, 7, 54))
    assert logits.shape == (4, 3, 4)
    logits.retain_grad()
    target = torch.tensor([[1, -1, -1]] * 4)
    loss = masked_cross_entropy(logits, target)
    loss.backward()
    assert logits.grad[:, 0].abs().sum() > 0
    assert logits.grad[:, 1:].abs().sum() == 0
    altered = logits.detach().clone()
    altered[:, 1:] = 999
    torch.testing.assert_close(masked_cross_entropy(altered, target), loss.detach())
    assert masked_cross_entropy(logits, torch.full_like(target, -1)).item() == 0


def test_frequency_and_optional_weights_use_only_supplied_training_support():
    targets = np.array([[0, 1, 2], [0, -1, -1], [1, 1, 2]])
    priors = frequency_probabilities(targets, 4)
    np.testing.assert_allclose(priors[0], [2 / 3, 1 / 3, 0, 0])
    assert class_weights(torch.tensor(targets), 4) is None
    weights = class_weights(torch.tensor(targets), 4, "sqrt_inverse", max_weight=2)
    assert torch.isfinite(weights).all() and weights.max() <= 2
    assert weights[0, 3] == 0 and weights[0, 1] > weights[0, 0]


def test_metrics_have_fixed_class_denominator_and_change_strata():
    samples = pd.DataFrame({"pilot_id": ["p1", "p1", "p2"]})
    target = np.array([[0, 0, -1], [1, 1, 1], [1, 1, 1]])
    probability = np.broadcast_to([0.7, 0.2, 0.1, 0], (3, 3, 4))
    summary, per_class, matrices = classification_metrics(
        probability, target, np.array([0, 0, -1]), samples, list("abcd"), "baseline")
    row = summary.query("pilot_id == 'population' and cohort == 'all' and horizon_s == 1").iloc[0]
    assert row.top1_accuracy == pytest.approx(1 / 3)
    assert row.top3_accuracy == 1
    assert row.macro_f1_catalogue == pytest.approx(0.5 / 4)
    assert row.macro_f1_supported == pytest.approx(0.5 / 2)
    assert matrices["population_5s"].sum() == 2
    groups = summary.query("pilot_id == 'population' and horizon_s == 1").set_index("cohort")
    assert groups.loc[["same", "changed", "unknown_current"], "count"].tolist() == [1, 1, 1]
    assert len(per_class) == 3 * 3 * 4


def test_preparation_keeps_test_unread_and_fits_only_training_histories(tmp_path):
    from pathlib import Path

    import pyarrow as pa
    import pyarrow.parquet as pq

    from bvr_behavior_prediction.forecasting.data import (
        grouped_split,
        read_json,
        sha256,
        write_json,
    )
    from bvr_behavior_prediction.skill_forecasting.data import prepare_dataset

    corpus, output = tmp_path / "corpus", tmp_path / "prepared"
    cases = [{"scenario_group": f"g{i:02}", "scenario": {"range_m": 18000}} for i in range(10)]
    groups = grouped_split(cases)
    reference = tmp_path / "reference.json"
    write_json(reference, groups)
    write_json(corpus / "config.json", {"observer_id": "B01", "target_id": "A01",
                                        "sample_interval_s": 0.1, "cases": cases})
    write_json(corpus / "schema.json", {})
    writers, episodes = {}, []
    for group, assignment in sorted(groups.items()):
        obs, track, decisions = episode()
        split = assignment["split"]
        episode_id = f"pilot_001_{group}"
        if split == "validation":
            obs[:, 0] = 900  # Would contaminate scaling if accidentally used.
        elif split == "test":
            obs[:] = np.nan  # Must never reach episode validation/preparation.
        observations = pd.DataFrame(obs, columns=[f"obs_{i:02}" for i in range(51)])
        for frame in (observations, track):
            frame["episode_id"] = episode_id
            frame["step"] = np.arange(121)
            frame["time_s"] = np.arange(121) * 0.1
        decisions["episode_id"] = episode_id
        decisions["skill"] = decisions.skill.map({"a": "lag_pursuit", "b": "turn_cold"})
        if split == "test":
            decisions["skill"] = "not_in_catalogue"
        for part, frame in [("observations", observations), ("tracks", track), ("target_decisions", decisions)]:
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if part not in writers:
                (corpus / part).mkdir()
                writers[part] = pq.ParquetWriter(corpus / part / "pilot_001.parquet", table.schema)
            writers[part].write_table(table)
        episodes.append({"episode_id": episode_id, "pilot_id": "pilot_001",
                         "scenario_group": group, "frames": 121})
    for writer in writers.values():
        writer.close()
    pd.DataFrame(episodes).to_parquet(corpus / "episodes.parquet", index=False)
    write_json(corpus / "manifest.json", {"complete": True, "episodes": 10,
        "files": {p.relative_to(corpus).as_posix(): {"sha256": sha256(p)}
                  for p in corpus.rglob("*") if p.is_file()}})
    manifest = prepare_dataset(corpus, output, reference)
    assert manifest["counts"]["test"]["samples"] == 0
    assert read_json(output / "scaling.json")["feature_mean"][0] == 0
    samples = pd.read_parquet(output / "samples.parquet")
    assert set(samples.split) == {"train", "validation"}
    support = pd.read_csv(output / "label_support.csv")
    assert len(support) == 2 * 2 * 3 * 33
    with pytest.raises(ValueError, match="empty"):
        prepare_dataset(corpus, output, reference)

    from bvr_behavior_prediction.skill_forecasting.inference import SkillPredictor
    from bvr_behavior_prediction.skill_forecasting.train import train_model, verify_run

    config = {"device": "cpu", "cpu_threads": 1, "seed": 42, "epochs": 1, "patience": 2,
              "min_delta": 0.001, "batch_size": 16, "evaluation_batch_size": 16,
              "learning_rate": 0.001, "weight_decay": 0.01, "gradient_clip": 1,
              "class_weighting": "none", "max_class_weight": 5,
              "model": {"width": 16, "layers": 1, "heads": 4, "dropout": 0}}
    run = tmp_path / "experiment" / "gru"
    root = Path(__file__).resolve().parents[1]
    result = train_model(root, output, run, "gru", config)
    assert result["epochs_completed"] == 1 and not result["test_evaluated"]
    assert verify_run(output, run)["validation_probabilities_exact"]
    predictor = SkillPredictor(run / "best.pt")
    raw, _, _ = episode()
    raw[:, 0] = 900
    prediction = predictor.predict(raw[:61])
    np.testing.assert_allclose(prediction["probabilities"],
                               np.load(run / "validation_probabilities.npy")[0], atol=2e-7)
    assert prediction["top3_skills"].shape == (3, 3)
    with pytest.raises(ValueError, match="at least 56"):
        predictor.predict(raw[:51])
    with pytest.raises(ValueError, match="empty"):
        train_model(root, output, run, "gru", config)
    assert train_model(root, output, run, "gru", config, resume=True)["best_epoch"] == 1
    with pytest.raises(ValueError, match="differs"):
        train_model(root, output, run, "gru", {**config, "seed": 43}, resume=True)
