"""Guard the distinction between a pure Transformer challenge and incumbent blends."""

import numpy as np
import pandas as pd
import pytest

from bvr_behavior_prediction.skill_forecasting.transformer_search import (
    incumbent_internal,
    mean_accuracy_interval,
    rank_confirmed,
    read_internal_member,
    shortlist_names,
)


def row(recipe, fold, accuracy, eligible, kind="primary_transformer", weight=1.0):
    return {"recipe": recipe, "fold": fold, "mean_accuracy": accuracy, "eligible": eligible,
            "kind": kind, "new_weight": weight, "gain": accuracy - 0.6,
            "nll": 1.0, "mean_macro_f1": 0.4}


def test_screen_uses_standalone_accuracy_and_keeps_families_separate():
    frame = pd.DataFrame([row("pure_high", 0, 0.65, False), row("pure_low", 0, 0.61, True),
                          row("pure_middle", 0, 0.63, False), row("hybrid", 0, 0.99, True)])
    pure, hybrid = shortlist_names(frame, ["pure_high", "pure_low", "pure_middle"], ["hybrid"])
    assert pure == ["pure_high", "pure_middle"]
    assert hybrid == ["hybrid"]


def test_primary_prefers_guarded_recipe_but_retains_failing_challenger_if_needed():
    frame = pd.DataFrame([row("higher", 0, 0.7, False), row("higher", 1, 0.7, True),
                          row("guarded", 0, 0.62, True), row("guarded", 1, 0.63, True)])
    assert rank_confirmed(frame, "primary_transformer", True)["recipe"] == "guarded"
    frame["eligible"] = False
    result = rank_confirmed(frame, "primary_transformer", True)
    assert result["recipe"] == "higher" and not result["eligible"]
    assert rank_confirmed(frame, "primary_transformer", False) is None


def test_secondary_requires_both_folds_and_cannot_mix_with_primary_ranking():
    frame = pd.DataFrame([row("pure", 0, 0.99, True), row("pure", 1, 0.99, True),
                          row("unstable", 0, 0.7, True, "secondary_blend", 0.5),
                          row("unstable", 1, 0.6, False, "secondary_blend", 0.5),
                          row("stable", 0, 0.63, True, "secondary_blend", 0.25),
                          row("stable", 1, 0.64, True, "secondary_blend", 0.25)])
    assert rank_confirmed(frame, "secondary_blend", False)["recipe"] == "stable"
    with pytest.raises(ValueError, match="both registered folds"):
        rank_confirmed(frame[frame.fold.eq(0)], "secondary_blend", False)
    with pytest.raises(ValueError, match="Duplicate"):
        rank_confirmed(pd.concat([frame, frame.iloc[[0]]]), "primary_transformer", True)


def test_mean_interval_averages_correlated_horizons_within_group_draws():
    targets = np.zeros((4, 3), dtype=np.int64)
    old = np.zeros((4, 3, 2)); new = np.zeros_like(old)
    old[:, 0, 0] = 1; new[:, 0, 1] = 1
    old[:, 1, 1] = 1; new[:, 1, 0] = 1
    old[:, 2, 0] = 1; new[:, 2, 0] = 1
    result = mean_accuracy_interval(new, old, targets,
                                    pd.DataFrame({"scenario_group": ["a", "a", "b", "b"]}), draws=100)
    assert result["mean_accuracy_gain"] == 0
    assert result["lower_95"] == result["upper_95"] == 0


def test_incumbent_is_half_each_family_not_uniform_over_fifteen(monkeypatch, tmp_path):
    from bvr_behavior_prediction.skill_forecasting import transformer_search as module

    seen = []

    def read(path, name, fold, seed, protocol, partition):
        seen.append((path, name, seed))
        probability = np.zeros((2, 3, 33), dtype=np.float32)
        probability[:, :, int(name == "tcn")] = 1
        return probability, np.array([4, 8])

    monkeypatch.setattr(module, "read_internal_member", read)
    monkeypatch.setattr(module, "read_json", lambda path: {"fit_group": "fit"})
    monkeypatch.setattr(module, "_check_selection_origins", lambda values, *args: values)
    protocol = {"baseline_mlp_components": ["a", "b", "c", "d"],
                "baseline_tcn_component": "tcn", "training_seeds": [42, 43, 44],
                "baseline_v3_directory": str(tmp_path / "v3"),
                "baseline_v4_directory": str(tmp_path / "v4")}
    probability, indices = incumbent_internal(tmp_path, protocol, 0)
    np.testing.assert_array_equal(probability[:, :, :2], np.full((2, 3, 2), 0.5))
    np.testing.assert_array_equal(indices, [4, 8])
    assert len(seen) == 15
    for path, name, seed in seen:
        expected = "v3" if name != "tcn" and seed == 42 else "v4"
        assert path.parent.parent == tmp_path / expected


def test_internal_reuse_rejects_wrong_groups_and_tampered_predictions(tmp_path):
    import torch

    from bvr_behavior_prediction.forecasting.data import sha256, write_json

    path = tmp_path / "inner_0/model_s42"
    path.mkdir(parents=True)
    candidate = {"architecture": "test"}
    protocol = {"candidates": {"model": candidate}, "data_manifest_sha256": "data"}
    write_json(tmp_path / "protocol.json", protocol)
    saved = {"candidate": candidate, "model_config": {}, "seed": 42, "epoch": 2,
             "temperature": 1.0, "scaling": {"fit_groups": ["fit"]},
             "data_manifest_sha256": "data"}
    torch.save(saved, path / "best.pt")
    probability = np.full((2, 3, 33), 1 / 33, dtype=np.float32)
    np.save(path / "selection_probabilities.npy", probability)
    np.save(path / "selection_indices.npy", np.array([3, 9]))
    run = {"candidate": candidate, "model": {}, "seed": 42, "fold": 0, "final": False,
           "fit_groups": ["fit"], "protocol_sha256": sha256(tmp_path / "protocol.json")}
    summary = {"name": "model", "seed": 42, "fold": 0, "final": False, "best_epoch": 2,
               "temperature": 1.0, "test_evaluated": False, "outer_validation_evaluated": False,
               "checkpoint_sha256": sha256(path / "best.pt"),
               "selection_probability_sha256": sha256(path / "selection_probabilities.npy"),
               "selection_indices_sha256": sha256(path / "selection_indices.npy")}
    write_json(path / "run.json", run)
    write_json(path / "summary.json", summary)
    partition = {"fit": "fit", "stop": "stop", "select": "select"}
    read_internal_member(path, "model", 0, 42, protocol, partition, current=True)
    write_json(path / "run.json", {**run, "fit_groups": ["fit", "select"]})
    with pytest.raises(ValueError, match="provenance"):
        read_internal_member(path, "model", 0, 42, protocol, partition, current=True)
    write_json(path / "run.json", run)
    np.save(path / "selection_indices.npy", np.array([9, 3]))
    with pytest.raises(ValueError, match="Cached internal prediction"):
        read_internal_member(path, "model", 0, 42, protocol, partition, current=True)
