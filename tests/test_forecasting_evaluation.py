"""Cached/streaming equivalence, bounded reuse, numerical diagnostics and display cadence."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from bvr_behavior_prediction import paired_skill_forecasting as p
from bvr_behavior_prediction import skill_intent_forecasting as intent
from bvr_behavior_prediction.enemy_missile_models import HGTHierarchicalIntentTransformer


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def example(classes=None, device="cpu"):
    classes = classes or ["beam_target_left", "beam_target_right", "launch", "notch_left", "notch_right"]
    policy, intent_policy = intent.hierarchy_policy(classes), intent.intent_policy(classes)
    rng = np.random.default_rng(42)
    features = rng.normal(0, .1, (480, 82 + len(classes))).astype(np.float32)
    features[:, 3], features[:, 6] = .8, .9
    features[:, 82:] = 0
    features[:, 82] = 1
    features[:, 23:30] = features[:, 61:68] = 0
    mean = rng.normal(0, .01, features.shape[1]).astype(np.float32)
    std = rng.uniform(.5, 1.5, features.shape[1]).astype(np.float32)
    targets = rng.integers(0, len(classes), (9, 3))
    targets[0, 1] = -1
    targets[:, 2] = classes.index("launch")  # An entirely empty direction horizon.
    anchors = np.array([50, 60, 70, 250, 260, 270, 450, 460, 470])
    data = SimpleNamespace(features=features, mean=mean, std=std, anchors=anchors, classes=classes,
        targets=targets, family_targets=intent.map_intent_targets(targets, policy["skill_to_family"]),
        direction_targets=intent.map_intent_targets(targets, policy["skill_to_direction"]),
        intent_targets=intent.map_intent_targets(targets, intent_policy["skill_to_intent"]),
        family_classes=policy["families"], direction_classes=["left", "right"], intent_classes=intent.INTENT_CLASSES,
        manifest={"features": [f"f{i}" for i in range(features.shape[1])], "skill_hierarchy": policy},
        samples=pd.DataFrame(dict(episode_id=np.repeat([1, 2, 3], 3), observer_id=["A01"]*6 + ["B01"]*3)),
        indices={"test": np.arange(9), "train": np.array([7, 0, 4, 1, 3, 2, 8, 5, 6])})

    def batch(ids, selected_device):
        steps = anchors[ids, None] + np.arange(1-p.HISTORY_STEPS, 1)
        return torch.from_numpy((features[steps]-mean)/std).to(selected_device), torch.as_tensor(targets[ids], device=selected_device)
    data.batch = batch
    model = HGTHierarchicalIntentTransformer(num_classes=len(classes), input_features=82+len(classes),
        skill_hierarchy=policy, width=12, heads=6, layers=1, graph_width=8, graph_heads=2,
        graph_layers=2, dropout=.25, observer_acceleration=True, target_relative_missiles=True,
        target_relative_missile_velocities=True, mean=mean, std=std).to(device)
    return data, model


def assert_equivalent(a, b, full=False, tolerance=2e-6):
    for task in (None, "intent", "family", "direction"):
        left, right = (a, b) if task is None else (a[task], b[task])
        for table in ("summary", "per_class"):
            pd.testing.assert_frame_equal(left[table], right[table], atol=tolerance, rtol=tolerance)
        np.testing.assert_array_equal(left["confusion"], right["confusion"])
        if full:
            np.testing.assert_allclose(left["probabilities"], right["probabilities"], atol=tolerance, rtol=tolerance)
            np.testing.assert_array_equal(left["targets"], right["targets"])
    for key in ("data_loss", "joint_skill_data_loss", "skill_data_loss", "family_data_loss", "handedness_data_loss"):
        assert a[key] == pytest.approx(b[key], abs=tolerance, rel=tolerance)
    pd.testing.assert_frame_equal(a["hierarchical_reward"], b["hierarchical_reward"], atol=tolerance, rtol=tolerance)
    np.testing.assert_array_equal(a["sample_indices"], b["sample_indices"])


@pytest.mark.parametrize("loss", ["joint_nll", "partial_credit"])
@pytest.mark.parametrize("batch_size,cap", [(1, 51), (4, 153), (7, 65536)])
def test_cached_streaming_and_full_predictions_match_existing_reports(loss, batch_size, cap):
    torch.manual_seed(4)
    data, model = example()
    kwargs = dict(evaluation_attention="math", intent_loss_weight=.25, hierarchical_loss=loss,
        family_reward=2., handedness_reward=3., class_weights=np.arange(15).reshape(3, 5)/10,
        handedness_weights=torch.tensor([[[.3, 2.1], [0, 1.]], [[1., 2.], [.5, 3.]], [[0., 0.], [0., 0.]]]))
    baseline = p.evaluate(model, data, "train", "cpu", batch_size=3, **kwargs)
    for cached in (False, True):
        streamed = p.evaluate(model, data, "train", "cpu", batch_size=batch_size, streaming=True,
            frame_cache=cached, cache_max_frames=cap, frame_batch_size=17, **kwargs)
        assert_equivalent(baseline, streamed)
        for task in (streamed, streamed["intent"], streamed["family"], streamed["direction"]):
            assert "probabilities" not in task and "losses" not in task
    full = p.evaluate(model, data, "train", "cpu", batch_size=batch_size, frame_cache=True,
        cache_max_frames=cap, frame_batch_size=17, **kwargs)
    assert_equivalent(baseline, full, full=True)


def test_cache_encodes_unique_rows_in_bounded_chunks_and_is_rebuilt_after_weight_changes(monkeypatch):
    data, model = example()
    counts = []
    original = model.encode_frame_rows
    def encode(frames):
        counts.append(len(frames))
        return original(frames)
    monkeypatch.setattr(model, "encode_frame_rows", encode)
    kwargs = dict(evaluation_attention="math", frame_cache=True, cache_max_frames=153, frame_batch_size=17, batch_size=3)
    first = p.evaluate(model, data, "test", "cpu", **kwargs)
    assert sum(counts) == 3*71 < 9*51 and max(counts) <= 17
    with torch.no_grad():
        model.projection.bias[0].add_(1.)
    counts.clear()
    second = p.evaluate(model, data, "test", "cpu", **kwargs)
    assert sum(counts) == 3*71
    assert not np.allclose(first["probabilities"], second["probabilities"])
    uncached = p.evaluate(model, data, "test", "cpu", evaluation_attention="math", batch_size=3)
    assert_equivalent(uncached, second, full=True)


def test_streaming_copies_only_status_flags_and_end_of_split_totals(monkeypatch):
    data, model = example()
    original_cpu = torch.Tensor.cpu
    transfers = []
    def cpu(tensor, *args, **kwargs):
        transfers.append((tensor.numel(), tensor.dtype))
        return original_cpu(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    monkeypatch.setattr(p, "_check_evaluation_tensor", lambda *args: pytest.fail("Detailed checks run only on failure"))
    p.evaluate(model, data, "test", "cpu", batch_size=2, streaming=True)
    assert transfers[:10] == [(1, torch.bool), (10, torch.bool)]*5
    assert len(transfers) == 12 and [x[1] for x in transfers[-2:]] == [torch.int64, torch.float64]


def test_cached_bad_input_identifies_all_affected_histories_and_original_feature():
    data, model = example()
    data.features[60, 3] = np.nan  # Histories anchored at 60 and 70 share this row.
    with pytest.raises(p.NonfiniteEvaluationError) as caught:
        p.evaluate(model, data, "test", "cpu", frame_cache=True, streaming=True,
                    batch_size=3, cache_max_frames=153, frame_batch_size=17)
    details = caught.value.details
    assert details["stage"] == "inputs" and details["frame_cache"]
    assert details["sample_indices"] == [1, 2] and details["frame_indices"] == [60]
    assert details["feature_names"] == ["f3"]
    assert [row["observer_id"] for row in details["samples"]] == ["A01", "A01"]


def test_cache_ignores_rows_outside_selected_histories_and_diagnoses_encoder_failure():
    data, model = example()
    data.features[100] = np.nan  # Outside all selected histories, so never encoded.
    baseline = p.evaluate(model, data, "test", "cpu", streaming=True, frame_cache=True)
    assert np.isfinite(baseline["summary"].nll).all()
    def invalid_encoding(module, inputs, output):
        return output * float("nan")
    model.projection.register_forward_hook(invalid_encoding)
    with pytest.raises(p.NonfiniteEvaluationError) as caught:
        p.evaluate(model, data, "test", "cpu", streaming=True, frame_cache=True,
                    batch_size=3, cache_max_frames=153, frame_batch_size=17)
    assert caught.value.details["stage"] == "frame_encoder_output"
    assert caught.value.details["frame_cache"] and caught.value.details["sample_indices"] == [0, 1]


@pytest.mark.parametrize("streaming", [False, True])
def test_batched_numerical_checks_keep_first_failure_and_sample_metadata(streaming):
    data, model = example()
    def nan_output(module, inputs, output):
        return output * float("nan")
    model.family_head.register_forward_hook(nan_output)
    with pytest.raises(p.NonfiniteEvaluationError) as caught:
        p.evaluate(model, data, "test", "cpu", streaming=streaming, batch_size=2)
    assert caught.value.details["stage"] == "logits"
    assert caught.value.details["sample_indices"] == [0, 1]


def test_no_mirrored_families_have_empty_direction_reports_with_streaming_cache():
    data, model = example(["launch", "maintain_heading"])
    baseline = p.evaluate(model, data, "test", "cpu")
    fast = p.evaluate(model, data, "test", "cpu", streaming=True, frame_cache=True, frame_batch_size=23)
    assert_equivalent(baseline, fast)
    assert not fast["direction"]["summary"]["count"].any()


def test_cpu_normalized_frame_cache_and_direct_rows_give_equivalent_evaluation():
    data, model = example()
    baseline = p.evaluate(model, data, "test", "cpu")
    pipeline = SimpleNamespace(cache=p.TensorFrameCache(data, "cpu", chunk_frames=19))
    fast = p.evaluate(model, data, "test", "cpu", frame_cache=True, streaming=True, pipeline=pipeline)
    assert_equivalent(baseline, fast)


def test_per_class_display_every_five_epochs_and_at_normal_or_early_completion():
    path = Path(__file__).resolve().parents[1] / "notebooks/27_skill_intent_target_relative_missile_hgt_nn_duel_skill_forecasting.ipynb"
    source = "".join(json.loads(path.read_text())["cells"][3]["source"]).removeprefix("%%time\n").split("tracking = ")[0]
    namespace = dict(pd=pd, CONFIG={"epochs": 7}, display=lambda *args: None)
    exec(compile(source, "notebook27-callback", "exec"), namespace)
    shown = []
    namespace["display_per_class"] = lambda table, epoch: shown.append(epoch)
    summary = pd.DataFrame([dict(split="test", horizon_s=1, nll=1., accuracy=.5, macro_f1=.5,
                                  supported_macro_f1=.5, weighted_f1=.5)])
    row = dict.fromkeys(("train_nll", "test_nll", "train_data_loss", "train_intent_nll", "train_family_nll",
        "train_handedness_data_loss", "test_hierarchical_credit", "train_l1_penalty", "generalization_gap",
        "learning_rate", "overtraining_epochs", "seconds"), 0.)
    for epoch in range(1, 8):
        namespace["epoch_output"]({**row, "epoch": epoch, "early_stopped": False}, summary, None)
    assert shown == [5, 7]
    shown.clear()
    namespace["epoch_output"]({**row, "epoch": 3, "early_stopped": True}, summary, None)
    assert shown == [3]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA parity needs a GPU")
def test_cuda_streaming_cached_full_precision_metrics_match():
    data, model = example(device="cuda")
    baseline = p.evaluate(model, data, "test", "cuda", evaluation_attention="math", batch_size=3)
    fast = p.evaluate(model, data, "test", "cuda", evaluation_attention="math", batch_size=4,
                      streaming=True, frame_cache=True, frame_batch_size=23)
    assert_equivalent(baseline, fast, tolerance=3e-5)
