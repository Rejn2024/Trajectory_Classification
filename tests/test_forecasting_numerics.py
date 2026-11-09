"""Evaluation precision, attention policy and actionable nonfinite failures."""

import json
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from bvr_behavior_prediction import paired_skill_forecasting as paired


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class FixedLogits(torch.nn.Module):
    num_classes = 2

    def __init__(self, values=(0., 1.)):
        super().__init__()
        self.values = values
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x.new_tensor(self.values).expand(x.shape[0], 3, 2).clone()


def examples():
    x = torch.zeros(10, 51, 2)
    targets = np.zeros((10, 3), dtype=np.int64)
    targets[7, 1] = -1
    data = SimpleNamespace(
        classes=["left", "right"], targets=targets,
        indices={"test": np.array([4, 7, 9])},
        manifest={"features": ["position", "speed"]},
        samples=pd.DataFrame({"episode_id": [f"episode_{i}" for i in range(10)],
                              "step": np.arange(10) + 60}),
    )
    data.batch = lambda ids, device: (x[ids].to(device),
                                    torch.tensor(targets[ids], device=device))
    return data, x


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_nonfinite_inputs_report_split_batch_and_sample_ids_before_forward(bad_value):
    data, x = examples()
    x[7, 12, 1] = bad_value
    model = FixedLogits()
    with pytest.raises(paired.NonfiniteEvaluationError) as caught:
        paired.evaluate(model, data, "test", "cpu", batch_size=2)
    details = caught.value.details
    assert details["stage"] == "inputs"
    assert details["split"] == "test"
    assert details["batch_index"] == 0
    assert details["sample_indices"] == [7]
    assert details["nonfinite_count"] == 1
    assert details["feature_indices"] == [1]
    assert details["feature_names"] == ["speed"]
    assert model.calls == 0
    json.dumps(details, allow_nan=False)


def test_nonfinite_logits_cannot_reach_aggregation():
    data, _ = examples()
    with pytest.raises(paired.NonfiniteEvaluationError) as caught:
        paired.evaluate(FixedLogits((float("nan"), 1.)), data, "test", "cpu", batch_size=2)
    assert caught.value.details["stage"] == "logits"
    assert caught.value.details["sample_indices"] == [4, 7]


def test_finite_extreme_logits_report_overflow_in_cross_entropy():
    data, _ = examples()
    # Both logits fit FP32, but the loss for class zero exceeds its finite range.
    model = FixedLogits((-3e38, 3e38))
    with pytest.raises(paired.NonfiniteEvaluationError) as caught:
        paired.evaluate(model, data, "test", "cpu", batch_size=2)
    assert caught.value.details["stage"] in ("loss", "losses")


def test_probability_kernel_failure_is_reported(monkeypatch):
    data, _ = examples()
    monkeypatch.setattr(torch.Tensor, "softmax", lambda tensor, *args, **kwargs:
                        torch.full_like(tensor, float("nan")))
    with pytest.raises(paired.NonfiniteEvaluationError) as caught:
        paired.evaluate(FixedLogits(), data, "test", "cpu", batch_size=2)
    assert caught.value.details["stage"] == "probabilities"


@pytest.mark.parametrize("raises", [False, True])
def test_math_attention_restores_backend_state_even_on_error(raises):
    original_mha = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(True)
    flags = (torch.backends.cuda.flash_sdp_enabled(),
             torch.backends.cuda.mem_efficient_sdp_enabled(),
             torch.backends.cuda.math_sdp_enabled())
    try:
        try:
            with paired.evaluation_context("cpu", "math"):
                assert not torch.backends.mha.get_fastpath_enabled()
                assert torch.backends.cuda.math_sdp_enabled()
                assert not torch.backends.cuda.flash_sdp_enabled()
                assert not torch.backends.cuda.mem_efficient_sdp_enabled()
                if raises:
                    raise RuntimeError("attention-context-test")
        except RuntimeError as error:
            assert raises and str(error) == "attention-context-test"
        assert torch.backends.mha.get_fastpath_enabled()
        assert flags == (torch.backends.cuda.flash_sdp_enabled(),
                         torch.backends.cuda.mem_efficient_sdp_enabled(),
                         torch.backends.cuda.math_sdp_enabled())
    finally:
        torch.backends.mha.set_fastpath_enabled(original_mha)


def test_evaluation_precision_overrides_training_precision(monkeypatch):
    data, _ = examples()
    requested = []

    def autocast(device, precision):
        requested.append(precision)
        return nullcontext()

    monkeypatch.setattr(paired, "autocast_context", autocast)
    result = paired.evaluate(FixedLogits(), data, "test", "cpu", precision="bf16",
                             evaluation_precision="fp32", evaluation_attention="math")
    assert requested == ["fp32"]
    assert np.isfinite(result["summary"].nll).all()


def test_stable_evaluation_preserves_weighted_masks_and_predictions():
    data, _ = examples()
    model = FixedLogits()
    weights = np.array([[.5, 2.], [2., .5], [1., 1.]])
    plain = paired.evaluate(model, data, "test", "cpu", batch_size=2)
    stable = paired.evaluate(model, data, "test", "cpu", batch_size=2,
                             evaluation_precision="fp32", evaluation_attention="math",
                             class_weights=weights)
    np.testing.assert_array_equal(stable["targets"], data.targets[data.indices["test"]])
    np.testing.assert_array_equal(stable["probabilities"], plain["probabilities"])
    pd.testing.assert_frame_equal(stable["summary"], plain["summary"])
    x, y = data.batch(data.indices["test"], "cpu")
    expected = paired.masked_loss(model(x), y, torch.tensor(weights)).item()
    assert stable["data_loss"] == pytest.approx(expected)


def test_nll_mean_does_not_overflow_for_finite_fp32_losses():
    labels = np.zeros((3, 3), dtype=np.int64)
    probabilities = np.tile(np.array([0., 1.], dtype=np.float32), (3, 3, 1))
    losses = np.full((3, 3), 2e38, dtype=np.float32)
    summary, scores, _ = paired.metric_tables(labels, probabilities, ["left", "right"],
                                             "test", losses=losses)
    assert np.isfinite(summary.nll).all()
    np.testing.assert_allclose(summary.nll, float(losses[0, 0]), rtol=1e-6)
    assert np.isfinite(scores.loc[scores.support.gt(0), "class_nll"]).all()


def training_data(tmp_path):
    directory = tmp_path / "prepared"
    directory.mkdir()
    paired.write_json(directory / "manifest.json", {"fixture": "finite evaluation"})
    data = SimpleNamespace(
        directory=directory, classes=["left", "right"],
        mean=np.zeros(len(paired.FEATURES), np.float32),
        std=np.ones(len(paired.FEATURES), np.float32),
        targets=np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
        indices={"train": np.array([0]), "test": np.array([1]), "validation": np.array([2])},
    )
    data.batch = lambda ids, device: (
        torch.zeros(len(ids), 51, len(paired.FEATURES), device=device),
        torch.tensor(data.targets[ids], device=device),
    )
    return data


def test_trainer_uses_default_fp32_math_evaluation(tmp_path, monkeypatch):
    data = training_data(tmp_path)
    original = paired.evaluate
    requested = []

    def observed(model, data, split, device, *args, **kwargs):
        requested.append((split, kwargs.get("precision"), kwargs.get("evaluation_attention")))
        return original(model, data, split, device, *args, **kwargs)

    monkeypatch.setattr(paired, "evaluate", observed)
    paired.train_model(data, tmp_path / "run", {
        "epochs": 1, "width": 12, "heads": 6, "layers": 1, "dropout": 0.,
        "precision": "bf16", "data_backend": "direct", "cpu_threads": 1,
    }, device="cpu")
    assert requested == [("train", "fp32", "math"), ("test", "fp32", "math")]


def test_trainer_persists_nonfinite_failure_evidence(tmp_path, monkeypatch):
    data = training_data(tmp_path)
    original = paired.evaluate

    def failing(model, data, split, device, *args, **kwargs):
        if split == "test":
            model = FixedLogits((float("nan"), 1.))
        return original(model, data, split, device, *args, **kwargs)

    monkeypatch.setattr(paired, "evaluate", failing)
    run = tmp_path / "failed"
    with pytest.raises(paired.NonfiniteEvaluationError) as caught:
        paired.train_model(data, run, {
            "epochs": 1, "width": 12, "heads": 6, "layers": 1, "dropout": 0.,
            "data_backend": "direct", "cpu_threads": 1,
        }, device="cpu")
    saved = paired.read_json(run / "nonfinite_evaluation.json")
    assert saved["stage"] == caught.value.details["stage"] == "logits"
    assert saved["split"] == "test" and saved["sample_indices"] == [1]
    assert not (run / "best.pt").exists()


def test_cpu_transformer_fp32_math_attention_is_finite():
    model = paired.PairedTransformer(num_classes=2, width=12, heads=6, layers=1,
                                      dropout=.25, input_features=2)
    data, x = examples()
    x[:] = torch.randn_like(x) * 100
    result = paired.evaluate(model, data, "test", "cpu", batch_size=2,
                             evaluation_precision="fp32", evaluation_attention="math")
    assert np.isfinite(result["probabilities"]).all()
    assert np.isfinite(result["summary"].nll).all()
