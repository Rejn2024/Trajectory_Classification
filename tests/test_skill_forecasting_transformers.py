"""Observation-history boundaries and attention behavior for Transformer recipes."""

import numpy as np
import pytest
import torch

from bvr_behavior_prediction.skill_forecasting.improved_features import (
    history_offsets,
    raw_history_features,
)
from bvr_behavior_prediction.skill_forecasting.transformer_models import (
    MaskedAttentionBlock,
    TransformerForecaster,
    history_patch_indices,
)

ARCHITECTURES = ["temporal_transformer", "patch_transformer", "hybrid_transformer"]


@pytest.fixture(autouse=True)
def small_deterministic_models():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(1729)
    yield
    torch.set_num_threads(previous)


def make_model(architecture, tokens=26, auxiliary=False):
    return TransformerForecaster(architecture, 7, tokens, width=16, layers=2,
                                 dropout=0.1, auxiliary=auxiliary)


def inputs(tokens=26):
    features = torch.randn(3, tokens, 7)
    valid = torch.ones(3, tokens, dtype=torch.bool)
    valid[0, :tokens - 3] = False
    valid[1, :tokens // 2] = False
    return features, valid


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("tokens", [26, 66])
@pytest.mark.parametrize("auxiliary", [False, True])
def test_finite_gradients_reach_attention_but_not_missing_observations(architecture, tokens, auxiliary):
    model = make_model(architecture, tokens, auxiliary)
    features, valid = inputs(tokens)
    features.requires_grad_()
    logits = model(features, valid)
    assert logits.shape == (3, 4 if auxiliary else 3, 33)
    labels = torch.arange(logits.shape[0] * logits.shape[1]) % 33
    torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels).backward()
    assert torch.isfinite(logits).all() and torch.isfinite(features.grad).all()
    assert torch.count_nonzero(features.grad[~valid]) == 0
    assert features.grad[valid].abs().sum() > 0
    for layer in model.encoder:
        assert layer.qkv.weight.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("tokens", [26, 66])
def test_padding_nan_values_and_all_missing_examples_cannot_change_output(architecture, tokens):
    model = make_model(architecture, tokens).eval()
    features, valid = inputs(tokens)
    valid[0] = False
    altered = features.clone()
    altered[~valid] = float("nan")
    with torch.no_grad():
        expected, actual = model(features, valid), model(altered, valid)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_eval_output_is_independent_of_other_examples_and_batch_order(architecture):
    model = make_model(architecture).eval()
    features, valid = inputs()
    altered = features.clone()
    altered[1:] *= 100
    with torch.no_grad():
        expected = model(features[:1], valid[:1])[0]
        actual = model(altered, valid)[0]
        reordered = model(features.flip(0), valid.flip(0))[-1]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(reordered, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("tokens,count", [(11, 6), (26, 9), (66, 17)])
def test_patches_preserve_every_token_and_do_not_cross_sampling_boundaries(tokens, count):
    patches = history_patch_indices(tokens)
    assert len(patches) == count
    torch.testing.assert_close(patches[patches >= 0], torch.arange(tokens))
    older = tokens - 11
    for patch in patches:
        indices = patch[patch >= 0]
        assert bool((indices < older).all()) or bool((indices >= older).all())
        if indices[0] < older:
            assert len(indices) == 5
        elif indices[0] == tokens - 1:
            assert len(indices) == 1
        else:
            assert len(indices) == 2 and indices[-1] < tokens - 1


def test_partial_patch_keeps_only_present_observations_and_metadata():
    model = make_model("patch_transformer")
    values = torch.arange(26 * 10).reshape(1, 26, 10).float()
    valid = torch.ones(1, 26, dtype=torch.bool)
    valid[:, :4] = False
    patches, present = model._patch_values(values, valid)
    assert present.all()
    first = patches[0, 0].reshape(5, 10)
    assert torch.count_nonzero(first[:4]) == 0
    torch.testing.assert_close(first[4], values[0, 4])
    last = patches[0, -1].reshape(5, 10)
    torch.testing.assert_close(last[0], values[0, -1])
    assert torch.count_nonzero(last[1:]) == 0


def test_attention_layers_are_independent_and_padding_bias_is_removed():
    model = make_model("temporal_transformer")
    assert not torch.equal(model.encoder[0].qkv.weight, model.encoder[1].qkv.weight)
    block = MaskedAttentionBlock(16, dropout=0).eval()
    values, valid = torch.randn(2, 26, 16), torch.ones(2, 26, dtype=torch.bool)
    valid[:, :10] = False
    with torch.no_grad():
        block.attention_norm.bias.fill_(4)
        block.feedforward_norm.bias.fill_(3)
        expected = block(values, valid)
        changed = values.clone()
        changed[:, :10] = float("nan")
        actual = block(changed, valid)
    assert torch.count_nonzero(actual[:, :10]) == 0
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_saved_adapter_matches_direct_observation_only_prediction(tmp_path, architecture):
    from bvr_behavior_prediction.skill_forecasting.improved_models import ImprovedForecaster
    from bvr_behavior_prediction.skill_forecasting.improvement import ImprovedPredictor

    time = np.arange(121) * 0.1
    observations = np.zeros((121, 51), dtype=np.float32)
    observations[:, 3] = 0.3
    observations[:, 9] = 0.3 - time * 0.002
    observations[:, 10] = 0.1
    observations[:, 12] = -0.1 + 0.02 * np.sin(time)
    observations[:, 15] = 0.3
    offsets = history_offsets(20, 5)
    frames, valid = raw_history_features(observations, offsets, "geometry")
    config = {"architecture": architecture, "input_size": frames.shape[1], "tokens": len(offsets),
              "num_classes": 4, "width": 16, "layers": 2, "dropout": 0.1, "auxiliary": False}
    model = ImprovedForecaster(**config).eval()
    with torch.no_grad():
        expected = (model(torch.tensor(frames[None]), torch.tensor(valid[None]))[0] / 1.3).softmax(-1).numpy()
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_state": model.state_dict(), "model_config": config, "temperature": 1.3,
                "candidate": {"features": "geometry"}, "offsets": offsets.tolist(),
                "scaling": {"mean": [0] * len(frames[0]), "std": [1] * len(frames[0])},
                "classes": list("abcd")}, checkpoint)
    actual = ImprovedPredictor([checkpoint]).predict(observations)
    np.testing.assert_array_equal(actual["probabilities"], expected)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_fixed_time_metadata_matches_the_actual_observation_offsets(architecture):
    for seconds, tokens in [(5, 11), (20, 26), (60, 66)]:
        model = make_model(architecture, tokens)
        offsets = torch.tensor(history_offsets(seconds, 5), dtype=torch.float32) / 10
        torch.testing.assert_close(model.time_coordinates[:, 0] * 60, offsets)
        assert model.time_coordinates[:, 0].max() == 0
    with pytest.raises(ValueError, match="sparse history"):
        make_model(architecture, tokens=51)
