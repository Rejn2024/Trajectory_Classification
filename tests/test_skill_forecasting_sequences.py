"""Padding, causality and batching boundaries for the new sequence models."""

import pytest
import torch

from bvr_behavior_prediction.skill_forecasting.improved_features import history_offsets
from bvr_behavior_prediction.skill_forecasting.sequence_models import (
    CausalTemporalBlock,
    SequenceForecaster,
    sparse_time_coordinates,
)

ARCHITECTURES = ["multiscale_mlp", "tcn", "gru"]


@pytest.fixture(autouse=True)
def deterministic_small_cpu_models():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(1729)
    yield
    torch.set_num_threads(previous)


def make_model(architecture, tokens=26, auxiliary=False):
    return SequenceForecaster(architecture, 7, tokens, width=16, layers=2,
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
def test_output_shape_and_gradients_ignore_invalid_input(architecture, tokens, auxiliary):
    model = make_model(architecture, tokens, auxiliary)
    features, valid = inputs(tokens)
    features.requires_grad_()
    logits = model(features, valid)
    assert logits.shape == (3, 4 if auxiliary else 3, 33)
    assert torch.isfinite(logits).all()
    labels = torch.arange(logits.shape[0] * logits.shape[1]) % 33
    torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels).backward()
    assert torch.isfinite(features.grad).all()
    assert torch.count_nonzero(features.grad[~valid]) == 0
    assert features.grad[valid].abs().sum() > 0
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("tokens", [26, 66])
def test_invalid_padding_values_cannot_change_predictions(architecture, tokens):
    model = make_model(architecture, tokens).eval()
    features, valid = inputs(tokens)
    altered = features.clone()
    altered[~valid] = float("nan")
    with torch.no_grad():
        expected = model(features, valid)
        actual = model(altered, valid)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_eval_predictions_do_not_depend_on_other_batch_members(architecture):
    model = make_model(architecture).eval()
    features, valid = inputs()
    altered = features.clone()
    altered[1:] *= 100
    with torch.no_grad():
        expected = model(features[:1], valid[:1])[0]
        batched = model(altered, valid)[0]
        reordered = model(features.flip(0), valid.flip(0))[-1]
    torch.testing.assert_close(batched, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(reordered, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("dilation", [1, 2, 8])
def test_causal_tcn_block_cannot_read_future_tokens(dilation):
    block = CausalTemporalBlock(8, dilation, dropout=0).eval()
    features = torch.randn(2, 26, 8, requires_grad=True)
    valid = torch.ones(2, 26, dtype=torch.bool)
    valid[:, :3] = False
    altered = features.detach().clone()
    altered[:, 15:] += 100
    expected = block(features, valid)
    actual = block(altered, valid)
    torch.testing.assert_close(actual[:, :15], expected[:, :15], atol=0, rtol=0)
    assert torch.count_nonzero(expected[:, :3]) == 0
    expected[:, :15].square().sum().backward()
    assert torch.count_nonzero(features.grad[:, 15:]) == 0
    assert torch.count_nonzero(features.grad[:, :3]) == 0
    assert features.grad[:, 3:15].abs().sum() > 0


def test_gru_matches_running_only_valid_tokens_without_padding_updates():
    model = make_model("gru").eval()
    features, valid = inputs()
    times = model.time_coordinates[None].expand(len(features), -1, -1)
    values = torch.cat((features, times, valid[:, :, None].float()), dim=-1)
    with torch.no_grad():
        final, pooled = model._gru_summary(values.masked_fill(~valid[:, :, None], 0), valid)
        for row in range(len(features)):
            # Reference feeds the actual suffix, never zero-valued prefix steps.
            hidden = values[row, valid[row]][None]
            for layer in model.encoder:
                hidden, _ = layer(hidden)
            torch.testing.assert_close(final[row], hidden[0, -1], atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(pooled[row], hidden[0].mean(0), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_entirely_masked_input_is_finite_and_independent_of_contents(architecture):
    model = make_model(architecture).eval()
    features, valid = inputs()
    valid[:] = False
    with torch.no_grad():
        expected = model(features, valid)
        actual = model(torch.full_like(features, float("nan")), valid)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("history,tokens", [(5, 11), (20, 26), (60, 66)])
def test_time_encoding_matches_existing_sparse_history_offsets(history, tokens):
    times = sparse_time_coordinates(tokens)
    expected = torch.tensor(history_offsets(history, 5), dtype=torch.float32) / 10
    torch.testing.assert_close(times[:, 0] * 60, expected)
    torch.testing.assert_close(times[1:, 1], expected[1:] - expected[:-1])
    assert times[0, 1] == 0
    assert times[-1, 0] == 0
    assert torch.all(times[:, 0] <= 0)


def test_unknown_token_layout_fails_instead_of_assuming_wrong_time_spacing():
    with pytest.raises(ValueError, match="sparse history"):
        make_model("tcn", tokens=51)
