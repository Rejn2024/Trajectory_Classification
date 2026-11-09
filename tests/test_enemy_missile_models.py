"""Schema, set symmetry, missing-slot safety and real graph dependencies."""

import io

import pytest
import torch

from bvr_behavior_prediction.enemy_missile_models import (
    MissileSetTransformer, TypedGraphTransformer, masked_summary,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def example():
    torch.manual_seed(9)
    raw = torch.randn(2, 51, 115) * .1
    raw[..., 19:47] = 0
    raw[..., 54:82] = 0
    raw[..., 19:26] = torch.tensor([.1, .03, -.02, .4, .1, .02, .11])
    raw[..., 33:40] = torch.tensor([.2, -.1, .02, .6, -.1, .01, .23])
    raw[..., 61:68] = torch.tensor([.3, -.1, -.03, -.2, .1, .02, .32])
    raw[..., 82:] = 0
    raw[..., 84] = 1
    mean, std = torch.randn(115), torch.rand(115) + .1
    mean[82:], std[82:] = 0, 1
    return raw, mean, std


def model(kind, mean=None, std=None):
    return kind(width=24, heads=6, layers=1, dropout=0,
                graph_width=8, graph_layers=2, mean=mean, std=std)


@pytest.mark.parametrize("kind", [MissileSetTransformer, TypedGraphTransformer])
def test_per_frame_permutations_ignore_slot_number_and_slot_specific_scaling(kind):
    raw, mean, std = example()
    network = model(kind, mean, std).eval()
    changed = raw.clone()
    # Independently permute slots at every instant, including empty slots. This
    # detects accidental per-slot recurrent histories as well as fixed readouts.
    for frame in range(51):
        for start in (19, 54):
            slots = raw[:, frame, start:start + 28].reshape(2, 4, 7)
            changed[:, frame, start:start + 28] = slots[:, torch.randperm(4)].reshape(2, 28)
    with torch.no_grad():
        before = network((raw - mean) / std)
        after = network((changed - mean) / std)
    assert before.shape == (2, 3, 33)
    torch.testing.assert_close(before, after, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("kind", [MissileSetTransformer, TypedGraphTransformer])
def test_standardized_absence_is_masked_and_empty_sets_have_finite_gradients(kind):
    raw, mean, std = example()
    raw[..., 19:47], raw[..., 54:82] = 0, 0
    normalized = ((raw - mean) / std).requires_grad_()
    assert normalized[..., 19:47].abs().sum() > 0
    network = model(kind, mean, std)
    _, _, _, present = network._decode(normalized)
    assert present[:, :2].all() and not present[:, 2:].any()
    output = network(normalized)
    output.square().mean().backward()
    assert torch.isfinite(output).all() and torch.isfinite(normalized.grad).all()
    assert not normalized.grad[..., 19:47].any()
    assert not normalized.grad[..., 54:82].any()
    assert all(torch.isfinite(p.grad).all() for p in network.parameters() if p.grad is not None)


def test_masked_summary_has_no_fake_maximum_for_empty_set():
    values = torch.full((2, 4, 8), -3.)
    present = torch.tensor([[True, False, False, False], [False] * 4])
    result = masked_summary(values, present)
    torch.testing.assert_close(result[0, :16], torch.full((16,), -3.))
    assert result[0, -1] == .25 and not result[1].any()


def test_graph_aircraft_updates_depend_on_neighbour_and_missile_messages():
    raw, mean, std = example()
    graph = model(TypedGraphTransformer, mean, std)
    raw, slots, nodes, present = graph._decode((raw - mean) / std)
    nodes = nodes.detach().requires_grad_()
    updated = graph._spatial(raw, slots, nodes, present)
    gradient = torch.autograd.grad(updated[:, 0, 0].sum(), nodes)[0]
    assert gradient[:, 1].abs().sum() > 0  # Opposing aircraft -> observer.
    assert gradient[:, 2].abs().sum() > 0  # Present friendly missile -> observer.
    assert not gradient[:, 3].any()       # Absent missile has no message.
    assert len(graph.sources) == 34


@pytest.mark.parametrize("kind", [MissileSetTransformer, TypedGraphTransformer])
def test_frame_encoder_never_joins_times_or_batch_members(kind):
    raw, mean, std = example()
    network = model(kind, mean, std).eval()
    normalized = (raw - mean) / std
    altered = normalized.clone()
    altered[0, 50] += .5
    with torch.no_grad():
        frames = network.encode_frames(normalized)
        changed = network.encode_frames(altered)
        before = network(normalized)
        after = network(altered)
    torch.testing.assert_close(frames[0, :50], changed[0, :50], atol=0, rtol=0)
    torch.testing.assert_close(before[1], after[1], atol=0, rtol=0)


@pytest.mark.parametrize("kind", [MissileSetTransformer, TypedGraphTransformer])
def test_scaling_buffers_and_predictions_survive_checkpoint_reload(kind):
    raw, mean, std = example()
    network = model(kind, mean, std).eval()
    stream = io.BytesIO()
    torch.save(network.state_dict(), stream)
    stream.seek(0)
    restored = model(kind).eval()
    restored.load_state_dict(torch.load(stream, weights_only=True))
    torch.testing.assert_close(restored.feature_mean, mean)
    torch.testing.assert_close(restored.feature_std, std)
    with torch.no_grad():
        torch.testing.assert_close(network((raw - mean) / std), restored((raw - mean) / std),
                                   atol=0, rtol=0)


@pytest.mark.parametrize("kind", [MissileSetTransformer, TypedGraphTransformer])
def test_bfloat16_autocast_keeps_empty_graph_finite(kind):
    network = model(kind)
    history = torch.zeros(1, 51, 115)
    history[..., 82] = 1
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = network(history)
    output.float().square().sum().backward()
    assert torch.isfinite(output).all()
    assert all(torch.isfinite(p.grad).all() for p in network.parameters() if p.grad is not None)


@pytest.mark.parametrize("kind", [MissileSetTransformer, TypedGraphTransformer])
def test_schema_and_scaling_are_validated(kind):
    with pytest.raises(ValueError, match="notebook-17"):
        kind(input_features=87)
    with pytest.raises(ValueError, match="mean"):
        kind(std=torch.zeros(115))
    network = model(kind)
    with pytest.raises(ValueError, match="Expected"):
        network(torch.zeros(1, 50, 115))
@pytest.mark.parametrize("model_class", [MissileSetTransformer, TypedGraphTransformer])
@pytest.mark.skipif(not hasattr(torch.export, "Dim"), reason="Dynamic export requires newer PyTorch")
def test_dynamic_batch_export(model_class):
    """The MLflow PT2 export path must accept batch sizes beyond its example."""
    torch.set_num_threads(1)
    model = model_class(width=12, heads=6, layers=1, dropout=0.0,
                        graph_width=8, graph_layers=2).eval()
    windows = torch.zeros(2, 51, 115)
    windows[:, :, 82] = 1
    program = torch.export.export(model, (windows,), dynamic_shapes=(
        {0: torch.export.Dim("batch", min=1, max=8)},))
    with torch.inference_mode():
        torch.testing.assert_close(program.module()(windows[:1]), model(windows[:1]))
