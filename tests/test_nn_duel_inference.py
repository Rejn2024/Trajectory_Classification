"""Batched actors preserve independent policies, request order and CPU decisions."""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from nn_duel_inference import BatchedActors, PolicyHeads

from bvr_behavior_prediction.rl.pilot import HybridSkillPilot


@pytest.fixture
def models():
    old_fastpath = torch.backends.mha.get_fastpath_enabled()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(91)
    result = {
        f"pilot_{i:03d}": HybridSkillPilot(
            51,
            hidden_size=8,
            history_steps=20,
            transformer_heads=2,
            transformer_layers=1,
            selected_skill_parameters=True,
        ).eval()
        for i in (1, 2)
    }
    yield result
    torch.backends.mha.set_fastpath_enabled(old_fastpath)
    torch.set_num_threads(old_threads)


def test_ensemble_preserves_request_order_and_separate_weights(models):
    actors = BatchedActors(models, "cpu")
    rng = np.random.default_rng(17)
    requests = [
        (name, rng.normal(size=(20, 51)).astype(np.float32))
        for name in ("pilot_002", "pilot_001", "pilot_002")
    ]
    heads, stats = actors.heads(requests)
    assert stats == {"rows": 3, "padded_rows": 4}
    for (name, history), (logits, means) in zip(requests, heads, strict=True):
        with torch.inference_mode():
            expected_logits, expected_means = PolicyHeads(models[name])(
                torch.as_tensor(history[None])
            )
        np.testing.assert_allclose(logits, expected_logits.numpy()[0], rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(means, expected_means.numpy()[0], rtol=2e-5, atol=2e-5)
    # Reordering and padding must not change which pilot's weights are applied.
    permuted, _ = actors.heads(requests[::-1])
    for a, b in zip(heads, permuted[::-1], strict=True):
        np.testing.assert_allclose(a[0], b[0], atol=2e-5)


def test_decoded_actions_match_original_pilots_and_parity_probe(models):
    actors = BatchedActors(models, "cpu")
    assert actors.parity_check(histories=3)["passed"]
    x = np.random.default_rng(4).normal(size=(20, 51)).astype(np.float32) * 0.2
    requests = [("pilot_002", x), ("pilot_001", x)]
    actions, _ = actors.act(requests)
    for (name, history), action in zip(requests, actions, strict=True):
        reference = models[name].act(history, deterministic=True)
        assert action[0] == reference[0]
        assert action[1].keys() == reference[1].keys()
        for parameter, value in action[1].items():
            if isinstance(value, float):
                assert value == pytest.approx(reference[1][parameter], rel=2e-5, abs=2e-5)
            else:
                assert value == reference[1][parameter]


def test_empty_batch_and_incompatible_history_are_handled(models):
    actors = BatchedActors(models, "cpu")
    assert actors.act([])[0] == []
    with pytest.raises(ValueError, match="history shapes"):
        actors.act([("pilot_001", np.zeros((20, 51))), ("pilot_002", np.zeros((19, 51)))])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_cuda_stacked_actors_pass_cpu_parity(models):
    actors = BatchedActors(models, "cuda")
    assert actors.parity_check(histories=8)["passed"]
