"""Train-only class weights and the actual masked, equal-horizon objective."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bvr_behavior_prediction import paired_skill_forecasting as paired


def test_per_horizon_weights_use_only_retained_training_labels():
    targets = np.array([
        [0, 0, 2], [0, 0, 2], [0, 0, 2], [0, 0, 2],
        [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2],
        [0, 1, 2], [1, 1, 0], [1, 1, -1], [2, 1, -1],
    ])
    train_ids = np.arange(12)

    class TrainingTargetsOnly:
        def __getitem__(self, ids):
            np.testing.assert_array_equal(ids, train_ids)
            return targets

    # Any access to another partition through the target array fails the assertion.
    data = SimpleNamespace(targets=TrainingTargetsOnly(), classes=['a', 'b', 'c', 'unseen'],
                           indices={'train': train_ids, 'test': np.array([12]),
                                    'validation': np.array([13])})
    config = {**paired.DEFAULT_TRAINING, 'class_weighting': 'sqrt_inverse'}
    state = paired.fit_class_weighting(data, config)
    counts, weights = np.array(state['counts']), np.array(state['weights'])
    np.testing.assert_array_equal(counts, [[9, 2, 1, 0], [4, 8, 0, 0], [1, 0, 9, 0]])
    frequencies = counts / counts.sum(1, keepdims=True)
    np.testing.assert_allclose((weights * frequencies).sum(1), 1.0)
    assert weights[0, 2] / weights[0, 0] == pytest.approx(3)
    assert weights[1, 0] > weights[1, 1]
    assert weights[2, 0] > weights[2, 2]
    assert not weights[counts == 0].any()
    table = paired.class_weight_table(state)
    assert table.shape == (12, 5)
    assert table.training_support.sum() == (targets >= 0).sum()


@pytest.mark.parametrize('cap', [1.0, 3.0, 5.0])
def test_extreme_rarity_is_capped_before_mean_normalisation(cap):
    labels = np.array([0] * 100 + [1])
    data = SimpleNamespace(targets=np.tile(labels[:, None], (1, 3)),
                           classes=['common', 'rare', 'unseen'], indices={'train': np.arange(101)})
    config = {**paired.DEFAULT_TRAINING, 'class_weighting': 'sqrt_inverse',
              'max_class_weight_ratio': cap}
    weights = np.array(paired.fit_class_weighting(data, config)['weights'])
    np.testing.assert_allclose(weights[:, 1] / weights[:, 0], cap)
    np.testing.assert_allclose((100 * weights[:, 0] + weights[:, 1]) / 101, 1)
    assert np.isfinite(weights).all() and (weights >= 0).all() and (weights <= cap).all()


@pytest.mark.parametrize('all_missing', [False, True])
def test_weighted_loss_matches_manual_objective_and_gradients(all_missing):
    torch.manual_seed(7)
    logits = torch.randn(4, 3, 4, requires_grad=True)
    targets = torch.tensor([[0, 0, -1], [0, 1, 2], [1, -1, 2], [2, 0, 0]])
    if all_missing:
        targets[:] = -1
    weights = torch.tensor([[.5, 1., 2., 0.], [1., 2., 0., 0.], [2., 0., .5, 0.]])
    actual = paired.masked_loss(logits, targets, weights)
    terms = []
    logp = logits.log_softmax(-1)
    for h in range(3):
        valid = targets[:, h] >= 0
        y = targets[valid, h]
        if len(y):
            terms.append((-logp[valid, h, y] * weights[h, y]).mean())
    expected = torch.stack(terms).mean() if terms else logits.sum() * 0
    torch.testing.assert_close(actual, expected)
    gradient = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    torch.testing.assert_close(gradient, torch.autograd.grad(expected, logits)[0])
    assert not gradient[targets < 0].any()


def test_single_class_minibatch_keeps_its_class_weight():
    logits = torch.zeros(1, 3, 2, requires_grad=True)
    weights = torch.tensor([[.5, 2.]] * 3)
    common = paired.masked_loss(logits, torch.zeros(1, 3, dtype=torch.long), weights)
    rare = paired.masked_loss(logits, torch.ones(1, 3, dtype=torch.long), weights)
    assert (rare / common).item() == pytest.approx(4)
    torch.testing.assert_close(paired.masked_loss(logits, torch.zeros(1, 3, dtype=torch.long)),
                               common * 2)


@pytest.mark.parametrize('setting,value', [
    ('class_weighting', 'unknown'), ('max_class_weight_ratio', 0),
    ('max_class_weight_ratio', float('nan')), ('max_class_weight_ratio', float('inf')),
    ('max_class_weight_ratio', True),
])
def test_invalid_settings_fail_before_accessing_data(tmp_path, setting, value):
    with pytest.raises(ValueError):
        paired.train_model(None, tmp_path / 'unused', {setting: value}, device='cpu')
    assert not (tmp_path / 'unused').exists()


def test_empty_training_horizon_is_rejected():
    data = SimpleNamespace(targets=np.full((2, 3), -1), classes=['a', 'b'],
                           indices={'train': np.arange(2)})
    with pytest.raises(ValueError, match='No retained training labels'):
        paired.fit_class_weighting(data, {**paired.DEFAULT_TRAINING, 'class_weighting': 'sqrt_inverse'})


def test_weighting_disabled_never_reads_targets():
    assert paired.fit_class_weighting(None, paired.DEFAULT_TRAINING) is None


def test_weighted_diagnostic_preserves_unweighted_evaluation_and_batch_invariance():
    import pandas as pd

    torch.manual_seed(19)
    logits = torch.randn(5, 3, 3)
    targets = np.array([[0, 0, -1], [0, 1, 2], [0, -1, 2], [1, 0, 0], [2, 1, 1]])

    class FixedModel(torch.nn.Module):
        num_classes = 3

        def forward(self, x):
            return logits[x[:, 0].long()]

    data = SimpleNamespace(targets=targets, classes=['a', 'b', 'c'],
                           indices={'train': np.arange(5)})
    data.batch = lambda ids, device: (torch.tensor(ids[:, None]), torch.tensor(targets[ids]))
    state = paired.fit_class_weighting(data, {**paired.DEFAULT_TRAINING,
                                             'class_weighting': 'sqrt_inverse'})
    weights = torch.tensor(state['weights'], dtype=torch.float32)
    plain = paired.evaluate(FixedModel(), data, 'train', 'cpu', batch_size=5)
    for batch_size in (1, 2, 5):
        weighted = paired.evaluate(FixedModel(), data, 'train', 'cpu', batch_size=batch_size,
                                   class_weights=state['weights'])
        pd.testing.assert_frame_equal(plain['summary'], weighted['summary'])
        np.testing.assert_array_equal(plain['probabilities'], weighted['probabilities'])
        assert weighted['data_loss'] == pytest.approx(paired.masked_loss(logits, torch.tensor(targets), weights).item())
    assert weighted['data_loss'] != pytest.approx(plain['data_loss'])
