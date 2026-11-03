"""Warmup preserves old schedules and fixes refit budgets before validation."""

import numpy as np
import pytest
import torch

from bvr_behavior_prediction.skill_forecasting.improvement import training_schedule


def rates(warmup, epochs, legacy=False):
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=0.0003)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=3e-5)
                 if legacy else training_schedule(optimizer, 60, warmup))
    values = []
    for _ in range(epochs):
        values.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return np.asarray(values)


def test_default_schedule_exactly_preserves_previous_experiments():
    np.testing.assert_array_equal(rates(0, 61), rates(0, 61, legacy=True))


def test_registered_warmup_and_refit_prefix_do_not_depend_on_validation():
    complete = rates(4, 61)
    np.testing.assert_allclose(complete[:5], [0.00003, 0.0000975, 0.000165, 0.0002325, 0.0003])
    assert np.all(np.diff(complete[4:]) < 0)
    assert complete[-1] == pytest.approx(0.00003)
    np.testing.assert_array_equal(rates(4, 42), complete[:42])


@pytest.mark.parametrize("warmup,factor", [(60, 0.1), (-1, 0.1), (4, 0), (4, 1.1)])
def test_invalid_warmup_is_rejected_before_fitting(warmup, factor):
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=0.0003)
    with pytest.raises(ValueError, match="Invalid warmup"):
        training_schedule(optimizer, 60, warmup, factor)
