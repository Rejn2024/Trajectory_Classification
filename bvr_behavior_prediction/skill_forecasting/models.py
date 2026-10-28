"""Three categorical heads sharing an observation-only sequence encoder."""

import torch
from torch import nn

from bvr_behavior_prediction.forecasting.models import MotionForecaster


class SkillForecaster(MotionForecaster):
    def __init__(self, num_classes=33, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.head[-1] = nn.Linear(self.head[-1].in_features, 3 * num_classes)

    def forward(self, observations):
        if observations.ndim != 3 or observations.shape[1] != self.history_steps:
            raise ValueError("Expected complete (batch, history_steps, features) windows")
        tokens = self.projection(observations)
        if self.architecture == "transformer":
            hidden = self.encoder(tokens + self.position)[:, -1]
        else:
            _, state = self.encoder(tokens)
            hidden = state[-1]
        return self.head(self.norm(hidden)).reshape(-1, 3, self.num_classes)


def class_weights(targets, num_classes, mode="none", max_weight=5.0):
    """Optional capped square-root weights fitted on training targets only."""
    if mode == "none":
        return None
    if mode != "sqrt_inverse" or max_weight < 1:
        raise ValueError("Expected none or sqrt_inverse weighting with max_weight >= 1")
    weights = []
    for h in range(3):
        valid = targets[:, h]
        counts = torch.bincount(valid[valid >= 0], minlength=num_classes).float()
        present = counts > 0
        weight = torch.zeros_like(counts)
        weight[present] = torch.sqrt(counts[present].sum() / counts[present])
        weight[present] /= (weight[present] * counts[present]).sum() / counts[present].sum()
        weights.append(weight.clamp(max=max_weight))
    return torch.stack(weights)


def masked_cross_entropy(logits, targets, weights=None):
    """Equal horizon contribution; unavailable labels (-1) have no gradient."""
    losses = []
    for h in range(3):
        valid = targets[:, h] >= 0
        if valid.any():
            losses.append(nn.functional.cross_entropy(
                logits[valid, h].float(), targets[valid, h],
                weight=None if weights is None else weights[h]))
    return torch.stack(losses).mean() if losses else logits.sum() * 0
