"""Regularized sequence and temporal-MLP alternatives with direct state readout."""

import torch
from torch import nn

from .models import SkillForecaster


class ResidualBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.network = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(width * 2, width), nn.Dropout(dropout))

    def forward(self, values):
        return values + self.network(values)


class ImprovedForecaster(nn.Module):
    def __init__(self, architecture, input_size, tokens, num_classes=33, width=128,
                 layers=2, dropout=0.2, auxiliary=False):
        super().__init__()
        self.architecture, self.num_classes = architecture, num_classes
        self.horizons = 4 if auxiliary else 3
        if architecture == "legacy":
            if auxiliary:
                raise ValueError("Legacy control has no auxiliary head")
            self.legacy = SkillForecaster(architecture="transformer", input_size=input_size,
                                         history_steps=tokens, num_classes=num_classes, width=width,
                                         layers=layers, dropout=dropout, heads=4)
        elif architecture == "mlp":
            self.encoder = nn.Sequential(nn.Linear(tokens * (input_size + 1), width), nn.GELU(),
                                         *[ResidualBlock(width, dropout) for _ in range(layers)],
                                         nn.LayerNorm(width))
            self.head = nn.Linear(width, self.horizons * num_classes)
        elif architecture == "transformer":
            self.projection = nn.Linear(input_size, width)
            self.position = nn.Parameter(torch.randn(1, tokens, width) * 0.02)
            block = nn.TransformerEncoderLayer(width, 4, width * 4, dropout, activation="gelu",
                                                batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
            for layer in self.encoder.layers:
                for parameter in layer.parameters():
                    if parameter.ndim > 1:
                        nn.init.xavier_uniform_(parameter)
            self.head = nn.Sequential(nn.LayerNorm(2 * width + input_size),
                                      nn.Linear(2 * width + input_size, width * 2), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(width * 2, self.horizons * num_classes))
        else:
            raise ValueError("Unknown architecture")

    def forward(self, features, valid):
        features = features.masked_fill(~valid[:, :, None], 0)
        if self.architecture == "legacy":
            return self.legacy(features)
        if self.architecture == "mlp":
            values = torch.cat((features, valid[:, :, None].to(features.dtype)), dim=-1)
            return self.head(self.encoder(values.flatten(1))).reshape(-1, self.horizons, self.num_classes)
        hidden = self.encoder(self.projection(features) + self.position, src_key_padding_mask=~valid)
        pooled = (hidden * valid[:, :, None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        summary = torch.cat((hidden[:, -1], pooled, features[:, -1]), dim=-1)
        return self.head(summary).reshape(-1, self.horizons, self.num_classes)


def supervised_loss(logits, future, current, smoothing=0.0, auxiliary_weight=0.0):
    losses = []
    for h in range(3):
        valid = future[:, h] >= 0
        if valid.any():
            losses.append(nn.functional.cross_entropy(logits[valid, h].float(), future[valid, h],
                                                       label_smoothing=smoothing))
    loss = torch.stack(losses).mean() if losses else logits.sum() * 0
    if auxiliary_weight:
        valid = current >= 0
        if valid.any():
            loss = loss + auxiliary_weight * nn.functional.cross_entropy(
                logits[valid, 3].float(), current[valid], label_smoothing=smoothing)
    return loss
