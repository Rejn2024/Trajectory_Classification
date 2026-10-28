"""Small residual forecasters sharing the same observation and output contracts."""

import torch
from torch import nn


class MotionForecaster(nn.Module):
    def __init__(self, architecture="transformer", input_size=54, history_steps=51,
                 width=128, layers=2, heads=4, dropout=0.1):
        super().__init__()
        self.architecture = architecture
        self.history_steps = history_steps
        self.projection = nn.Linear(input_size, width)
        if architecture == "transformer":
            self.position = nn.Parameter(torch.randn(1, history_steps, width) * 0.02)
            block = nn.TransformerEncoderLayer(
                width, heads, width * 4, dropout, activation="gelu", batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
            # TransformerEncoder clones blocks; initialise each independently.
            for layer in self.encoder.layers:
                for parameter in layer.parameters():
                    if parameter.ndim > 1:
                        nn.init.xavier_uniform_(parameter)
        elif architecture == "gru":
            self.encoder = nn.GRU(width, width, num_layers=layers, batch_first=True,
                                  dropout=dropout if layers > 1 else 0.0)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        self.norm = nn.LayerNorm(width)
        self.head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(width, 3 * 6))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, observations):
        if observations.ndim != 3 or observations.shape[1] != self.history_steps:
            raise ValueError("Expected complete (batch, history_steps, features) windows")
        tokens = self.projection(observations)
        if self.architecture == "transformer":
            # Every token is at or before the forecast origin; no future tokens exist.
            hidden = self.encoder(tokens + self.position)[:, -1]
        else:
            _, state = self.encoder(tokens)
            hidden = state[-1]
        return self.head(self.norm(hidden)).reshape(-1, 3, 6)


def masked_loss(prediction, target, valid):
    """Balance horizons, position/velocity, and axes in scaled physical residuals."""
    error = nn.functional.smooth_l1_loss(prediction.float(), target.float(), reduction="none").mean(-1)
    counts = valid.sum(0)
    losses = (error * valid).sum(0) / counts.clamp_min(1)
    return losses[counts > 0].mean()
