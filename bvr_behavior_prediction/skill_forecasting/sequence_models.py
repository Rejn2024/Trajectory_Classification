"""Sequence alternatives using the existing sparse, observation-only histories.

Histories contain eleven recent half-second tokens and, optionally, older
one-second tokens. No labels, pilot identifiers or future observations enter
these models. The constructor deliberately matches ``ImprovedForecaster``.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence


class DenseResidualBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 2 * width), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2 * width, width), nn.Dropout(dropout),
        )

    def forward(self, values):
        return values + self.network(values)


class CausalTemporalBlock(nn.Module):
    """One left-padded dilated convolution; normalization is per token only."""

    def __init__(self, width, dilation=1, dropout=0.2):
        super().__init__()
        self.left_padding = 2 * dilation
        self.norm = nn.LayerNorm(width)
        self.temporal = nn.Conv1d(width, 2 * width, kernel_size=3, dilation=dilation)
        self.output = nn.Linear(2 * width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, valid):
        invalid = ~valid[:, :, None]
        values = values.masked_fill(invalid, 0)
        # LayerNorm's learned bias must not turn padding into recurrent evidence.
        normalized = self.norm(values).masked_fill(invalid, 0)
        hidden = self.temporal(F.pad(normalized.transpose(1, 2), (self.left_padding, 0)))
        hidden = self.dropout(F.gelu(hidden.transpose(1, 2)))
        hidden = self.dropout(self.output(hidden))
        return (values + hidden).masked_fill(invalid, 0)


def sparse_time_coordinates(tokens):
    """Known relative time and preceding gap, not additional sensor features.

The supported layouts correspond exactly to history_offsets(history, 5).
Rejecting other lengths prevents silently treating dense 10 Hz histories as
the sparse baseline layout. Relative time uses a fixed sixty-second scale;
gaps are in seconds and the first token has no preceding gap.
"""
    if tokens not in (11, 26, 66):
        raise ValueError("Sequence models require 11, 26 or 66 sparse history tokens")
    older = torch.arange(-(tokens - 6), -5, dtype=torch.float32)
    offsets = torch.cat((older, torch.linspace(-5, 0, 11)))
    gaps = torch.cat((torch.zeros(1), offsets[1:] - offsets[:-1]))
    return torch.stack((offsets / 60, gaps), dim=-1)


class SequenceForecaster(nn.Module):
    """Multiscale MLP, causal TCN or packed GRU with a current-state bypass.

``layers`` is the residual-block count for the MLP/TCN and recurrent-layer
count for the GRU. Four TCN blocks cover 31 tokens; six cover 127 tokens.
All predictions still use only observations at or before the forecast origin.
"""

    def __init__(self, architecture, input_size, tokens, num_classes=33, width=128,
                 layers=2, dropout=0.2, auxiliary=False):
        super().__init__()
        if architecture not in ("multiscale_mlp", "tcn", "gru"):
            raise ValueError("Unknown sequence architecture")
        if width < 4 or layers < 1:
            raise ValueError("Sequence models require width >= 4 and layers >= 1")
        self.architecture, self.num_classes = architecture, num_classes
        self.input_size, self.tokens = input_size, tokens
        self.horizons = 4 if auxiliary else 3
        self.register_buffer("time_coordinates", sparse_time_coordinates(tokens))
        if architecture == "multiscale_mlp":
            branch_width, current_width = 3 * width // 4, width // 4
            self.recent_encoder = nn.Sequential(
                nn.Linear(11 * (input_size + 1), branch_width), nn.GELU(),
                nn.LayerNorm(branch_width),
            )
            self.older_encoder = nn.Sequential(
                nn.Linear((tokens - 11) * (input_size + 1), branch_width), nn.GELU(),
                nn.LayerNorm(branch_width),
            ) if tokens > 11 else None
            self.current_encoder = nn.Sequential(nn.Linear(input_size, current_width), nn.GELU())
            self.head = nn.Sequential(
                nn.Linear(2 * branch_width + current_width, width), nn.GELU(),
                *[DenseResidualBlock(width, dropout) for _ in range(layers)],
                nn.LayerNorm(width), nn.Linear(width, self.horizons * num_classes),
            )
        else:
            # Time/gap metadata identifies the change in spacing at five seconds.
            if architecture == "tcn":
                self.projection = nn.Linear(input_size + 3, width)
                self.encoder = nn.ModuleList(
                    CausalTemporalBlock(width, 2 ** index, dropout) for index in range(layers)
                )
            else:
                # Explicit inter-layer dropout avoids cuDNN's recurrent dropout
                # state, which causes a shutdown failure on the target Windows GPU.
                self.encoder = nn.ModuleList(
                    nn.GRU(input_size + 3 if index == 0 else width, width, batch_first=True)
                    for index in range(layers)
                )
                self.recurrent_dropout = nn.Dropout(dropout)
            self.head = nn.Sequential(
                nn.LayerNorm(2 * width + input_size),
                nn.Linear(2 * width + input_size, 2 * width), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(2 * width, self.horizons * num_classes),
            )

    def _gru_summary(self, values, valid):
        """Compact valid tokens before packing, so left padding never updates state."""
        lengths = valid.sum(1)
        positions = torch.arange(self.tokens, device=values.device)[None].expand_as(valid)
        order = (positions + (~valid).long() * self.tokens).argsort(dim=1)
        compact = values.gather(1, order[:, :, None].expand(-1, -1, values.shape[-1]))
        packed = pack_padded_sequence(
            compact, lengths.clamp_min(1).cpu(), batch_first=True, enforce_sorted=False,
        )
        for index, layer in enumerate(self.encoder):
            encoded, _ = layer(packed)
            if index + 1 < len(self.encoder):
                packed = PackedSequence(
                    self.recurrent_dropout(encoded.data), encoded.batch_sizes,
                    encoded.sorted_indices, encoded.unsorted_indices,
                )
        hidden, _ = pad_packed_sequence(encoded, batch_first=True, total_length=self.tokens)
        packed_valid = positions < lengths[:, None]
        hidden = hidden.masked_fill(~packed_valid[:, :, None], 0)
        final = hidden[torch.arange(len(values), device=values.device), lengths.clamp_min(1) - 1]
        pooled = hidden.sum(1) / lengths[:, None].clamp_min(1)
        return final, pooled

    def forward(self, features, valid):
        if features.ndim != 3 or features.shape[1:] != (self.tokens, self.input_size):
            raise ValueError("Features do not match the configured token and feature dimensions")
        if valid.shape != features.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("Validity must be a boolean mask with one entry per token")
        features = features.masked_fill(~valid[:, :, None], 0)
        validity = valid[:, :, None].to(features.dtype)
        if self.architecture == "multiscale_mlp":
            values = torch.cat((features, validity), dim=-1)
            recent = self.recent_encoder(values[:, -11:].flatten(1))
            older = (self.older_encoder(values[:, :-11].flatten(1))
                     if self.older_encoder is not None else torch.zeros_like(recent))
            current = self.current_encoder(features[:, -1])
            summary = torch.cat((older, recent, current), dim=-1)
        else:
            times = self.time_coordinates.to(features.dtype)[None].expand(len(features), -1, -1)
            values = torch.cat((features, times, validity), dim=-1)
            values = values.masked_fill(~valid[:, :, None], 0)
            if self.architecture == "tcn":
                hidden = self.projection(values).masked_fill(~valid[:, :, None], 0)
                for block in self.encoder:
                    hidden = block(hidden, valid)
                final = hidden[:, -1]
                pooled = hidden.sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
            else:
                final, pooled = self._gru_summary(values, valid)
            summary = torch.cat((final, pooled, features[:, -1]), dim=-1)
        return self.head(summary).reshape(-1, self.horizons, self.num_classes)
