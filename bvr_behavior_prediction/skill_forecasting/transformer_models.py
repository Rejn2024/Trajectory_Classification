"""Time-aware Transformer alternatives for the fixed observation-only histories.

Attention is bidirectional within the supplied history, whose timestamps are all
at or before the forecast origin. It never receives future observations or labels.
The patch layout respects the older one-second/recent half-second sampling split.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .sequence_models import DenseResidualBlock, sparse_time_coordinates


class MaskedAttentionBlock(nn.Module):
    """Independently initialized pre-norm attention and feed-forward residuals."""

    def __init__(self, width, dropout=0.2, heads=4):
        super().__init__()
        if width % heads:
            raise ValueError("Attention width must be divisible by its head count")
        self.heads, self.head_width, self.dropout_probability = heads, width // heads, dropout
        self.attention_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.attention_output = nn.Linear(width, width)
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, width * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width * 4, width),
        )
        self.dropout = nn.Dropout(dropout)
        self.attention_scale = nn.Parameter(torch.full((width,), 0.1))
        self.feedforward_scale = nn.Parameter(torch.full((width,), 0.1))

    def forward(self, values, valid):
        invalid = ~valid[:, :, None]
        values = values.masked_fill(invalid, 0)
        normalized = self.attention_norm(values).masked_fill(invalid, 0)
        batch, tokens, width = values.shape
        qkv = self.qkv(normalized).reshape(batch, tokens, 3, self.heads, self.head_width)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        # A fully padded example uses one dummy key, then all of its outputs are
        # removed. This keeps softmax well-defined without exposing padding data.
        safe_keys = valid.clone()
        safe_keys[:, 0] |= ~valid.any(1)
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=safe_keys[:, None, None, :],
            dropout_p=self.dropout_probability if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, tokens, width)
        values = (values + self.dropout(self.attention_output(attended))
                  * self.attention_scale).masked_fill(invalid, 0)
        hidden = self.feedforward(self.feedforward_norm(values).masked_fill(invalid, 0))
        return (values + self.dropout(hidden) * self.feedforward_scale).masked_fill(invalid, 0)


def history_patch_indices(tokens):
    """Partition every sparse token exactly once; -1 marks unused patch slots.

Older one-second observations form patches of five. Ten recent half-second
observations form five pairs, and the current observation remains separate.
The layouts are 6, 9 and 17 patches for 11, 26 and 66 input tokens respectively.
"""
    sparse_time_coordinates(tokens)  # Reject unknown temporal layouts.
    older = tokens - 11
    patches = [list(range(start, min(start + 5, older))) for start in range(0, older, 5)]
    patches += [list(range(start, start + 2)) for start in range(older, tokens - 1, 2)]
    patches.append([tokens - 1])
    return torch.tensor([patch + [-1] * (5 - len(patch)) for patch in patches], dtype=torch.long)


class TransformerForecaster(nn.Module):
    """Temporal, multiscale-patch or explicitly hybrid Transformer classifier.

The constructor matches ImprovedForecaster. Temporal and patch variants use
the attention representation plus a direct current-state readout. The hybrid
also has a separate residual MLP over the original sparse history. Flattening
contextual tokens preserves their positions instead of reducing the entire
trajectory to only its final and mean states.
"""

    def __init__(self, architecture, input_size, tokens, num_classes=33, width=128,
                 layers=2, dropout=0.2, auxiliary=False):
        super().__init__()
        if architecture not in ("temporal_transformer", "patch_transformer", "hybrid_transformer"):
            raise ValueError("Unknown Transformer architecture")
        if width < 4 or width % 4 or layers < 1:
            raise ValueError("Transformers require width divisible by four and positive depth")
        self.architecture, self.input_size, self.tokens = architecture, input_size, tokens
        self.num_classes, self.horizons = num_classes, 4 if auxiliary else 3
        self.register_buffer("time_coordinates", sparse_time_coordinates(tokens))
        if architecture == "patch_transformer":
            self.register_buffer("patch_indices", history_patch_indices(tokens))
            encoded_tokens = len(self.patch_indices)
            self.projection = nn.Linear(5 * (input_size + 3), width)
        else:
            encoded_tokens = tokens
            self.projection = nn.Linear(input_size + 3, width)
        self.position = nn.Parameter(torch.randn(1, encoded_tokens, width) * 0.02)
        self.encoder = nn.ModuleList(MaskedAttentionBlock(width, dropout) for _ in range(layers))
        self.output_norm = nn.LayerNorm(width)
        self.attention_readout = nn.Sequential(
            nn.Linear(encoded_tokens * width, width), nn.GELU(), nn.Dropout(dropout),
        )
        if architecture == "hybrid_transformer":
            self.history_mlp = nn.Sequential(
                nn.Linear(tokens * (input_size + 1), width), nn.GELU(),
                DenseResidualBlock(width, dropout), DenseResidualBlock(width, dropout),
                nn.LayerNorm(width),
            )
        readout_width = (2 * width if architecture == "hybrid_transformer" else width) + input_size
        self.head = nn.Sequential(
            nn.LayerNorm(readout_width), nn.Linear(readout_width, width), nn.GELU(),
            nn.Dropout(dropout), DenseResidualBlock(width, dropout), nn.LayerNorm(width),
            nn.Linear(width, self.horizons * num_classes),
        )

    def _patch_values(self, values, valid):
        indices = self.patch_indices.clamp_min(0)
        patch_valid = valid[:, indices] & (self.patch_indices >= 0)[None]
        patched = values[:, indices].masked_fill(~patch_valid[:, :, :, None], 0)
        return patched.flatten(2), patch_valid.any(2)

    def forward(self, features, valid):
        if features.ndim != 3 or features.shape[1:] != (self.tokens, self.input_size):
            raise ValueError("Features do not match configured history dimensions")
        if valid.shape != features.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("Validity must be a boolean mask with one entry per token")
        features = features.masked_fill(~valid[:, :, None], 0)
        validity = valid[:, :, None].to(features.dtype)
        times = self.time_coordinates.to(features.dtype)[None].expand(len(features), -1, -1)
        values = torch.cat((features, times, validity), dim=-1).masked_fill(~valid[:, :, None], 0)
        if self.architecture == "patch_transformer":
            values, encoded_valid = self._patch_values(values, valid)
        else:
            encoded_valid = valid
        hidden = (self.projection(values) + self.position).masked_fill(~encoded_valid[:, :, None], 0)
        for layer in self.encoder:
            hidden = layer(hidden, encoded_valid)
        hidden = self.output_norm(hidden).masked_fill(~encoded_valid[:, :, None], 0)
        summaries = [self.attention_readout(hidden.flatten(1)), features[:, -1]]
        if self.architecture == "hybrid_transformer":
            original = torch.cat((features, validity), dim=-1).flatten(1)
            summaries.append(self.history_mlp(original))
        return self.head(torch.cat(summaries, dim=-1)).reshape(-1, self.horizons, self.num_classes)
