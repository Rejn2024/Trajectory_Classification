import torch
from torch import nn


class CausalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__(); padding = (kernel_size - 1) * dilation
        self.padding = padding
        self.net = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size,
                                            padding=padding, dilation=dilation),
                                 nn.ReLU(), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)[..., :-self.padding] if self.padding else self.net(x)


class TCNEncoder(nn.Module):
    def __init__(self, input_dim, channels=(64, 128), kernel_size=3, dropout=0.1):
        super().__init__(); blocks = []; previous = input_dim
        for i, channel in enumerate(channels):
            blocks.append(CausalBlock(previous, channel, kernel_size, 2 ** i, dropout)); previous = channel
        self.network = nn.Sequential(*blocks); self.output_dim = previous
    def forward(self, x): return self.network(x.transpose(1, 2))[..., -1]

