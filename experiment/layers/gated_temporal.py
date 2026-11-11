import torch
import torch.nn as nn


class GatedTemporalConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.filter_conv = nn.Conv2d(
            in_channels,
            out_channels,
            (kernel_size, 1),
            padding=(padding, 0),
        )

        self.gate_conv = nn.Conv2d(
            in_channels,
            out_channels,
            (kernel_size, 1),
            padding=(padding, 0),
        )

    def forward(self, x):

        x = x.permute(0, 3, 1, 2)

        filter_out = torch.tanh(
            self.filter_conv(x)
        )

        gate = torch.sigmoid(
            self.gate_conv(x)
        )

        out = filter_out * gate

        out = out.permute(0, 2, 3, 1)

        return out