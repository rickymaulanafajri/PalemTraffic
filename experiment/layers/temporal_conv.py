import torch
import torch.nn as nn


class TemporalConv(nn.Module):
    """
    Temporal Convolution

    Input:
        (B,T,N,F)

    Output:
        (B,T,N,Cout)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
    ):
        super().__init__()

        padding = ((kernel_size - 1) * dilation) // 2

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, 1),
            dilation=(dilation, 1),
            padding=(padding, 0),
        )

        self.activation = nn.ReLU()

    def forward(self, x):

        # (B,T,N,F)
        x = x.permute(0, 3, 1, 2)

        x = self.conv(x)

        x = self.activation(x)

        x = x.permute(0, 2, 3, 1)

        return x