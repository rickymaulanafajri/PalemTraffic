import torch
import torch.nn as nn

from layers.temporal_conv import TemporalConv
from layers.graph_conv import GraphConvolution


class STConvBlock(nn.Module):
    """
    Spatio-Temporal Convolution Block

    Architecture:
        TemporalConv
            ↓
        GraphConvolution
            ↓
        ReLU
            ↓
        TemporalConv
            ↓
        Residual Connection
            ↓
        LayerNorm
            ↓
        Dropout

    Input:
        x   : (B, T, N, Cin)
        adj : (N, N)

    Output:
        (B, T, N, Cout)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        dropout=0.3,
    ):
        super().__init__()

        # First temporal convolution
        self.temporal1 = TemporalConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

        # Spatial graph convolution
        self.graph_conv = GraphConvolution(
            in_channels=out_channels,
            out_channels=out_channels,
        )

        # Second temporal convolution
        self.temporal2 = TemporalConv(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

        # Activation
        self.activation = nn.ReLU()

        # Residual projection
        if in_channels != out_channels:
            self.residual = nn.Linear(
                in_channels,
                out_channels
            )
        else:
            self.residual = nn.Identity()

        # Normalization
        self.norm = nn.LayerNorm(out_channels)

        # Regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        """
        Parameters
        ----------
        x : Tensor
            Shape (B,T,N,Cin)

        adj : Tensor
            Shape (N,N)

        Returns
        -------
        Tensor
            Shape (B,T,N,Cout)
        """

        residual = self.residual(x)

        out = self.temporal1(x)

        out = self.graph_conv(out, adj)

        out = self.activation(out)

        out = self.temporal2(out)

        out = out + residual

        out = self.norm(out)

        out = self.dropout(out)

        return out