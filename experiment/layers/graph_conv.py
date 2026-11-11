import torch
import torch.nn as nn


class GraphConvolution(nn.Module):
    """
    Basic Graph Convolution Layer

    Input:
        x : (B, T, N, Cin)
        adj : (N, N)

    Output:
        (B, T, N, Cout)
    """

    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()

        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x, adj):

        # Graph aggregation
        x = torch.einsum("ij,btjf->btif", adj, x)

        # Feature transformation
        x = self.linear(x)

        return x