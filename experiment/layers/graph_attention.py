import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class GraphAttention(nn.Module):
    """
    Graph Attention Layer

    Input:
        x : (B,T,N,F)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        heads=4,
        concat=False,
    ):
        super().__init__()

        self.gat = GATConv(
            in_channels,
            out_channels,
            heads=heads,
            concat=concat,
        )

    def forward(self, x, edge_index):

        B, T, N, F = x.shape

        outputs = []

        for b in range(B):

            sequence = []

            for t in range(T):

                h = self.gat(
                    x[b, t],
                    edge_index,
                )

                sequence.append(h)

            outputs.append(torch.stack(sequence))

        return torch.stack(outputs)