import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


class GraphSAGEConv(nn.Module):
    """
    GraphSAGE Layer

    Wrapper around PyTorch Geometric's SAGEConv.

    Input
    -----
        x : (B, T, N, F)

    edge_index
        (2, E)

    Output
    ------
        (B, T, N, out_channels)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        aggr="mean",
        normalize=False,
        bias=True,
    ):
        super().__init__()

        self.sage = SAGEConv(
            in_channels=in_channels,
            out_channels=out_channels,
            aggr=aggr,
            normalize=normalize,
            bias=bias,
        )

    def forward(
        self,
        x,
        edge_index,
    ):

        B, T, N, F = x.shape

        outputs = []

        for b in range(B):

            sequence = []

            for t in range(T):

                h = self.sage(
                    x[b, t],
                    edge_index,
                )

                sequence.append(h)

            outputs.append(
                torch.stack(sequence, dim=0)
            )

        return torch.stack(outputs, dim=0)