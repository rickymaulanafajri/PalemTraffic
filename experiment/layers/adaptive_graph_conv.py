import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGraphConvolution(nn.Module):
    """
    Adaptive Graph Convolution used in Graph WaveNet.

    Learns an adaptive adjacency matrix from trainable node embeddings.

    Input
    -----
    x:
        (B, T, N, Cin)

    Output
    ------
    (B, T, N, Cout)
    """

    def __init__(
        self,
        num_nodes,
        in_channels,
        out_channels,
        embedding_dim=10,
        bias=True,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim

        # Learnable node embeddings
        self.nodevec1 = nn.Parameter(
            torch.randn(num_nodes, embedding_dim)
        )

        self.nodevec2 = nn.Parameter(
            torch.randn(embedding_dim, num_nodes)
        )

        # Feature projection
        self.linear = nn.Linear(
            in_channels,
            out_channels,
            bias=bias,
        )

    def adaptive_adj(self):
        """
        Compute adaptive adjacency matrix.

        A = softmax(relu(E1 @ E2))
        """

        adj = torch.mm(
            self.nodevec1,
            self.nodevec2
        )

        adj = F.relu(adj)

        adj = F.softmax(adj, dim=1)

        return adj

    def forward(self, x):

        adj = self.adaptive_adj()

        # Graph aggregation
        x = torch.einsum(
            "ij,btjf->btif",
            adj,
            x,
        )

        x = self.linear(x)

        return x