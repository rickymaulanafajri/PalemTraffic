import torch
import torch.nn as nn
import torch.nn.functional as F


class STSGGraphConv(nn.Module):
    """
    Graph Convolution used in STSGCN.

    Implements

        H = AHW

    where A is the localized spatial-temporal graph.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        activation="glu",
        bias=True,
    ):
        super().__init__()

        self.activation = activation.lower()

        if self.activation == "glu":

            self.linear = nn.Linear(
                in_channels,
                out_channels * 2,
                bias=bias,
            )

        else:

            self.linear = nn.Linear(
                in_channels,
                out_channels,
                bias=bias,
            )

    def forward(
        self,
        x,
        adj,
    ):
        """
        x
            (3N,C)

        adj
            (3N,3N)
        """

        x = torch.matmul(adj, x)

        x = self.linear(x)

        if self.activation == "glu":

            lhs, rhs = torch.chunk(
                x,
                chunks=2,
                dim=-1,
            )

            x = lhs * torch.sigmoid(rhs)

        else:

            x = F.relu(x)

        return x


class STSGCM(nn.Module):
    """
    Spatial-Temporal Synchronous Graph
    Convolution Module.
    """

    def __init__(
        self,
        num_nodes,
        in_channels,
        out_channels,
        num_layers=3,
        activation="glu",
    ):
        super().__init__()

        self.num_nodes = num_nodes

        self.layers = nn.ModuleList()

        current_dim = in_channels

        for _ in range(num_layers):

            self.layers.append(

                STSGGraphConv(
                    current_dim,
                    out_channels,
                    activation=activation,
                )

            )

            current_dim = out_channels

    def aggregate(
        self,
        outputs,
    ):
        """
        Element-wise max pooling.

        Paper Eq.(5)
        """

        outputs = torch.stack(
            outputs,
            dim=0,
        )

        outputs = torch.max(
            outputs,
            dim=0,
        )[0]

        return outputs

    def crop(
        self,
        x,
    ):
        """
        Keep only middle timestep.

        Input

            (3N,C)

        Output

            (N,C)
        """

        start = self.num_nodes

        end = start + self.num_nodes

        return x[start:end]

    def forward(
        self,
        x,
        adj,
    ):
        """
        x

            (3N,C)

        adj

            (3N,3N)
        """

        outputs = []

        h = x

        for layer in self.layers:

            h = layer(
                h,
                adj,
            )

            outputs.append(h)

        h = self.aggregate(outputs)

        h = self.crop(h)

        return h