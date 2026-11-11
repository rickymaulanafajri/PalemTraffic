import torch
import torch.nn as nn

from models.base_model import BaseModel
from layers.temporal_conv import TemporalConv
from layers.graph_attention import GraphAttention


class STGAT(BaseModel):
    """
    Spatial-Temporal Graph Attention Network

    Input
    -----
        x : (B,T,N,F)

    Output
    ------
        (B,H,N)
    """

    def __init__(
        self,
        num_nodes,
        input_dim,
        history,
        horizon,
        hidden_dim=64,
        heads=4,
        dropout=0.3,
    ):
        super().__init__(
            num_nodes=num_nodes,
            input_dim=input_dim,
            history=history,
            horizon=horizon,
        )

        self.temporal1 = TemporalConv(
            input_dim,
            hidden_dim,
        )

        self.gat = GraphAttention(
            hidden_dim,
            hidden_dim,
            heads=heads,
            concat=False,
        )

        self.temporal2 = TemporalConv(
            hidden_dim,
            hidden_dim,
        )

        self.dropout = nn.Dropout(dropout)

        self.output = nn.Linear(
            history * hidden_dim,
            horizon,
        )

    def forward(
        self,
        x,
        edge_index,
    ):
        """
        Parameters
        ----------
        x
            (B,T,N,F)

        edge_index
            (2,E)
        """

        self.validate_input(x)

        x = self.temporal1(x)

        x = self.gat(
            x,
            edge_index,
        )

        x = self.temporal2(x)

        x = self.dropout(x)

        B, T, N, C = x.shape

        x = x.permute(
            0,
            2,
            1,
            3,
        )

        x = x.reshape(
            B,
            N,
            T * C,
        )

        x = self.output(x)

        x = x.permute(
            0,
            2,
            1,
        )

        return x