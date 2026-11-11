import torch
import torch.nn as nn

from models.base_model import BaseModel
from layers.graphsage_conv import GraphSAGEConv


class GraphSAGELSTM(BaseModel):
    """
    GraphSAGE + LSTM

    Input
    -----
        (B,T,N,F)

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
        num_layers=1,
        dropout=0.3,
        aggr="mean",
    ):
        super().__init__(
            num_nodes=num_nodes,
            input_dim=input_dim,
            history=history,
            horizon=horizon,
        )

        self.hidden_dim = hidden_dim

        # Spatial Encoder
        self.graphsage = GraphSAGEConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            aggr=aggr,
        )

        # Temporal Encoder
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # Prediction
        self.output = nn.Linear(
            hidden_dim,
            horizon,
        )

    def forward(
        self,
        x,
        edge_index,
    ):

        self.validate_input(x)

        x = self.graphsage(
            x,
            edge_index,
        )

        B, T, N, C = x.shape

        # (B,T,N,C)
        # ->
        # (B,N,T,C)

        x = x.permute(
            0,
            2,
            1,
            3,
        )

        # ->
        # (B*N,T,C)

        x = x.reshape(
            B * N,
            T,
            C,
        )

        x, _ = self.lstm(x)

        x = x[:, -1]

        x = self.dropout(x)

        x = self.output(x)

        # ->
        # (B,N,H)

        x = x.reshape(
            B,
            N,
            self.horizon,
        )

        # ->
        # (B,H,N)

        x = x.permute(
            0,
            2,
            1,
        )

        return x