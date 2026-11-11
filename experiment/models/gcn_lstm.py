import torch
import torch.nn as nn

from models.base_model import BaseModel
from layers.graph_conv import GraphConvolution


class GCNLSTM(BaseModel):
    """
    Graph Convolution + LSTM

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
    ):
        super().__init__(
            num_nodes=num_nodes,
            input_dim=input_dim,
            history=history,
            horizon=horizon,
        )

        self.hidden_dim = hidden_dim

        # Spatial Encoder
        self.gcn = GraphConvolution(
            in_channels=input_dim,
            out_channels=hidden_dim,
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

        # Prediction Head
        self.output = nn.Linear(
            hidden_dim,
            horizon,
        )

    def forward(
        self,
        x,
        adj,
    ):
        """
        x
            (B,T,N,F)

        adj
            (N,N)
        """

        self.validate_input(x)

        # -----------------------------
        # Spatial Encoding
        # -----------------------------

        x = self.gcn(
            x,
            adj,
        )

        B, T, N, C = x.shape

        # -----------------------------
        # LSTM per Node
        # -----------------------------

        x = x.permute(
            0,
            2,
            1,
            3,
        )

        # (B,N,T,C)
        # ->
        # (B*N,T,C)

        x = x.reshape(
            B * N,
            T,
            C,
        )

        x, _ = self.lstm(x)

        # Last timestep

        x = x[:, -1]

        x = self.dropout(x)

        # Prediction

        x = self.output(x)

        # (B*N,H)
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