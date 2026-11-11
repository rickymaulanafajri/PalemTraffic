import torch
import torch.nn as nn

from models.base_model import BaseModel


class STID(BaseModel):
    """
    Spatial-Temporal Identity (STID)

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
        num_layers=3,
        dropout=0.3,
    ):
        super().__init__(
            num_nodes=num_nodes,
            input_dim=input_dim,
            history=history,
            horizon=horizon,
        )

        self.hidden_dim = hidden_dim

        # Feature embedding
        self.input_proj = nn.Linear(
            input_dim,
            hidden_dim,
        )

        # MLP Encoder
        mlp = []

        for _ in range(num_layers):

            mlp.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])

        self.encoder = nn.Sequential(*mlp)

        # Prediction Head
        self.output = nn.Linear(
            history * hidden_dim,
            horizon,
        )

    def forward(
        self,
        x,
        *_,
    ):
        """
        Ignore adjacency / edge_index.
        """

        self.validate_input(x)

        x = self.input_proj(x)

        x = self.encoder(x)

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