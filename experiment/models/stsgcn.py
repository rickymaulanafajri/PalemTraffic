import torch
import torch.nn as nn

from models.base_model import BaseModel
from layers.stsgcl import STSGCL


class STSGCN(BaseModel):
    """
    Spatial-Temporal Synchronous Graph Convolutional Network

    AAAI 2020

    Input
    -----
        (B, T, N, F)

    Output
    ------
        (B, H, N)
    """

    def __init__(
        self,
        num_nodes,
        input_dim,
        history,
        horizon,
        hidden_dim=64,
        num_layers=2,
        num_gcn_layers=3,
        activation="glu",
    ):
        super().__init__(
            num_nodes=num_nodes,
            input_dim=input_dim,
            history=history,
            horizon=horizon,
        )

        self.hidden_dim = hidden_dim
        self.history = history
        self.horizon = horizon

        # --------------------------------------------------
        # Input Projection
        # --------------------------------------------------

        self.input_proj = nn.Linear(
            input_dim,
            hidden_dim,
        )

        # --------------------------------------------------
        # STSGCL Stack
        # --------------------------------------------------

        self.stsgcl_layers = nn.ModuleList()

        current_history = history

        for _ in range(num_layers):

            self.stsgcl_layers.append(

                STSGCL(
                    num_nodes=num_nodes,
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    history=current_history,
                    num_gcn_layers=num_gcn_layers,
                    activation=activation,
                )

            )

            current_history -= 2

        self.final_history = current_history

        if self.final_history <= 0:

            raise ValueError(
                "Too many STSGCL layers for the given history."
            )

        # --------------------------------------------------
        # Output Projection
        # --------------------------------------------------

        self.output = nn.Linear(
            self.final_history * hidden_dim,
            horizon,
        )

    def forward(
        self,
        x,
        localized_adj,
    ):
        """
        Parameters
        ----------
        x
            (B,T,N,F)

        localized_adj
            (3N,3N)
        """

        self.validate_input(x)

        x = self.input_proj(x)

        for layer in self.stsgcl_layers:

            x = layer(
                x,
                localized_adj,
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

        x = x.reshape(
            B,
            N,
            T * C,
        )

        x = self.output(x)

        # (B,N,H)
        # ->
        # (B,H,N)

        x = x.permute(
            0,
            2,
            1,
        )

        return x