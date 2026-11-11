import torch
import torch.nn as nn

from models.base_model import BaseModel
from layers.st_conv_block import STConvBlock


class STGCN(BaseModel):
    """
    Spatio-Temporal Graph Convolution Network

    Input
        (B,T,N,F)

    Output
        (B,H,N)
    """

    def __init__(
        self,
        num_nodes,
        input_dim,
        history,
        horizon,
        hidden_dim=64,
        num_blocks=2,
        kernel_size=3,
        dropout=0.3,
    ):

        super().__init__(
            num_nodes=num_nodes,
            input_dim=input_dim,
            history=history,
            horizon=horizon,
        )

        self.blocks = nn.ModuleList()

        in_channels = input_dim

        for _ in range(num_blocks):

            self.blocks.append(

                STConvBlock(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )

            )

            in_channels = hidden_dim

        # Prediction layer
        self.output = nn.Linear(
            history * hidden_dim,
            horizon
        )

    def forward(self, x, adj):

        """
        x:
            (B,T,N,F)

        adj:
            (N,N)
        """

        self.validate_input(x)

        for block in self.blocks:

            x = block(x, adj)

        B, T, N, C = x.shape

        # (B,T,N,C)
        # ->
        # (B,N,T*C)

        x = x.permute(0, 2, 1, 3)

        x = x.reshape(B, N, T * C)

        x = self.output(x)

        # (B,N,H)
        #->
        # (B,H,N)

        x = x.permute(0, 2, 1)

        return x