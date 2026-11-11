import torch
import torch.nn as nn

from layers.stsgcm import STSGCM


class STSGCL(nn.Module):
    """
    Spatial-Temporal Synchronous Graph Convolution Layer

    Input
    -----
        (B,T,N,C)

    Output
    ------
        (B,T-2,N,Cout)
    """

    def __init__(
        self,
        num_nodes,
        input_dim,
        output_dim,
        history,
        num_gcn_layers=3,
        activation="glu",
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.history = history

        # One STSGCM for each sliding window
        self.modules_list = nn.ModuleList()

        for _ in range(history - 2):

            self.modules_list.append(

                STSGCM(
                    num_nodes=num_nodes,
                    in_channels=input_dim,
                    out_channels=output_dim,
                    num_layers=num_gcn_layers,
                    activation=activation,
                )

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
            (B,T,N,C)

        localized_adj
            (3N,3N)

        Returns
        -------
            (B,T-2,N,Cout)
        """

        B, T, N, C = x.shape

        outputs = []

        for t in range(T - 2):

            window = x[:, t:t + 3]

            # (B,3,N,C)
            # ->
            # (B,3N,C)

            window = window.reshape(
                B,
                3 * N,
                C,
            )

            batch_output = []

            for b in range(B):

                out = self.modules_list[t](
                    window[b],
                    localized_adj,
                )

                batch_output.append(out)

            batch_output = torch.stack(
                batch_output,
                dim=0,
            )

            outputs.append(batch_output)

        outputs = torch.stack(
            outputs,
            dim=1,
        )

        return outputs