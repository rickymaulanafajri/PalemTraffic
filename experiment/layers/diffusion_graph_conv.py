import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionGraphConvolution(nn.Module):
    """
    Diffusion Graph Convolution
    (Graph WaveNet - Wu et al., 2019)

    Supports:
        - Forward diffusion
        - Backward diffusion
        - Adaptive adjacency (optional)

    Input
    -----
    x:
        (B, T, N, Cin)

    supports:
        list of adjacency matrices
        each (N, N)

    Output
    ------
    (B, T, N, Cout)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        diffusion_steps=2,
        bias=True,
    ):
        super().__init__()

        self.diffusion_steps = diffusion_steps

        # Number of propagated features
        self.mlp = nn.Linear(
            (2 * diffusion_steps + 1) * in_channels,
            out_channels,
            bias=bias,
        )

    def propagate(self, x, support):
        """
        Diffusion propagation.

        support:
            (N,N)

        x:
            (B,T,N,F)
        """

        out = [x]

        x1 = x

        for _ in range(self.diffusion_steps):

            x1 = torch.einsum(
                "ij,btjf->btif",
                support,
                x1,
            )

            out.append(x1)

        return out

    def forward(
        self,
        x,
        supports,
    ):
        """
        Parameters
        ----------
        x:
            (B,T,N,C)

        supports:
            list of adjacency matrices

            Example:
                [forward_adj,
                 backward_adj]

            or

                [forward_adj,
                 backward_adj,
                 adaptive_adj]
        """

        diffusion_features = [x]

        for support in supports:

            propagated = self.propagate(
                x,
                support,
            )

            diffusion_features.extend(
                propagated[1:]
            )

        x = torch.cat(
            diffusion_features,
            dim=-1,
        )

        x = self.mlp(x)

        return x