import torch.nn as nn


class BaseModel(nn.Module):
    """
    Base class for all models in the benchmark framework.

    Every model should inherit from this class.

    Input:
        (Batch, History, Nodes, Features)

    Output:
        (Batch, Horizon, Nodes)
    """

    def __init__(
        self,
        num_nodes,
        input_dim,
        history,
        horizon,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.history = history
        self.horizon = horizon

    @property
    def name(self):
        return self.__class__.__name__

    def count_parameters(self):
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

    def validate_input(self, x):
        """
        Validate the input tensor shape.
        Expected shape:
            (Batch, History, Nodes, Features)
        """

        if x.ndim != 4:
            raise ValueError(
                f"{self.name} expects a 4D tensor "
                f"(Batch, History, Nodes, Features), "
                f"but got {tuple(x.shape)}"
            )

        if x.shape[1] != self.history:
            raise ValueError(
                f"Expected history={self.history}, "
                f"but got {x.shape[1]}"
            )

        if x.shape[2] != self.num_nodes:
            raise ValueError(
                f"Expected {self.num_nodes} nodes, "
                f"but got {x.shape[2]}"
            )

        if x.shape[3] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} features, "
                f"but got {x.shape[3]}"
            )

    def forward(self, x):
        raise NotImplementedError(
            "Subclasses must implement forward()."
        )