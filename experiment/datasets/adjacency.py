import os
import numpy as np
import pandas as pd
import torch


class GraphLoader:
    """
    Graph Utility Class

    Supports:

    1. Identity Graph
    2. CSV Adjacency Matrix
    3. CSV Edge Index
    4. Adjacency -> Edge Index
    5. Edge Index -> Adjacency
    """

    def __init__(self, num_nodes):

        self.num_nodes = num_nodes

    # ======================================================
    # Identity Graph
    # ======================================================

    def identity(self):

        return torch.eye(
            self.num_nodes,
            dtype=torch.float32,
        )

    def identity_edge_index(self):

        idx = torch.arange(
            self.num_nodes,
            dtype=torch.long,
        )

        return torch.stack(
            [idx, idx],
            dim=0,
        )

    # ======================================================
    # CSV Loader
    # ======================================================

    def from_adjacency_csv(
        self,
        csv_path,
    ):

        if not os.path.exists(csv_path):

            raise FileNotFoundError(csv_path)

        A = pd.read_csv(
            csv_path,
            header=None,
        ).values

        return torch.FloatTensor(A)

    def from_edge_index_csv(
        self,
        csv_path,
    ):

        if not os.path.exists(csv_path):

            raise FileNotFoundError(csv_path)

        edge = pd.read_csv(
            csv_path,
            header=None,
        ).values

        return torch.LongTensor(
            edge.T
        )

    # ======================================================
    # Conversion
    # ======================================================

    def adjacency_to_edge_index(
        self,
        adjacency,
    ):

        if isinstance(
            adjacency,
            np.ndarray,
        ):

            adjacency = torch.FloatTensor(
                adjacency
            )

        edge_index = adjacency.nonzero(
            as_tuple=False
        ).t()

        return edge_index.long()

    def edge_index_to_adjacency(
        self,
        edge_index,
    ):

        adjacency = torch.zeros(
            (
                self.num_nodes,
                self.num_nodes,
            ),
            dtype=torch.float32,
        )

        adjacency[
            edge_index[0],
            edge_index[1],
        ] = 1.0

        return adjacency

    # ======================================================
    # Utility
    # ======================================================

    def print_info(
        self,
        graph,
    ):

        if graph.dim() == 2:

            print("Nodes :", graph.shape[0])

            print(
                "Edges :",
                int(graph.sum().item()),
            )

        else:

            print(graph.shape)