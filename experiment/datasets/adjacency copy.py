import os
import numpy as np
import pandas as pd
import torch


class GraphLoader:
    """
    Utility class for loading graph structures.

    Supported:
        1. adjacency matrix (.csv)
        2. edge index (.csv)
        3. identity graph
    """

    def __init__(self, num_nodes):

        self.num_nodes = num_nodes

    def identity(self):

        A = np.eye(self.num_nodes, dtype=np.float32)

        return torch.FloatTensor(A)

    def load_adjacency(self, csv_path):

        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)

        A = pd.read_csv(csv_path, header=None).values

        return torch.FloatTensor(A)

    def load_edge_index(self, csv_path):

        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)

        edge = pd.read_csv(csv_path, header=None).values

        edge = torch.LongTensor(edge.T)

        return edge