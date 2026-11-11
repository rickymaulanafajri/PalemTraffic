import torch
from torch.utils.data import Dataset


class TrafficDataset(Dataset):

    def __init__(
        self,
        data,
        history=12,
        horizon=12,
        target_feature=0,
    ):

        self.data = data
        self.history = history
        self.horizon = horizon
        self.target_feature = target_feature

    def __len__(self):

        return len(self.data) - self.history - self.horizon + 1

    def __getitem__(self, idx):

        x = self.data[
            idx:idx + self.history
        ]

        y = self.data[
            idx + self.history:
            idx + self.history + self.horizon,
            :,
            self.target_feature
        ]

        return (
            torch.FloatTensor(x),
            torch.FloatTensor(y)
        )