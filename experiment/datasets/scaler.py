import numpy as np


class StandardScaler:
    """
    Standardization using mean and standard deviation.

    x_scaled = (x - mean) / std
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):

        self.mean = np.mean(data, axis=(0, 1), keepdims=True)
        self.std = np.std(data, axis=(0, 1), keepdims=True)

        self.std[self.std == 0] = 1.0

    def transform(self, data):

        return (data - self.mean) / self.std

    def inverse_transform(self, data):

        return data * self.std + self.mean

    def fit_transform(self, data):

        self.fit(data)
        return self.transform(data)