import numpy as np

from torch.utils.data import DataLoader

from datasets.dataset import TrafficDataset
from datasets.scaler import StandardScaler


def load_dataset(
    npz_path,
    history=12,
    horizon=12,
    batch_size=64,
    train_ratio=0.7,
    val_ratio=0.1,
):

    data = np.load(npz_path)

    if isinstance(data, np.lib.npyio.NpzFile):
        data = data[list(data.keys())[0]]

    scaler = StandardScaler()

    train_end = int(len(data) * train_ratio)

    scaler.fit(data[:train_end])

    data = scaler.transform(data)

    val_end = int(len(data) * (train_ratio + val_ratio))

    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]

    train_dataset = TrafficDataset(
        train,
        history,
        horizon,
    )

    val_dataset = TrafficDataset(
        val,
        history,
        horizon,
    )

    test_dataset = TrafficDataset(
        test,
        history,
        horizon,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    metadata = {
    "num_nodes": data.shape[1],
    "input_dim": data.shape[2],
    "history": history,
    "horizon": horizon,
}   

    return (
        train_loader,
        val_loader,
        test_loader,
        scaler,
        metadata,
    )