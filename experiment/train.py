import argparse

import torch
import torch.nn as nn

from datasets.dataloader import load_dataset
from datasets.adjacency import GraphLoader

from models import build_model

from utils.seed import set_seed
from utils.trainer import Trainer


def build_graph(model_name, metadata):
    """
    Build the graph representation required by the selected model.
    """

    loader = GraphLoader(metadata["num_nodes"])

    adjacency_models = {
        "stgcn",
        "stsgcn",
        "gcn_lstm",
    }

    edge_index_models = {
        "stgat",
        "graphsage_lstm",
    }

    if model_name in adjacency_models:
        return loader.identity()

    elif model_name in edge_index_models:
        return loader.identity_edge_index()

    else:
        return None


def main():

    parser = argparse.ArgumentParser(
        description="Train Traffic Forecasting Models"
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Model name",
    )

    parser.add_argument(
        "--data",
        default="data/transgraphai_data_30days_multivariate.npz",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("Loading Dataset")
    print("=" * 60)

    (
        train_loader,
        val_loader,
        test_loader,
        scaler,
        metadata,
    ) = load_dataset(
        npz_path=args.data,
        history=12,
        horizon=12,
        batch_size=args.batch_size,
    )

    print(f"Number of Nodes : {metadata['num_nodes']}")
    print(f"Input Features  : {metadata['input_dim']}")
    print(f"History Length  : {metadata['history']}")
    print(f"Horizon         : {metadata['horizon']}")
    print()

    print("=" * 60)
    print(f"Building Model : {args.model}")
    print("=" * 60)

    model = build_model(
        model_name=args.model,
        **metadata,
    )

    graph = build_graph(
        args.model,
        metadata,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    criterion = nn.MSELoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        graph=graph,
        patience=20,
    )

    print()
    print("=" * 60)
    print("Training")
    print("=" * 60)

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
    )

    print()
    print("=" * 60)
    print("Testing")
    print("=" * 60)

    trainer.test(
        test_loader,
    )


if __name__ == "__main__":

    main()