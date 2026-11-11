import argparse
import os
import time

import pandas as pd
import torch
import torch.nn as nn

from datasets.dataloader import load_dataset
from datasets.adjacency import GraphLoader

from models import build_model

from utils.seed import set_seed
from utils.trainer import Trainer
from utils.metrics import mae, rmse, mape


MODELS = [
    "stgcn",
    "stsgcn",
    "stgat",
    "gcn_lstm",
    "graphsage_lstm",
    "stid",
]


def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def build_graph(model_name, metadata):

    loader = GraphLoader(metadata["num_nodes"])

    adjacency_models = {
        "stgcn",
        "stsgcn",
        "gcn_lstm",
    }

    edge_models = {
        "stgat",
        "graphsage_lstm",
    }

    if model_name in adjacency_models:
        return loader.identity()

    elif model_name in edge_models:
        return loader.identity_edge_index()

    return None


@torch.no_grad()
def evaluate_model(
    model,
    dataloader,
    graph,
    device,
):

    model.eval()

    if graph is not None:
        graph=graph.to(device)

    preds = []
    labels = []

    start = time.perf_counter()

    for x, y in dataloader:

        x = x.to(device)

        if graph is None:

            pred = model(x)

        else:

            pred = model(
                x,
                graph,
            )

        preds.append(
            pred.cpu()
        )

        labels.append(y.cpu())

    inference_time = time.perf_counter() - start

    preds = torch.cat(preds)

    labels = torch.cat(labels)

    return {

        "MAE": float(
            mae(preds, labels)
        ),

        "RMSE": float(
            rmse(preds, labels)
        ),

        "MAPE": float(
            mape(preds, labels)
        ),

        "Inference Time (s)": inference_time,

    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
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
        "--data",
        default="data/transgraphai_data_30days_multivariate.npz",
    )

    args = parser.parse_args()

    set_seed(42)

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )

    (
        train_loader,
        val_loader,
        test_loader,
        scaler,
        metadata,
    ) = load_dataset(

        npz_path=args.data,

        batch_size=args.batch_size,

    )

    benchmark = []

    for model_name in MODELS:

        print()
        print("=" * 80)
        print(f"Benchmark : {model_name.upper()}")
        print("=" * 80)

        model = build_model(
            model_name=model_name,
            **metadata,
        )

        graph = build_graph(
            model_name,
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
        )

        start = time.perf_counter()

        summary = trainer.fit(
            train_loader,
            val_loader,
            epochs=args.epochs,
        )

        #training_time = time.perf_counter() - start
        training_time=summary["training_time"]

        metrics = evaluate_model(
            model,
            test_loader,
            graph,
            device,
        )

        benchmark.append({

            "Model": model.name,

            "Parameters": count_parameters(model),

            "Best Epoch": summary["best_epoch"],

            "Best Loss": summary["best_loss"],

            "Training Time (s)": training_time,

            **metrics,

        })

    os.makedirs(
        "results",
        exist_ok=True,
    )

    benchmark = pd.DataFrame(
        benchmark
    )

    benchmark.to_csv(

        "results/benchmark.csv",

        index=False,

    )

    print()
    print("=" * 80)
    print("Benchmark Finished")
    print("=" * 80)

    print(benchmark)

    print()

    print(
        "Results saved to results/benchmark.csv"
    )


if __name__ == "__main__":

    main()