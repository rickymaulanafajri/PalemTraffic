import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

from datasets.dataloader import load_dataset
from datasets.adjacency import GraphLoader

from models import build_model

from utils.metrics import mae, rmse, mape
from utils.seed import set_seed


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
def evaluate(
    model,
    dataloader,
    graph,
    device,
):

    model.eval()

    predictions = []
    targets = []

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

        predictions.append(
            pred.cpu()
        )

        targets.append(
            y.cpu()
        )

    inference_time = time.perf_counter() - start

    predictions = torch.cat(predictions)

    targets = torch.cat(targets)

    error = torch.abs(
        predictions - targets
    )

    result = {

        "MAE": float(
            mae(predictions, targets)
        ),

        "RMSE": float(
            rmse(predictions, targets)
        ),

        "MAPE": float(
            mape(predictions, targets)
        ),

        "Inference Time (s)": inference_time,

        "Parameters": count_parameters(model),

    }

    return result, predictions, targets, error


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--data",
        default="data/transgraphai_data_30days_multivariate.npz",
    )

    args = parser.parse_args()

    set_seed(1)

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

    model = build_model(
        model_name=args.model,
        **metadata,
    )

    graph = build_graph(
        args.model,
        metadata,
    )

    model.to(device)

    if graph is not None:
        graph = graph.to(device)

    checkpoint = args.checkpoint

    if checkpoint is None:

        checkpoint = os.path.join(
            "checkpoints",
            f"{model.name.lower()}_best.pt"
        )

    print("=" * 60)
    print("Loading Checkpoint")
    print("=" * 60)
    print(checkpoint)

    ckpt = torch.load(
        checkpoint,
        map_location=device,
    )

    model.load_state_dict(
        ckpt["model_state_dict"]
    )

    print("Checkpoint Loaded.")
    print()

    print("=" * 60)
    print("Evaluating")
    print("=" * 60)

    (
        result,
        predictions,
        targets,
        error,
    ) = evaluate(

        model,
        test_loader,
        graph,
        device,

    )

    print()

    print("=" * 60)
    print("Evaluation Result")
    print("=" * 60)

    for k, v in result.items():

        print(f"{k:20s}: {v}")

    #######################################################
    # Save evaluation csv
    #######################################################

    os.makedirs(
        "results",
        exist_ok=True,
    )

    csv_file = os.path.join(
        "results",
        f"{args.model}_evaluation.csv",
    )

    pd.DataFrame(
        [result]
    ).to_csv(
        csv_file,
        index=False,
    )

    #######################################################
    # Save predictions
    #######################################################

    prediction_dir = os.path.join(
        "results",
        "predictions",
    )

    os.makedirs(
        prediction_dir,
        exist_ok=True,
    )

    np.save(
        os.path.join(
            prediction_dir,
            f"{args.model}_prediction.npy",
        ),
        predictions.numpy(),
    )

    np.save(
        os.path.join(
            prediction_dir,
            f"{args.model}_ground_truth.npy",
        ),
        targets.numpy(),
    )

    np.save(
        os.path.join(
            prediction_dir,
            f"{args.model}_error.npy",
        ),
        error.numpy(),
    )

    #######################################################

    print()
    print(f"Evaluation saved : {csv_file}")

    print()

    print("Prediction files")

    print("------------------------------")

    print(f"{args.model}_prediction.npy")

    print(f"{args.model}_ground_truth.npy")

    print(f"{args.model}_error.npy")


if __name__ == "__main__":

    main()