import os
import time
import copy
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.metrics import mae, rmse, mape


class Trainer:

    def __init__(
    self,
    model,
    optimizer,
    criterion,
    device,
    graph=None,
    scheduler=None,
    checkpoint_dir="checkpoints",
    results_dir="results",
    patience=20,
):

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.graph=graph
        if self.graph is not None:
            self.graph=self.graph.to(device)
        self.scheduler = scheduler

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.results_dir = results_dir

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        self.patience = patience

        self.best_loss = float("inf")
        self.best_epoch = 0
        self.best_state = None

        self.model.to(device)

    ########################################################
    # Internal Forward
    ########################################################

    def _forward(self, batch):

        x, y = batch

        x    = x.to(self.device)
        y = y.to(self.device)

        if self.graph is None:

            prediction = self.model(x)

        else:

            prediction = self.model(
            x,
            self.graph,
        )

        return prediction, y

    ########################################################
    # Train
    ########################################################

    def train_epoch(
        self,
        dataloader,
    ):

        self.model.train()

        total_loss = 0

        for batch in tqdm(
            dataloader,
            leave=False,
        ):

            prediction, target = self._forward(batch)

            loss = self.criterion(
                prediction,
                target,
            )

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    ########################################################
    # Validation
    ########################################################

    @torch.no_grad()
    def validate(
        self,
        dataloader,
    ):

        self.model.eval()

        total_loss = 0

        predictions = []
        targets = []

        for batch in tqdm(
            dataloader,
            leave=False,
        ):

            prediction, target = self._forward(batch)

            loss = self.criterion(
                prediction,
                target,
            )

            total_loss += loss.item()

            predictions.append(prediction.cpu())

            targets.append(target.cpu())

        predictions = torch.cat(predictions)

        targets = torch.cat(targets)

        metrics = {

            "loss": total_loss / len(dataloader),

            "mae": mae(
                predictions,
                targets,
            ),

            "rmse": rmse(
                predictions,
                targets,
            ),

            "mape": mape(
                predictions,
                targets,
            ),
        }

        return metrics

    ########################################################
    # Fit
    ########################################################

    def fit(
        self,
        train_loader,
        val_loader,
        epochs,
    ):

        patience_counter = 0

        history = []

        start = time.time()

        for epoch in range(epochs):

            train_loss = self.train_epoch(
                train_loader,
            )

            val_metrics = self.validate(
                val_loader,
            )

            if self.scheduler is not None:

                self.scheduler.step()

            print(

                f"Epoch {epoch+1}/{epochs} | "
                f"Train {train_loss:.4f} | "
                f"Val {val_metrics['loss']:.4f} | "
                f"MAE {val_metrics['mae']:.4f}"

            )

            history.append({

                "epoch": epoch + 1,

                "train_loss": train_loss,

                **val_metrics,

            })

            if val_metrics["loss"] < self.best_loss:

                self.best_loss = val_metrics["loss"]

                self.best_epoch = epoch + 1

                patience_counter = 0

                self.best_state = copy.deepcopy(
                    self.model.state_dict()
                )

                self.save_checkpoint()

            else:

                patience_counter += 1

            if patience_counter >= self.patience:

                print("Early stopping.")

                break

        elapsed = time.time() - start

        print(f"Training Time : {elapsed:.2f} sec")

        if self.best_state is not None:

            self.model.load_state_dict(
                self.best_state
            )

        history_df = pd.DataFrame(history)

        history_file = os.path.join(
            self.results_dir,
            f"{self.model.name.lower()}_history.csv",
        )

        history_df.to_csv(
            history_file,
            index=False,
        )

        summary = {
            "history": history_df,
            "best_epoch": self.best_epoch,
            "best_loss": self.best_loss,
            "training_time": elapsed,
        }

        print()
        print("=" * 60)
        print("Training Summary")
        print("=" * 60)
        print(f"Best Epoch    : {self.best_epoch}")
        print(f"Best Loss     : {self.best_loss:.6f}")
        print(f"Training Time : {elapsed:.2f} sec")
        print(f"History Saved : {history_file}")

        return summary

    ########################################################
    # Test
    ########################################################

    @torch.no_grad()
    def test(
        self,
        test_loader,
    ):

        metrics = self.validate(
            test_loader,
        )

        print()

        print("=" * 60)

        print("Test Result")

        print("=" * 60)

        for k, v in metrics.items():

            print(f"{k:10s}: {v:.6f}")

        return metrics

    ########################################################
    # Predict
    ########################################################

    @torch.no_grad()
    def predict(
        self,
        dataloader,
    ):

        self.model.eval()

        outputs = []

        for batch in dataloader:

            prediction, _ = self._forward(batch)

            outputs.append(
                prediction.cpu()
            )

        return torch.cat(outputs)

    ########################################################
    # Checkpoint
    ########################################################

    def save_checkpoint(self):

        filename = os.path.join(

            self.checkpoint_dir,

            f"{self.model.name.lower()}_best.pt"

        )

        torch.save(

            {

                "model_name": self.model.name,
                "model_state_dict": self.model.state_dict(),
                "best_loss":self.best_loss,
                "best_epoch":self.best_epoch,

            },

            filename,

        )

    ########################################################
    # Load
    ########################################################

    def load_checkpoint(
        self,
        filename,
    ):

        checkpoint = torch.load(

            filename,

            map_location=self.device,

        )

        self.model.load_state_dict(

            checkpoint["model_state_dict"]

        )

        self.best_loss = checkpoint["best_loss"]

        self.best_epoch = checkpoint["best_epoch"]

        print(

            f"Loaded checkpoint from epoch "

            f"{self.best_epoch}"

        )