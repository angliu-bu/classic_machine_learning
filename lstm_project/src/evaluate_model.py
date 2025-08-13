"""Evaluate a trained LSTM model and generate diagnostics."""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task: str) -> Dict[str, float]:
    """Compute evaluation metrics for regression or classification."""
    metrics: Dict[str, float] = {}
    y_pred_flat = y_pred.squeeze()
    metrics["mse"] = mean_squared_error(y_true, y_pred_flat)
    metrics["rmse"] = float(np.sqrt(metrics["mse"]))
    metrics["mae"] = mean_absolute_error(y_true, y_pred_flat)
    if task == "classification":
        if y_pred.shape[-1] > 1:
            y_classes = y_pred.argmax(axis=-1)
        else:
            y_classes = (y_pred_flat > 0.5).astype(int)
        metrics["accuracy"] = accuracy_score(y_true, y_classes)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained LSTM model.")
    parser.add_argument("--data_file", default="data/engineered_data.npz", help="Dataset to evaluate against (.npz).")
    parser.add_argument("--model_file", default="models/best_model.keras", help="Trained model file.")
    parser.add_argument("--history_file", default="results/history.csv", help="Training history CSV for loss plot.")
    parser.add_argument("--results_dir", default="results", help="Directory to save evaluation outputs.")
    parser.add_argument("--task", choices=["regression", "classification"], default="regression")
    args = parser.parse_args()

    data = np.load(args.data_file)
    X, y = data["X"], data["y"]

    model = tf.keras.models.load_model(args.model_file)
    preds = model.predict(X)

    metrics = compute_metrics(y, preds, args.task)

    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Plot training/validation loss
    hist_df = pd.read_csv(args.history_file)
    plt.figure()
    plt.plot(hist_df["loss"], label="train_loss")
    if "val_loss" in hist_df:
        plt.plot(hist_df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, "loss_eval.png"))
    plt.close()

    # Plot predicted vs actual values
    plt.figure()
    plt.plot(y[:100], label="actual")
    plt.plot(preds[:100], label="predicted")
    plt.xlabel("Sample")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, "pred_vs_actual.png"))
    plt.close()

    print("Evaluation metrics:", metrics)


if __name__ == "__main__":
    main()

