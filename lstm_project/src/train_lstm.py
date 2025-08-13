"""Train an LSTM model on engineered sequence data."""
from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def build_model(
    input_shape: Tuple[int, int],
    layers: int,
    units: int,
    dropout: float,
    learning_rate: float,
    task: str,
    num_classes: int,
) -> tf.keras.Model:
    """Create a compiled LSTM model."""
    model = Sequential()
    for i in range(layers):
        return_sequences = i < layers - 1
        model.add(
            LSTM(
                units,
                return_sequences=return_sequences,
                input_shape=input_shape if i == 0 else None,
            )
        )
        if dropout > 0:
            model.add(Dropout(dropout))

    if task == "classification":
        if num_classes <= 2:
            model.add(Dense(1, activation="sigmoid"))
            loss = "binary_crossentropy"
        else:
            model.add(Dense(num_classes, activation="softmax"))
            loss = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]
    else:
        model.add(Dense(1))
        loss = "mse"
        metrics = ["mse"]

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss, metrics=metrics)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an LSTM model.")
    parser.add_argument("--data_file", default="data/engineered_data.npz", help="Path to engineered dataset (.npz).")
    parser.add_argument("--model_dir", default="models", help="Directory to store trained models.")
    parser.add_argument("--results_dir", default="results", help="Directory to store training results.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--units", type=int, default=50)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--validation_split", type=float, default=0.2)
    parser.add_argument("--task", choices=["regression", "classification"], default="regression")
    args = parser.parse_args()

    data = np.load(args.data_file)
    X, y = data["X"], data["y"]

    if args.task == "classification":
        num_classes = len(np.unique(y))
    else:
        num_classes = 1

    model = build_model(
        input_shape=(X.shape[1], X.shape[2]),
        layers=args.layers,
        units=args.units,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        task=args.task,
        num_classes=num_classes,
    )

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    checkpoint_path = os.path.join(args.model_dir, "best_model.keras")
    callbacks = [
        ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_loss"),
        EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss"),
    ]

    history = model.fit(
        X,
        y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final model
    model.save(os.path.join(args.model_dir, "final_model.keras"))

    # Save training history
    hist_df = pd.DataFrame(history.history)
    hist_csv = os.path.join(args.results_dir, "history.csv")
    hist_df.to_csv(hist_csv, index=False)

    # Plot loss curves
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(hist_df["loss"], label="train_loss")
    if "val_loss" in hist_df:
        plt.plot(hist_df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, "loss.png"))
    plt.close()

    print(f"Training complete. Best model stored at {checkpoint_path}")


if __name__ == "__main__":
    main()

