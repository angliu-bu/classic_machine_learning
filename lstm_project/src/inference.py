"""Run batch inference using a trained LSTM model."""
from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf


def load_input_sequences(path: Optional[str], input_shape: tuple[int, int]) -> np.ndarray:
    """Load input sequences from a file or generate a random sample."""
    if path and os.path.isfile(path):
        if path.endswith(".npz"):
            data = np.load(path)
            X = data["X"]
        else:
            df = pd.read_csv(path)
            X = df.values.reshape((-1, input_shape[0], input_shape[1]))
    else:
        # Generate a random sample
        X = np.random.rand(1, *input_shape)
        print("No input file provided; using a random sample:")
        print(X)
    return X


def main() -> None:
    parser = argparse.ArgumentParser(description="Perform inference with a trained LSTM model.")
    parser.add_argument("--model_file", default="models/best_model.keras", help="Path to trained model.")
    parser.add_argument("--input_file", help="Optional input file (.npz or .csv) containing sequences.")
    parser.add_argument(
        "--output_file", default="results/predictions.csv", help="File to save prediction results.")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model_file)
    seq_len, n_features = model.input_shape[1:3]
    X = load_input_sequences(args.input_file, (seq_len, n_features))

    preds = model.predict(X)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    pd.DataFrame(preds, columns=["prediction"]).to_csv(args.output_file, index=False)

    print("Predictions saved to", args.output_file)
    print("Sample prediction:", preds[0])


if __name__ == "__main__":
    main()

