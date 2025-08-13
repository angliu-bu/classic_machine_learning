"""Data preprocessing utilities for LSTM models.

This script loads a time-series dataset, handles missing values, normalizes
features, creates sliding windows for sequence modelling, and saves the
processed dataset to disk.
"""
from __future__ import annotations

import argparse
import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values using interpolation and back/forward fill."""
    return df.interpolate(method="linear").fillna(method="bfill").fillna(method="ffill")


def scale_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """Scale features using :class:`~sklearn.preprocessing.StandardScaler`."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df.values)
    scaled_df = pd.DataFrame(scaled, columns=df.columns, index=df.index)
    return scaled_df, scaler


def create_sliding_windows(
    df: pd.DataFrame, target_col: str, seq_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate sliding windows from a dataframe.

    Parameters
    ----------
    df: pd.DataFrame
        Preprocessed dataframe.
    target_col: str
        Name of the target column.
    seq_len: int
        Number of time steps per window.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Arrays ``X`` (samples, seq_len, features) and ``y`` (targets).
    """
    data = df.values
    target_idx = df.columns.get_loc(target_col)
    X, y = [], []
    for i in range(len(df) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len, target_idx])
    return np.array(X), np.array(y)


def main() -> None:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Preprocess time-series data for LSTM models.")
    parser.add_argument("--input_file", required=True, help="Path to raw CSV file.")
    parser.add_argument("--target_column", required=True, help="Name of the target column.")
    parser.add_argument("--sequence_length", type=int, default=30, help="Length of sliding window.")
    parser.add_argument("--output_dir", default="data", help="Directory to save preprocessed data.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input_file)
    df = handle_missing_values(df)
    scaled_df, scaler = scale_features(df)

    # Save scaled dataframe
    preprocessed_csv = os.path.join(args.output_dir, "preprocessed.csv")
    scaled_df.to_csv(preprocessed_csv, index=False)

    # Create sliding windows
    X, y = create_sliding_windows(scaled_df, args.target_column, args.sequence_length)
    np.savez(os.path.join(args.output_dir, "preprocessed_data.npz"), X=X, y=y)

    # Persist scaler for reuse during inference
    joblib.dump(scaler, os.path.join(args.output_dir, "scaler.pkl"))

    print(f"Saved preprocessed data to {args.output_dir}")


if __name__ == "__main__":
    main()

