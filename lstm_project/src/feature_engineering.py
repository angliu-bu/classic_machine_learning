"""Feature engineering for time-series data.

This script adds temporal features such as lags, moving averages and Fourier
components. It also creates padded sequences ready for LSTM input.
"""
from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences


def add_lag_features(df: pd.DataFrame, lags: int) -> pd.DataFrame:
    """Add lag features for each column."""
    for col in df.columns:
        for lag in range(1, lags + 1):
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df


def add_moving_averages(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Append moving average features."""
    for col in df.columns:
        df[f"{col}_ma_{window}"] = df[col].rolling(window=window).mean()
    return df


def add_fourier_features(df: pd.DataFrame, cols: List[str], n: int) -> pd.DataFrame:
    """Attach the first ``n`` Fourier magnitude coefficients for ``cols``."""
    for col in cols:
        fft = np.abs(np.fft.rfft(df[col].fillna(0).values))
        for i in range(min(n, len(fft))):
            df[f"{col}_fft_{i}"] = fft[i]
    return df


def create_sequences(
    df: pd.DataFrame, target_col: str, seq_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Create padded sequences for LSTM models."""
    sequences, labels = [], []
    for i in range(len(df)):
        start = max(0, i - seq_len)
        seq = df.iloc[start:i].values.tolist()
        sequences.append(seq)
        labels.append(df.iloc[i][target_col])
    X = pad_sequences(sequences, maxlen=seq_len, dtype="float32", padding="pre", truncating="pre")
    y = np.array(labels)
    return X, y


def main() -> None:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Feature engineering for time-series data.")
    parser.add_argument("--input_file", required=True, help="Path to preprocessed CSV file.")
    parser.add_argument("--target_column", required=True, help="Target column name.")
    parser.add_argument("--lags", type=int, default=3, help="Number of lag features to add.")
    parser.add_argument("--moving_avg_window", type=int, default=5, help="Window size for moving average.")
    parser.add_argument("--fft_components", type=int, default=0, help="Number of Fourier components to include.")
    parser.add_argument("--sequence_length", type=int, default=30, help="Sequence length for LSTM input.")
    parser.add_argument(
        "--output_file",
        default="data/engineered_data.npz",
        help="File to save engineered sequences (NumPy .npz).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_file)
    df = add_lag_features(df, args.lags)
    df = add_moving_averages(df, args.moving_avg_window)
    if args.fft_components > 0:
        df = add_fourier_features(df, df.columns.tolist(), args.fft_components)

    df = df.dropna().reset_index(drop=True)

    X, y = create_sequences(df, args.target_column, args.sequence_length)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    np.savez(args.output_file, X=X, y=y)

    print(f"Engineered data saved to {args.output_file}")


if __name__ == "__main__":
    main()

