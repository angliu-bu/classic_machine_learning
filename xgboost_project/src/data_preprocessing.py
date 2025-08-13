"""Data preprocessing script for the XGBoost project.

Example
-------
Prepare train and optional test data::

    python src/data_preprocessing.py --train data/train.csv --target target_col --test data/test.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import DATA_DIR, MODELS_DIR, ensure_directories


def load_dataset(path: Path, target: str) -> tuple[pd.DataFrame, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {path}")
    y = df[target]
    X = df.drop(columns=[target])
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore", sparse=False),
        ),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    return preprocessor


def save_npz(path: Path, X: np.ndarray, y: pd.Series) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, y=y.to_numpy())


def preprocess(train_path: Path, target: str, test_path: Path | None) -> None:
    ensure_directories()
    X_train, y_train = load_dataset(train_path, target)
    preprocessor = build_preprocessor(X_train)
    X_train_proc = preprocessor.fit_transform(X_train)
    save_npz(DATA_DIR / "processed_train.npz", X_train_proc, y_train)

    if test_path:
        X_test, y_test = load_dataset(test_path, target)
        X_test_proc = preprocessor.transform(X_test)
        save_npz(DATA_DIR / "processed_test.npz", X_test_proc, y_test)

    joblib.dump(preprocessor, MODELS_DIR / "preprocess.joblib")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess CSV data")
    parser.add_argument("--train", type=Path, required=True, help="Path to training CSV")
    parser.add_argument("--target", type=str, required=True, help="Target column name")
    parser.add_argument("--test", type=Path, help="Optional path to test CSV")
    return parser.parse_args()


def main() -> None:
    args = get_args()
    preprocess(args.train, args.target, args.test)


if __name__ == "__main__":
    main()
