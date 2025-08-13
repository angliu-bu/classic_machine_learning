"""Feature engineering script.

Example
-------
Update the preprocessing pipeline with engineered features::

    python src/feature_engineering.py
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from utils import MODELS_DIR


def engineer_features(X: np.ndarray) -> np.ndarray:
    """Add simple interaction and ratio features.

    Parameters
    ----------
    X : np.ndarray
        Input feature array.

    Returns
    -------
    np.ndarray
        Array with additional engineered features appended.
    """
    if X.shape[1] < 2:
        return X
    a, b = X[:, 0], X[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    prod = a * b
    return np.hstack([X, ratio[:, None], prod[:, None]])


def main() -> None:
    preprocess_path = MODELS_DIR / "preprocess.joblib"
    if not preprocess_path.exists():
        raise FileNotFoundError("Preprocessor not found. Run data_preprocessing first.")
    preprocessor = joblib.load(preprocess_path)
    fe_transformer = FunctionTransformer(engineer_features)
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("feature_engineering", fe_transformer),
    ])
    joblib.dump(pipeline, preprocess_path)
    print(f"Updated preprocessing pipeline saved to {preprocess_path}")


if __name__ == "__main__":
    main()
