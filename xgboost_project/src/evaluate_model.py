"""Evaluate the trained XGBoost model.

Example
-------
Evaluate with an explicit test set::

    python src/evaluate_model.py --test data/test.csv --target target_col
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from xgboost import XGBClassifier

from utils import DATA_DIR, MODELS_DIR, REPORTS_DIR, save_json_metrics

sns.set_theme()


def load_processed_test() -> tuple[np.ndarray, np.ndarray]:
    path = DATA_DIR / "processed_test.npz"
    if not path.exists():
        raise FileNotFoundError("Processed test data not found and no test CSV provided.")
    data = np.load(path)
    return data["X"], data["y"]


def load_model() -> XGBClassifier:
    model_path = MODELS_DIR / "xgb_model.json"
    model = XGBClassifier()
    model.load_model(model_path)
    return model


def evaluate(X: np.ndarray, y: np.ndarray, model: XGBClassifier) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
    }
    return metrics, preds, proba


def plot_curves(y: np.ndarray, proba: np.ndarray) -> None:
    preds = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(REPORTS_DIR / "confusion_matrix.png")
    plt.close()

    RocCurveDisplay.from_predictions(y, proba)
    plt.savefig(REPORTS_DIR / "roc_curve.png")
    plt.close()

    PrecisionRecallDisplay.from_predictions(y, proba)
    plt.savefig(REPORTS_DIR / "pr_curve.png")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate XGBoost model")
    parser.add_argument("--test", type=Path, help="Path to raw test CSV")
    parser.add_argument("--target", type=str, help="Target column name if using raw CSV")
    args = parser.parse_args()

    if args.test and not args.target:
        parser.error("--target is required when --test is provided")

    if args.test:
        df = pd.read_csv(args.test)
        if args.target not in df.columns:
            raise ValueError(f"Target column '{args.target}' not found in test CSV")
        y = df[args.target].to_numpy()
        X = df.drop(columns=[args.target])
        pipeline = joblib.load(MODELS_DIR / "preprocess.joblib")
        X = pipeline.transform(X)
    else:
        X, y = load_processed_test()

    model = load_model()
    metrics, preds, proba = evaluate(X, y, model)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    save_json_metrics(REPORTS_DIR / "metrics.json", metrics)
    plot_curves(y, proba)
    print("Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
