"""Train an XGBoost classifier.

Example
-------
Train the model using processed training data::

    python src/train_xgboost.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier

from utils import DATA_DIR, MODELS_DIR, save_json_metrics


def load_processed(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Processed data not found: {path}")
    data = np.load(path)
    return data["X"], data["y"]


def train_model() -> None:
    X, y = load_processed(DATA_DIR / "processed_train.npz")
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        scale_pos_weight=pos_weight,
    )
    param_grid = {
        "max_depth": [3, 5],
        "learning_rate": [0.1, 0.01],
        "n_estimators": [100, 200],
    }
    grid = GridSearchCV(xgb, param_grid=param_grid, scoring="f1", cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model: XGBClassifier = grid.best_estimator_
    y_pred = best_model.predict(X_valid)
    val_f1 = f1_score(y_valid, y_pred)

    model_path = MODELS_DIR / "xgb_model.json"
    best_model.save_model(model_path)

    meta = {
        "best_params": grid.best_params_,
        "f1_score": val_f1,
        "classes": best_model.classes_.tolist(),
        "n_features": int(X.shape[1]),
    }
    save_json_metrics(MODELS_DIR / "model_meta.json", meta)
    print(f"Model saved to {model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost model")
    parser.parse_args()  # no arguments currently
    train_model()


if __name__ == "__main__":
    main()
