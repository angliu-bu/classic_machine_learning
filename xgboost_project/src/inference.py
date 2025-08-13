"""Run batch inference using the trained model.

Example
-------
Predict on new data::

    python src/inference.py --input data/new_rows.csv --output predictions.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier

from utils import MODELS_DIR


def predict(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    pipeline = joblib.load(MODELS_DIR / "preprocess.joblib")
    model = XGBClassifier()
    model.load_model(MODELS_DIR / "xgb_model.json")
    df = pd.read_csv(input_path)
    X = pipeline.transform(df)
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)
    result = df.copy()
    result["prediction"] = preds
    result["probability"] = proba
    result.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on new data")
    parser.add_argument("--input", type=Path, required=True, help="Path to input CSV")
    parser.add_argument("--output", type=Path, required=True, help="Path to save predictions CSV")
    args = parser.parse_args()
    predict(args.input, args.output)


if __name__ == "__main__":
    main()
