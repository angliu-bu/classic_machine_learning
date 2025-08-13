# XGBoost Project

This project provides an end-to-end pipeline for training and evaluating an XGBoost classifier. It includes scripts for data preprocessing, feature engineering, model training, evaluation, and inference.

## Folder Structure

```
xgboost_project/
  README.md
  .gitignore
  requirements.txt
  data/
  models/
  reports/
  src/
```

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Example Usage

```bash
python src/data_preprocessing.py --train data/train.csv --target target_col
python src/feature_engineering.py
python src/train_xgboost.py
python src/evaluate_model.py --test data/test.csv --target target_col
python src/inference.py --input data/new_rows.csv --output predictions.csv
```

Artifacts are saved under `models/` and `reports/` directories.
