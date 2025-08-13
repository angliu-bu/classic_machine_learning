# LSTM Project

This project demonstrates a complete workflow for training and evaluating a Long Short-Term Memory (LSTM) neural network on time-series data. It includes preprocessing, feature engineering, model training, evaluation, and inference scripts.

## Project Structure

- `data/` — placeholder directory for raw and preprocessed datasets.
- `src/` — Python source files implementing the workflow.
- `models/` — trained model artifacts.
- `results/` — generated plots, metrics, and predictions.

## Dataset

The project is designed for sequential or time-series datasets stored in CSV format. Each dataset should contain a target column representing the value to be predicted.

## Usage

1. **Preprocess data**
   ```bash
   python src/data_preprocessing.py --input_file data/raw.csv --target_column value --sequence_length 30
   ```

2. **Engineer features**
   ```bash
   python src/feature_engineering.py --input_file data/preprocessed.csv --target_column value --lags 3 --moving_avg_window 5
   ```

3. **Train the LSTM model**
   ```bash
   python src/train_lstm.py --data_file data/engineered_data.npz --epochs 50
   ```

4. **Evaluate the model**
   ```bash
   python src/evaluate_model.py --data_file data/engineered_data.npz --model_file models/best_model.keras
   ```

5. **Run inference**
   ```bash
   python src/inference.py --model_file models/best_model.keras --input_file data/sample.csv
   ```

All outputs such as trained models, plots, and metrics are stored in the `models/` and `results/` directories.

