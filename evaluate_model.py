from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.config import DATASET_RAW_PATH, METRICS_PATH, MODEL_PATH, TARGET_COLUMN
from src.data_loader import load_credit_data
from src.evaluation import evaluate_model
from src.preprocessing import split_train_test


def main():
    dataset = load_credit_data(DATASET_RAW_PATH)
    _, X_test, _, y_test = split_train_test(dataset, target_column=TARGET_COLUMN, test_size=0.2, random_state=42)

    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"No trained model found at {model_path}. Run train_model.py first.")

    model = joblib.load(model_path)
    metrics = evaluate_model(model, X_test, y_test)
    print(metrics)

    print(f"Saved metrics file: {METRICS_PATH}")


if __name__ == "__main__":
    main()
