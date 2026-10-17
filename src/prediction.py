from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .config import FEATURE_NAMES, TARGET_COLUMN


def validate_prediction_input(record: dict[str, Any]) -> dict[str, Any]:
    """Validate a single record before predicting credit risk."""
    if not isinstance(record, dict):
        raise TypeError("Prediction input must be a dictionary of field values.")

    required_fields = [field for field in FEATURE_NAMES if field != TARGET_COLUMN]
    missing = [field for field in required_fields if field not in record]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    normalized = {}
    for field in required_fields:
        value = record[field]
        if pd.isna(value):
            raise ValueError(f"Field '{field}' cannot be empty.")
        normalized[field] = value

    numeric_fields = [
        "duration_in_month",
        "credit_amount",
        "installment_rate_in_percentage_of_disposable_income",
        "residence_since",
        "age_in_years",
        "number_of_existing_credits_at_this_bank",
        "number_of_people_liable_to_provide_maintenance_for",
    ]
    for field in numeric_fields:
        numeric_value = normalized[field]
        try:
            numeric_value = float(numeric_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Field '{field}' must be numeric.") from exc
        if numeric_value < 0:
            raise ValueError(f"Field '{field}' cannot be negative.")
        normalized[field] = numeric_value

    return normalized


def predict_single(record: dict[str, Any], model=None, preprocessor=None):
    """Predict a single row and return a dictionary with class and probability."""
    validated_record = validate_prediction_input(record)

    if model is None:
        return {
            "predicted_class": 1,
            "predicted_probability": 0.5,
            "risk_label": "Bad credit (demo fallback)",
            "explanation": "This is a placeholder prediction because no trained model file was provided.",
        }

    df = pd.DataFrame([validated_record])
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(df)[0, 1])
        prediction = int(model.predict(df)[0])
    else:
        probability = 0.5
        prediction = int(model.predict(df)[0])

    risk_label = "Bad credit" if prediction == 1 else "Good credit"
    return {
        "predicted_class": prediction,
        "predicted_probability": probability,
        "risk_label": risk_label,
        "explanation": "Probability of bad-credit classification. It is an estimate, not a guarantee of real-world default risk.",
    }


def batch_predict(df: pd.DataFrame, model=None, preprocessor=None) -> pd.DataFrame:
    """Run batched inference on a dataframe and append prediction columns."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Batch prediction requires a pandas DataFrame.")

    required_columns = [field for field in FEATURE_NAMES if field != TARGET_COLUMN]
    missing_columns = [field for field in required_columns if field not in df.columns]
    if missing_columns:
        raise ValueError(f"Input CSV is missing required columns: {missing_columns}")

    output = df.copy()
    if model is None:
        output["predicted_class"] = 1
        output["predicted_probability"] = 0.5
        output["risk_label"] = "Bad credit (demo fallback)"
        return output

    predictions = model.predict(df[required_columns])
    probabilities = model.predict_proba(df[required_columns])[:, 1] if hasattr(model, "predict_proba") else [0.5] * len(output)
    output["predicted_class"] = predictions.astype(int)
    output["predicted_probability"] = probabilities.astype(float)
    output["risk_label"] = output["predicted_class"].map({0: "Good credit", 1: "Bad credit"})
    return output


def load_model_artifact(path: str | Path):
    """Load a saved model artifact, raising a helpful error if it is absent."""
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Saved model artifact not found at {artifact_path}. Train the model first using train_model.py or the project pipeline."
        )

    import joblib

    return joblib.load(artifact_path)
