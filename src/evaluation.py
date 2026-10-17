from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def summarize_model_metrics(metric_dictionary):
    """Return a summary dictionary suitable for app display and JSON serialization."""
    if not metric_dictionary:
        return {"selected_model": "none", "n_models": 0}

    selected_model = max(metric_dictionary.items(), key=lambda item: item[1].get("roc_auc", item[1].get("accuracy", 0)))[0]
    return {
        "selected_model": selected_model,
        "n_models": len(metric_dictionary),
        "models": metric_dictionary,
    }


def evaluate_model(model, X_test, y_test):
    """Compute metrics for a single trained pipeline."""
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities))
        except ValueError:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics


def evaluate_at_threshold(y_true, probabilities, threshold: float):
    """Recompute binary classification metrics for a chosen positive-class cutoff."""
    if not 0 <= threshold <= 1:
        raise ValueError("The classification threshold must be between 0 and 1.")

    actual = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    if actual.ndim != 1 or probabilities.ndim != 1 or actual.shape != probabilities.shape:
        raise ValueError("Labels and probabilities must be one-dimensional arrays with matching lengths.")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite values between 0 and 1.")

    predictions = (probabilities > threshold).astype(int)
    return {
        "predictions": predictions,
        "accuracy": float(accuracy_score(actual, predictions)),
        "precision": float(precision_score(actual, predictions, zero_division=0)),
        "recall": float(recall_score(actual, predictions, zero_division=0)),
        "f1": float(f1_score(actual, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(actual, predictions, labels=[0, 1]).tolist(),
    }


def save_metrics_json(metrics, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, default=str)
    return path


def compute_model_comparison_table(results):
    rows = []
    for model_name, payload in results.items():
        rows.append(
            {
                "model": model_name,
                "accuracy": payload.get("accuracy", 0.0),
                "precision": payload.get("precision", 0.0),
                "recall": payload.get("recall", 0.0),
                "f1": payload.get("f1", 0.0),
                "roc_auc": payload.get("roc_auc", 0.0),
            }
        )
    return rows
