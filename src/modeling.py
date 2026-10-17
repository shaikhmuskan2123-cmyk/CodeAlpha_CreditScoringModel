from __future__ import annotations

from collections import OrderedDict

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .preprocessing import FlexiblePreprocessor, split_train_test


def get_model_candidates():
    return OrderedDict(
        [
            ("logistic_regression", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)),
            ("decision_tree", DecisionTreeClassifier(random_state=42, class_weight="balanced")),
            ("random_forest", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1)),
        ]
    )


def train_candidate_models(df, target_column: str = "credit_risk"):
    """Train and return a dictionary of scikit-learn pipelines for the candidate models."""
    X_train, _, y_train, _ = split_train_test(df, target_column=target_column, test_size=0.2, random_state=42)
    trained = {}

    for name, model in get_model_candidates().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", FlexiblePreprocessor()),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        trained[name] = pipeline

    return trained


def select_best_model(cv_results):
    if not cv_results:
        raise ValueError("No cross-validation results were provided.")
    return max(cv_results.items(), key=lambda item: item[1]["roc_auc_mean"])[0]


def evaluate_cv_performance(df, target_column: str = "credit_risk", cv_folds: int = 5):
    X_train, _, y_train, _ = split_train_test(df, target_column=target_column, test_size=0.2, random_state=42)
    cv = StratifiedKFold(n_splits=min(cv_folds, y_train.value_counts().min()), shuffle=True, random_state=42)

    results = {}
    for name, model in get_model_candidates().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", FlexiblePreprocessor()),
                ("model", model),
            ]
        )
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
        results[name] = {
            "roc_auc_mean": float(scores.mean()),
            "roc_auc_std": float(scores.std()),
            "cv_scores": [float(value) for value in scores],
        }
    return results
