from __future__ import annotations

import json
from pathlib import Path

import joblib

from src.config import DATASET_RAW_PATH, METRICS_PATH, MODEL_METADATA_PATH, MODEL_PATH, PLOTS_DIR, REPORTS_DIR, TARGET_COLUMN
from src.data_loader import load_credit_data
from src.evaluation import evaluate_model, save_metrics_json
from src.modeling import evaluate_cv_performance, select_best_model, train_candidate_models
from src.visualization import get_feature_importance_data, save_confusion_matrix, save_feature_importance, save_model_comparison, save_roc_curve, save_target_distribution


def main():
    dataset = load_credit_data(DATASET_RAW_PATH)
    X_train, X_test, y_train, y_test = __import__("src.preprocessing", fromlist=["split_train_test"]).split_train_test(dataset, target_column=TARGET_COLUMN, test_size=0.2, random_state=42)

    candidate_models = train_candidate_models(dataset, target_column=TARGET_COLUMN)
    cv_results = evaluate_cv_performance(dataset, target_column=TARGET_COLUMN, cv_folds=5)
    selected_model_name = select_best_model(cv_results)
    selected_pipeline = candidate_models[selected_model_name]
    selected_pipeline.fit(X_train, y_train)

    all_metrics = {}
    for model_name, model in candidate_models.items():
        metrics = evaluate_model(model, X_test, y_test)
        metrics["cv_roc_auc_mean"] = cv_results[model_name]["roc_auc_mean"]
        metrics["cv_roc_auc_std"] = cv_results[model_name]["roc_auc_std"]
        metrics["selected"] = model_name == selected_model_name
        all_metrics[model_name] = metrics

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "selected_model": selected_model_name,
        "cv_results": cv_results,
        "test_results": all_metrics,
        "class_distribution": dataset[TARGET_COLUMN].value_counts().sort_index().to_dict(),
        "heldout_evaluation": {
            "y_true": [int(value) for value in y_test],
            "probabilities": {
                name: [float(value) for value in model.predict_proba(X_test)[:, 1]]
                for name, model in candidate_models.items()
            },
        },
        "feature_importance": {
            name: [
                {"feature": str(row.feature), "importance": float(row.importance)}
                for row in get_feature_importance_data(model).itertuples(index=False)
            ]
            for name, model in candidate_models.items()
        },
    }

    save_metrics_json(summary, METRICS_PATH)
    joblib.dump(selected_pipeline, MODEL_PATH)
    metadata = {
        "selected_model": selected_model_name,
        "selection_metric": "mean_stratified_5_fold_roc_auc",
        "cross_validation": cv_results[selected_model_name],
        "training_rows": len(X_train),
        "heldout_rows": len(X_test),
        "feature_count": X_train.shape[1],
        "target_mapping": {"good_credit": 0, "bad_credit": 1},
    }
    with MODEL_METADATA_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    save_target_distribution(dataset, PLOTS_DIR / "target_distribution.png")
    save_model_comparison(
        {
            name: {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics.get("roc_auc"),
            }
            for name, metrics in all_metrics.items()
        },
        PLOTS_DIR / "model_comparison.png",
    )
    save_feature_importance(selected_pipeline, PLOTS_DIR / "feature_importance.png")

    for model_name, metrics in all_metrics.items():
        if metrics.get("confusion_matrix") is not None:
            labels = [0, 1]
            save_confusion_matrix(metrics["confusion_matrix"], labels, PLOTS_DIR / f"{model_name}_confusion_matrix.png")

    if all_metrics[selected_model_name].get("roc_auc") is not None:
        y_prob = selected_pipeline.predict_proba(X_test)[:, 1]
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        save_roc_curve(fpr, tpr, PLOTS_DIR / "roc_curve.png")

    print(f"Selected model: {selected_model_name}")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved summary metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
