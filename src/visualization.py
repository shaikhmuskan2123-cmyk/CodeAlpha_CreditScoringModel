from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _style_current_figure():
    figure = plt.gcf()
    figure.patch.set_facecolor("#142019")
    for axis in figure.axes:
        axis.set_facecolor("#142019")
        axis.tick_params(colors="#bac8bc")
        axis.xaxis.label.set_color("#d1ddd2")
        axis.yaxis.label.set_color("#d1ddd2")
        axis.title.set_color("#eef4fb")
        for spine in axis.spines.values():
            spine.set_color("#394d40")
        for label in axis.get_xticklabels() + axis.get_yticklabels():
            label.set_color("#bac8bc")
        legend = axis.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor("#1b2b22")
            legend.get_frame().set_edgecolor("#394d40")
            for text in legend.get_texts():
                text.set_color("#dce7f2")


def save_target_distribution(df, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sns.countplot(
        data=df,
        x="credit_risk",
        hue="credit_risk",
        palette={0: "#28d3ac", 1: "#ff6b78"},
        legend=False,
    )
    plt.title("Target distribution")
    plt.xlabel("Credit risk")
    plt.ylabel("Count")
    _style_current_figure()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_model_comparison(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(results.keys())
    scores = [result.get("accuracy", 0.0) for result in results.values()]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=labels, y=scores, hue=labels, palette=["#59c68a", "#35d5a0", "#c3a0ff"], legend=False)
    plt.title("Model comparison (accuracy)")
    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    plt.xticks(rotation=20)
    _style_current_figure()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def get_feature_importance_data(model):
    estimator = model.named_steps.get("model", model) if hasattr(model, "named_steps") else model

    if hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_)
    elif hasattr(estimator, "coef_"):
        importances = np.asarray(estimator.coef_)
        importances = np.mean(np.abs(importances), axis=0) if importances.ndim > 1 else np.abs(importances)
    else:
        raise ValueError("The model does not expose feature importances or coefficients.")

    if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
        feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    else:
        feature_names = [f"feature_{index}" for index in range(len(importances))]
    if len(feature_names) != len(importances):
        raise ValueError("The number of feature names does not match the model importance values.")

    return pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
        "importance", ascending=False, ignore_index=True
    )


def save_feature_importance(model, output_path, top_n=20):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    importance_data = get_feature_importance_data(model).head(max(1, top_n)).sort_values("importance")
    plt.figure(figsize=(10, max(5, len(importance_data) * 0.28)))
    sns.barplot(data=importance_data, x="importance", y="feature", color="#35d5a0")
    plt.title("Top Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Transformed feature")
    _style_current_figure()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_confusion_matrix(cm, labels, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=sns.color_palette(["#19291f", "#267750"], as_cmap=True),
        annot_kws={"color": "#eef4fb", "weight": "bold"},
        linewidths=1,
        linecolor="#394d40",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Confusion matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    _style_current_figure()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_roc_curve(fpr, tpr, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="#e8b65f", lw=2, label="ROC curve")
    plt.plot([0, 1], [0, 1], color="#829487", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    _style_current_figure()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
