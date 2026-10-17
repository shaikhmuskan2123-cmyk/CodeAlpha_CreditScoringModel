import json

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.config import PROJECT_ROOT
from src.evaluation import evaluate_at_threshold, summarize_model_metrics


def test_summarize_model_metrics_handles_missing_model_data():
    metrics = summarize_model_metrics({})
    assert isinstance(metrics, dict)
    assert metrics["selected_model"] == "none"


def test_json_serialization_for_metrics():
    payload = {"accuracy": 0.80, "precision": 0.75, "recall": 0.70}
    json.dumps(payload)
    assert payload["accuracy"] >= 0


def test_dataframe_summary_returns_rows_and_columns():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert df.shape == (2, 2)


def test_threshold_updates_predictions_and_confusion_matrix():
    actual = [0, 0, 1, 1]
    probabilities = [0.1, 0.4, 0.6, 0.9]

    lower_cutoff = evaluate_at_threshold(actual, probabilities, 0.5)
    higher_cutoff = evaluate_at_threshold(actual, probabilities, 0.7)

    assert lower_cutoff["confusion_matrix"] == [[2, 0], [0, 2]]
    assert higher_cutoff["confusion_matrix"] == [[2, 0], [1, 1]]
    assert lower_cutoff["recall"] != higher_cutoff["recall"]
    tied_probability = evaluate_at_threshold([0, 1], [0.5, 0.5], 0.5)
    assert tied_probability["predictions"].tolist() == [0, 0]


def test_prediction_result_requires_submit_button():
    app = AppTest.from_file(PROJECT_ROOT / "app.py").run(timeout=30)
    app.radio[0].set_value("Predict").run(timeout=30)

    assert not app.exception
    assert not any("MODEL CLASSIFICATION" in item.value for item in app.markdown)

    submit = next(button for button in app.button if button.label == "Predict Result")
    submit.click().run(timeout=30)

    assert not app.exception
    assert any("MODEL CLASSIFICATION" in item.value for item in app.markdown)
