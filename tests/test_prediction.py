import pandas as pd
import pytest

from src.prediction import batch_predict, predict_single, validate_prediction_input
from predict import parse_args


@pytest.fixture
def synthetic_input_frame():
    return pd.DataFrame({
        "status_of_existing_checking_account": ["A11", "A12"],
        "duration_in_month": [12, 24],
        "credit_history": ["A30", "A31"],
        "purpose": ["A40", "A41"],
        "credit_amount": [500, 900],
        "savings_account_bonds": ["A61", "A62"],
        "present_employment_since": ["A71", "A72"],
        "installment_rate_in_percentage_of_disposable_income": [2, 3],
        "personal_status_and_sex": ["A91", "A92"],
        "other_debtors_guarantors": ["A101", "A102"],
        "residence_since": [1, 2],
        "property": ["A121", "A122"],
        "age_in_years": [23, 34],
        "other_installment_plans": ["A141", "A142"],
        "housing": ["A151", "A152"],
        "number_of_existing_credits_at_this_bank": [1, 2],
        "job": ["A171", "A172"],
        "number_of_people_liable_to_provide_maintenance_for": [1, 2],
        "telephone": ["A191", "A192"],
        "foreign_worker": ["A201", "A202"],
    })


def test_prediction_validation_and_probability_ranges(synthetic_input_frame):
    row = synthetic_input_frame.iloc[0].to_dict()
    validated = validate_prediction_input(row)
    assert validated["duration_in_month"] == 12
    with pytest.raises(ValueError):
        validate_prediction_input({**row, "duration_in_month": -1})

    result = predict_single(row, model=None, preprocessor=None)
    assert "predicted_class" in result
    assert "predicted_probability" in result
    assert 0 <= result["predicted_probability"] <= 1


def test_batch_predict_output_contains_predicted_columns(synthetic_input_frame):
    result = batch_predict(synthetic_input_frame, model=None, preprocessor=None)
    assert {"predicted_class", "predicted_probability", "risk_label"}.issubset(result.columns)


def test_cli_accepts_explicit_csv_batch_option():
    args = parse_args(["--input", "records.csv", "--batch"])
    assert args.input == "records.csv"
    assert args.batch is True
