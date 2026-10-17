import pandas as pd
import pytest

from src.config import FEATURE_NAMES
from src.data_loader import load_credit_data, map_credit_target, validate_credit_dataframe


@pytest.fixture
def sample_credit_frame():
    data = {
        "status_of_existing_checking_account": ["A11", "A12", "A13", "A14", "A11"],
        "duration_in_month": [12, 24, 6, 48, 18],
        "credit_history": ["A30", "A31", "A32", "A33", "A34"],
        "purpose": ["A40", "A41", "A42", "A43", "A44"],
        "credit_amount": [500, 700, 1200, 2500, 800],
        "savings_account_bonds": ["A61", "A62", "A63", "A64", "A65"],
        "present_employment_since": ["A71", "A72", "A73", "A74", "A75"],
        "installment_rate_in_percentage_of_disposable_income": [2, 3, 4, 1, 2],
        "personal_status_and_sex": ["A91", "A92", "A93", "A94", "A95"],
        "other_debtors_guarantors": ["A101", "A102", "A103", "A104", "A101"],
        "residence_since": [1, 2, 2, 3, 1],
        "property": ["A121", "A122", "A123", "A124", "A121"],
        "age_in_years": [23, 34, 41, 52, 29],
        "other_installment_plans": ["A141", "A142", "A143", "A144", "A141"],
        "housing": ["A151", "A152", "A153", "A154", "A151"],
        "number_of_existing_credits_at_this_bank": [1, 2, 1, 3, 2],
        "job": ["A171", "A172", "A173", "A174", "A171"],
        "number_of_people_liable_to_provide_maintenance_for": [1, 2, 1, 2, 1],
        "telephone": ["A191", "A192", "A193", "A194", "A191"],
        "foreign_worker": ["A201", "A202", "A203", "A204", "A201"],
        "credit_risk": [1, 2, 1, 2, 1],
    }
    return pd.DataFrame(data)


def test_target_mapping_and_schema(sample_credit_frame):
    mapped = map_credit_target(sample_credit_frame.copy())
    assert list(mapped.columns) == FEATURE_NAMES
    assert set(mapped["credit_risk"].unique()) == {0, 1}
    assert mapped["credit_risk"].notna().all()


def test_validate_credit_dataframe_detects_missing_required_columns(sample_credit_frame):
    bad_frame = sample_credit_frame.drop(columns=["credit_risk"])
    with pytest.raises(ValueError):
        validate_credit_dataframe(bad_frame)


def test_load_credit_data_persists_mapped_dataset(sample_credit_frame, tmp_path):
    raw_path = tmp_path / "german_credit_raw.data"
    processed_path = tmp_path / "processed" / "german_credit_processed.csv"
    sample_credit_frame.to_csv(raw_path, sep=" ", header=False, index=False)

    dataset = load_credit_data(raw_path, processed_path=processed_path)

    assert processed_path.exists()
    assert set(dataset["credit_risk"]) == {0, 1}
    persisted = pd.read_csv(processed_path)
    assert persisted["credit_risk"].tolist() == dataset["credit_risk"].tolist()
