import pandas as pd

from src.modeling import train_candidate_models
from src.visualization import get_feature_importance_data, save_feature_importance


def test_model_training_on_synthetic_fixture(tmp_path):
    df = pd.DataFrame({
        "status_of_existing_checking_account": ["A11", "A12", "A13", "A14", "A11", "A12", "A13", "A14", "A11", "A12", "A13", "A14"],
        "duration_in_month": [12, 18, 24, 30, 14, 22, 9, 36, 15, 27, 11, 42],
        "credit_history": ["A30", "A31", "A32", "A33", "A30", "A31", "A32", "A33", "A30", "A31", "A32", "A33"],
        "purpose": ["A40", "A41", "A42", "A43", "A40", "A41", "A42", "A43", "A40", "A41", "A42", "A43"],
        "credit_amount": [500, 800, 900, 1200, 550, 850, 740, 1500, 680, 900, 1050, 1700],
        "savings_account_bonds": ["A61", "A62", "A63", "A64", "A61", "A62", "A63", "A64", "A61", "A62", "A63", "A64"],
        "present_employment_since": ["A71", "A72", "A73", "A74", "A71", "A72", "A73", "A74", "A71", "A72", "A73", "A74"],
        "installment_rate_in_percentage_of_disposable_income": [2, 3, 4, 2, 1, 3, 4, 2, 2, 3, 2, 4],
        "personal_status_and_sex": ["A91", "A92", "A93", "A94", "A91", "A92", "A93", "A94", "A91", "A92", "A93", "A94"],
        "other_debtors_guarantors": ["A101", "A102", "A103", "A104", "A101", "A102", "A103", "A104", "A101", "A102", "A103", "A104"],
        "residence_since": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
        "property": ["A121", "A122", "A123", "A124", "A121", "A122", "A123", "A124", "A121", "A122", "A123", "A124"],
        "age_in_years": [22, 25, 31, 45, 19, 27, 36, 47, 24, 33, 39, 52],
        "other_installment_plans": ["A141", "A142", "A143", "A144", "A141", "A142", "A143", "A144", "A141", "A142", "A143", "A144"],
        "housing": ["A151", "A152", "A153", "A154", "A151", "A152", "A153", "A154", "A151", "A152", "A153", "A154"],
        "number_of_existing_credits_at_this_bank": [1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 3],
        "job": ["A171", "A172", "A173", "A174", "A171", "A172", "A173", "A174", "A171", "A172", "A173", "A174"],
        "number_of_people_liable_to_provide_maintenance_for": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        "telephone": ["A191", "A192", "A193", "A194", "A191", "A192", "A193", "A194", "A191", "A192", "A193", "A194"],
        "foreign_worker": ["A201", "A202", "A203", "A204", "A201", "A202", "A203", "A204", "A201", "A202", "A203", "A204"],
        "credit_risk": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })

    models = train_candidate_models(df, target_column="credit_risk")
    assert set(models.keys()) >= {"logistic_regression", "decision_tree", "random_forest"}
    assert len(models["logistic_regression"]) >= 1
    importance_data = get_feature_importance_data(models["logistic_regression"])
    assert not importance_data.empty
    assert importance_data["importance"].ge(0).all()
    plot_path = tmp_path / "feature_importance.png"
    save_feature_importance(models["random_forest"], plot_path)
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
