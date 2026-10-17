from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"

DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
DATASET_RAW_PATH = RAW_DATA_DIR / "german_credit.csv"
DATASET_PROCESSED_PATH = PROCESSED_DATA_DIR / "german_credit_processed.csv"
MODEL_PATH = MODELS_DIR / "credit_model_pipeline.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
METRICS_PATH = REPORTS_DIR / "model_metrics.json"
PLOTS_DIR = REPORTS_DIR / "plots"

TARGET_COLUMN = "credit_risk"
GOOD_CREDIT_LABEL = 1
BAD_CREDIT_LABEL = 2
BINARY_TARGET_MAP = {1: 0, 2: 1}

# Use feature names defined in the dataset documentation.
FEATURE_NAMES = [
    "status_of_existing_checking_account",
    "duration_in_month",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_account_bonds",
    "present_employment_since",
    "installment_rate_in_percentage_of_disposable_income",
    "personal_status_and_sex",
    "other_debtors_guarantors",
    "residence_since",
    "property",
    "age_in_years",
    "other_installment_plans",
    "housing",
    "number_of_existing_credits_at_this_bank",
    "job",
    "number_of_people_liable_to_provide_maintenance_for",
    "telephone",
    "foreign_worker",
    "credit_risk",
]

TRAIN_SIZE = 0.8
RANDOM_SEED = 42

# Validation and UI text
EDUCATIONAL_DISCLAIMER = (
    "This project is for educational and research use only. It does not constitute a lending decision or a financial recommendation."
)
