from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from .config import (
    DATASET_RAW_PATH,
    DATASET_PROCESSED_PATH,
    DATASET_URL,
    FEATURE_NAMES,
    RAW_DATA_DIR,
    TARGET_COLUMN,
)


def download_dataset(force: bool = False) -> Path:
    """Download the German credit dataset from the official UCI source if needed."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATASET_RAW_PATH.exists() and not force:
        return DATASET_RAW_PATH

    response = requests.get(DATASET_URL, timeout=30)
    response.raise_for_status()
    DATASET_RAW_PATH.write_bytes(response.content)
    return DATASET_RAW_PATH


def load_raw_dataset(path: str | Path | None = None, force_download: bool = False) -> pd.DataFrame:
    """Load the raw German Credit dataset into a pandas DataFrame."""
    dataset_path = Path(path) if path is not None else download_dataset(force=force_download)
    if not dataset_path.exists():
        dataset_path = download_dataset(force=True)

    df = pd.read_csv(dataset_path, sep=r"\s+", header=None, names=FEATURE_NAMES)
    return df


def validate_credit_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Check that the dataset contains the required schema and valid labels."""
    if df is None or df.empty:
        raise ValueError("The dataset is empty.")

    missing = [column for column in FEATURE_NAMES if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df.copy()
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column missing: {TARGET_COLUMN}")

    target_values = set(pd.Series(df[TARGET_COLUMN]).dropna().unique())
    valid_target_values = {0, 1, 2, 3}  # 0/1 are already mapped in some sources, and 1/2 appear in the UCI file.
    unexpected = sorted(target_values - valid_target_values)
    if unexpected:
        raise ValueError(f"Unexpected target labels found: {unexpected}")

    for column in FEATURE_NAMES:
        if column == TARGET_COLUMN:
            continue
        if df[column].isnull().all():
            raise ValueError(f"Column '{column}' contains only missing values.")

    return df


def map_credit_target(df: pd.DataFrame) -> pd.DataFrame:
    """Map the original German Credit target labels to Good=0 and Bad=1."""
    validated = validate_credit_dataframe(df)
    mapped = validated.copy()
    target_values = set(mapped[TARGET_COLUMN].dropna().unique())

    if target_values.issubset({0, 1}):
        mapped[TARGET_COLUMN] = mapped[TARGET_COLUMN].astype(int)
        return mapped

    if not target_values.issubset({1, 2}):
        raise ValueError(f"Unsupported target values for credit-risk mapping: {sorted(target_values)}")

    mapped[TARGET_COLUMN] = mapped[TARGET_COLUMN].map({1: 0, 2: 1})
    if mapped[TARGET_COLUMN].isnull().any():
        raise ValueError("Credit target mapping produced missing values.")

    return mapped


def load_credit_data(
    path: str | Path | None = None,
    force_download: bool = False,
    processed_path: str | Path | None = DATASET_PROCESSED_PATH,
) -> pd.DataFrame:
    """Download or load the dataset and coerce it to the validated project schema."""
    raw_data = load_raw_dataset(path=path, force_download=force_download)
    clean_data = map_credit_target(raw_data)
    if processed_path is not None:
        output_path = Path(processed_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        clean_data.to_csv(output_path, index=False)
    return clean_data
