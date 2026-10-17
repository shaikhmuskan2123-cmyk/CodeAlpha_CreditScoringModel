from __future__ import annotations

from typing import Iterable, List, Tuple

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class FlexiblePreprocessor(BaseEstimator, TransformerMixin):
    """Infer numeric/categorical columns at fit time and build a robust sklearn pipeline."""

    def __init__(self, numeric_features: Iterable[str] | None = None, categorical_features: Iterable[str] | None = None):
        self.numeric_features = list(numeric_features) if numeric_features is not None else None
        self.categorical_features = list(categorical_features) if categorical_features is not None else None
        self._transformer = None

    def _infer_feature_types(self, X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        numeric = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
        categorical = [col for col in X.columns if col not in numeric]
        return numeric, categorical

    def fit(self, X, y=None):
        X_df = X.copy()
        if self.numeric_features is None or self.categorical_features is None:
            numeric_features, categorical_features = self._infer_feature_types(X_df)
            self.numeric_features = numeric_features
            self.categorical_features = categorical_features

        transformers = []
        if self.numeric_features:
            transformers.append(
                (
                    "numeric",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    self.numeric_features,
                )
            )
        if self.categorical_features:
            transformers.append(
                (
                    "categorical",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    self.categorical_features,
                )
            )

        if not transformers:
            raise ValueError("No feature columns were available for preprocessing.")

        self._transformer = ColumnTransformer(transformers=transformers, remainder="drop")
        self._transformer.fit(X_df, y)
        return self

    def transform(self, X):
        if self._transformer is None:
            self.fit(X)
        return self._transformer.transform(X)

    def fit_transform(self, X, y=None, **kwargs):
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        if self._transformer is None:
            raise RuntimeError("The preprocessor must be fitted before requesting feature names.")
        return self._transformer.get_feature_names_out(input_features)


def build_preprocessor(X: pd.DataFrame | None = None, numeric_features: Iterable[str] | None = None, categorical_features: Iterable[str] | None = None):
    """Create a flexible preprocessor object that infers feature types when possible."""
    if X is not None:
        return FlexiblePreprocessor(numeric_features=numeric_features, categorical_features=categorical_features).fit(X)
    return FlexiblePreprocessor(numeric_features=numeric_features, categorical_features=categorical_features)


def split_train_test(df: pd.DataFrame, target_column: str = "credit_risk", test_size: float = 0.2, random_state: int = 42):
    """Split features and target while preserving class balance in the target."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataframe.")
    if df.empty:
        raise ValueError("The dataframe is empty, so no train/test split can be created.")

    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test
