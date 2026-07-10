"""
preprocessing.py
----------------
Reusable data cleaning, feature engineering, and preprocessing pipeline for
the Telco Customer Churn dataset.

Why this file exists:
    Training and inference (the Streamlit app) must apply *exactly* the same
    transformations to raw data, or predictions will be wrong. This module
    builds a single scikit-learn `ColumnTransformer` that is fit once during
    training and then saved with `joblib`, so the exact same fitted
    transformation can be reloaded and reused at inference time -- no
    duplicated logic, no train/serve skew.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from src.config import (
    ID_COLUMN,
    TARGET_COLUMN,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply dataset-specific cleaning rules that must happen *before* the
    scikit-learn pipeline runs (these are not simple column transforms).

    Steps:
        1. Drop the customerID identifier column (not predictive).
        2. Coerce TotalCharges from text to numeric (blank strings for
           brand-new tenure==0 customers become NaN, then are imputed to 0,
           since a customer with zero tenure has by definition paid $0
           total so far).
        3. Drop exact duplicate rows, if any.

    Args:
        df: Raw DataFrame as loaded from `data_loader.load_raw_data`.

    Returns:
        A cleaned copy of the DataFrame, ready for feature engineering.
    """
    df = df.copy()

    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_missing_total_charges = df["TotalCharges"].isnull().sum()
    if n_missing_total_charges > 0:
        logger.info(
            "Imputing %d missing TotalCharges values (new customers, "
            "tenure == 0) with 0.0",
            n_missing_total_charges,
        )
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    before = len(df)
    df = df.drop_duplicates()
    if len(df) < before:
        logger.info("Dropped %d duplicate rows", before - len(df))

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features that are not present in the raw dataset but carry
    additional predictive signal identified during EDA.

    New features:
        - AvgMonthlySpend: TotalCharges / max(tenure, 1). Separates
          "spend rate" from "time on the books" (see EDA report section 4).
        - TenureGroup: Bucketed tenure (New / Established / Loyal) so tree
          models can split on a coarser, monotonic-with-risk category.
        - TotalServicesSubscribed: Count of add-on services subscribed to
          (OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
          StreamingTV, StreamingMovies), used as a simple "engagement" proxy.
        - IsNewCustomer: Binary flag for tenure <= 3 months, the highest
          observed churn-risk window from EDA.

    Args:
        df: Cleaned DataFrame (post `clean_raw_data`).

    Returns:
        DataFrame with engineered feature columns appended.
    """
    df = df.copy()

    df["AvgMonthlySpend"] = df["TotalCharges"] / df["tenure"].clip(lower=1)

    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 48, 100],
        labels=["New", "Established", "Loyal"],
    ).astype(str)

    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["TotalServicesSubscribed"] = (df[service_cols] == "Yes").sum(axis=1)

    df["IsNewCustomer"] = (df["tenure"] <= 3).astype(int)

    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Build (but do not fit) the scikit-learn ColumnTransformer that scales
    numeric features and one-hot encodes categorical features, including
    the newly engineered `TenureGroup` category.

    Returns:
        An unfitted `ColumnTransformer`.
    """
    numeric_cols = NUMERIC_FEATURES + [
        "AvgMonthlySpend", "TotalServicesSubscribed", "IsNewCustomer",
    ]
    categorical_cols = CATEGORICAL_FEATURES + ["TenureGroup"]

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    # Keep pandas DataFrame output (with real column names) flowing through
    # the pipeline, instead of a bare numpy array. This keeps SHAP/LIME
    # feature names correct and avoids sklearn's "fitted without feature
    # names" warning when the fitted model later sees named columns.
    preprocessor.set_output(transform="pandas")
    return preprocessor


def get_feature_target_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Run cleaning + feature engineering, then split into X (features) and
    y (binary-encoded target).

    Args:
        df: Raw DataFrame.

    Returns:
        Tuple of (X, y) where y is 1 for churn == 'Yes', else 0.
    """
    df = clean_raw_data(df)
    df = engineer_features(df)

    y = (df[TARGET_COLUMN] == "Yes").astype(int)
    X = df.drop(columns=[TARGET_COLUMN])
    return X, y


def split_train_test(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train/test split (stratified on churn to preserve class
    balance in both sets, since the target is moderately imbalanced).

    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


if __name__ == "__main__":
    from src.data_loader import load_raw_data

    raw = load_raw_data()
    X_all, y_all = get_feature_target_split(raw)
    X_tr, X_te, y_tr, y_te = split_train_test(X_all, y_all)
    logger.info("X_train: %s | X_test: %s", X_tr.shape, X_te.shape)
    logger.info("Engineered columns: %s", list(X_all.columns))
