"""
data_loader.py
--------------
Responsible for loading the raw Telco Customer Churn dataset from disk and
performing lightweight structural validation before it is handed off to the
EDA or preprocessing stages.

Why this file exists:
    Keeping I/O logic in one place means the CSV path, expected schema, and
    basic sanity checks are defined once. If the data source ever changes
    (e.g., swapped for a database query or a different file format), only
    this module needs to change -- nothing downstream does.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_PATH, TARGET_COLUMN
from src.utils.logger import get_logger

logger = get_logger(__name__)

EXPECTED_COLUMNS: list[str] = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]


def load_raw_data(path: str | None = None) -> pd.DataFrame:
    """
    Load the raw Telco Customer Churn CSV file into a DataFrame.

    Args:
        path: Optional override path. Defaults to `config.RAW_DATA_PATH`.

    Returns:
        The raw, unmodified DataFrame.

    Raises:
        FileNotFoundError: If the CSV cannot be found at the resolved path.
        ValueError: If the file is missing expected columns.
    """
    file_path = Path(path) if path else RAW_DATA_PATH

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {file_path}")

    logger.info("Loading raw dataset from %s", file_path)
    df = pd.read_csv(file_path)

    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Dataset is missing expected columns: {sorted(missing_cols)}"
        )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    logger.info("Loaded dataset with shape %s", df.shape)
    return df


def basic_data_report(df: pd.DataFrame) -> dict:
    """
    Produce a quick structural summary of the dataset: shape, dtypes,
    missing values, and duplicate row count. Used by EDA and unit tests.

    Args:
        df: The DataFrame to summarize.

    Returns:
        A dictionary summarizing key structural facts about the DataFrame.
    """
    report = {
        "n_rows": df.shape[0],
        "n_columns": df.shape[1],
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values_per_column": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
    logger.info(
        "Data report: %d rows, %d columns, %d duplicates",
        report["n_rows"], report["n_columns"], report["duplicate_rows"],
    )
    return report


if __name__ == "__main__":
    data = load_raw_data()
    print(basic_data_report(data))
