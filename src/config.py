"""
config.py
---------
Centralized configuration for the Customer Churn Prediction project.

Why this file exists:
    Hard-coding paths and constants across multiple scripts (EDA, preprocessing,
    training, explainability, the Streamlit app) leads to duplication and bugs
    when the project is moved or restructured. This module defines a single
    source of truth for file paths, column groupings, and modeling constants,
    so every other module imports from here instead of redefining values.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Directory structure
# ---------------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT_DIR / "data"
MODELS_DIR: Path = ROOT_DIR / "models"
ASSETS_DIR: Path = ROOT_DIR / "assets"
REPORTS_DIR: Path = ROOT_DIR / "reports"

# ---------------------------------------------------------------------------
# Data files
# ---------------------------------------------------------------------------
RAW_DATA_PATH: Path = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
CLEANED_DATA_PATH: Path = DATA_DIR / "telco_churn_cleaned.csv"

# ---------------------------------------------------------------------------
# Model artifact files
# ---------------------------------------------------------------------------
BEST_MODEL_PATH: Path = MODELS_DIR / "best_model.pkl"
PREPROCESSOR_PATH: Path = MODELS_DIR / "preprocessor.pkl"
LABEL_ENCODER_PATH: Path = MODELS_DIR / "label_encoder.pkl"
MODEL_METADATA_PATH: Path = MODELS_DIR / "model_metadata.json"
FEATURE_NAMES_PATH: Path = MODELS_DIR / "feature_names.json"

# ---------------------------------------------------------------------------
# Column groupings (based on IBM Telco Customer Churn dataset schema)
# ---------------------------------------------------------------------------
TARGET_COLUMN: str = "Churn"
ID_COLUMN: str = "customerID"

NUMERIC_FEATURES: list[str] = ["tenure", "MonthlyCharges", "TotalCharges"]

BINARY_FEATURES: list[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]

MULTI_CATEGORY_FEATURES: list[str] = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]

CATEGORICAL_FEATURES: list[str] = BINARY_FEATURES + MULTI_CATEGORY_FEATURES

ENGINEERED_FEATURES: list[str] = [
    "AvgMonthlySpend",
    "TenureGroup",
    "TotalServicesSubscribed",
    "IsNewCustomer",
]

# ---------------------------------------------------------------------------
# Modeling constants
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
CV_FOLDS: int = 5

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

# ---------------------------------------------------------------------------
# Ensure directories exist (safe no-op if already present)
# ---------------------------------------------------------------------------
for _dir in (DATA_DIR, MODELS_DIR, ASSETS_DIR, REPORTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
