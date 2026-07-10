"""
predict_pipeline.py
--------------------
A single, high-level entry point that turns one raw customer record (as a
dict, e.g. from a Streamlit form) into a churn prediction plus SHAP and LIME
explanations. This is the module both the Streamlit app and the test suite
import -- neither talks to `preprocessing.py` / `explain.py` directly -- so
there is exactly one place where "raw input -> prediction + explanation"
logic lives.

Why this file exists:
    The Streamlit app should not need to know about ColumnTransformers,
    SHAP explainer internals, or engineered feature formulas. This module
    hides all of that behind one function: `predict_and_explain()`.
"""

from __future__ import annotations

import json
from functools import lru_cache

import joblib
import pandas as pd

from src.config import (
    BEST_MODEL_PATH,
    PREPROCESSOR_PATH,
    MODEL_METADATA_PATH,
)
from src.preprocessing import clean_raw_data, engineer_features
from src.explain import (
    _get_transformed_frame,
    build_shap_explainer,
    explain_single_prediction_lime,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

REQUIRED_RAW_FIELDS: list[str] = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


class InvalidCustomerInputError(ValueError):
    """Raised when a raw customer record is missing fields or has bad types."""


@lru_cache(maxsize=1)
def _load_model_and_preprocessor():
    """Load and cache the trained model + preprocessor (loaded once per process)."""
    if not BEST_MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Run `python -m src.train` first."
        )
    model = joblib.load(BEST_MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


@lru_cache(maxsize=1)
def get_model_metadata() -> dict:
    """Load the saved training metadata (model comparison results, etc.)."""
    if not MODEL_METADATA_PATH.exists():
        return {}
    return json.loads(MODEL_METADATA_PATH.read_text())


def validate_customer_input(raw_record: dict) -> None:
    """
    Validate a raw customer record dict before it enters the pipeline.

    Args:
        raw_record: Dict of raw feature name -> value (e.g. from a form).

    Raises:
        InvalidCustomerInputError: If required fields are missing or values
            are out of a sane range.
    """
    missing = [f for f in REQUIRED_RAW_FIELDS if f not in raw_record]
    if missing:
        raise InvalidCustomerInputError(f"Missing required fields: {missing}")

    numeric_checks = {
        "tenure": (0, 100),
        "MonthlyCharges": (0, 1000),
        "TotalCharges": (0, 50000),
    }
    for field, (lo, hi) in numeric_checks.items():
        try:
            value = float(raw_record[field])
        except (TypeError, ValueError) as exc:
            raise InvalidCustomerInputError(
                f"Field '{field}' must be numeric, got: {raw_record[field]!r}"
            ) from exc
        if not (lo <= value <= hi):
            raise InvalidCustomerInputError(
                f"Field '{field}' = {value} is outside the expected range "
                f"[{lo}, {hi}]"
            )

    if int(raw_record.get("SeniorCitizen", 0)) not in (0, 1):
        raise InvalidCustomerInputError("SeniorCitizen must be 0 or 1")


def predict_and_explain(raw_record: dict, explain_with_lime: bool = True) -> dict:
    """
    Run the full pipeline: validate -> clean -> engineer features ->
    preprocess -> predict -> SHAP explain -> (optionally) LIME explain.

    Args:
        raw_record: Dict of raw feature values for one customer, matching
            `REQUIRED_RAW_FIELDS`.
        explain_with_lime: Whether to also compute a LIME explanation
            (slightly slower; disable for bulk predictions).

    Returns:
        Dict containing:
            - churn_probability: float in [0, 1]
            - predicted_label: "Yes" or "No"
            - risk_level: "Low" / "Medium" / "High"
            - shap_values: list of (feature, shap_value) for this record
            - shap_base_value: the model's expected value (baseline)
            - lime_explanation: list of (feature_description, weight), or
              None if `explain_with_lime` is False
    """
    validate_customer_input(raw_record)

    model, preprocessor = _load_model_and_preprocessor()

    df_raw = pd.DataFrame([raw_record])
    df_clean = clean_raw_data(
        df_raw.assign(customerID="TEMP", Churn="No")
    )
    df_engineered = engineer_features(df_clean)
    X_row = df_engineered.drop(columns=["Churn"])

    X_row_t = _get_transformed_frame(preprocessor, X_row)

    proba = float(model.predict_proba(X_row_t)[0, 1])
    predicted_label = "Yes" if proba >= 0.5 else "No"

    if proba < 0.33:
        risk_level = "Low"
    elif proba < 0.66:
        risk_level = "Medium"
    else:
        risk_level = "High"

    explainer = build_shap_explainer(model, X_row_t)
    shap_exp = explainer(X_row_t)
    if len(shap_exp.values.shape) == 3:
        values = shap_exp.values[0, :, 1]
        base_value = float(shap_exp.base_values[0, 1])
    else:
        values = shap_exp.values[0]
        base_value = float(shap_exp.base_values[0])

    shap_pairs = sorted(
        zip(X_row_t.columns, values), key=lambda p: abs(p[1]), reverse=True
    )
    shap_values_out = [(feat, float(val)) for feat, val in shap_pairs[:10]]

    lime_result = None
    if explain_with_lime:
        try:
            from src.data_loader import load_raw_data
            from src.preprocessing import get_feature_target_split, split_train_test

            raw_df = load_raw_data()
            X_all, y_all = get_feature_target_split(raw_df)
            X_train, _, _, _ = split_train_test(X_all, y_all)
            X_train_t = _get_transformed_frame(preprocessor, X_train.sample(
                n=min(500, len(X_train)), random_state=42
            ))
            lime_result = explain_single_prediction_lime(X_train_t, model, X_row_t)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("LIME explanation failed, continuing without it: %s", exc)
            lime_result = None

    logger.info(
        "Prediction complete: probability=%.3f, label=%s, risk=%s",
        proba, predicted_label, risk_level,
    )

    return {
        "churn_probability": proba,
        "predicted_label": predicted_label,
        "risk_level": risk_level,
        "shap_values": shap_values_out,
        "shap_base_value": base_value,
        "lime_explanation": lime_result,
    }
