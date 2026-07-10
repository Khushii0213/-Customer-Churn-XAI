"""
explain.py
----------
Explainable AI (XAI) module. Wraps SHAP (global + local explanations) and
LIME (local explanations) around the saved best model so that every
prediction -- not just the model's aggregate behavior -- can be explained
in human-readable terms.

Why this file exists:
    A churn probability score alone is not actionable for a retention team.
    This module answers "why did the model predict this customer will
    churn?" at both the population level (SHAP summary / feature
    importance) and the individual level (SHAP waterfall/force plot, LIME
    local explanation), which is the core differentiator of this project
    versus a plain churn classifier.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

from src.config import (
    ASSETS_DIR,
    BEST_MODEL_PATH,
    PREPROCESSOR_PATH,
    FEATURE_NAMES_PATH,
)
from src.data_loader import load_raw_data
from src.preprocessing import get_feature_target_split, split_train_test
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_artifacts():
    """Load the persisted best model and fitted preprocessor from disk."""
    model = joblib.load(BEST_MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


def _get_transformed_frame(preprocessor, X_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw features into the same pandas DataFrame format the model
    was fitted on. `preprocessor` is configured with
    `set_output(transform="pandas")`, so `.transform()` already returns a
    DataFrame whose column names exactly match what the model saw during
    `fit()` -- no renaming here, or the model's internal feature-name check
    will reject the input at prediction time.
    """
    return preprocessor.transform(X_raw)


def build_shap_explainer(model, background_data: pd.DataFrame):
    """
    Build a SHAP TreeExplainer (works for tree-based models: Random Forest,
    Decision Tree, XGBoost). Falls back to a model-agnostic Explainer for
    linear models such as Logistic Regression.

    Args:
        model: The fitted best model.
        background_data: A representative sample of transformed training
            data used as the background distribution for SHAP.

    Returns:
        A fitted SHAP explainer object.
    """
    try:
        explainer = shap.TreeExplainer(model)
        logger.info("Using SHAP TreeExplainer for %s", type(model).__name__)
    except Exception:
        logger.info(
            "Model type %s not tree-based; falling back to generic Explainer",
            type(model).__name__,
        )
        explainer = shap.Explainer(model.predict_proba, background_data)
    return explainer


def generate_global_shap_plots(sample_size: int = 300) -> dict:
    """
    Generate the global SHAP explanation charts (summary plot and mean
    |SHAP value| feature importance bar plot) using a sample of the test
    set, and save them to `assets/`.

    Args:
        sample_size: Number of test rows to sample for SHAP computation
            (SHAP can be slow on the full test set for some explainers).

    Returns:
        Dict of {plot_name: saved_file_path}.
    """
    model, preprocessor = load_artifacts()
    raw_df = load_raw_data()
    X, y = get_feature_target_split(raw_df)
    _, X_test, _, _ = split_train_test(X, y)

    X_test_t = _get_transformed_frame(preprocessor, X_test)
    sample = X_test_t.sample(n=min(sample_size, len(X_test_t)), random_state=42)

    explainer = build_shap_explainer(model, sample)
    shap_values = explainer.shap_values(sample)

    # For binary classifiers, shap_values may be a list [class0, class1]
    if isinstance(shap_values, list):
        shap_values_churn = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values_churn = shap_values[:, :, 1]
    else:
        shap_values_churn = shap_values

    # --- Summary plot ---
    plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_values_churn, sample, show=False)
    summary_path = ASSETS_DIR / "shap_summary_plot.png"
    plt.tight_layout()
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()

    # --- Global feature importance (mean |SHAP value|) ---
    mean_abs_shap = np.abs(shap_values_churn).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": sample.columns,
        "importance": mean_abs_shap,
    }).sort_values("importance", ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(importance_df["feature"], importance_df["importance"], color="#2E86AB")
    ax.set_title("Global Feature Importance (Mean |SHAP Value|)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean |SHAP value| (impact on churn prediction)")
    fig.tight_layout()
    importance_path = ASSETS_DIR / "shap_feature_importance.png"
    fig.savefig(importance_path, dpi=150)
    plt.close(fig)

    logger.info("Saved SHAP summary plot to %s", summary_path)
    logger.info("Saved SHAP feature importance plot to %s", importance_path)

    return {
        "summary_plot": str(summary_path),
        "feature_importance_plot": str(importance_path),
    }


def explain_single_prediction_shap(
    model, preprocessor, X_row_raw: pd.DataFrame, save_prefix: str = "single"
) -> dict:
    """
    Generate a SHAP waterfall plot explaining one individual prediction.

    Args:
        model: Fitted model.
        preprocessor: Fitted preprocessor.
        X_row_raw: A single-row raw-feature DataFrame (before preprocessing).
        save_prefix: Filename prefix for the saved waterfall plot.

    Returns:
        Dict with the churn probability, predicted label, and saved plot path.
    """
    X_t = _get_transformed_frame(preprocessor, X_row_raw)
    explainer = build_shap_explainer(model, X_t)
    shap_values = explainer(X_t)

    # Handle multi-output SHAP explanation objects for binary classifiers
    if len(shap_values.values.shape) == 3:
        single_explanation = shap.Explanation(
            values=shap_values.values[0, :, 1],
            base_values=shap_values.base_values[0, 1],
            data=shap_values.data[0],
            feature_names=shap_values.feature_names,
        )
    else:
        single_explanation = shap_values[0]

    plt.figure(figsize=(9, 6))
    shap.plots.waterfall(single_explanation, show=False, max_display=12)
    waterfall_path = ASSETS_DIR / f"{save_prefix}_shap_waterfall.png"
    plt.tight_layout()
    plt.savefig(waterfall_path, dpi=150, bbox_inches="tight")
    plt.close()

    proba = model.predict_proba(X_t)[0, 1]
    label = "Yes (Will Churn)" if proba >= 0.5 else "No (Will Stay)"

    logger.info(
        "Saved individual SHAP waterfall plot to %s (churn probability: %.3f)",
        waterfall_path, proba,
    )

    return {
        "churn_probability": float(proba),
        "predicted_label": label,
        "waterfall_plot_path": str(waterfall_path),
    }


def explain_single_prediction_lime(
    X_train_transformed: pd.DataFrame, model, X_row_transformed: pd.DataFrame
) -> list[tuple[str, float]]:
    """
    Generate a LIME local explanation for one individual prediction.

    LIME complements SHAP by fitting a simple, locally faithful surrogate
    model around the single prediction, offering an independent sanity
    check on which features drove the churn probability.

    Args:
        X_train_transformed: Transformed training data (for LIME's
            background statistics).
        model: Fitted model exposing `predict_proba`.
        X_row_transformed: A single transformed row to explain.

    Returns:
        List of (feature_description, contribution_weight) tuples, sorted
        by absolute contribution.
    """
    explainer = LimeTabularExplainer(
        training_data=X_train_transformed.values,
        feature_names=list(X_train_transformed.columns),
        class_names=["No Churn", "Churn"],
        mode="classification",
        random_state=42,
    )

    explanation = explainer.explain_instance(
        data_row=X_row_transformed.values[0],
        predict_fn=model.predict_proba,
        num_features=10,
    )

    return explanation.as_list()


if __name__ == "__main__":
    generate_global_shap_plots()

    # Demonstrate single-prediction explanation on one test row
    model, preprocessor = load_artifacts()
    raw_df = load_raw_data()
    X, y = get_feature_target_split(raw_df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    sample_row = X_test.iloc[[0]]
    result = explain_single_prediction_shap(model, preprocessor, sample_row)
    print(result)

    X_train_t = _get_transformed_frame(preprocessor, X_train)
    X_row_t = _get_transformed_frame(preprocessor, sample_row)
    lime_result = explain_single_prediction_lime(X_train_t, model, X_row_t)
    print("\nLIME explanation:")
    for feature, weight in lime_result:
        print(f"  {feature}: {weight:.4f}")
