"""
train.py
--------
Trains multiple classification models on the preprocessed Telco Customer
Churn data, evaluates them on a held-out test set using Accuracy, Precision,
Recall, F1, and ROC-AUC, and automatically saves the best-performing model
(plus its fitted preprocessor) to disk with joblib.

Why this file exists:
    A production churn model should never be chosen "by feel." This module
    encodes an explicit, repeatable selection rule (highest ROC-AUC, a
    threshold-independent metric well suited to an imbalanced target) so
    that re-running training on updated data always yields a defensible,
    reproducible model choice.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.config import (
    RANDOM_STATE,
    BEST_MODEL_PATH,
    PREPROCESSOR_PATH,
    MODEL_METADATA_PATH,
    FEATURE_NAMES_PATH,
    MODEL_DISPLAY_NAMES,
)
from src.data_loader import load_raw_data
from src.preprocessing import (
    build_preprocessor,
    get_feature_target_split,
    split_train_test,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelResult:
    """Container for one model's evaluation metrics."""
    model_key: str
    display_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


def get_candidate_models() -> dict:
    """
    Define the candidate models to train and compare.

    Returns:
        Dict mapping internal model key -> unfitted estimator instance.
    """
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=8, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, random_state=RANDOM_STATE,
            class_weight="balanced", n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            random_state=RANDOM_STATE, eval_metric="logloss",
        ),
    }


def evaluate_model(model, X_test_transformed, y_test) -> ModelResult:
    """
    Compute the standard classification metric suite for a fitted model.

    Args:
        model: A fitted classifier exposing predict() and predict_proba().
        X_test_transformed: Preprocessed test features.
        y_test: True binary labels.

    Returns:
        A populated `ModelResult`.
    """
    y_pred = model.predict(X_test_transformed)
    y_proba = model.predict_proba(X_test_transformed)[:, 1]

    return ModelResult(
        model_key="",  # filled by caller
        display_name="",  # filled by caller
        accuracy=round(accuracy_score(y_test, y_pred), 4),
        precision=round(precision_score(y_test, y_pred), 4),
        recall=round(recall_score(y_test, y_pred), 4),
        f1=round(f1_score(y_test, y_pred), 4),
        roc_auc=round(roc_auc_score(y_test, y_proba), 4),
    )


def train_and_compare_models() -> tuple[str, list[ModelResult]]:
    """
    Full training pipeline: load data, preprocess, train every candidate
    model, evaluate, and identify the best model by ROC-AUC.

    Returns:
        Tuple of (best_model_key, list_of_all_results).
    """
    raw_df = load_raw_data()
    X, y = get_feature_target_split(raw_df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train, y_train)
    X_test_t = preprocessor.transform(X_test)

    results: list[ModelResult] = []
    fitted_models: dict = {}

    for key, model in get_candidate_models().items():
        logger.info("Training %s ...", MODEL_DISPLAY_NAMES[key])
        model.fit(X_train_t, y_train)
        result = evaluate_model(model, X_test_t, y_test)
        result.model_key = key
        result.display_name = MODEL_DISPLAY_NAMES[key]
        results.append(result)
        fitted_models[key] = model
        logger.info(
            "%s -> Accuracy: %.4f | Precision: %.4f | Recall: %.4f | "
            "F1: %.4f | ROC-AUC: %.4f",
            result.display_name, result.accuracy, result.precision,
            result.recall, result.f1, result.roc_auc,
        )

    best_result = max(results, key=lambda r: r.roc_auc)
    best_key = best_result.model_key
    best_model = fitted_models[best_key]

    logger.info(
        "Best model selected: %s (ROC-AUC = %.4f)",
        best_result.display_name, best_result.roc_auc,
    )

    # Persist artifacts needed for inference
    joblib.dump(best_model, BEST_MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    feature_names = list(preprocessor.get_feature_names_out())
    FEATURE_NAMES_PATH.write_text(json.dumps(feature_names, indent=2))

    metadata = {
        "best_model_key": best_key,
        "best_model_display_name": best_result.display_name,
        "all_results": [asdict(r) for r in results],
        "n_train_samples": int(X_train.shape[0]),
        "n_test_samples": int(X_test.shape[0]),
        "n_features_raw": int(X_train.shape[1]),
        "n_features_transformed": len(feature_names),
    }
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    logger.info("Saved best model to %s", BEST_MODEL_PATH)
    logger.info("Saved preprocessor to %s", PREPROCESSOR_PATH)
    logger.info("Saved metadata to %s", MODEL_METADATA_PATH)

    return best_key, results


def print_comparison_table(results: list[ModelResult]) -> None:
    """Pretty-print the model comparison table to the console."""
    df = pd.DataFrame([asdict(r) for r in results]).drop(columns=["model_key"])
    df = df.sort_values("roc_auc", ascending=False).reset_index(drop=True)
    print("\n=== Model Comparison (sorted by ROC-AUC) ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    best_key, all_results = train_and_compare_models()
    print_comparison_table(all_results)
    print(f"\nBest model: {MODEL_DISPLAY_NAMES[best_key]}")
