from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.config import ASSETS_DIR, MODEL_DISPLAY_NAMES
from src.predict_pipeline import (
    predict_and_explain,
    validate_customer_input,
    InvalidCustomerInputError,
    get_model_metadata,
)

# Page configuration + design tokens

st.set_page_config(
    page_title="Churn Radar | Customer Churn Prediction",
    page_icon="\U0001F4E1",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#0F766E"       # teal -- primary brand / actions
PRIMARY_DARK = "#0B4F4A"
BG = "#0F3D3E"             
PANEL = "#141E33"
TEXT = "#E6EDF3"
MUTED = "#93A2B8"
RISK_HIGH = "#EF4444"
RISK_MED = "#F59E0B"
RISK_LOW = "#22C55E"

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background-color: {BG};
    color: {TEXT};
}}

section[data-testid="stSidebar"] {{
    background-color: {PANEL};
    border-right: 1px solid rgba(255,255,255,0.06);
}}

h1, h2, h3 {{
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: 0.2px;
}}

.churn-hero {{
    background: linear-gradient(135deg, {PRIMARY_DARK} 0%, {BG} 70%);
    border: 1px solid rgba(15,118,110,0.4);
    border-radius: 14px;
    padding: 28px 32px;
    margin-bottom: 20px;
}}

.metric-card {{
    background-color: {PANEL};
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 18px 20px;
}}

.badge {{
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.85rem;
    font-family: 'Space Grotesk', sans-serif;
}}

.badge-high {{ background-color: rgba(239,68,68,0.15); color: {RISK_HIGH}; border: 1px solid {RISK_HIGH}; }}
.badge-med  {{ background-color: rgba(245,158,11,0.15); color: {RISK_MED}; border: 1px solid {RISK_MED}; }}
.badge-low  {{ background-color: rgba(34,197,94,0.15); color: {RISK_LOW}; border: 1px solid {RISK_LOW}; }}

hr {{ border-color: rgba(255,255,255,0.08); }}

div.stButton > button {{
    background-color: {PRIMARY};
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: 600;
    padding: 0.55rem 1.4rem;
}}
div.stButton > button:hover {{
    background-color: {PRIMARY_DARK};
    color: white;
}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Sidebar navigation
with st.sidebar:
    st.markdown("## \U0001F4E1 Churn Radar")
    st.caption("Customer Churn Prediction with Explainable AI")
    page = st.radio(
        "Navigate",
        ["Home", "Predict Churn", "About the Project", "Model Information"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Built with scikit-learn, XGBoost, SHAP, LIME & Streamlit")


def risk_badge(risk_level: str) -> str:
    """Return an HTML badge span for the given risk level."""
    css_class = {"High": "badge-high", "Medium": "badge-med", "Low": "badge-low"}[risk_level]
    return f'<span class="badge {css_class}">{risk_level} Risk</span>'



# HOME PAGE
if page == "Home":
    st.markdown(
        """
        <div class="churn-hero">
            <h1 style="margin-bottom:6px;">Predict churn. Explain every prediction.</h1>
            <p style="color:#B7C4D6; font-size:1.05rem; max-width:720px;">
            Churn Radar scores telecom customers on churn risk and shows exactly
            which factors are driving each score -- powered by SHAP and LIME,
            not a black box.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    meta = get_model_metadata()
    best_result = next(
        (r for r in meta.get("all_results", []) if r["model_key"] == meta.get("best_model_key")),
        {},
    )
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Best Model", meta.get("best_model_display_name", "N/A"))
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("ROC-AUC", f"{best_result.get('roc_auc', 0):.3f}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Recall (catches churners)", f"{best_result.get('recall', 0):.1%}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Training Samples", f"{meta.get('n_train_samples', 0):,}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### How it works")
    steps = st.columns(4)
    step_texts = [
        ("1. Enter customer details", "Fill in contract, billing, and service info in the Predict Churn tab."),
        ("2. Model scores risk", "The trained model returns a churn probability and risk tier."),
        ("3. SHAP explains it", "See exactly which features pushed the score up or down."),
        ("4. LIME double-checks", "An independent local explanation confirms the reasoning."),
    ]
    for col, (title, desc) in zip(steps, step_texts):
        with col:
            st.markdown(f"**{title}**")
            st.caption(desc)

    st.markdown("### Key churn drivers (population-level)")
    st.image(str(ASSETS_DIR / "shap_feature_importance.png"), use_container_width=True)


# PREDICT CHURN PAGE
elif page == "Predict Churn":
    st.markdown("## Predict Churn for a Customer")
    st.caption("Fill in the customer's profile, then click Predict.")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Demographics**")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Has Partner", ["Yes", "No"])
            dependents = st.selectbox("Has Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)

        with c2:
            st.markdown("**Services**")
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        with c3:
            st.markdown("**Billing**")
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            )
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 500.0, 70.0, step=1.0)
            total_charges = st.number_input(
                "Total Charges ($)", 0.0, 20000.0, float(monthly_charges * max(tenure, 1)), step=10.0
            )

        submitted = st.form_submit_button("Predict Churn")

    if submitted:
        raw_record = {
            "gender": gender,
            "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        try:
            validate_customer_input(raw_record)
        except InvalidCustomerInputError as exc:
            st.error(f"Invalid input: {exc}")
            st.stop()

        with st.spinner("Scoring customer and generating explanations..."):
            try:
                result = predict_and_explain(raw_record)
            except FileNotFoundError as exc:
                st.error(str(exc))
                st.stop()
            except Exception as exc:  # pragma: no cover - defensive UI guard
                st.error(f"Prediction failed: {exc}")
                st.stop()

        st.markdown("---")
        st.markdown("### Result")

        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            proba_pct = result["churn_probability"] * 100
            st.markdown(
                f"""
                <div class="metric-card">
                    <p style="color:{MUTED}; margin-bottom:4px;">Churn Probability</p>
                    <h2 style="margin:0;">{proba_pct:.1f}%</h2>
                    <p style="margin-top:10px;">Prediction:
                        <strong>{'Will Churn' if result['predicted_label']=='Yes' else 'Will Stay'}</strong>
                    </p>
                    <p>{risk_badge(result['risk_level'])}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with res_col2:
            shap_df = pd.DataFrame(result["shap_values"], columns=["Feature", "SHAP Value"])
            shap_df = shap_df.sort_values("SHAP Value")
            fig, ax = plt.subplots(figsize=(7, 4.2))
            colors = [RISK_HIGH if v > 0 else RISK_LOW for v in shap_df["SHAP Value"]]
            ax.barh(shap_df["Feature"], shap_df["SHAP Value"], color=colors)
            ax.axvline(0, color="white", linewidth=0.8)
            ax.set_title("Top factors driving this prediction", fontsize=11, color="black")
            ax.set_xlabel("SHAP value (impact on churn probability)")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("#### Why this prediction? (SHAP)")
        st.caption(
            "Positive (red) bars push the prediction toward churn; negative "
            "(green) bars push it toward retention. Base value: "
            f"{result['shap_base_value']:.3f}"
        )

        if result["lime_explanation"]:
            st.markdown("#### Independent check (LIME)")
            lime_df = pd.DataFrame(result["lime_explanation"], columns=["Condition", "Weight"])
            st.dataframe(lime_df, use_container_width=True, hide_index=True)
            st.caption(
                "LIME fits a simple local model around this specific customer "
                "as an independent cross-check on the SHAP explanation above."
            )

# ABOUT PAGE
elif page == "About the Project":
    st.markdown("## About This Project")
    st.markdown(
        """
Churn Radar is an end-to-end machine learning system that predicts whether
a telecom customer will churn, and explains **why** for every single
prediction using Explainable AI (SHAP and LIME).

**Dataset:** IBM Telco Customer Churn dataset (7,043 customers, 21 raw
attributes covering demographics, subscribed services, contract terms, and
billing).

**Pipeline:**
1. Data cleaning & feature engineering (`src/preprocessing.py`)
2. Multi-model training & comparison (`src/train.py`) --
   Logistic Regression, Decision Tree, Random Forest, XGBoost
3. Automatic best-model selection by ROC-AUC
4. Explainability layer (`src/explain.py`) -- global SHAP summary &
   feature importance, per-prediction SHAP waterfall, and LIME local
   explanations
5. This Streamlit dashboard (`app.py`) for interactive predictions

**Why explainability matters here:** a churn score alone tells a retention
team *who* is at risk but not *what to do about it*. Pairing every
prediction with a feature-level explanation turns the model into an
actionable tool instead of a black box.
        """
    )
    st.markdown("### Project Structure")
    st.code(
        """Customer-Churn-XAI/
├── data/                 # Raw dataset
├── models/               # Saved model, preprocessor, metadata
├── notebooks/            # Exploratory notebooks (optional)
├── src/
│   ├── config.py         # Paths & constants
│   ├── data_loader.py    # Load & validate raw data
│   ├── eda.py             # Exploratory Data Analysis
│   ├── preprocessing.py  # Cleaning, feature engineering, pipeline
│   ├── train.py          # Multi-model training & comparison
│   ├── explain.py         # SHAP & LIME explainability
│   ├── predict_pipeline.py  # Unified inference entry point
│   └── utils/logger.py    # Shared logging utility
├── assets/               # Generated charts
├── reports/              # EDA report, project report
├── tests/                # Unit tests
├── app.py                # Streamlit dashboard (this app)
└── requirements.txt""",
        language="text",
    )

# MODEL INFORMATION PAGE
elif page == "Model Information":
    st.markdown("## Model Information")
    meta = get_model_metadata()

    if not meta:
        st.warning("No trained model found. Run `python -m src.train` first.")
    else:
        st.markdown(f"**Selected model:** {meta['best_model_display_name']} "
                    f"(chosen automatically by highest ROC-AUC)")

        results_df = pd.DataFrame(meta["all_results"]).drop(columns=["model_key"])
        results_df = results_df.rename(columns={
            "display_name": "Model", "accuracy": "Accuracy", "precision": "Precision",
            "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC",
        }).sort_values("ROC-AUC", ascending=False)
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        st.markdown("### Model Comparison Chart")
        st.image(str(ASSETS_DIR / "model_comparison.png"), use_container_width=True)

        st.markdown("### Global Feature Importance (SHAP)")
        st.image(str(ASSETS_DIR / "shap_summary_plot.png"), use_container_width=True)

        st.markdown("### Dataset Stats")
        col1, col2, col3 = st.columns(3)
        col1.metric("Training samples", f"{meta['n_train_samples']:,}")
        col2.metric("Test samples", f"{meta['n_test_samples']:,}")
        col3.metric("Transformed features", meta["n_features_transformed"])
