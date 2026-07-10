"""
eda.py
------
Performs Exploratory Data Analysis (EDA) on the Telco Customer Churn dataset
and saves all charts to `assets/` plus a written explanation of each chart to
`reports/eda_report.md`.

Why this file exists:
    EDA is a required, auditable step in any real ML project -- it justifies
    the preprocessing and feature engineering decisions made later. Running
    this script regenerates every chart and the written report from scratch,
    so the analysis is always reproducible and never manually edited.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless backend, no display server needed

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import ASSETS_DIR, REPORTS_DIR, NUMERIC_FEATURES, TARGET_COLUMN
from src.data_loader import load_raw_data, basic_data_report
from src.utils.logger import get_logger

logger = get_logger(__name__)

sns.set_theme(style="whitegrid")
PALETTE = {"No": "#2E86AB", "Yes": "#E63946"}


def _clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    TotalCharges is stored as text in the raw file and contains blank
    strings for 11 brand-new customers (tenure == 0). Coerce to numeric
    for analysis purposes only (the reusable preprocessing pipeline in
    `preprocessing.py` handles this the same way for modeling).
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def missing_and_duplicates_report(df: pd.DataFrame) -> str:
    """Return a markdown section summarizing missing values and duplicates."""
    report = basic_data_report(df)
    missing = {k: v for k, v in report["missing_values_per_column"].items() if v > 0}

    lines = [
        "## 1. Data Quality Overview",
        f"- **Rows:** {report['n_rows']}  |  **Columns:** {report['n_columns']}",
        f"- **Duplicate rows:** {report['duplicate_rows']}",
    ]
    if missing:
        lines.append("- **Missing values (raw, before coercion):**")
        for col, cnt in missing.items():
            lines.append(f"  - {col}: {cnt}")
    else:
        lines.append(
            "- **Missing values (raw):** none detected by `isnull()`, however "
            "`TotalCharges` is stored as text and contains 11 blank-string "
            "entries (all customers with `tenure == 0`, i.e. brand-new "
            "sign-ups who have not yet been billed). These become true NaNs "
            "once coerced to numeric and are handled explicitly in "
            "preprocessing."
        )
    return "\n".join(lines)


def statistics_summary(df: pd.DataFrame) -> str:
    """Return a markdown table of descriptive statistics for numeric columns."""
    stats = df[NUMERIC_FEATURES].describe().round(2)
    return "## 2. Descriptive Statistics (Numeric Features)\n\n" + stats.to_markdown()


def plot_churn_distribution(df: pd.DataFrame) -> str:
    """Bar chart of overall churn class balance."""
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = df[TARGET_COLUMN].value_counts()
    pct = (counts / counts.sum() * 100).round(1)
    bars = ax.bar(counts.index, counts.values,
                   color=[PALETTE.get(c, "#888") for c in counts.index])
    for bar, p in zip(bars, pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                 f"{p}%", ha="center", fontweight="bold")
    ax.set_title("Customer Churn Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Number of Customers")
    fig.tight_layout()
    path = ASSETS_DIR / "churn_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_numeric_distributions(df: pd.DataFrame) -> str:
    """Histograms of tenure, MonthlyCharges, TotalCharges split by churn."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col in zip(axes, NUMERIC_FEATURES):
        for label, color in PALETTE.items():
            subset = df[df[TARGET_COLUMN] == label][col].dropna()
            ax.hist(subset, bins=30, alpha=0.6, label=label, color=color)
        ax.set_title(f"{col} Distribution by Churn")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        ax.legend(title="Churn")
    fig.tight_layout()
    path = ASSETS_DIR / "numeric_distributions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    """Correlation heatmap of numeric features + binary-encoded churn."""
    corr_df = df[NUMERIC_FEATURES].copy()
    corr_df["Churn_Binary"] = (df[TARGET_COLUMN] == "Yes").astype(int)
    corr = corr_df.corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap (Numeric Features & Churn)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = ASSETS_DIR / "correlation_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_categorical_churn_rates(df: pd.DataFrame) -> str:
    """Churn rate by Contract type, InternetService, and PaymentMethod."""
    cols = ["Contract", "InternetService", "PaymentMethod"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, col in zip(axes, cols):
        churn_rate = (
            df.groupby(col)[TARGET_COLUMN]
            .apply(lambda s: (s == "Yes").mean() * 100)
            .sort_values(ascending=False)
        )
        bars = ax.bar(churn_rate.index, churn_rate.values, color="#E63946")
        ax.set_title(f"Churn Rate by {col}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Churn Rate (%)")
        ax.tick_params(axis="x", rotation=30)
        for bar, v in zip(bars, churn_rate.values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1,
                     f"{v:.1f}%", ha="center", fontsize=9)
    fig.tight_layout()
    path = ASSETS_DIR / "categorical_churn_rates.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_tenure_vs_monthly_charges(df: pd.DataFrame) -> str:
    """Scatter of tenure vs MonthlyCharges colored by churn."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for label, color in PALETTE.items():
        subset = df[df[TARGET_COLUMN] == label]
        ax.scatter(subset["tenure"], subset["MonthlyCharges"],
                    alpha=0.4, s=15, color=color, label=label)
    ax.set_title("Tenure vs Monthly Charges by Churn Status",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Monthly Charges ($)")
    ax.legend(title="Churn")
    fig.tight_layout()
    path = ASSETS_DIR / "tenure_vs_monthly_charges.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def build_eda_report(df: pd.DataFrame) -> None:
    """Generate every EDA chart and write the full markdown report."""
    logger.info("Generating EDA visualizations...")
    df_clean = _clean_total_charges(df)

    churn_path = plot_churn_distribution(df_clean)
    numeric_path = plot_numeric_distributions(df_clean)
    corr_path = plot_correlation_heatmap(df_clean)
    cat_path = plot_categorical_churn_rates(df_clean)
    scatter_path = plot_tenure_vs_monthly_charges(df_clean)

    churn_rate = (df_clean[TARGET_COLUMN] == "Yes").mean() * 100

    report = f"""# Exploratory Data Analysis Report
## Customer Churn Prediction -- IBM Telco Dataset

{missing_and_duplicates_report(df_clean)}

{statistics_summary(df_clean)}

## 3. Churn Distribution

![Churn Distribution](../assets/churn_distribution.png)

**Explanation:** Overall, {churn_rate:.1f}% of customers in the dataset have
churned. This is a moderate class imbalance (roughly 3:1 non-churn to churn),
which is why the model comparison in Phase 4 evaluates Precision, Recall,
F1, and ROC-AUC in addition to Accuracy -- accuracy alone would be misleading
on an imbalanced target.

## 4. Numeric Feature Distributions

![Numeric Distributions](../assets/numeric_distributions.png)

**Explanation:** Customers who churn are heavily concentrated at **low
tenure** (many leave within the first few months), and skew toward **higher
MonthlyCharges**. TotalCharges for churned customers clusters at lower
values, which is largely a byproduct of their short tenure rather than an
independent effect. This motivates engineering an `AvgMonthlySpend` feature
(TotalCharges / tenure) to separate "spend rate" from "time on the books."

## 5. Correlation Heatmap

![Correlation Heatmap](../assets/correlation_heatmap.png)

**Explanation:** `tenure` shows the strongest negative correlation with
churn (longer-tenured customers churn less), while `MonthlyCharges` shows a
positive correlation with churn. `TotalCharges` is highly correlated with
`tenure` itself (naturally, since it accumulates over time), which is a
signal we account for during feature engineering to avoid redundant,
collinear inputs.

## 6. Churn Rate by Key Categorical Drivers

![Categorical Churn Rates](../assets/categorical_churn_rates.png)

**Explanation:**
- **Contract:** Month-to-month customers churn at a dramatically higher
  rate than one-year or two-year contract holders -- contract length is one
  of the strongest churn predictors in the dataset.
- **InternetService:** Customers with Fiber optic service churn more than
  DSL or no-internet customers, likely reflecting price sensitivity or
  service quality/competition in the fiber segment.
- **PaymentMethod:** Electronic check payers churn noticeably more than
  customers on automatic bank transfer or credit card payments, which may
  correlate with less "sticky" payment relationships.

## 7. Tenure vs Monthly Charges

![Tenure vs Monthly Charges](../assets/tenure_vs_monthly_charges.png)

**Explanation:** Churned customers (red) cluster in the low-tenure,
mid-to-high monthly charge region. Long-tenured customers rarely churn
regardless of price, suggesting that the first several months of the
customer relationship are the highest-risk window for retention efforts.

---
*Report generated automatically by `src/eda.py`. Re-run the script to
regenerate charts and statistics if the underlying dataset changes.*
"""

    report_path = REPORTS_DIR / "eda_report.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("EDA report written to %s", report_path)
    logger.info(
        "Charts saved: %s",
        [churn_path, numeric_path, corr_path, cat_path, scatter_path],
    )


if __name__ == "__main__":
    raw_df = load_raw_data()
    build_eda_report(raw_df)
