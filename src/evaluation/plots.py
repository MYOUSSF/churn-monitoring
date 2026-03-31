"""
Visualisation module — 8 publication-quality plots saved to results/plots/.

1.  eda_overview.png           — class balance, tenure dist, monthly charges by churn
2.  roc_pr_curves.png          — ROC + PR curves on test set
3.  calibration_curve.png      — reliability diagram (calibrated vs uncalibrated)
4.  shap_summary.png           — beeswarm SHAP summary
5.  shap_waterfall.png         — waterfall for one high-risk customer
6.  score_drift.png            — PSI of churn score across cohorts
7.  auroc_degradation.png      — AUROC per cohort with retrain trigger line
8.  feature_drift_heatmap.png  — PSI heatmap for key features × cohorts
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, precision_recall_curve, auc

warnings.filterwarnings("ignore")

PLOT_DIR = Path(__file__).resolve().parents[2] / "results" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────

PALETTE   = {"No": "#4A90D9", "Yes": "#E05C5C"}
CMAP_MONO = "Blues"
RETRAIN_COLOR = "#E05C5C"
WARN_COLOR    = "#F5A623"
OK_COLOR      = "#4A90D9"

def _style():
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.color":        "#cccccc",
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.labelsize":    11,
        "legend.frameon":    False,
    })

_style()


# ── 1. EDA overview ───────────────────────────────────────────────────────────

def plot_eda(train_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Telco Churn — EDA overview (training set)", fontsize=14, y=1.02)

    # Class balance
    ax = axes[0]
    counts = train_df["Churn"].value_counts()
    bars = ax.bar(
        ["No churn", "Churned"],
        [counts.get(0, counts.get("No", 0)), counts.get(1, counts.get("Yes", 0))],
        color=[PALETTE["No"], PALETTE["Yes"]],
        width=0.5, edgecolor="white",
    )
    ax.set_title("Class balance")
    ax.set_ylabel("Customers")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 15, f"{int(bar.get_height()):,}",
                ha="center", fontsize=10)
    ax.set_ylim(0, max(b.get_height() for b in bars) * 1.18)

    # Tenure distribution by churn
    ax = axes[1]
    churn_col = "Churn" if train_df["Churn"].dtype == int else "Churn"
    for label, color, val in [("No churn", PALETTE["No"], 0), ("Churned", PALETTE["Yes"], 1)]:
        mask = train_df["Churn"] == val
        if mask.sum() == 0:
            mask = train_df["Churn"] == ("Yes" if val == 1 else "No")
        ax.hist(train_df.loc[mask, "tenure"], bins=30, alpha=0.65,
                color=color, label=label, edgecolor="white")
    ax.set_title("Tenure distribution by churn")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Count")
    ax.legend()

    # Monthly charges by churn
    ax = axes[2]
    no_mask  = (train_df["Churn"] == 0) | (train_df["Churn"] == "No")
    yes_mask = (train_df["Churn"] == 1) | (train_df["Churn"] == "Yes")
    ax.hist(train_df.loc[no_mask, "MonthlyCharges"],  bins=30, alpha=0.65,
            color=PALETTE["No"],  label="No churn", edgecolor="white")
    ax.hist(train_df.loc[yes_mask, "MonthlyCharges"], bins=30, alpha=0.65,
            color=PALETTE["Yes"], label="Churned",  edgecolor="white")
    ax.set_title("Monthly charges by churn")
    ax.set_xlabel("Monthly charges ($)")
    ax.set_ylabel("Count")
    ax.legend()

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "01_eda_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 01_eda_overview.png")


# ── 2. ROC + PR curves ────────────────────────────────────────────────────────

def plot_roc_pr(metrics: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC
    ax = axes[0]
    fpr, tpr = metrics["fpr"], metrics["tpr"]
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color="#4A90D9", lw=2, label=f"XGBoost (AUROC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#4A90D9")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve — test set")
    ax.legend()

    # PR
    ax = axes[1]
    prec, rec = metrics["precision"], metrics["recall"]
    pr_auc = auc(rec, prec)
    ax.plot(rec, prec, color="#E05C5C", lw=2, label=f"XGBoost (AUPRC = {pr_auc:.3f})")
    base_rate = metrics["y_true"].mean()
    ax.axhline(base_rate, ls="--", color="k", lw=1, alpha=0.5,
               label=f"Baseline ({base_rate:.3f})")
    ax.fill_between(rec, prec, alpha=0.08, color="#E05C5C")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve — test set")
    ax.legend()

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "02_roc_pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 02_roc_pr_curves.png")


# ── 3. Calibration curve ──────────────────────────────────────────────────────

def plot_calibration(
    uncalibrated_model,
    calibrated_model,
    test_df: pd.DataFrame,
    features: list[str],
) -> None:
    X = test_df[features].fillna(0)
    y = test_df["Churn"]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    for model, label, color, ls in [
        (uncalibrated_model, "XGBoost (raw)",       "#E05C5C", "--"),
        (calibrated_model,   "XGBoost (calibrated)","#4A90D9", "-"),
    ]:
        prob_pos = model.predict_proba(X)[:, 1]
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y, prob_pos, n_bins=10
        )
        ax.plot(mean_predicted_value, fraction_of_positives,
                marker="o", ms=5, lw=2, color=color, ls=ls, label=label)

    ax.plot([0, 1], [0, 1], "k:", lw=1, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve (reliability diagram)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "03_calibration_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 03_calibration_curve.png")


# ── 4. SHAP summary (beeswarm) ────────────────────────────────────────────────

def plot_shap_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    max_display: int = 15,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(
        shap_values, X,
        max_display=max_display,
        show=False,
        plot_size=None,
    )
    plt.title("SHAP feature importance (beeswarm)", fontsize=13, pad=10)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "04_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 04_shap_summary.png")


# ── 5. SHAP waterfall (single prediction) ─────────────────────────────────────

def plot_shap_waterfall(
    explainer,
    X: pd.DataFrame,
    idx: int = 0,
    max_display: int = 12,
) -> None:
    sv = explainer(X.iloc[[idx]])
    fig, ax = plt.subplots(figsize=(9, 5))
    shap.plots.waterfall(sv[0], max_display=max_display, show=False)
    plt.title(f"SHAP waterfall — customer #{idx} (high-risk example)", fontsize=12)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "05_shap_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 05_shap_waterfall.png")


# ── 6. Score PSI across cohorts ───────────────────────────────────────────────

def plot_score_drift(report_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    colors = [
        RETRAIN_COLOR if s == "retrain"
        else WARN_COLOR if s == "warning"
        else OK_COLOR
        for s in report_df["score_status"]
    ]
    bars = ax.bar(report_df["cohort"], report_df["score_psi"],
                  color=colors, width=0.6, edgecolor="white")

    ax.axhline(0.10, ls="--", color=WARN_COLOR,    lw=1.5, label="Warning threshold (0.10)")
    ax.axhline(0.20, ls="--", color=RETRAIN_COLOR, lw=1.5, label="Retrain threshold (0.20)")

    ax.set_xlabel("Production cohort")
    ax.set_ylabel("Score PSI")
    ax.set_title("Churn score distribution drift (PSI) across cohorts")
    ax.set_xticks(report_df["cohort"])
    ax.set_xticklabels([f"Cohort {c}" for c in report_df["cohort"]], rotation=20)

    legend_patches = [
        mpatches.Patch(color=OK_COLOR,      label="Stable (PSI < 0.10)"),
        mpatches.Patch(color=WARN_COLOR,    label="Warning (0.10–0.20)"),
        mpatches.Patch(color=RETRAIN_COLOR, label="Retrain (PSI > 0.20)"),
    ]
    ax.legend(handles=legend_patches, loc="upper left")

    # Annotate values
    for bar, val in zip(bars, report_df["score_psi"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.004,
                f"{val:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "06_score_drift.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 06_score_drift.png")


# ── 7. AUROC degradation ──────────────────────────────────────────────────────

def plot_auroc_degradation(report_df: pd.DataFrame, threshold: float = 0.70) -> None:
    df = report_df.dropna(subset=["auroc"]).copy()

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [RETRAIN_COLOR if r else "#4A90D9"
              for r in df["retrain_triggered"]]
    ax.plot(df["cohort"], df["auroc"], color="#4A90D9", lw=2, zorder=1)
    ax.scatter(df["cohort"], df["auroc"], c=colors, s=80, zorder=2,
               label="Cohort AUROC")

    ax.axhline(threshold, ls="--", color=RETRAIN_COLOR, lw=1.5,
               label=f"Retrain threshold ({threshold})")

    # Shade retrain region
    retrain_mask = df["retrain_triggered"]
    if retrain_mask.any():
        ax.fill_between(
            df["cohort"], threshold, df["auroc"],
            where=retrain_mask, alpha=0.15, color=RETRAIN_COLOR,
            label="Retrain triggered",
        )

    ax.set_xlabel("Production cohort")
    ax.set_ylabel("AUROC")
    ax.set_title("Model AUROC degradation across production cohorts")
    ax.set_xticks(df["cohort"])
    ax.set_xticklabels([f"Cohort {c}" for c in df["cohort"]], rotation=20)
    ax.set_ylim(max(0.5, df["auroc"].min() - 0.05), min(1.0, df["auroc"].max() + 0.05))
    ax.legend()

    # Annotate
    for _, row in df.iterrows():
        ax.text(row["cohort"], row["auroc"] + 0.006,
                f"{row['auroc']:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "07_auroc_degradation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 07_auroc_degradation.png")


# ── 8. Feature drift heatmap ──────────────────────────────────────────────────

def plot_feature_drift_heatmap(
    reference_cohort: pd.DataFrame,
    scored_cohorts:   list[pd.DataFrame],
    features:         list[str],
) -> None:
    from src.monitoring.drift import psi as psi_fn

    key_features = [
        f for f in ["MonthlyCharges", "tenure", "num_services",
                    "TotalCharges", "charges_per_month_tenure",
                    "SeniorCitizen", "PhoneService"]
        if f in reference_cohort.columns
    ]

    psi_matrix = []
    cohort_labels = []
    for cohort in scored_cohorts:
        row = []
        for feat in key_features:
            ref_vals = reference_cohort[feat].fillna(0).values
            cur_vals = cohort[feat].fillna(0).values
            row.append(psi_fn(ref_vals, cur_vals))
        psi_matrix.append(row)
        cohort_labels.append(f"Cohort {int(cohort['cohort'].iloc[0])}")

    psi_df = pd.DataFrame(psi_matrix, index=cohort_labels, columns=key_features)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        psi_df, annot=True, fmt=".3f", cmap="YlOrRd",
        linewidths=0.5, ax=ax,
        vmin=0, vmax=0.35,
        cbar_kws={"label": "PSI"},
    )
    ax.set_title("Feature drift heatmap (PSI vs reference cohort)")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Production cohort")
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "08_feature_drift_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 08_feature_drift_heatmap.png")


# ── Master runner ─────────────────────────────────────────────────────────────

def generate_all(
    train_df,
    test_df,
    uncalibrated_model,
    calibrated_model,
    shap_values,
    explainer,
    test_metrics,
    report_df,
    scored_cohorts,
    features,
) -> None:
    print("\nGenerating plots …")
    plot_eda(train_df)
    plot_roc_pr(test_metrics)
    plot_calibration(uncalibrated_model, calibrated_model, test_df, features)

    X_smoted_df = pd.DataFrame(
        shap_values,
        columns=features[:shap_values.shape[1]] if hasattr(shap_values, "shape") else features,
    )
    # Use first cohort as reference for SHAP
    X_ref = scored_cohorts[0][features].fillna(0)
    X_ref_aligned = X_ref.iloc[:, :shap_values.shape[1]] if shap_values.shape[1] <= len(X_ref.columns) else X_ref
    feat_cols = features[:shap_values.shape[1]]
    plot_shap_summary(shap_values, pd.DataFrame(shap_values, columns=feat_cols))
    plot_shap_waterfall(explainer, X_ref[feat_cols].reset_index(drop=True))
    plot_score_drift(report_df)
    plot_auroc_degradation(report_df)
    plot_feature_drift_heatmap(scored_cohorts[0], scored_cohorts, features)
    print(f"\nAll plots saved to {PLOT_DIR}")
