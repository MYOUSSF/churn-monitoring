"""
Visualisation module — 10 publication-quality plots.

 1. eda_overview.png           — churn rate, usage distributions, revenue
 2. horizon_churn_rates.png    — churn rate at each horizon
 3. roc_pr_curves.png          — ROC + PR curves (XGB vs baseline)
 4. calibration_curve.png      — reliability diagram (corrected calibration)
 5. shap_summary.png           — beeswarm SHAP summary
 6. shap_waterfall.png         — waterfall for high-risk customer
 7. survival_curves.png        — survival curves by risk segment
 8. score_drift.png            — PSI of churn score across cohorts
 9. auroc_degradation.png      — AUROC per cohort with retrain threshold
10. feature_drift_heatmap.png  — PSI heatmap for key features × cohorts
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
from sklearn.metrics import auc

warnings.filterwarnings("ignore")

PLOT_DIR = Path(__file__).resolve().parents[2] / "results" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [30, 60, 90, 180]

# ── Palette ────────────────────────────────────────────────────────────────────
C_BLUE    = "#2E86AB"
C_RED     = "#E05C5C"
C_ORANGE  = "#F5A623"
C_GREEN   = "#4CAF82"
C_PURPLE  = "#7B5EA7"
C_GRAY    = "#8E9AAF"

def _style():
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "#FAFAFA",
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


# ── 1. EDA overview ────────────────────────────────────────────────────────────

def plot_eda(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Cell2Cell — EDA overview", fontsize=14, y=1.01)

    # Churn balance
    ax = axes[0]
    counts = df["churndep"].value_counts().sort_index()
    bars   = ax.bar(["No churn", "Churned"], counts.values,
                    color=[C_BLUE, C_RED], width=0.5, edgecolor="white")
    ax.set_title("Class balance")
    ax.set_ylabel("Customers")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 200,
                f"{int(bar.get_height()):,}", ha="center", fontsize=10)
    ax.set_ylim(0, counts.max() * 1.15)

    # MOU by churn
    ax = axes[1]
    for val, label, color in [(0, "No churn", C_BLUE), (1, "Churned", C_RED)]:
        ax.hist(df.loc[df["churndep"] == val, "mou"].clip(0, 2000),
                bins=40, alpha=0.65, color=color, label=label, edgecolor="white")
    ax.set_title("Minutes of use (MOU) by churn")
    ax.set_xlabel("MOU (clipped at 2000)")
    ax.set_ylabel("Count")
    ax.legend()

    # Revenue by churn
    ax = axes[2]
    for val, label, color in [(0, "No churn", C_BLUE), (1, "Churned", C_RED)]:
        ax.hist(df.loc[df["churndep"] == val, "revenue"].clip(0, 200),
                bins=40, alpha=0.65, color=color, label=label, edgecolor="white")
    ax.set_title("Monthly revenue by churn")
    ax.set_xlabel("Revenue ($)")
    ax.set_ylabel("Count")
    ax.legend()

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "01_eda_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 01_eda_overview.png")


# ── 2. Horizon churn rates ─────────────────────────────────────────────────────

def plot_horizon_rates(df: pd.DataFrame) -> None:
    rates  = [df[f"churn_{h}d"].mean() for h in HORIZONS]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = [C_GREEN, C_BLUE, C_ORANGE, C_RED]
    bars    = ax.bar([f"{h}d" for h in HORIZONS], rates,
                     color=colors, width=0.5, edgecolor="white")
    ax.set_title("Churn rate by prediction horizon (Weibull event log)")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Churn rate")
    ax.set_ylim(0, max(rates) * 1.20)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{rate:.1%}", ha="center", fontsize=11, fontweight="bold")

    ax.axhline(df["churndep"].mean(), ls="--", color=C_GRAY, lw=1.5,
               label=f"Snapshot churn rate ({df['churndep'].mean():.1%})")
    ax.legend()
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "02_horizon_churn_rates.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 02_horizon_churn_rates.png")


# ── 3. ROC + PR curves (XGB vs baseline) ──────────────────────────────────────

def plot_roc_pr(xgb_metrics: dict, baseline_metrics: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    h = xgb_metrics["horizon"]

    # ROC
    ax = axes[0]
    fpr, tpr = xgb_metrics["fpr"], xgb_metrics["tpr"]
    ax.plot(fpr, tpr, color=C_BLUE, lw=2,
            label=f"XGBoost (AUROC={xgb_metrics['auroc']:.3f})")
    # Baseline
    b_fpr, b_tpr, _ = __import__(
        "sklearn.metrics", fromlist=["roc_curve"]
    ).roc_curve(baseline_metrics["y_true"], baseline_metrics["y_score"])
    ax.plot(b_fpr, b_tpr, color=C_GRAY, lw=1.5, ls="--",
            label=f"Logistic (AUROC={baseline_metrics['auroc']:.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1, alpha=0.4)
    ax.fill_between(fpr, tpr, alpha=0.07, color=C_BLUE)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC curve — {h}d horizon (test set)")
    ax.legend()

    # PR
    ax = axes[1]
    prec, rec = xgb_metrics["precision"], xgb_metrics["recall"]
    ax.plot(rec, prec, color=C_RED, lw=2,
            label=f"XGBoost (AUPRC={xgb_metrics['auprc']:.3f})")
    b_prec, b_rec, _ = __import__(
        "sklearn.metrics", fromlist=["precision_recall_curve"]
    ).precision_recall_curve(baseline_metrics["y_true"], baseline_metrics["y_score"])
    ax.plot(b_rec, b_prec, color=C_GRAY, lw=1.5, ls="--",
            label=f"Logistic (AUPRC={baseline_metrics['auprc']:.3f})")
    base_rate = xgb_metrics["y_true"].mean()
    ax.axhline(base_rate, ls=":", color="k", lw=1, alpha=0.4,
               label=f"Baseline ({base_rate:.3f})")
    ax.fill_between(rec, prec, alpha=0.07, color=C_RED)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall — {h}d horizon (test set)")
    ax.legend()

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "03_roc_pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 03_roc_pr_curves.png")


# ── 4. Calibration curve ───────────────────────────────────────────────────────

def plot_calibration(
    raw_model,
    calibrated_model,
    test_df:  pd.DataFrame,
    features: list[str],
    horizon:  int = 90,
) -> None:
    target = f"churn_{horizon}d"
    X = test_df[features].fillna(0)
    y = test_df[target]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for model, label, color, ls in [
        (raw_model,        f"XGBoost raw ({horizon}d)",        C_RED,  "--"),
        (calibrated_model, f"XGBoost calibrated ({horizon}d)", C_BLUE, "-"),
    ]:
        prob_pos = model.predict_proba(X)[:, 1]
        frac_pos, mean_pred = calibration_curve(y, prob_pos, n_bins=10)
        ax.plot(mean_pred, frac_pos, marker="o", ms=5, lw=2,
                color=color, ls=ls, label=label)

    ax.plot([0, 1], [0, 1], "k:", lw=1, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve — corrected isotonic calibration")
    ax.legend()
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "04_calibration_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 04_calibration_curve.png")


# ── 5. SHAP summary ────────────────────────────────────────────────────────────

def plot_shap_summary(
    shap_values: np.ndarray,
    X:           pd.DataFrame,
    horizon:     int = 90,
    max_display: int = 15,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(shap_values, X, max_display=max_display,
                      show=False, plot_size=None)
    plt.title(f"SHAP feature importance — {horizon}d horizon (beeswarm)",
              fontsize=13, pad=10)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "05_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 05_shap_summary.png")


# ── 6. SHAP waterfall ──────────────────────────────────────────────────────────

def plot_shap_waterfall(
    explainer,
    X:           pd.DataFrame,
    horizon:     int = 90,
    max_display: int = 12,
) -> None:
    # Find a high-risk customer
    raw_scores = explainer.shap_values(X)
    total_shap = raw_scores.sum(axis=1)
    idx        = int(np.argmax(total_shap))

    sv = explainer(X.iloc[[idx]])
    fig, ax = plt.subplots(figsize=(9, 5))
    shap.plots.waterfall(sv[0], max_display=max_display, show=False)
    plt.title(f"SHAP waterfall — {horizon}d horizon — highest-risk customer",
              fontsize=12)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "06_shap_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 06_shap_waterfall.png")


# ── 7. Survival curves by risk segment ────────────────────────────────────────

def plot_survival_curves(
    aft,
    test_df:      pd.DataFrame,
    surv_features: list[str],
    duration_col: str = "days_to_churn",
    event_col:    str = "event_observed",
) -> None:
    try:
        df_plot = test_df[surv_features + [duration_col, event_col]].fillna(0).copy()
        df_plot[duration_col] = df_plot[duration_col].clip(lower=1)

        # Split into risk tertiles by median survival
        median_surv = aft.predict_median(df_plot)
        tertiles    = pd.qcut(median_surv, q=3, labels=["High risk", "Medium risk", "Low risk"])

        fig, ax = plt.subplots(figsize=(10, 6))
        colors  = [C_RED, C_ORANGE, C_BLUE]
        times   = np.arange(1, 731)

        for segment, color in zip(["High risk", "Medium risk", "Low risk"], colors):
            mask    = tertiles == segment
            seg_df  = df_plot[mask.values].copy()
            if len(seg_df) == 0:
                continue
            # Mean survival function for the segment
            surv_fn = aft.predict_survival_function(seg_df, times=times)
            mean_s  = surv_fn.mean(axis=1)
            ax.plot(times, mean_s, color=color, lw=2, label=segment)
            ax.fill_between(times, mean_s, alpha=0.08, color=color)

        for h, ls in zip(HORIZONS, [":", "--", "-.", "-"]):
            ax.axvline(h, ls=ls, color=C_GRAY, lw=1, alpha=0.6,
                       label=f"{h}d horizon")

        ax.set_xlabel("Days from observation")
        ax.set_ylabel("Survival probability P(T > t)")
        ax.set_title("Weibull AFT survival curves by risk segment")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper right")
        plt.tight_layout()
        fig.savefig(PLOT_DIR / "07_survival_curves.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved 07_survival_curves.png")
    except Exception as e:
        print(f"  Skipped survival curves plot: {e}")


# ── 8. Score drift ─────────────────────────────────────────────────────────────

def plot_score_drift(report_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    color_map = {"stable": C_BLUE, "warning": C_ORANGE, "retrain": C_RED}
    colors    = [color_map.get(s, C_GRAY) for s in report_df["score_status"]]
    bars      = ax.bar(report_df["cohort"], report_df["score_psi"],
                       color=colors, width=0.6, edgecolor="white")
    ax.axhline(0.10, ls="--", color=C_ORANGE, lw=1.5, label="Warning (0.10)")
    ax.axhline(0.20, ls="--", color=C_RED,    lw=1.5, label="Retrain (0.20)")
    ax.set_xlabel("Production cohort")
    ax.set_ylabel("Score PSI")
    ax.set_title("Churn score PSI across cohorts (Cell2Cell)")
    ax.set_xticks(report_df["cohort"])
    ax.set_xticklabels([f"C{c}" for c in report_df["cohort"]])
    legend_patches = [
        mpatches.Patch(color=C_BLUE,   label="Stable (PSI < 0.10)"),
        mpatches.Patch(color=C_ORANGE, label="Warning (0.10–0.20)"),
        mpatches.Patch(color=C_RED,    label="Retrain (PSI > 0.20)"),
    ]
    ax.legend(handles=legend_patches, loc="upper left")
    for bar, val in zip(bars, report_df["score_psi"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003, f"{val:.3f}",
                ha="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "08_score_drift.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 08_score_drift.png")


# ── 9. AUROC degradation ───────────────────────────────────────────────────────

def plot_auroc_degradation(
    report_df: pd.DataFrame,
    threshold: float = 0.70,
) -> None:
    df = report_df.dropna(subset=["auroc"]).copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [C_RED if r else C_BLUE for r in df["retrain_triggered"]]
    ax.plot(df["cohort"], df["auroc"], color=C_BLUE, lw=2, zorder=1)
    ax.scatter(df["cohort"], df["auroc"], c=colors, s=90, zorder=2)
    ax.axhline(threshold, ls="--", color=C_RED, lw=1.5,
               label=f"Retrain threshold ({threshold})")
    retrain_mask = df["retrain_triggered"]
    if retrain_mask.any():
        ax.fill_between(df["cohort"], threshold, df["auroc"],
                        where=retrain_mask, alpha=0.15, color=C_RED,
                        label="Retrain triggered")
    ax.set_xlabel("Production cohort")
    ax.set_ylabel("AUROC")
    ax.set_title("AUROC degradation across production cohorts")
    ax.set_xticks(df["cohort"])
    ax.set_xticklabels([f"C{c}" for c in df["cohort"]])
    for _, row in df.iterrows():
        ax.text(row["cohort"], row["auroc"] + 0.005,
                f"{row['auroc']:.3f}", ha="center", fontsize=9)
    ax.legend()
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "09_auroc_degradation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 09_auroc_degradation.png")


# ── 10. Feature drift heatmap ──────────────────────────────────────────────────

def plot_feature_drift_heatmap(
    reference_cohort: pd.DataFrame,
    scored_cohorts:   list[pd.DataFrame],
) -> None:
    from src.monitoring.drift import psi as psi_fn, KEY_DRIFT_FEATURES

    key_features = [f for f in KEY_DRIFT_FEATURES
                    if f in reference_cohort.columns]

    psi_matrix, cohort_labels = [], []
    for cohort in scored_cohorts:
        row = [psi_fn(reference_cohort[f].fillna(0).values,
                      cohort[f].fillna(0).values)
               for f in key_features]
        psi_matrix.append(row)
        cohort_labels.append(f"C{int(cohort['cohort'].iloc[0])}")

    psi_df = pd.DataFrame(psi_matrix, index=cohort_labels, columns=key_features)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(psi_df, annot=True, fmt=".3f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, vmin=0, vmax=0.40,
                cbar_kws={"label": "PSI"})
    ax.set_title("Feature drift heatmap (PSI vs reference cohort) — Cell2Cell")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Production cohort")
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "10_feature_drift_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 10_feature_drift_heatmap.png")


# ── Master runner ──────────────────────────────────────────────────────────────

def generate_all(
    train_df,
    test_df,
    raw_model,
    calibrated_model,
    baseline,
    baseline_metrics,
    shap_values,
    explainer,
    test_metrics,
    report_df,
    scored_cohorts,
    features,
    horizon,
    aft=None,
    surv_features=None,
) -> None:
    print("\nGenerating plots …")

    full_df = pd.concat([train_df, test_df], ignore_index=True)
    plot_eda(full_df)
    plot_horizon_rates(full_df)
    plot_roc_pr(test_metrics, baseline_metrics)
    plot_calibration(raw_model, calibrated_model, test_df, features, horizon)

    X_shap = pd.DataFrame(shap_values,
                          columns=features[:shap_values.shape[1]])
    plot_shap_summary(shap_values, X_shap, horizon)

    X_ref = scored_cohorts[0][features[:shap_values.shape[1]]].fillna(0).reset_index(drop=True)
    plot_shap_waterfall(explainer, X_ref, horizon)

    if aft is not None and surv_features is not None:
        plot_survival_curves(aft, test_df, surv_features)

    plot_score_drift(report_df)
    plot_auroc_degradation(report_df)
    plot_feature_drift_heatmap(scored_cohorts[0], scored_cohorts)

    print(f"\n  All plots saved → {PLOT_DIR}")
