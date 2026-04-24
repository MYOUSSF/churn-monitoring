"""
Interactive Churn Monitoring Dashboard — Cell2Cell edition.

Streamlit app that lets users:
  - Adjust drift thresholds and watch the retrain trigger fire
  - Explore business impact with a live cost calculator
  - Compare XGBoost vs survival model at any prediction horizon
  - Inspect feature drift heatmap and SHAP importance

Run locally:
    streamlit run dashboard.py

Deploy to Streamlit Cloud (free):
    1. Push repo to GitHub
    2. Go to share.streamlit.io → New app → select repo → dashboard.py
    3. Share the link in your README

Deploy to Hugging Face Spaces (free):
    1. Create a Space with Streamlit SDK
    2. Push repo files to the Space
    3. Share the link
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Monitor — Cell2Cell",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: #1e2130;
    border-radius: 10px;
    padding: 18px 22px;
    margin: 6px 0;
  }
  .retrain-alert {
    background: #3d1a1a;
    border-left: 4px solid #e05c5c;
    border-radius: 6px;
    padding: 12px 18px;
    margin: 8px 0;
  }
  .stable-badge {
    background: #1a3d2b;
    border-left: 4px solid #4caf82;
    border-radius: 6px;
    padding: 12px 18px;
    margin: 8px 0;
  }
  .warning-badge {
    background: #3d2e1a;
    border-left: 4px solid #f5a623;
    border-radius: 6px;
    padding: 12px 18px;
    margin: 8px 0;
  }
  [data-testid="stSidebar"] { background-color: #0e1117; }
  h1 { color: #e8eaf6; }
  h2 { color: #c5cae9; border-bottom: 1px solid #2d3250; padding-bottom: 6px; }
  h3 { color: #9fa8da; }
</style>
""", unsafe_allow_html=True)

# ── Colours ────────────────────────────────────────────────────────────────────
C_BLUE   = "#2E86AB"
C_RED    = "#E05C5C"
C_ORANGE = "#F5A623"
C_GREEN  = "#4CAF82"
C_PURPLE = "#7B5EA7"
C_GRAY   = "#8E9AAF"

plt.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor":   "#1e2130",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "axes.grid":        True,
    "grid.alpha":       0.2,
    "grid.color":       "#444",
    "text.color":       "#c5cae9",
    "axes.labelcolor":  "#9fa8da",
    "xtick.color":      "#9fa8da",
    "ytick.color":      "#9fa8da",
    "font.size":        10,
    "axes.titlesize":   12,
    "legend.frameon":   False,
    "legend.facecolor": "#1e2130",
    "legend.edgecolor": "#444",
})


# ── Data loading (cached) ──────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading Cell2Cell data and training model…")
def load_everything(n_cohorts: int, drift_start: int, horizon: int):
    """Run the full pipeline once and cache all results."""
    import mlflow
    mlflow.set_tracking_uri("file:///tmp/mlflow_dashboard")

    from src.data.loader import load_pipeline
    from src.models.churn_model import train, evaluate, evaluate_baseline, score_cohorts
    from src.monitoring.drift import build_cohort_reports, reports_to_dataframe
    from src.business.business_metrics import (
        BusinessImpactCalculator, track_business_impact_over_cohorts
    )

    train_df, test_df, cohorts, features, hz = load_pipeline(
        n_cohorts=n_cohorts,
        drift_start=drift_start,
        horizon=horizon,
    )

    model, calibrated, baseline, shap_values, explainer, train_metrics = train(
        train_df, features, horizon=horizon,
        experiment_name="dashboard", run_name=f"dash_{horizon}d",
    )

    test_metrics     = evaluate(calibrated, test_df, features, horizon=horizon)
    baseline_metrics = evaluate_baseline(baseline, test_df, features, horizon=horizon)
    X_test           = test_df[features].fillna(0)
    baseline_metrics["y_true"]  = test_df[f"churn_{horizon}d"].values
    baseline_metrics["y_score"] = baseline.predict_proba(X_test)[:, 1]

    scored_cohorts = score_cohorts(calibrated, cohorts, features, horizon=horizon)

    return dict(
        train_df=train_df,
        test_df=test_df,
        cohorts=cohorts,
        features=features,
        model=model,
        calibrated=calibrated,
        baseline=baseline,
        shap_values=shap_values,
        explainer=explainer,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        baseline_metrics=baseline_metrics,
        scored_cohorts=scored_cohorts,
    )


def build_reports(scored_cohorts, features, auroc_threshold, horizon):
    from src.monitoring.drift import build_cohort_reports, reports_to_dataframe
    reports = build_cohort_reports(
        reference_cohort=scored_cohorts[0],
        scored_cohorts=scored_cohorts,
        features=features,
        auroc_threshold=auroc_threshold,
        horizon=horizon,
    )
    return reports_to_dataframe(reports)


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Pipeline Config")

    horizon = st.select_slider(
        "Prediction horizon",
        options=[30, 60, 90, 180],
        value=90,
        help="The number of days ahead the model predicts churn",
    )
    n_cohorts   = st.slider("Number of cohorts",   4, 10, 6)
    drift_start = st.slider("Drift starts at cohort", 1, n_cohorts - 1, 3)

    st.markdown("---")
    st.markdown("## 🚨 Monitoring Thresholds")

    psi_threshold  = st.slider("PSI retrain threshold", 0.10, 0.40, 0.20, 0.01,
                                help="Score PSI above this fires a retrain alert")
    auroc_threshold = st.slider("AUROC retrain threshold", 0.55, 0.85, 0.70, 0.01,
                                 help="AUROC below this fires a retrain alert")

    st.markdown("---")
    st.markdown("## 💰 Business Assumptions")

    ltv             = st.number_input("Customer LTV ($)", 200, 5000, 1200, 100)
    offer_cost      = st.number_input("Retention offer cost ($)", 10, 500, 75, 5)
    monthly_at_risk = st.number_input("Customers scored / month", 500, 50000, 5000, 500)

    st.markdown("---")
    st.caption("📉 Cell2Cell Churn Monitor  |  [GitHub](https://github.com/MYOUSSF/churn-monitoring)")


# ── Load data ──────────────────────────────────────────────────────────────────

data = load_everything(n_cohorts, drift_start, horizon)
report_df = build_reports(
    data["scored_cohorts"], data["features"], auroc_threshold, horizon
)

# Recompute retrain logic with the live PSI threshold from sidebar
from src.monitoring.drift import psi as psi_fn, psi_label, KEY_DRIFT_FEATURES

def _recompute_retrain(report_df, psi_thresh, auroc_thresh):
    df = report_df.copy()
    df["retrain_triggered"] = (
        (df["score_psi"] > psi_thresh)
        | (df["auroc"].notna() & (df["auroc"] < auroc_thresh))
        | (df["n_drifted_features"] / 10 > 0.20)
    )
    df["score_status"] = df["score_psi"].apply(
        lambda v: "stable" if v < 0.10 else ("warning" if v < psi_thresh else "retrain")
    )
    return df

report_df = _recompute_retrain(report_df, psi_threshold, auroc_threshold)

# ── Header ─────────────────────────────────────────────────────────────────────

st.title("📉 Cell2Cell Churn Monitoring Dashboard")
st.markdown(
    f"**{horizon}-day horizon** · "
    f"**{n_cohorts} cohorts** · "
    f"drift injected from cohort **{drift_start}** · "
    f"PSI threshold **{psi_threshold:.2f}** · "
    f"AUROC threshold **{auroc_threshold:.2f}**"
)

n_retrain  = report_df["retrain_triggered"].sum()
first_fire = report_df[report_df["retrain_triggered"]]["cohort"].min() if n_retrain else None

if n_retrain:
    st.markdown(
        f'<div class="retrain-alert">⚠️ <b>Retrain triggered</b> — '
        f'{int(n_retrain)} of {n_cohorts} cohorts exceed thresholds. '
        f'First alert at cohort {int(first_fire)}.</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="stable-badge">✓ <b>Model stable</b> — no retrain triggered '
        'across all cohorts.</div>',
        unsafe_allow_html=True,
    )

# ── Top KPI row ────────────────────────────────────────────────────────────────

st.markdown("---")
k1, k2, k3, k4, k5 = st.columns(5)

test_m = data["test_metrics"]
base_m = data["baseline_metrics"]

k1.metric("XGB AUROC", f"{test_m['auroc']:.3f}",
          f"{test_m['auroc'] - base_m['auroc']:+.3f} vs baseline")
k2.metric("XGB AUPRC",  f"{test_m['auprc']:.3f}")
k3.metric("Brier Score", f"{test_m['brier']:.3f}",
          f"{base_m['brier'] - test_m['brier']:+.3f} vs baseline", delta_color="inverse")
k4.metric("CV AUROC",
          f"{data['train_metrics']['cv_auroc_mean']:.3f}",
          f"± {data['train_metrics']['cv_auroc_std']:.3f}")
k5.metric("Optimal threshold",
          f"{test_m['optimal_threshold']:.2f}",
          "vs 0.50 default")

st.markdown("---")

# ── Tab layout ─────────────────────────────────────────────────────────────────

tab_monitor, tab_business, tab_model, tab_features = st.tabs([
    "🔴 Drift Monitor", "💰 Business Impact", "📊 Model Performance", "🔬 Feature Analysis"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DRIFT MONITOR
# ═══════════════════════════════════════════════════════════════════════════════

with tab_monitor:
    st.markdown("### Cohort Monitoring Summary")
    st.caption(
        "Adjust PSI and AUROC thresholds in the sidebar to watch the retrain trigger "
        "fire or clear in real time."
    )

    # Colour-coded status table
    def _colour_status(val):
        if val == "retrain":
            return "background-color: #3d1a1a; color: #e05c5c; font-weight: bold"
        elif val == "warning":
            return "background-color: #3d2e1a; color: #f5a623"
        return "background-color: #1a3d2b; color: #4caf82"

    def _colour_retrain(val):
        if val:
            return "background-color: #3d1a1a; color: #e05c5c; font-weight: bold"
        return "background-color: #1a3d2b; color: #4caf82"

    display_cols = ["cohort", "churn_rate", "mean_score", "score_psi",
                    "score_status", "auroc", "n_drifted_features", "retrain_triggered"]
    styled = (
        report_df[display_cols]
        .style
        .applymap(_colour_status,   subset=["score_status"])
        .applymap(_colour_retrain,  subset=["retrain_triggered"])
        .format({
            "churn_rate":  "{:.1%}",
            "mean_score":  "{:.3f}",
            "score_psi":   "{:.4f}",
            "auroc":       "{:.3f}",
        })
    )
    st.dataframe(styled, use_container_width=True, height=280)

    col_psi, col_auroc = st.columns(2)

    # PSI chart
    with col_psi:
        st.markdown("#### Score PSI by Cohort")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        color_map = {"stable": C_GREEN, "warning": C_ORANGE, "retrain": C_RED}
        colors = [color_map.get(s, C_GRAY) for s in report_df["score_status"]]
        bars = ax.bar(report_df["cohort"], report_df["score_psi"],
                      color=colors, width=0.6, edgecolor="#0e1117", linewidth=0.5)
        ax.axhline(0.10, ls="--", color=C_ORANGE, lw=1.2, alpha=0.8, label="Warning 0.10")
        ax.axhline(psi_threshold, ls="--", color=C_RED, lw=1.5, label=f"Retrain {psi_threshold:.2f}")
        for bar, val in zip(bars, report_df["score_psi"]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003, f"{val:.3f}",
                    ha="center", fontsize=8, color="#c5cae9")
        ax.set_xlabel("Cohort")
        ax.set_ylabel("PSI")
        ax.legend(fontsize=9)
        ax.set_xticks(report_df["cohort"])
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # AUROC chart
    with col_auroc:
        st.markdown("#### AUROC by Cohort")
        df_a = report_df.dropna(subset=["auroc"])
        fig, ax = plt.subplots(figsize=(6, 3.5))
        dot_colors = [C_RED if r else C_BLUE for r in df_a["retrain_triggered"]]
        ax.plot(df_a["cohort"], df_a["auroc"], color=C_BLUE, lw=2, zorder=1)
        ax.scatter(df_a["cohort"], df_a["auroc"], c=dot_colors, s=80, zorder=2)
        ax.axhline(auroc_threshold, ls="--", color=C_RED, lw=1.5,
                   label=f"Retrain threshold {auroc_threshold:.2f}")
        retrain_mask = df_a["retrain_triggered"]
        if retrain_mask.any():
            ax.fill_between(df_a["cohort"], auroc_threshold, df_a["auroc"],
                            where=retrain_mask, alpha=0.2, color=C_RED)
        for _, row in df_a.iterrows():
            ax.text(row["cohort"], row["auroc"] + 0.004,
                    f"{row['auroc']:.3f}", ha="center", fontsize=8, color="#c5cae9")
        ax.set_xlabel("Cohort")
        ax.set_ylabel("AUROC")
        ax.legend(fontsize=9)
        ax.set_xticks(df_a["cohort"])
        y_min = max(0.5, df_a["auroc"].min() - 0.06)
        y_max = min(1.0, df_a["auroc"].max() + 0.06)
        ax.set_ylim(y_min, y_max)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Feature drift heatmap
    st.markdown("#### Feature Drift Heatmap (PSI vs Reference Cohort)")
    key_features = [f for f in KEY_DRIFT_FEATURES
                    if f in data["scored_cohorts"][0].columns]
    psi_matrix, cohort_labels = [], []
    for cohort in data["scored_cohorts"]:
        row = [psi_fn(data["scored_cohorts"][0][f].fillna(0).values,
                      cohort[f].fillna(0).values)
               for f in key_features]
        psi_matrix.append(row)
        cohort_labels.append(f"C{int(cohort['cohort'].iloc[0])}")
    psi_df = pd.DataFrame(psi_matrix, index=cohort_labels, columns=key_features)

    import seaborn as sns
    fig, ax = plt.subplots(figsize=(12, 3.5))
    sns.heatmap(psi_df, annot=True, fmt=".3f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, vmin=0, vmax=0.40,
                cbar_kws={"label": "PSI", "shrink": 0.8},
                annot_kws={"size": 9})
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption(
        "PSI < 0.10 = stable · 0.10–0.20 = warning · > 0.20 = retrain. "
        "Drift injected into revenue, MOU, equipment age, and customer care from "
        f"cohort {drift_start} onward."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BUSINESS IMPACT
# ═══════════════════════════════════════════════════════════════════════════════

with tab_business:
    from src.business.business_metrics import (
        BusinessImpactCalculator, track_business_impact_over_cohorts
    )
    calc       = BusinessImpactCalculator(ltv, offer_cost, monthly_at_risk)
    biz_report = calc.full_report(test_m["y_true"], test_m["y_score"])
    opt        = biz_report.optimal
    at_half    = biz_report.at_half

    st.markdown("### 💰 Business Impact Calculator")
    st.caption(
        f"Assumptions: LTV=${ltv:,} · Offer cost=${offer_cost:,} · "
        f"{monthly_at_risk:,} customers scored/month. Adjust in sidebar."
    )

    # Summary KPIs
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Baseline monthly loss",
              f"${biz_report.baseline_monthly_loss:,.0f}",
              "No model — all churners lost")
    b2.metric(f"Optimal savings / mo",
              f"${opt.monthly_savings:,.0f}",
              f"threshold = {opt.threshold:.2f}")
    b3.metric("Optimal annual savings",
              f"${opt.monthly_savings * 12:,.0f}")
    b4.metric("Savings left on table (at 0.50)",
              f"${opt.monthly_savings - at_half.monthly_savings:,.0f}",
              "by using default 0.5 threshold", delta_color="inverse")

    col_sweep, col_cohort = st.columns([3, 2])

    with col_sweep:
        st.markdown("#### Threshold Sweep — Monthly Savings vs Precision/Recall")
        sweep = biz_report.threshold_sweep
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax2 = ax1.twinx()

        ax1.fill_between(sweep["threshold"], sweep["monthly_savings"],
                         alpha=0.25, color=C_GREEN)
        ax1.plot(sweep["threshold"], sweep["monthly_savings"],
                 color=C_GREEN, lw=2, label="Monthly savings ($)")
        ax1.axvline(opt.threshold, ls="--", color=C_GREEN, lw=1.5, alpha=0.8,
                    label=f"Optimal ({opt.threshold:.2f})")
        ax1.axvline(0.5, ls=":", color=C_GRAY, lw=1.2, alpha=0.7, label="Default 0.50")
        ax1.set_xlabel("Decision threshold")
        ax1.set_ylabel("Monthly savings ($)", color=C_GREEN)
        ax1.tick_params(axis="y", labelcolor=C_GREEN)

        ax2.plot(sweep["threshold"], sweep["recall"],
                 color=C_BLUE, lw=1.5, ls="--", label="Recall", alpha=0.8)
        ax2.plot(sweep["threshold"], sweep["precision"],
                 color=C_ORANGE, lw=1.5, ls="--", label="Precision", alpha=0.8)
        ax2.set_ylabel("Precision / Recall", color="#c5cae9")
        ax2.tick_params(axis="y", labelcolor="#c5cae9")
        ax2.set_ylim(0, 1.05)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_cohort:
        st.markdown("#### Savings Erosion Under Drift")
        biz_cohort_df = track_business_impact_over_cohorts(
            data["scored_cohorts"], data["features"], horizon=horizon,
            ltv=ltv, offer_cost=offer_cost, monthly_at_risk=monthly_at_risk,
        )
        if not biz_cohort_df.empty:
            fig, ax = plt.subplots(figsize=(5, 4))
            dot_colors = []
            for c in biz_cohort_df["cohort"]:
                row = report_df[report_df["cohort"] == c]
                dot_colors.append(
                    C_RED if (len(row) and row["retrain_triggered"].iloc[0]) else C_BLUE
                )
            ax.bar(biz_cohort_df["cohort"], biz_cohort_df["monthly_savings"],
                   color=dot_colors, width=0.6, edgecolor="#0e1117")
            ax.set_xlabel("Cohort")
            ax.set_ylabel("Monthly savings ($)")
            ax.set_title("As drift degrades model, savings erode")
            for x, y in zip(biz_cohort_df["cohort"], biz_cohort_df["monthly_savings"]):
                ax.text(x, y + max(y * 0.02, 10), f"${y:,.0f}",
                        ha="center", fontsize=8, color="#c5cae9")
            legend_patches = [
                mpatches.Patch(color=C_BLUE, label="Stable"),
                mpatches.Patch(color=C_RED,  label="Retrain triggered"),
            ]
            ax.legend(handles=legend_patches, fontsize=8)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # Detail table
    st.markdown("#### Decision Detail at Key Thresholds")
    key_thresholds = [opt.threshold, 0.25, 0.30, 0.40, 0.50]
    rows = []
    for t in sorted(set(key_thresholds)):
        r = calc._compute_at_threshold(test_m["y_true"], test_m["y_score"], t)
        rows.append({
            "Threshold":         f"{t:.2f}",
            "Recall":            f"{r.recall:.1%}",
            "Precision":         f"{r.precision:.1%}",
            "Churners caught/mo": f"{r.churners_caught:,}",
            "Wasted offers/mo":  f"{r.wasted_offers:,}",
            "Monthly cost":      f"${r.monthly_cost:,.0f}",
            "Monthly savings":   f"${r.monthly_savings:,.0f}",
            "Optimal?":          "✅" if abs(t - opt.threshold) < 0.01 else "",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.info(
        f"💡 **Key insight:** Using the model with an optimised threshold of "
        f"**{opt.threshold:.2f}** instead of the naive default of **0.50** saves an "
        f"additional **${opt.monthly_savings - at_half.monthly_savings:,.0f}/month** "
        f"(**${(opt.monthly_savings - at_half.monthly_savings)*12:,.0f}/year**) by "
        f"catching {opt.churners_caught - at_half.churners_caught:,} more churners "
        f"who would otherwise be missed."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════

with tab_model:
    st.markdown("### Model Performance — Test Set")

    col_roc, col_cal = st.columns(2)

    with col_roc:
        st.markdown("#### ROC Curve — XGBoost vs Logistic Baseline")
        from sklearn.metrics import roc_curve, auc
        fpr, tpr = test_m["fpr"], test_m["tpr"]
        b_fpr, b_tpr, _ = roc_curve(base_m["y_true"], base_m["y_score"])

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.plot(fpr, tpr, color=C_BLUE, lw=2,
                label=f"XGBoost (AUROC={test_m['auroc']:.3f})")
        ax.plot(b_fpr, b_tpr, color=C_GRAY, lw=1.5, ls="--",
                label=f"Logistic (AUROC={base_m['auroc']:.3f})")
        ax.plot([0, 1], [0, 1], ":", lw=1, color=C_GRAY, alpha=0.4)
        ax.fill_between(fpr, tpr, alpha=0.08, color=C_BLUE)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(fontsize=9)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_cal:
        st.markdown("#### Calibration — Corrected vs Raw")
        from sklearn.calibration import calibration_curve
        X_test = data["test_df"][data["features"]].fillna(0)
        y_test = data["test_df"][f"churn_{horizon}d"]

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        for mdl, label, color, ls in [
            (data["model"],      f"XGB raw ({horizon}d)",        C_RED,   "--"),
            (data["calibrated"], f"XGB calibrated ({horizon}d)", C_BLUE,  "-"),
        ]:
            prob_pos = mdl.predict_proba(X_test)[:, 1]
            frac, mean_pred = calibration_curve(y_test, prob_pos, n_bins=10)
            ax.plot(mean_pred, frac, marker="o", ms=4, lw=2,
                    color=color, ls=ls, label=label)
        ax.plot([0, 1], [0, 1], ":", lw=1, color=C_GRAY, alpha=0.5,
                label="Perfect calibration")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.legend(fontsize=9)
        ax.set_title("Corrected holdout isotonic calibration")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    col_pr, col_shap = st.columns(2)

    with col_pr:
        st.markdown("#### Precision-Recall Curve")
        from sklearn.metrics import precision_recall_curve, auc
        prec, rec = test_m["precision"], test_m["recall"]
        b_prec, b_rec, _ = precision_recall_curve(base_m["y_true"], base_m["y_score"])

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.plot(rec, prec, color=C_RED, lw=2,
                label=f"XGBoost (AUPRC={test_m['auprc']:.3f})")
        ax.plot(b_rec, b_prec, color=C_GRAY, lw=1.5, ls="--",
                label=f"Logistic (AUPRC={base_m['auprc']:.3f})")
        base_rate = test_m["y_true"].mean()
        ax.axhline(base_rate, ls=":", color=C_GRAY, lw=1, alpha=0.5,
                   label=f"Random ({base_rate:.3f})")
        ax.fill_between(rec, prec, alpha=0.08, color=C_RED)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(fontsize=9)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_shap:
        st.markdown("#### SHAP Feature Importance")
        try:
            import shap
            shap_vals = data["shap_values"]
            feat_imp  = np.abs(shap_vals).mean(axis=0)
            top_idx   = np.argsort(feat_imp)[-15:]
            top_feats = [data["features"][i] for i in top_idx]
            top_vals  = feat_imp[top_idx]

            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            bars = ax.barh(top_feats, top_vals, color=C_PURPLE, edgecolor="#0e1117")
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title(f"Top 15 features — {horizon}d horizon")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as e:
            st.warning(f"SHAP plot unavailable: {e}")

    st.markdown("#### Horizon Churn Rates (Weibull Event Log)")
    full_df  = pd.concat([data["train_df"], data["test_df"]], ignore_index=True)
    horizons = [30, 60, 90, 180]
    rates    = [full_df[f"churn_{h}d"].mean() for h in horizons if f"churn_{h}d" in full_df]

    fig, ax = plt.subplots(figsize=(7, 3))
    colors_h = [C_GREEN, C_BLUE, C_ORANGE, C_RED]
    bars = ax.bar([f"{h}d" for h in horizons[:len(rates)]], rates,
                  color=colors_h[:len(rates)], width=0.5, edgecolor="#0e1117")
    ax.axhline(full_df["churndep"].mean() if "churndep" in full_df.columns else 0,
               ls="--", color=C_GRAY, lw=1.2, label="Snapshot churn rate")
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.004,
                f"{rate:.1%}", ha="center", fontsize=10, color="#c5cae9",
                fontweight="bold")
    ax.set_ylabel("Churn rate")
    ax.legend(fontsize=9)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption(
        "Horizon labels generated by the Weibull event log. Shape k=1.5 → "
        "increasing hazard over time (aging / contract-expiry effect)."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — FEATURE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_features:
    st.markdown("### Feature Analysis")

    col_eda1, col_eda2 = st.columns(2)
    full_df = pd.concat([data["train_df"], data["test_df"]], ignore_index=True)
    target_col = f"churn_{horizon}d"

    with col_eda1:
        feat_choice = st.selectbox(
            "Distribution by churn status",
            options=["mou", "revenue", "eqpdays", "custcare",
                     "overage", "months", "revenue_per_mou", "custcare_rate"],
        )
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        if feat_choice in full_df.columns and target_col in full_df.columns:
            for val, label, color in [(0, "No churn", C_BLUE), (1, "Churned", C_RED)]:
                data_slice = full_df.loc[full_df[target_col] == val, feat_choice].dropna()
                clip_val   = data_slice.quantile(0.99)
                ax.hist(data_slice.clip(upper=clip_val), bins=40, alpha=0.6,
                        color=color, label=label, edgecolor="none")
        ax.set_xlabel(feat_choice)
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_eda2:
        st.markdown("**Feature correlation with churn**")
        num_feats = [f for f in data["features"]
                     if f in full_df.columns
                     and full_df[f].dtype in [np.float64, np.float32, np.int64, np.int32]
                     and target_col in full_df.columns][:40]
        if num_feats and target_col in full_df.columns:
            corrs = full_df[num_feats + [target_col]].corr()[target_col].drop(target_col)
            top_corr = corrs.abs().nlargest(15)
            top_corr_vals = corrs[top_corr.index]

            fig, ax = plt.subplots(figsize=(5.5, 4))
            colors_c = [C_RED if v > 0 else C_BLUE for v in top_corr_vals.values]
            ax.barh(top_corr_vals.index, top_corr_vals.values,
                    color=colors_c, edgecolor="#0e1117")
            ax.axvline(0, color=C_GRAY, lw=0.8)
            ax.set_xlabel(f"Pearson correlation with churn_{horizon}d")
            ax.set_title("Top 15 features by |correlation|")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.markdown("#### Score Distribution by Cohort")
    fig, ax = plt.subplots(figsize=(12, 3.5))
    for i, cohort in enumerate(data["scored_cohorts"]):
        if "churn_score" not in cohort.columns:
            continue
        color  = C_RED if report_df.loc[report_df["cohort"] == i, "retrain_triggered"].values[0] else C_BLUE
        alpha  = 0.85 if i == 0 else 0.55
        lw     = 2.5  if i == 0 else 1.5
        label  = f"C{i} (ref)" if i == 0 else f"C{i}"
        scores = cohort["churn_score"].clip(0, 1)
        ax.hist(scores, bins=50, density=True, histtype="step",
                color=color, lw=lw, alpha=alpha, label=label)
    ax.set_xlabel("Churn score")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, ncol=3)
    ax.set_title("Score distribution shift — reference vs drifted cohorts (red = retrain triggered)")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
