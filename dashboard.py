"""
Interactive Churn Monitoring Dashboard — Cell2Cell edition.

New in this version:
  Tab 1 — Drift Monitor  + Live cohort simulation (▶ Simulate Deployment button)
  Tab 2 — Business Impact + Campaign budget simulator (efficient-frontier slider)
  Tab 3 — Customer Explorer — top-risk table + per-customer SHAP waterfall
  Tab 4 — Survival Curves  — personalised survival curve with feature sliders
  Tab 5 — Model Performance (existing)
  Tab 6 — Feature Analysis  (existing)

Run:
    streamlit run dashboard.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick

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

# ── Dependency check ───────────────────────────────────────────────────────────
try:
    import xgboost  # noqa: F401
except ImportError:
    st.error(
        "**xgboost is not installed.**\n\n"
        "```bash\npip install xgboost\n```"
    )
    st.stop()

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #f7f8fc; }

  [data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e2e8f0;
  }
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3,
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] span { color: #1a202c !important; }

  .retrain-alert {
    background: #fff5f5; border-left: 4px solid #e53e3e;
    border-radius: 6px; padding: 14px 20px; margin: 10px 0;
    color: #742a2a; font-weight: 500;
  }
  .stable-badge {
    background: #f0fff4; border-left: 4px solid #38a169;
    border-radius: 6px; padding: 14px 20px; margin: 10px 0;
    color: #276749; font-weight: 500;
  }
  .warning-badge {
    background: #fffbeb; border-left: 4px solid #d97706;
    border-radius: 6px; padding: 14px 20px; margin: 10px 0;
    color: #7c2d12; font-weight: 500;
  }
  .customer-card {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 8px; padding: 16px 20px; margin: 8px 0;
  }
  h1 { color: #1a202c; font-weight: 700; }
  h2 { color: #2d3748; font-weight: 600;
       border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; }
  h3 { color: #4a5568; font-weight: 600; }
  [data-testid="metric-container"] {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 8px; padding: 12px 16px;
  }
</style>
""", unsafe_allow_html=True)

# ── Palette ────────────────────────────────────────────────────────────────────
C_BLUE   = "#3B82F6"
C_TEAL   = "#0D9488"
C_AMBER  = "#D97706"
C_ROSE   = "#DC2626"
C_VIOLET = "#7C3AED"
C_SLATE  = "#64748B"
C_GREEN  = "#22C55E"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#f8fafc",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": True, "axes.spines.bottom": True,
    "axes.edgecolor": "#cbd5e1", "axes.grid": True,
    "grid.alpha": 0.4, "grid.color": "#e2e8f0",
    "text.color": "#1e293b", "axes.labelcolor": "#334155",
    "xtick.color": "#475569", "ytick.color": "#475569",
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "semibold",
    "legend.frameon": True, "legend.framealpha": 0.92,
    "legend.facecolor": "white", "legend.edgecolor": "#e2e8f0",
})

fmt_dollar = mtick.FuncFormatter(lambda x, _: f"${x:,.0f}")


# ── Data loading (cached) ──────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading Cell2Cell data and training model…")
def load_everything(n_cohorts: int, drift_start: int, horizon: int):
    import mlflow
    mlflow.set_tracking_uri("file:///tmp/mlflow_dashboard")

    from src.data.loader import load_pipeline
    from src.models.churn_model import train, evaluate, evaluate_baseline, score_cohorts

    train_df, test_df, cohorts, features, hz = load_pipeline(
        n_cohorts=n_cohorts, drift_start=drift_start, horizon=horizon,
    )
    model, calibrated, baseline, shap_values, explainer, train_metrics = train(
        train_df, features, horizon=horizon,
        experiment_name="dashboard", run_name=f"dash_{horizon}d",
    )
    test_metrics     = evaluate(calibrated, test_df, features, horizon=horizon)
    baseline_metrics = evaluate_baseline(baseline, test_df, features, horizon=horizon)
    X_test = test_df[features].fillna(0)
    baseline_metrics["y_true"]  = test_df[f"churn_{horizon}d"].values
    baseline_metrics["y_score"] = baseline.predict_proba(X_test)[:, 1]
    scored_cohorts = score_cohorts(calibrated, cohorts, features, horizon=horizon)

    # Pre-compute per-customer SHAP on reference cohort (top 200 for speed)
    ref_cohort   = scored_cohorts[0].copy()
    X_ref        = ref_cohort[features].fillna(0)
    ref_sorted   = ref_cohort.sort_values("churn_score", ascending=False).head(200)
    X_ref_top    = ref_sorted[features].fillna(0).reset_index(drop=True)
    shap_top     = explainer.shap_values(X_ref_top)

    return dict(
        train_df=train_df, test_df=test_df, cohorts=cohorts,
        features=features, model=model, calibrated=calibrated,
        baseline=baseline, shap_values=shap_values, explainer=explainer,
        train_metrics=train_metrics, test_metrics=test_metrics,
        baseline_metrics=baseline_metrics, scored_cohorts=scored_cohorts,
        ref_sorted=ref_sorted.reset_index(drop=True),
        shap_top=shap_top,
    )


@st.cache_data(show_spinner=False)
def load_survival_model(_train_df, _features):
    """Fit WeibullAFT (cached separately — slower to train)."""
    try:
        from src.models.survival import train_survival
        aft, surv_features, metrics = train_survival(_train_df, _features)
        return aft, surv_features, metrics
    except Exception as e:
        return None, None, {"error": str(e)}


def build_reports(scored_cohorts, features, auroc_threshold, horizon):
    from src.monitoring.drift import build_cohort_reports, reports_to_dataframe
    return reports_to_dataframe(build_cohort_reports(
        reference_cohort=scored_cohorts[0],
        scored_cohorts=scored_cohorts,
        features=features,
        auroc_threshold=auroc_threshold,
        horizon=horizon,
    ))


def _recompute(df, psi_thresh, auroc_thresh):
    df = df.copy()
    df["retrain_triggered"] = (
        (df["score_psi"] > psi_thresh)
        | (df["auroc"].notna() & (df["auroc"] < auroc_thresh))
        | (df["n_drifted_features"] / 10 > 0.20)
    )
    df["score_status"] = df["score_psi"].apply(
        lambda v: "stable" if v < 0.10 else ("warning" if v < psi_thresh else "retrain")
    )
    return df


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Pipeline Config")
    horizon     = st.select_slider("Prediction horizon (days)",
                                   options=[30, 60, 90, 180], value=90)
    n_cohorts   = st.slider("Number of cohorts", 4, 10, 6)
    drift_start = st.slider("Drift starts at cohort", 1, n_cohorts - 1, 3)

    st.markdown("---")
    st.markdown("## 🚨 Monitoring Thresholds")
    psi_threshold   = st.slider("PSI retrain threshold",   0.10, 0.40, 0.20, 0.01)
    auroc_threshold = st.slider("AUROC retrain threshold", 0.55, 0.85, 0.70, 0.01)

    st.markdown("---")
    st.markdown("## 💰 Business Assumptions")
    ltv             = st.number_input("Customer LTV ($)",          200, 5000, 1200, 100)
    offer_cost      = st.number_input("Retention offer cost ($)",   10,  500,   75,   5)
    monthly_at_risk = st.number_input("Customers scored / month",  500,50000, 5000, 500)

    st.markdown("---")
    load_surv = st.checkbox("Load survival model (slower)", value=False,
                            help="Enables the Survival Curves tab")
    st.caption("📉 Cell2Cell Churn Monitor")


# ── Load data ──────────────────────────────────────────────────────────────────

data      = load_everything(n_cohorts, drift_start, horizon)
report_df = build_reports(data["scored_cohorts"], data["features"],
                          auroc_threshold, horizon)
report_df = _recompute(report_df, psi_threshold, auroc_threshold)

from src.monitoring.drift import psi as psi_fn, KEY_DRIFT_FEATURES, psi_label

aft, surv_features = None, None
if load_surv:
    with st.spinner("Fitting WeibullAFT survival model…"):
        aft, surv_features, _ = load_survival_model(
            data["train_df"], data["features"]
        )

# ── Header ─────────────────────────────────────────────────────────────────────

st.title("📉 Cell2Cell Churn Monitoring Dashboard")
st.markdown(
    f"**{horizon}-day horizon** · {n_cohorts} cohorts · "
    f"drift from cohort **{drift_start}** · "
    f"PSI **{psi_threshold:.2f}** · AUROC **{auroc_threshold:.2f}**"
)

n_retrain  = report_df["retrain_triggered"].sum()
first_fire = (report_df[report_df["retrain_triggered"]]["cohort"].min()
              if n_retrain else None)

if n_retrain:
    st.markdown(
        f'<div class="retrain-alert">⚠️ <b>Retrain triggered</b> — '
        f'{int(n_retrain)} of {n_cohorts} cohorts exceed thresholds. '
        f'First alert at cohort {int(first_fire)}.</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="stable-badge">✓ <b>Model stable</b> — '
        'no retrain triggered.</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")
test_m = data["test_metrics"]
base_m = data["baseline_metrics"]

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("XGB AUROC",    f"{test_m['auroc']:.3f}",
          f"{test_m['auroc'] - base_m['auroc']:+.3f} vs baseline")
k2.metric("XGB AUPRC",    f"{test_m['auprc']:.3f}")
k3.metric("Brier Score",  f"{test_m['brier']:.3f}",
          f"{base_m['brier'] - test_m['brier']:+.3f} vs baseline",
          delta_color="inverse")
k4.metric("CV AUROC",
          f"{data['train_metrics']['cv_auroc_mean']:.3f}",
          f"± {data['train_metrics']['cv_auroc_std']:.3f}")
k5.metric("Optimal threshold",
          f"{test_m['optimal_threshold']:.3f}")

st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────────

(tab_monitor, tab_business, tab_customers,
 tab_survival, tab_model, tab_features) = st.tabs([
    "🔴 Drift Monitor",
    "💰 Business Impact",
    "🔍 Customer Explorer",
    "📈 Survival Curves",
    "📊 Model Performance",
    "🔬 Feature Analysis",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DRIFT MONITOR  (+ live simulation)
# ═══════════════════════════════════════════════════════════════════════════════

with tab_monitor:
    st.markdown("### Cohort Monitoring Summary")

    # ── Static table ──────────────────────────────────────────────────────────
    def _style_status(val):
        if val == "retrain":
            return "background-color:#fee2e2;color:#991b1b;font-weight:600"
        elif val == "warning":
            return "background-color:#fef3c7;color:#92400e"
        return "background-color:#d1fae5;color:#065f46"

    def _style_retrain(val):
        return ("background-color:#fee2e2;color:#991b1b;font-weight:600"
                if val else "background-color:#d1fae5;color:#065f46")

    display_cols = ["cohort", "churn_rate", "mean_score", "score_psi",
                    "score_status", "auroc", "n_drifted_features", "retrain_triggered"]
    styled = (
        report_df[display_cols].style
        .applymap(_style_status,  subset=["score_status"])
        .applymap(_style_retrain, subset=["retrain_triggered"])
        .format({"churn_rate": "{:.1%}", "mean_score": "{:.3f}",
                 "score_psi": "{:.4f}", "auroc": "{:.3f}"})
    )
    st.dataframe(styled, use_container_width=True, height=265)

    col_psi, col_auroc = st.columns(2)

    with col_psi:
        st.markdown("#### Score PSI by Cohort")
        color_map  = {"stable": C_TEAL, "warning": C_AMBER, "retrain": C_ROSE}
        bar_colors = [color_map.get(s, C_SLATE) for s in report_df["score_status"]]
        fig, ax = plt.subplots(figsize=(6, 3.8))
        bars = ax.bar(report_df["cohort"], report_df["score_psi"],
                      color=bar_colors, width=0.55, edgecolor="white")
        ax.axhline(0.10, ls="--", color=C_AMBER, lw=1.4, label="Warning (0.10)")
        ax.axhline(psi_threshold, ls="--", color=C_ROSE, lw=1.6,
                   label=f"Retrain ({psi_threshold:.2f})")
        for bar, val in zip(bars, report_df["score_psi"]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.003, f"{val:.3f}",
                    ha="center", fontsize=8.5, color="#334155")
        ax.legend(handles=[
            mpatches.Patch(color=C_TEAL,  label="Stable"),
            mpatches.Patch(color=C_AMBER, label="Warning"),
            mpatches.Patch(color=C_ROSE,  label="Retrain"),
        ], fontsize=9)
        ax.set_xlabel("Cohort"); ax.set_ylabel("PSI")
        ax.set_xticks(report_df["cohort"])
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_auroc:
        st.markdown("#### AUROC by Cohort")
        df_a = report_df.dropna(subset=["auroc"])
        dot_c = [C_ROSE if r else C_TEAL for r in df_a["retrain_triggered"]]
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.plot(df_a["cohort"], df_a["auroc"], color=C_BLUE, lw=2, zorder=1)
        ax.scatter(df_a["cohort"], df_a["auroc"],
                   c=dot_c, s=80, zorder=2, edgecolors="white", linewidth=1.2)
        ax.axhline(auroc_threshold, ls="--", color=C_ROSE, lw=1.6,
                   label=f"Retrain ({auroc_threshold:.2f})")
        if df_a["retrain_triggered"].any():
            ax.fill_between(df_a["cohort"], auroc_threshold, df_a["auroc"],
                            where=df_a["retrain_triggered"],
                            alpha=0.12, color=C_ROSE)
        for _, row in df_a.iterrows():
            ax.text(row["cohort"], row["auroc"] + 0.004,
                    f"{row['auroc']:.3f}", ha="center", fontsize=8.5)
        ax.set_xlabel("Cohort"); ax.set_ylabel("AUROC")
        ax.legend(fontsize=9); ax.set_xticks(df_a["cohort"])
        ax.set_ylim(max(0.50, df_a["auroc"].min()-0.06),
                    min(1.00, df_a["auroc"].max()+0.06))
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Feature drift heatmap
    st.markdown("#### Feature Drift Heatmap")
    import seaborn as sns
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
    fig, ax = plt.subplots(figsize=(12, 3.8))
    sns.heatmap(psi_df, annot=True, fmt=".3f",
                cmap=sns.light_palette(C_BLUE, as_cmap=True),
                linewidths=0.5, ax=ax, vmin=0, vmax=0.40,
                cbar_kws={"label": "PSI", "shrink": 0.8},
                annot_kws={"size": 9, "color": "#1e293b"})
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption("PSI < 0.10 stable · 0.10–0.20 monitor · > 0.20 retrain")

    # ── ▶ LIVE SIMULATION ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ▶ Live Deployment Simulation")
    st.caption(
        "Press the button to simulate cohorts arriving one by one, exactly as "
        "they would in production. Watch the retrain alert fire when drift "
        "crosses the thresholds set in the sidebar."
    )

    sim_col1, sim_col2 = st.columns([1, 3])
    with sim_col1:
        sim_speed  = st.select_slider("Speed", options=["Slow", "Normal", "Fast"],
                                      value="Normal")
        run_sim    = st.button("▶  Simulate Deployment", type="primary",
                               use_container_width=True)

    delay_map  = {"Slow": 1.8, "Normal": 1.0, "Fast": 0.4}
    sim_delay  = delay_map[sim_speed]

    if run_sim:
        scored_cohorts = data["scored_cohorts"]
        ref_scores     = scored_cohorts[0]["churn_score"].values

        psi_placeholder     = sim_col2.empty()
        status_placeholder  = st.empty()
        log_placeholder     = st.empty()

        event_log  = []
        psi_hist   = []
        auroc_hist = []

        for i, cohort in enumerate(scored_cohorts):
            cur_scores  = cohort["churn_score"].values
            score_psi   = psi_fn(ref_scores, cur_scores)
            status      = psi_label(score_psi)

            from sklearn.metrics import roc_auc_score
            target_col = f"churn_{horizon}d"
            try:
                auroc = roc_auc_score(cohort[target_col], cur_scores)
            except Exception:
                auroc = None

            psi_hist.append(score_psi)
            auroc_hist.append(auroc)

            key_feats = [f for f in KEY_DRIFT_FEATURES if f in cohort.columns]
            from src.monitoring.drift import detect_feature_drift
            drift_results = detect_feature_drift(
                scored_cohorts[0], cohort, key_feats
            )
            n_drifted = sum(1 for r in drift_results if r.drifted)

            retrain = (
                score_psi > psi_threshold
                or (auroc is not None and auroc < auroc_threshold)
                or (n_drifted / max(len(key_feats), 1) > 0.20)
            )

            # Live PSI bar chart
            fig, ax = plt.subplots(figsize=(8, 2.8))
            x = np.arange(i + 1)
            bar_c = []
            for j, p in enumerate(psi_hist):
                s = psi_label(p)
                bar_c.append(
                    C_ROSE if s == "retrain" else
                    C_AMBER if s == "warning" else C_TEAL
                )
            ax.bar(x, psi_hist, color=bar_c, width=0.5, edgecolor="white")
            ax.axhline(psi_threshold, ls="--", color=C_ROSE, lw=1.4,
                       label=f"Retrain {psi_threshold:.2f}")
            ax.axhline(0.10, ls=":", color=C_AMBER, lw=1.2,
                       label="Warning 0.10")
            ax.set_xticks(x)
            ax.set_xticklabels([f"C{j}" for j in range(i+1)])
            ax.set_ylabel("Score PSI")
            ax.set_title(f"Live monitoring — cohort {i} arrived")
            ax.legend(fontsize=8)
            fig.tight_layout()
            psi_placeholder.pyplot(fig, use_container_width=True)
            plt.close(fig)

            # Status banner
            if retrain:
                reasons = []
                if score_psi > psi_threshold:
                    reasons.append(f"PSI={score_psi:.3f}>{psi_threshold:.2f}")
                if auroc and auroc < auroc_threshold:
                    reasons.append(f"AUROC={auroc:.3f}<{auroc_threshold:.2f}")
                if n_drifted / max(len(key_feats), 1) > 0.20:
                    reasons.append(f"{n_drifted}/{len(key_feats)} features drifted")
                status_placeholder.markdown(
                    f'<div class="retrain-alert">🚨 <b>Cohort {i}: RETRAIN TRIGGERED</b> '
                    f'— {" | ".join(reasons)}</div>',
                    unsafe_allow_html=True,
                )
                event_log.append(f"⚠ C{i}: RETRAIN — {', '.join(reasons)}")
            else:
                auroc_str = f"{auroc:.3f}" if auroc is not None else "N/A"
                status_placeholder.markdown(
                    f'<div class="stable-badge">✓ <b>Cohort {i}: Stable</b> '
                    f'— PSI={score_psi:.4f} | AUROC={auroc_str}</div>',
                    unsafe_allow_html=True,
                )
                event_log.append(
                    f"✓ C{i}: stable — PSI={score_psi:.4f}, "
                    f"AUROC={auroc_str}, drifted_feats={n_drifted}"
                )

            log_placeholder.code("\n".join(event_log), language=None)
            time.sleep(sim_delay)

        status_placeholder.markdown(
            f'<div class="{"retrain-alert" if any("RETRAIN" in e for e in event_log) else "stable-badge"}">'
            f'<b>Simulation complete.</b> {sum(1 for e in event_log if "RETRAIN" in e)} '
            f'retrain alert(s) across {len(scored_cohorts)} cohorts.</div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BUSINESS IMPACT  (+ campaign budget simulator)
# ═══════════════════════════════════════════════════════════════════════════════

with tab_business:
    from src.business.business_metrics import (
        BusinessImpactCalculator, track_business_impact_over_cohorts,
    )
    calc       = BusinessImpactCalculator(ltv, offer_cost, monthly_at_risk)
    biz_report = calc.full_report(test_m["y_true"], test_m["y_score"])
    opt        = biz_report.optimal
    at_half    = biz_report.at_half

    st.markdown("### 💰 Business Impact Calculator")
    st.caption(f"LTV=${ltv:,} · Offer=${offer_cost:,} · "
               f"{monthly_at_risk:,} customers/month")

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Baseline monthly loss",
              f"${biz_report.baseline_monthly_loss:,.0f}", "No model")
    b2.metric("Optimal savings / month",
              f"${opt.monthly_savings:,.0f}",
              f"threshold={opt.threshold:.3f}")
    b3.metric("Annual savings", f"${opt.monthly_savings * 12:,.0f}")
    b4.metric("Extra vs naive 0.50",
              f"${opt.monthly_savings - at_half.monthly_savings:,.0f}",
              "per month", delta_color="inverse")

    col_sweep, col_cohort = st.columns([3, 2])

    with col_sweep:
        st.markdown("#### Threshold Sweep — Savings vs Precision/Recall")
        sweep = biz_report.threshold_sweep
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax2 = ax1.twinx()
        ax1.fill_between(sweep["threshold"], sweep["monthly_savings"],
                         alpha=0.12, color=C_TEAL)
        ax1.plot(sweep["threshold"], sweep["monthly_savings"],
                 color=C_TEAL, lw=2.5, label="Monthly savings ($)")
        ax1.axvline(opt.threshold, ls="--", color=C_TEAL, lw=2,
                    label=f"Optimal τ={opt.threshold:.3f}")
        ax1.set_xlabel("Decision threshold τ")
        ax1.set_ylabel("Monthly savings ($)", color=C_TEAL)
        ax1.tick_params(axis="y", labelcolor=C_TEAL)
        ax1.yaxis.set_major_formatter(fmt_dollar)
        ax2.plot(sweep["threshold"], sweep["recall"],
                 color=C_BLUE,  lw=1.5, ls="--", label="Recall")
        ax2.plot(sweep["threshold"], sweep["precision"],
                 color=C_AMBER, lw=1.5, ls="--", label="Precision")
        ax2.set_ylabel("Precision / Recall", color="#334155")
        ax2.tick_params(axis="y", labelcolor="#334155")
        ax2.set_ylim(0, 1.05)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
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
            bdf = biz_cohort_df.reset_index(drop=True)
            x   = np.arange(len(bdf))
            retrain_mask = [
                bool(report_df.loc[report_df["cohort"] == c,
                                   "retrain_triggered"].values[0])
                for c in bdf["cohort"]
            ]
            bar_c = [C_ROSE if r else C_TEAL for r in retrain_mask]
            fig, ax = plt.subplots(figsize=(5, 4))
            bars = ax.bar(x, bdf["monthly_savings"], color=bar_c,
                          width=0.5, edgecolor="white")
            ax.axhline(bdf["monthly_savings"].iloc[0], ls="--",
                       color=C_SLATE, lw=1.3, label="Reference")
            ax.set_xticks(x)
            ax.set_xticklabels([f"C{int(c)}" for c in bdf["cohort"]])
            ax.set_ylabel("Monthly savings ($)")
            ax.yaxis.set_major_formatter(fmt_dollar)
            for bar, val in zip(bars, bdf["monthly_savings"]):
                if bar.get_height() > 0:
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + max(abs(val)*0.02, 30),
                            f"${val:,.0f}", ha="center", fontsize=8.5)
            ax.legend(handles=[
                mpatches.Patch(color=C_TEAL, label="Stable"),
                mpatches.Patch(color=C_ROSE, label="Retrain"),
            ], fontsize=9)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # ── Campaign budget simulator ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎯 Campaign Budget Simulator")
    st.caption(
        "Set a monthly offer budget. The simulator finds how many customers "
        "to contact to maximise churners caught within that budget — the "
        "efficient frontier between recall and cost."
    )

    budget_col, frontier_col = st.columns([1, 3])

    with budget_col:
        budget = st.number_input(
            "Monthly offer budget ($)",
            min_value=int(offer_cost * 10),
            max_value=int(monthly_at_risk * offer_cost),
            value=min(50_000, int(monthly_at_risk * offer_cost // 2)),
            step=1_000,
        )
        max_offers     = int(budget // offer_cost)
        total_churners = int(monthly_at_risk * test_m["y_true"].mean())

        st.metric("Max offers at this budget", f"{max_offers:,}")
        st.metric("Expected churners/month",   f"{total_churners:,}")

    with frontier_col:
        # Compute efficient frontier: for each budget level, what's the
        # best recall achievable?
        y_true  = test_m["y_true"]
        y_score = test_m["y_score"]
        n_test  = len(y_true)

        budgets     = np.linspace(offer_cost * 50,
                                  monthly_at_risk * offer_cost, 80)
        recalls     = []
        precisions  = []
        savings_ef  = []
        n_contacts  = []

        sorted_idx  = np.argsort(y_score)[::-1]

        for b in budgets:
            max_o  = int(b // offer_cost)
            # Scale from test set to monthly_at_risk population
            scale  = monthly_at_risk / n_test
            n_contact = min(max_o, n_test)
            top_idx  = sorted_idx[:n_contact]
            tp       = y_true[top_idx].sum()
            total_p  = y_true.sum()
            rec      = tp / max(total_p, 1)
            prec     = tp / max(n_contact, 1)
            # Scaled savings
            caught   = int(tp * scale)
            wasted   = int((n_contact - tp) * scale)
            savings  = caught * ltv - (caught + wasted) * offer_cost
            recalls.append(rec)
            precisions.append(prec)
            savings_ef.append(max(savings, 0))
            n_contacts.append(int(n_contact * scale))

        # Find the current budget point
        curr_idx = int(np.argmin(np.abs(budgets - budget)))

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        # Efficient frontier: recall vs budget
        ax = axes[0]
        ax.plot(budgets / 1000, recalls, color=C_BLUE, lw=2.5)
        ax.axvline(budget / 1000, ls="--", color=C_ROSE, lw=1.8,
                   label=f"Your budget ${budget:,}")
        ax.scatter([budget / 1000], [recalls[curr_idx]],
                   color=C_ROSE, s=100, zorder=5)
        ax.annotate(f"Recall={recalls[curr_idx]:.1%}\n{n_contacts[curr_idx]:,} contacts",
                    xy=(budget / 1000, recalls[curr_idx]),
                    xytext=(10, -30), textcoords="offset points",
                    fontsize=9, color=C_ROSE,
                    arrowprops=dict(arrowstyle="->", color=C_ROSE))
        ax.set_xlabel("Monthly offer budget ($k)")
        ax.set_ylabel("Recall (churners caught)")
        ax.set_title("Efficient frontier — recall vs budget")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        ax.legend(fontsize=9)

        # Savings vs budget
        ax = axes[1]
        ax.fill_between(budgets / 1000, savings_ef, alpha=0.15, color=C_GREEN)
        ax.plot(budgets / 1000, savings_ef, color=C_GREEN, lw=2.5,
                label="Net monthly savings")
        ax.axvline(budget / 1000, ls="--", color=C_ROSE, lw=1.8,
                   label=f"Your budget ${budget:,}")
        ax.scatter([budget / 1000], [savings_ef[curr_idx]],
                   color=C_ROSE, s=100, zorder=5)
        ax.annotate(f"${savings_ef[curr_idx]:,.0f}/mo",
                    xy=(budget / 1000, savings_ef[curr_idx]),
                    xytext=(10, 10), textcoords="offset points",
                    fontsize=9, color=C_GREEN,
                    arrowprops=dict(arrowstyle="->", color=C_GREEN))
        ax.set_xlabel("Monthly offer budget ($k)")
        ax.set_ylabel("Net monthly savings ($)")
        ax.set_title("Net savings vs offer budget")
        ax.yaxis.set_major_formatter(fmt_dollar)
        ax.legend(fontsize=9)

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Summary at selected budget
        best_savings_idx = int(np.argmax(savings_ef))
        st.info(
            f"**At ${budget:,}/month budget:** contact **{n_contacts[curr_idx]:,}** customers "
            f"· catch **{recalls[curr_idx]:.1%}** of churners "
            f"· net savings **${savings_ef[curr_idx]:,.0f}/month** · "
            f"optimal budget is **${budgets[best_savings_idx]:,.0f}** "
            f"(${savings_ef[best_savings_idx]:,.0f}/month)"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CUSTOMER EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════

with tab_customers:
    st.markdown("### 🔍 High-Risk Customer Explorer")
    st.caption(
        "Top 200 highest-risk customers from the reference cohort, ranked by "
        "predicted churn probability. Select a row to see their personalised "
        "SHAP explanation — why the model flagged this customer."
    )

    ref_sorted = data["ref_sorted"]
    shap_top   = data["shap_top"]
    features   = data["features"]

    # Build display table
    display_feats = [f for f in [
        "churn_score", "months", "revenue", "mou", "eqpdays", "custcare",
        "retention_contact", "made_retention_call", "outbound_ratio",
        "revenue_per_mou", "drop_rate", "custcare_rate",
    ] if f in ref_sorted.columns]

    display_df = ref_sorted[display_feats].copy()
    display_df.index = range(1, len(display_df) + 1)
    display_df.insert(0, "rank", display_df.index)

    # Colour the score column
    def _score_colour(val):
        if isinstance(val, float):
            if val > 0.20:
                return "background-color:#fee2e2;color:#991b1b;font-weight:600"
            elif val > 0.12:
                return "background-color:#fef3c7;color:#92400e"
        return ""

    styled_customers = (
        display_df.style
        .applymap(_score_colour, subset=["churn_score"])
        .format({
            "churn_score":  "{:.3f}",
            "revenue":      "${:.0f}",
            "mou":          "{:.0f}",
            "eqpdays":      "{:.0f}",
            "custcare":     "{:.0f}",
            "revenue_per_mou": "{:.3f}",
            "drop_rate":    "{:.3f}",
            "custcare_rate":"{:.3f}",
            "outbound_ratio":"{:.2f}",
        }, na_rep="—")
    )
    st.dataframe(styled_customers, use_container_width=True, height=320)

    # ── Per-customer SHAP waterfall ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🎯 Customer-Level Explanation")

    cust_col1, cust_col2 = st.columns([1, 3])

    with cust_col1:
        selected_rank = st.number_input(
            "Select customer rank (1 = highest risk)",
            min_value=1, max_value=len(ref_sorted), value=1, step=1,
        )
        idx = selected_rank - 1
        cust_row = ref_sorted.iloc[idx]
        score    = cust_row["churn_score"]

        risk_level = ("🔴 High" if score > 0.20 else
                      "🟡 Medium" if score > 0.12 else "🟢 Low")

        st.markdown(f"""
        <div class="customer-card">
        <b>Rank #{selected_rank}</b><br>
        <b>Risk:</b> {risk_level}<br>
        <b>Churn score:</b> {score:.3f}<br>
        <b>Tenure:</b> {cust_row.get('months', 'N/A'):.0f} months<br>
        <b>Revenue:</b> ${cust_row.get('revenue', 0):.0f}<br>
        <b>MOU:</b> {cust_row.get('mou', 0):.0f} mins<br>
        <b>Equipment age:</b> {cust_row.get('eqpdays', 0):.0f} days<br>
        <b>Custcare calls:</b> {cust_row.get('custcare', 0):.0f}<br>
        <b>Retention contact:</b> {'Yes' if cust_row.get('retention_contact', 0) else 'No'}
        </div>
        """, unsafe_allow_html=True)

    with cust_col2:
        st.markdown("**SHAP waterfall — why this customer is flagged**")
        try:
            import shap
            sv = data["explainer"](
                ref_sorted[features].fillna(0).iloc[[idx]]
            )
            fig, ax = plt.subplots(figsize=(9, 5))
            shap.plots.waterfall(sv[0], max_display=12, show=False)
            plt.title(
                f"Rank #{selected_rank} — churn score {score:.3f} "
                f"({risk_level})",
                fontsize=11
            )
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as e:
            st.warning(f"SHAP waterfall unavailable: {e}")

    # ── Feature comparison: this customer vs population ────────────────────────
    st.markdown("#### 📊 This Customer vs Population Percentiles")
    comparison_feats = [f for f in [
        "months", "revenue", "mou", "eqpdays", "custcare",
        "revenue_per_mou", "drop_rate", "custcare_rate",
    ] if f in ref_sorted.columns]

    pop_data   = data["scored_cohorts"][0]
    rows_cmp   = []
    for feat in comparison_feats:
        if feat not in pop_data.columns:
            continue
        val      = float(cust_row.get(feat, np.nan))
        pop_vals = pop_data[feat].dropna()
        pct      = float((pop_vals <= val).mean() * 100)
        rows_cmp.append({
            "Feature":       feat,
            "Customer":      f"{val:.2f}",
            "Population median": f"{pop_vals.median():.2f}",
            "Percentile":    f"{pct:.0f}th",
            "Flag":          "⬆ High" if pct > 75 else ("⬇ Low" if pct < 25 else "✓ Normal"),
        })

    if rows_cmp:
        cmp_df = pd.DataFrame(rows_cmp)
        def _flag_color(val):
            if "High" in str(val):
                return "color:#991b1b;font-weight:600"
            elif "Low" in str(val):
                return "color:#065f46"
            return ""
        st.dataframe(
            cmp_df.style.applymap(_flag_color, subset=["Flag"]),
            use_container_width=True, hide_index=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SURVIVAL CURVES (personalised)
# ═══════════════════════════════════════════════════════════════════════════════

with tab_survival:
    st.markdown("### 📈 Personalised Survival Curve Explorer")

    if not load_surv or aft is None:
        st.info(
            "Enable **Load survival model** in the sidebar to use this tab. "
            "It trains a WeibullAFT model and lets you explore personalised "
            "survival curves for any hypothetical customer."
        )
    else:
        st.caption(
            "Adjust the customer profile sliders to see how their predicted "
            "survival curve changes — P(still active) over time. "
            "Compare two profiles side by side."
        )

        from src.models.survival import predict_survival

        # Build feature defaults from population medians
        ref_pop   = data["scored_cohorts"][0]
        surv_cols = [f for f in surv_features if f in ref_pop.columns]

        def _median(col):
            return float(ref_pop[col].median()) if col in ref_pop.columns else 0.0

        col_sliders, col_curve = st.columns([1, 3])

        with col_sliders:
            st.markdown("**Customer A — profile**")
            months_a  = st.slider("Tenure (months)", 0, 72,
                                   int(_median("months")), key="ma")
            revenue_a = st.slider("Monthly revenue ($)", 0, 200,
                                   int(_median("revenue")), key="ra")
            mou_a     = st.slider("Minutes of use", 0, 2000,
                                   int(_median("mou")), key="moua")
            eqpdays_a = st.slider("Equipment age (days)", 0, 800,
                                   int(_median("eqpdays")), key="ea")
            custcare_a = st.slider("Customer care calls", 0, 20,
                                    int(_median("custcare")), key="ca")

            st.markdown("**Customer B — compare**")
            months_b  = st.slider("Tenure (months)", 0, 72,
                                   min(int(_median("months")) + 12, 72), key="mb")
            revenue_b = st.slider("Monthly revenue ($)", 0, 200,
                                   min(int(_median("revenue")) + 30, 200), key="rb")
            mou_b     = st.slider("Minutes of use", 0, 2000,
                                   max(int(_median("mou")) - 200, 0), key="moub")
            eqpdays_b = st.slider("Equipment age (days)", 0, 800,
                                   min(int(_median("eqpdays")) + 200, 800), key="eb")
            custcare_b = st.slider("Customer care calls", 0, 20,
                                    min(int(_median("custcare")) + 5, 20), key="cb")

        with col_curve:
            # Build synthetic customer rows
            def _make_row(months, revenue, mou, eqpdays, custcare):
                row = {c: _median(c) for c in surv_cols}
                row.update({
                    "months":          months,
                    "revenue":         revenue,
                    "mou":             mou,
                    "eqpdays":         eqpdays,
                    "custcare":        custcare,
                    "revenue_per_mou": revenue / (mou + 1),
                    "custcare_rate":   custcare / (months + 1),
                    "eqp_age_ratio":   eqpdays / (months * 30 + 1),
                    "days_to_churn":   365,
                    "event_observed":  1,
                })
                return pd.DataFrame([row])

            df_a = _make_row(months_a, revenue_a, mou_a, eqpdays_a, custcare_a)
            df_b = _make_row(months_b, revenue_b, mou_b, eqpdays_b, custcare_b)

            # Ensure columns match surv_features
            for c in surv_features:
                if c not in df_a.columns:
                    df_a[c] = 0.0
                    df_b[c] = 0.0

            df_a = df_a[surv_features + ["days_to_churn", "event_observed"]].fillna(0)
            df_b = df_b[surv_features + ["days_to_churn", "event_observed"]].fillna(0)
            df_a["days_to_churn"] = df_a["days_to_churn"].clip(lower=1)
            df_b["days_to_churn"] = df_b["days_to_churn"].clip(lower=1)

            try:
                times      = np.arange(1, 731)
                surv_a     = aft.predict_survival_function(df_a, times=times).mean(axis=1)
                surv_b     = aft.predict_survival_function(df_b, times=times).mean(axis=1)
                median_a   = float(aft.predict_median(df_a).iloc[0])
                median_b   = float(aft.predict_median(df_b).iloc[0])

                fig, ax = plt.subplots(figsize=(9, 5))
                ax.plot(times, surv_a, color=C_BLUE, lw=2.5, label="Customer A")
                ax.plot(times, surv_b, color=C_ROSE, lw=2.5, label="Customer B")
                ax.fill_between(times, surv_a, alpha=0.08, color=C_BLUE)
                ax.fill_between(times, surv_b, alpha=0.08, color=C_ROSE)

                # Median survival lines
                ax.axvline(median_a, ls="--", color=C_BLUE, lw=1.4, alpha=0.7)
                ax.axvline(median_b, ls="--", color=C_ROSE, lw=1.4, alpha=0.7)
                ax.text(median_a + 5, 0.52,
                        f"A median\n{median_a:.0f}d", fontsize=8,
                        color=C_BLUE)
                ax.text(median_b + 5, 0.42,
                        f"B median\n{median_b:.0f}d", fontsize=8,
                        color=C_ROSE)

                # Horizon reference lines
                horizons = [30, 60, 90, 180]
                for h in horizons:
                    ax.axvline(h, ls=":", color=C_SLATE, lw=1, alpha=0.5)
                    ax.text(h + 3, 0.98, f"{h}d", fontsize=8, color=C_SLATE)

                ax.set_xlabel("Days from observation")
                ax.set_ylabel("P(still active) = S(t)")
                ax.set_title("Personalised survival curves — Customer A vs B")
                ax.set_ylim(0, 1.02)
                ax.legend(fontsize=11)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                # Churn probabilities at each horizon
                st.markdown("#### Predicted churn probability at each horizon")
                h_rows = []
                for h in [30, 60, 90, 180]:
                    t_idx  = np.argmin(np.abs(times - h))
                    p_a    = 1 - float(surv_a.iloc[t_idx])
                    p_b    = 1 - float(surv_b.iloc[t_idx])
                    h_rows.append({
                        "Horizon": f"{h} days",
                        "Customer A": f"{p_a:.1%}",
                        "Customer B": f"{p_b:.1%}",
                        "Difference": f"{p_b - p_a:+.1%}",
                    })
                st.dataframe(pd.DataFrame(h_rows),
                             use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Survival prediction failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════

with tab_model:
    st.markdown("### Model Performance — Test Set")

    col_roc, col_cal = st.columns(2)

    with col_roc:
        st.markdown("#### ROC Curve")
        from sklearn.metrics import roc_curve
        fpr, tpr        = test_m["fpr"], test_m["tpr"]
        b_fpr, b_tpr, _ = roc_curve(base_m["y_true"], base_m["y_score"])
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.plot(fpr, tpr, color=C_BLUE, lw=2.5,
                label=f"XGBoost (AUROC={test_m['auroc']:.3f})")
        ax.plot(b_fpr, b_tpr, color=C_SLATE, lw=1.8, ls="--",
                label=f"Logistic (AUROC={base_m['auroc']:.3f})")
        ax.plot([0,1],[0,1],":",lw=1,color=C_SLATE,alpha=0.5)
        ax.fill_between(fpr, tpr, alpha=0.08, color=C_BLUE)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=9)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_cal:
        st.markdown("#### Calibration — Platt vs Raw")
        from sklearn.calibration import calibration_curve
        X_t = data["test_df"][data["features"]].fillna(0)
        y_t = data["test_df"][f"churn_{horizon}d"]
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        for mdl, label, color, ls in [
            (data["model"],      f"XGB raw",        C_ROSE, "--"),
            (data["calibrated"], f"XGB Platt",       C_BLUE, "-"),
        ]:
            prob = mdl.predict_proba(X_t)[:, 1]
            frac, mean_pred = calibration_curve(y_t, prob, n_bins=10)
            ax.plot(mean_pred, frac, marker="o", ms=4, lw=2,
                    color=color, ls=ls, label=label)
        ax.plot([0,1],[0,1],":",lw=1,color=C_SLATE,alpha=0.5,
                label="Perfect")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.legend(fontsize=9)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    col_pr, col_shap = st.columns(2)

    with col_pr:
        st.markdown("#### Precision-Recall Curve")
        from sklearn.metrics import precision_recall_curve
        prec, rec        = test_m["precision"], test_m["recall"]
        b_prec, b_rec, _ = precision_recall_curve(base_m["y_true"], base_m["y_score"])
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.plot(rec, prec, color=C_VIOLET, lw=2.5,
                label=f"XGBoost (AUPRC={test_m['auprc']:.3f})")
        ax.plot(b_rec, b_prec, color=C_SLATE, lw=1.8, ls="--",
                label=f"Logistic (AUPRC={base_m['auprc']:.3f})")
        base_rate = test_m["y_true"].mean()
        ax.axhline(base_rate, ls=":", color=C_SLATE, lw=1, alpha=0.6,
                   label=f"Random ({base_rate:.3f})")
        ax.fill_between(rec, prec, alpha=0.08, color=C_VIOLET)
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.legend(fontsize=9)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_shap:
        st.markdown("#### SHAP Feature Importance")
        try:
            shap_vals = data["shap_values"]
            feat_imp  = np.abs(shap_vals).mean(axis=0)
            top_idx   = np.argsort(feat_imp)[-15:]
            top_feats = [data["features"][i] for i in top_idx]
            top_vals  = feat_imp[top_idx]
            fig, ax   = plt.subplots(figsize=(5.5, 4.5))
            ax.barh(top_feats, top_vals, color=C_VIOLET,
                    edgecolor="white", linewidth=0.6)
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title(f"Top 15 — {horizon}d horizon")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as e:
            st.warning(f"SHAP unavailable: {e}")

    # Horizon churn rates
    st.markdown("#### Horizon Churn Rates (Weibull Event Log)")
    full_df = pd.concat([data["train_df"], data["test_df"]], ignore_index=True)
    horizons_list = [30, 60, 90, 180]
    rates = [full_df[f"churn_{h}d"].mean()
             for h in horizons_list if f"churn_{h}d" in full_df]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    bar_colors_h = [C_TEAL, C_BLUE, C_AMBER, C_ROSE]
    bars = ax.bar([f"{h}d" for h in horizons_list[:len(rates)]], rates,
                  color=bar_colors_h[:len(rates)],
                  width=0.45, edgecolor="white", linewidth=0.8)
    if "churndep" in full_df.columns:
        ax.axhline(full_df["churndep"].mean(), ls="--", color=C_SLATE,
                   lw=1.3, label="Snapshot churn rate")
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.004, f"{rate:.1%}",
                ha="center", fontsize=10, fontweight="bold", color="#1e293b")
    ax.set_ylabel("Churn rate"); ax.legend(fontsize=9)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — FEATURE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_features:
    st.markdown("### Feature Analysis")

    col_eda1, col_eda2 = st.columns(2)
    full_df    = pd.concat([data["train_df"], data["test_df"]], ignore_index=True)
    target_col = f"churn_{horizon}d"

    with col_eda1:
        available = [f for f in [
            "mou", "revenue", "eqpdays", "custcare", "overage", "months",
            "revenue_per_mou", "custcare_rate", "retention_contact",
            "outbound_ratio", "unanswered_rate", "revenue_declining",
        ] if f in full_df.columns]
        feat_choice = st.selectbox("Distribution by churn status",
                                   options=available)
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        if feat_choice in full_df.columns and target_col in full_df.columns:
            for val, label, color in [(0,"No churn",C_BLUE),(1,"Churned",C_ROSE)]:
                s        = full_df.loc[full_df[target_col]==val, feat_choice].dropna()
                clip_val = s.quantile(0.99)
                ax.hist(s.clip(upper=clip_val), bins=40, alpha=0.55,
                        color=color, label=label, edgecolor="none")
        ax.set_xlabel(feat_choice); ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_eda2:
        st.markdown("**Feature correlation with churn**")
        num_feats = [f for f in data["features"]
                     if f in full_df.columns
                     and full_df[f].dtype in [np.float64, np.float32,
                                               np.int64, np.int32]
                     and target_col in full_df.columns][:40]
        if num_feats and target_col in full_df.columns:
            corrs         = (full_df[num_feats + [target_col]]
                             .corr()[target_col].drop(target_col))
            top_corr      = corrs.abs().nlargest(15)
            top_corr_vals = corrs[top_corr.index]
            fig, ax = plt.subplots(figsize=(5.5, 4.2))
            bc = [C_ROSE if v > 0 else C_BLUE for v in top_corr_vals.values]
            ax.barh(top_corr_vals.index, top_corr_vals.values,
                    color=bc, edgecolor="white", linewidth=0.6)
            ax.axvline(0, color=C_SLATE, lw=0.8)
            ax.set_xlabel(f"Pearson r with churn_{horizon}d")
            ax.set_title("Top 15 by |correlation|")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.markdown("#### Score Distribution by Cohort")
    palette = [C_TEAL, C_BLUE, C_VIOLET, C_AMBER, C_ROSE, C_SLATE]
    fig, ax = plt.subplots(figsize=(12, 3.8))
    for i, cohort in enumerate(data["scored_cohorts"]):
        if "churn_score" not in cohort.columns:
            continue
        is_retrain = bool(
            report_df.loc[report_df["cohort"]==i,
                          "retrain_triggered"].values[0]
        )
        color = C_ROSE if is_retrain else palette[i % len(palette)]
        ax.hist(cohort["churn_score"].clip(0, 1), bins=50, density=True,
                histtype="step", color=color,
                lw=2.5 if i == 0 else 1.5,
                alpha=0.9 if i == 0 else 0.65,
                label=f"C{i} (ref)" if i == 0 else f"C{i}")
    ax.set_xlabel("Churn score"); ax.set_ylabel("Density")
    ax.legend(fontsize=8.5, ncol=3)
    ax.set_title("Score distribution shift — red = retrain triggered")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)