"""
Real-time streaming simulation of production churn monitoring.

Invoked via:  python analyze.py --stream
Or directly:  python -m src.monitoring.stream

Scores one cohort at a time with a configurable delay, printing live
monitoring updates to the terminal with colour-coded status banners.
Makes the "production monitoring" story visceral — you can watch the
PSI creep up cohort by cohort and the retrain alert fire in real time.

Terminal output mimics what a deployed monitoring job would log to
stdout / a log aggregator like Datadog or CloudWatch.
"""

from __future__ import annotations

import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ANSI colour codes — degrade gracefully on non-TTY
_IS_TTY = sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _IS_TTY else text


def green(t):  return _c(t, "32")
def yellow(t): return _c(t, "33")
def red(t):    return _c(t, "31")
def bold(t):   return _c(t, "1")
def dim(t):    return _c(t, "2")
def cyan(t):   return _c(t, "36")


# ── Streaming runner ───────────────────────────────────────────────────────────

def run_stream(
    model,
    scored_cohorts:  list[pd.DataFrame],
    report_df:       pd.DataFrame,
    features:        list[str],
    horizon:         int   = 90,
    delay:           float = 2.5,
    ltv:             float = 1_200,
    offer_cost:      float = 75,
    monthly_at_risk: int   = 5_000,
) -> None:
    """
    Stream cohort-by-cohort monitoring updates to stdout.

    Parameters
    ----------
    model           : fitted calibrated churn model
    scored_cohorts  : list of scored cohort DataFrames
    report_df       : pre-built monitoring report DataFrame
    features        : feature column names
    horizon         : prediction horizon in days
    delay           : seconds between cohort arrivals (simulates real time)
    ltv             : customer LTV for business impact ($)
    offer_cost      : retention offer cost ($)
    monthly_at_risk : customers scored per month for $ projections
    """
    from src.monitoring.drift import psi, psi_label, detect_feature_drift, KEY_DRIFT_FEATURES
    from src.business.business_metrics import BusinessImpactCalculator

    calc       = BusinessImpactCalculator(ltv, offer_cost, monthly_at_risk)
    target     = f"churn_{horizon}d"
    ref_scores = scored_cohorts[0]["churn_score"].values
    base_date  = datetime(2024, 1, 1)

    _print_stream_header(horizon, delay, ltv, offer_cost, monthly_at_risk)

    for i, cohort in enumerate(scored_cohorts):
        cohort_date = base_date + timedelta(days=30 * i)
        row         = report_df[report_df["cohort"] == i]

        _print_cohort_header(i, cohort_date, len(cohort))
        time.sleep(delay * 0.3)

        # Score distribution
        cur_scores    = cohort["churn_score"].values
        score_psi_val = psi(ref_scores, cur_scores)
        status        = psi_label(score_psi_val)
        _print_score_drift(score_psi_val, status, cur_scores)
        time.sleep(delay * 0.2)

        # Feature drift
        key_feats = [f for f in KEY_DRIFT_FEATURES if f in cohort.columns]
        drift_res = detect_feature_drift(scored_cohorts[0], cohort, key_feats)
        drifted   = [r for r in drift_res if r.drifted]
        _print_feature_drift(drift_res[:5], drifted)
        time.sleep(delay * 0.2)

        # Model performance
        auroc = row["auroc"].iloc[0] if len(row) else None
        _print_model_performance(auroc, horizon)
        time.sleep(delay * 0.1)

        # Business impact
        if target in cohort.columns and len(np.unique(cohort[target])) > 1:
            report = calc.full_report(cohort[target].values, cur_scores)
            _print_business_impact(report)
        time.sleep(delay * 0.1)

        # Retrain decision
        retrain = row["retrain_triggered"].iloc[0] if len(row) else False
        _print_retrain_decision(retrain, score_psi_val, auroc, drifted, i)

        if i < len(scored_cohorts) - 1:
            print()
            print(dim(f"  ── next cohort arriving in {delay:.0f}s "
                      f"(simulated +30 days) ──"))
            time.sleep(delay)
        else:
            print()
            _print_stream_footer(report_df)


# ── Print helpers ──────────────────────────────────────────────────────────────

def _print_stream_header(horizon, delay, ltv, offer_cost, monthly_at_risk):
    width = 65
    print()
    print(bold("═" * width))
    print(bold(f"  🔴 LIVE  Cell2Cell Churn Monitor  [{horizon}d horizon]"))
    print(bold("═" * width))
    print(dim(f"  Simulating production scoring — 1 cohort = 30 days"))
    print(dim(f"  Delay between cohorts : {delay:.1f}s"))
    print(dim(f"  LTV=${ltv:,}  OfferCost=${offer_cost:,}  "
              f"AtRisk={monthly_at_risk:,}/mo"))
    print(bold("─" * width))


def _print_cohort_header(idx: int, date: datetime, n: int):
    ts = datetime.now().strftime("%H:%M:%S")
    print()
    print(bold(cyan(f"  [{ts}]  COHORT {idx}  —  {date.strftime('%b %Y')}  "
                    f"({n:,} customers)")))
    print(dim("  " + "─" * 55))


def _print_score_drift(score_psi: float, status: str, scores: np.ndarray):
    bar   = _psi_bar(score_psi)
    label = {"stable": green("✓ STABLE"), "warning": yellow("⚠ WARNING"),
             "retrain": red("✗ RETRAIN")}[status]
    print(f"  Score PSI   {bar}  {score_psi:.4f}  [{label}]")
    print(dim(f"  Score dist  mean={scores.mean():.3f}  "
              f"p50={np.median(scores):.3f}  "
              f"p90={np.percentile(scores, 90):.3f}"))


def _print_feature_drift(top5, drifted):
    if not top5:
        return
    print(f"  Feature drift  (top 5 by PSI):")
    for r in top5:
        bar    = _psi_bar(r.psi, width=12)
        marker = red(" ← DRIFTED") if r.drifted else ""
        print(f"    {r.feature:<28} PSI={r.psi:.4f}{marker}")


def _print_model_performance(auroc: float | None, horizon: int):
    if auroc is None:
        print(dim(f"  AUROC       n/a"))
        return
    color = red if auroc < 0.70 else (yellow if auroc < 0.75 else green)
    bar   = _auroc_bar(auroc)
    print(f"  AUROC {horizon}d  {bar}  {color(f'{auroc:.4f}')}")


def _print_business_impact(report):
    opt = report.optimal
    savings_str = f"${opt.monthly_savings:,.0f}/mo"
    color = green if opt.monthly_savings > 0 else red
    print(f"  Biz impact  threshold={opt.threshold:.2f}  "
          f"recall={opt.recall:.0%}  "
          f"savings={color(savings_str)}")
    print(dim(f"              caught={opt.churners_caught:,}  "
              f"wasted_offers={opt.wasted_offers:,}"))


def _print_retrain_decision(retrain, score_psi, auroc, drifted, cohort_id):
    print()
    if retrain:
        reasons = []
        if score_psi > 0.20:
            reasons.append(f"Score PSI={score_psi:.3f}>0.20")
        if auroc is not None and auroc < 0.70:
            reasons.append(f"AUROC={auroc:.3f}<0.70")
        if drifted:
            reasons.append(f"{len(drifted)} feature(s) drifted")
        reason_str = " | ".join(reasons)
        print(red(f"  ⚠  RETRAIN TRIGGERED — cohort {cohort_id}"))
        print(red(f"     Reason: {reason_str}"))
        print(red( "     Action: queue retraining job, shadow-deploy challenger"))
    else:
        print(green(f"  ✓  Model stable — no action required"))


def _print_stream_footer(report_df: pd.DataFrame):
    width = 65
    n_retrain  = report_df["retrain_triggered"].sum()
    first_fire = report_df[report_df["retrain_triggered"]]["cohort"].min()

    print(bold("═" * width))
    print(bold("  SIMULATION COMPLETE"))
    print(bold("─" * width))
    print(f"  Cohorts monitored : {len(report_df)}")
    print(f"  Retrain triggers  : {red(str(n_retrain)) if n_retrain else green('0')}")
    if n_retrain:
        print(f"  First trigger     : cohort {int(first_fire)}")
    print(bold("═" * width))


def _psi_bar(value: float, width: int = 16) -> str:
    """ASCII progress bar for PSI value (max display = 0.40)."""
    filled = int(min(value / 0.40, 1.0) * width)
    bar    = "█" * filled + "░" * (width - filled)
    if value > 0.20:
        return red(f"[{bar}]")
    elif value > 0.10:
        return yellow(f"[{bar}]")
    return green(f"[{bar}]")


def _auroc_bar(value: float, width: int = 16) -> str:
    """ASCII bar for AUROC (range 0.5–1.0)."""
    filled = int(((value - 0.5) / 0.5) * width)
    filled = max(0, min(filled, width))
    bar    = "█" * filled + "░" * (width - filled)
    if value < 0.70:
        return red(f"[{bar}]")
    elif value < 0.75:
        return yellow(f"[{bar}]")
    return green(f"[{bar}]")
