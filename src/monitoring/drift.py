"""
Model and data drift monitoring.

Implements:
  - Population Stability Index (PSI) for score and feature drift
  - Kolmogorov-Smirnov test for numeric feature drift
  - Chi-squared test for categorical feature drift
  - AUROC degradation tracking across cohorts
  - Retraining trigger logic
"""

import warnings
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, field
from typing import Optional

warnings.filterwarnings("ignore")

HORIZONS = [30, 60, 90, 180]

PSI_THRESHOLDS = {
    "stable":  0.10,
    "warning": 0.20,
}


# ── PSI ────────────────────────────────────────────────────────────────────────

def psi(
    expected: np.ndarray,
    actual:   np.ndarray,
    n_bins:   int   = 10,
    eps:      float = 1e-6,
) -> float:
    """
    Population Stability Index.
    PSI = Σ (actual% - expected%) * ln(actual% / expected%)

    Bin edges are quantiles of the *expected* (reference) distribution,
    so bins are equally populated in the reference and any shift in the
    actual distribution is detected.
    """
    breakpoints = np.linspace(0, 1, n_bins + 1)
    bin_edges   = np.quantile(expected, breakpoints)
    bin_edges[0]  -= 1e-8
    bin_edges[-1] += 1e-8

    exp_counts = np.histogram(expected, bins=bin_edges)[0]
    act_counts = np.histogram(actual,   bins=bin_edges)[0]

    exp_pct = (exp_counts / max(len(expected), 1)).clip(eps)
    act_pct = (act_counts / max(len(actual),   1)).clip(eps)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def psi_label(value: float) -> str:
    if value < PSI_THRESHOLDS["stable"]:
        return "stable"
    elif value < PSI_THRESHOLDS["warning"]:
        return "warning"
    return "retrain"


# ── Feature drift ──────────────────────────────────────────────────────────────

@dataclass
class FeatureDriftResult:
    feature:   str
    test:      str
    statistic: float
    p_value:   float
    psi:       float
    drifted:   bool


def detect_feature_drift(
    reference: pd.DataFrame,
    current:   pd.DataFrame,
    features:  list[str],
    alpha:     float = 0.05,
) -> list[FeatureDriftResult]:
    results = []
    for feat in features:
        if feat not in reference.columns or feat not in current.columns:
            continue
        ref_col = reference[feat].fillna(0)
        cur_col = current[feat].fillna(0)
        n_unique = ref_col.nunique()

        if n_unique > 10:
            stat, pval = stats.ks_2samp(ref_col, cur_col)
            feat_psi   = psi(ref_col.values, cur_col.values)
            test       = "ks"
        else:
            cats       = sorted(set(ref_col.unique()) | set(cur_col.unique()))
            ref_counts = ref_col.value_counts().reindex(cats, fill_value=0)
            cur_counts = cur_col.value_counts().reindex(cats, fill_value=0)
            stat, pval = stats.chi2_contingency(
                np.array([ref_counts, cur_counts])
            )[:2]
            feat_psi   = psi(
                np.repeat(cats, ref_counts.values.astype(int)),
                np.repeat(cats, cur_counts.values.astype(int)),
            ) if ref_counts.sum() > 0 and cur_counts.sum() > 0 else 0.0
            test = "chi2"

        drifted = (pval < alpha) and (feat_psi > PSI_THRESHOLDS["stable"])
        results.append(FeatureDriftResult(
            feature=feat, test=test,
            statistic=float(stat), p_value=float(pval),
            psi=float(feat_psi), drifted=drifted,
        ))

    return sorted(results, key=lambda r: -r.psi)


# ── Cohort reports ─────────────────────────────────────────────────────────────

@dataclass
class CohortReport:
    cohort_id:         int
    n:                 int
    churn_rate:        float
    mean_score:        float
    score_psi:         float
    score_status:      str
    auroc:             Optional[float]
    auprc:             Optional[float]
    brier:             Optional[float]
    drifted_features:  list[str] = field(default_factory=list)
    retrain_triggered: bool = False


# Key features to monitor for drift — the richest Cell2Cell signals
KEY_DRIFT_FEATURES = [
    "revenue", "mou", "eqpdays", "custcare", "overage",
    "revenue_per_mou", "custcare_rate", "eqp_age_ratio",
    "drop_rate", "overage_rate",
]


def build_cohort_reports(
    reference_cohort: pd.DataFrame,
    scored_cohorts:   list[pd.DataFrame],
    features:         list[str],
    auroc_threshold:  float = 0.70,
    horizon:          int   = 90,
) -> list[CohortReport]:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, brier_score_loss
    )

    target      = f"churn_{horizon}d"
    ref_scores  = reference_cohort["churn_score"].values
    reports     = []

    key_features = [f for f in KEY_DRIFT_FEATURES if f in reference_cohort.columns]

    for cohort in scored_cohorts:
        cur_scores    = cohort["churn_score"].values
        score_psi_val = psi(ref_scores, cur_scores)

        drift_results = detect_feature_drift(
            reference_cohort, cohort, key_features
        )
        drifted = [r.feature for r in drift_results if r.drifted]

        y = cohort.get(target, None)
        if y is not None and y.nunique() > 1:
            try:
                auroc = float(roc_auc_score(y, cur_scores))
                auprc = float(average_precision_score(y, cur_scores))
                brier = float(brier_score_loss(y, cur_scores))
            except Exception:
                auroc = auprc = brier = None
        else:
            auroc = auprc = brier = None

        drift_fraction = len(drifted) / max(len(key_features), 1)
        retrain = (
            score_psi_val > PSI_THRESHOLDS["warning"]
            or (auroc is not None and auroc < auroc_threshold)
            or drift_fraction > 0.20
        )

        reports.append(CohortReport(
            cohort_id=int(cohort["cohort"].iloc[0]),
            n=len(cohort),
            churn_rate=float(cohort.get(target, pd.Series([0])).mean()),
            mean_score=float(cur_scores.mean()),
            score_psi=score_psi_val,
            score_status=psi_label(score_psi_val),
            auroc=auroc,
            auprc=auprc,
            brier=brier,
            drifted_features=drifted,
            retrain_triggered=retrain,
        ))

    return reports


def reports_to_dataframe(reports: list[CohortReport]) -> pd.DataFrame:
    rows = []
    for r in reports:
        rows.append({
            "cohort":             r.cohort_id,
            "n":                  r.n,
            "churn_rate":         r.churn_rate,
            "mean_score":         r.mean_score,
            "score_psi":          r.score_psi,
            "score_status":       r.score_status,
            "auroc":              r.auroc,
            "auprc":              r.auprc,
            "brier":              r.brier,
            "n_drifted_features": len(r.drifted_features),
            "drifted_features":   ", ".join(r.drifted_features),
            "retrain_triggered":  r.retrain_triggered,
        })
    return pd.DataFrame(rows)
