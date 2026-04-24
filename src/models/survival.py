"""
Survival analysis model using lifelines WeibullAFTFitter.

Why survival analysis for churn?
---------------------------------
The XGB horizon classifier answers: "will this customer churn within N days?"
The survival model answers:         "what is the full probability curve of
                                     survival over any time horizon?"

Key advantage: a single fitted survival model can produce churn probability
at 30d, 60d, 90d, 180d — or any other horizon — without retraining.

Censoring is handled correctly: customers where churndep=0 have not churned
yet at observation time. They are *censored* — we know they survived at least
as long as their tenure, but not when/if they will churn. Ignoring censoring
(treating them as "churn=never") biases the model toward underestimating risk.
The Weibull AFT fitter handles this natively via the event_col parameter.

Output
------
- Median survival time per customer
- Survival probability at each horizon
- Hazard ratio interpretation via coefficients
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

HORIZONS = [30, 60, 90, 180]


def train_survival(
    train_df:    pd.DataFrame,
    features:    list[str],
    duration_col: str = "days_to_churn",
    event_col:    str = "event_observed",
    penalizer:    float = 0.1,
) -> tuple:
    """
    Fit a Weibull Accelerated Failure Time (AFT) model.

    The AFT model parameterises ln(T) = X*beta + sigma*epsilon where
    T is time-to-churn. Covariates accelerate or decelerate the event.

    Parameters
    ----------
    train_df     : preprocessed DataFrame with days_to_churn + event_observed
    features     : feature columns to use as covariates
    duration_col : time-to-event column
    event_col    : 1 if event observed, 0 if censored
    penalizer    : L2 regularisation strength

    Returns
    -------
    aft          : fitted WeibullAFTFitter
    train_metrics: dict of concordance index + AIC
    """
    try:
        from lifelines import WeibullAFTFitter
        from lifelines.utils import concordance_index
    except ImportError:
        raise ImportError("lifelines is required: pip install lifelines")

    # Use a subset of features — survival models are more sensitive to
    # collinearity than tree models. Use key behavioral features only.
    survival_features = _select_survival_features(train_df, features)

    df_fit = train_df[survival_features + [duration_col, event_col]].copy()
    df_fit = df_fit.fillna(0)

    # Clip duration to positive values (lifelines requirement)
    df_fit[duration_col] = df_fit[duration_col].clip(lower=1)

    print(f"  Fitting WeibullAFT on {len(df_fit):,} rows, "
          f"{len(survival_features)} features …")

    aft = WeibullAFTFitter(penalizer=penalizer)
    aft.fit(
        df_fit,
        duration_col=duration_col,
        event_col=event_col,
    )

    # Concordance index on training data
    median_survival = aft.predict_median(df_fit)
    ci = concordance_index(
        df_fit[duration_col],
        median_survival,
        df_fit[event_col],
    )

    metrics = {
        "concordance_index": float(ci),
        "aic":               float(aft.AIC_),
        "n_train":           len(df_fit),
        "n_features":        len(survival_features),
    }

    print(f"  Concordance index: {ci:.4f}  (0.5=random, 1.0=perfect)")
    print(f"  AIC: {aft.AIC_:.1f}")

    return aft, survival_features, metrics


def predict_survival(
    aft,
    df:           pd.DataFrame,
    features:     list[str],
    horizons:     list[int] = None,
    duration_col: str = "days_to_churn",
    event_col:    str = "event_observed",
) -> pd.DataFrame:
    """
    Predict churn probability at each horizon for every customer.

    Returns DataFrame with columns:
      churn_prob_Xd  : P(churn within X days) = 1 - S(X)
      median_survival: expected days until churn
    """
    if horizons is None:
        horizons = HORIZONS

    survival_features = features
    df_pred = df[survival_features + [duration_col, event_col]].fillna(0).copy()
    df_pred[duration_col] = df_pred[duration_col].clip(lower=1)

    # Survival function: S(t) = P(T > t)
    surv_fn = aft.predict_survival_function(df_pred)  # shape: (time_points, n_customers)

    results = pd.DataFrame(index=df.index)

    for h in horizons:
        # Find closest time point in survival function index
        times   = surv_fn.index.values
        closest = times[np.argmin(np.abs(times - h))]
        s_at_h  = surv_fn.loc[closest].values          # S(h) per customer
        results[f"churn_prob_{h}d"] = 1 - s_at_h       # P(churn by h)

    # Median survival time
    results["median_survival_days"] = aft.predict_median(df_pred).values

    return results


def evaluate_survival(
    aft,
    test_df:      pd.DataFrame,
    features:     list[str],
    duration_col: str = "days_to_churn",
    event_col:    str = "event_observed",
    horizons:     list[int] = None,
) -> dict:
    """
    Evaluate survival model on test set.

    Returns concordance index + AUROC at each horizon (comparing
    predicted churn probability vs actual horizon label).
    """
    from lifelines.utils import concordance_index
    from sklearn.metrics import roc_auc_score

    if horizons is None:
        horizons = HORIZONS

    df_test = test_df[features + [duration_col, event_col]].fillna(0).copy()
    df_test[duration_col] = df_test[duration_col].clip(lower=1)

    median_survival = aft.predict_median(df_test)
    ci = concordance_index(
        df_test[duration_col], median_survival, df_test[event_col]
    )

    preds   = predict_survival(aft, test_df, features, horizons,
                               duration_col, event_col)
    metrics = {"concordance_index": float(ci)}

    for h in horizons:
        label_col = f"churn_{h}d"
        pred_col  = f"churn_prob_{h}d"
        if label_col in test_df.columns and pred_col in preds.columns:
            try:
                auroc = roc_auc_score(test_df[label_col], preds[pred_col])
                metrics[f"auroc_{h}d"] = float(auroc)
            except Exception:
                pass

    return metrics, preds


def _select_survival_features(
    df: pd.DataFrame,
    all_features: list[str],
    max_features: int = 20,
) -> list[str]:
    """
    Select a focused subset of features for the survival model.
    Survival models are more sensitive to collinearity and scale than
    tree models — use interpretable, lower-collinearity features.
    """
    preferred = [
        "months", "revenue", "mou", "eqpdays", "custcare",
        "overage", "hnd_price", "changem", "changer",
        "dropvce", "custcare_rate", "revenue_per_mou",
        "eqp_age_ratio", "mou_trend", "overage_rate",
        "drop_rate", "age1", "lor", "income",
        "phones", "uniqsubs",
    ]
    selected = [f for f in preferred if f in all_features]

    # Fill up to max_features with remaining numeric features if needed
    if len(selected) < max_features:
        remaining = [f for f in all_features
                     if f not in selected
                     and df[f].dtype in [np.float64, np.float32,
                                          np.int64, np.int32]
                     and not f.startswith("churn_")
                     and f not in ("days_to_churn", "event_observed")]
        selected += remaining[:max_features - len(selected)]

    return selected[:max_features]
