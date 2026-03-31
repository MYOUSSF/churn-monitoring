"""
Data loader for Telco Customer Churn dataset.

Downloads IBM Watson Telco churn data, engineers features, and creates
temporal cohorts that simulate production drift for monitoring experiments.
"""

import os
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_PATH = DATA_DIR / "telco_churn_raw.csv"

TELCO_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)


# ── Feature groups ────────────────────────────────────────────────────────────

NUMERIC_COLS = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "num_services", "charges_per_month_tenure",
]

BINARY_COLS = [
    "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "PaperlessBilling",
    "gender_Male",
]

CATEGORICAL_COLS = [
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod",
]

TARGET_COL = "Churn"


def download_data(force: bool = False) -> pd.DataFrame:
    """Download raw Telco Churn CSV (once) and return as DataFrame."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_PATH.exists() and not force:
        return pd.read_csv(RAW_PATH)

    try:
        import urllib.request
        print(f"Downloading Telco Churn data from GitHub …")
        urllib.request.urlretrieve(TELCO_URL, RAW_PATH)
        print(f"  Saved to {RAW_PATH}")
        return pd.read_csv(RAW_PATH)
    except Exception as e:
        print(f"  Download failed ({e}). Generating synthetic fallback …")
        return _generate_synthetic(n=7043, seed=42)


def _generate_synthetic(n: int = 7043, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic replica of IBM Telco schema + marginal distributions.
    Used as fallback when the remote CSV is unavailable.
    """
    rng = np.random.default_rng(seed)

    tenure = rng.integers(0, 72, size=n)
    monthly = rng.uniform(18, 118, size=n).round(2)
    total = (monthly * tenure + rng.normal(0, 10, n)).clip(0).round(2)
    senior = rng.binomial(1, 0.16, n)
    churn_prob = (
        0.05
        + 0.35 * (tenure < 12)
        + 0.20 * (monthly > 70)
        + 0.10 * senior
        - 0.15 * (tenure > 48)
    ).clip(0, 1)
    churn = rng.binomial(1, churn_prob)

    yes_no = lambda p: rng.choice(["Yes", "No"], size=n, p=[p, 1 - p])
    df = pd.DataFrame({
        "customerID":       [f"SYN-{i:05d}" for i in range(n)],
        "gender":           rng.choice(["Male", "Female"], size=n),
        "SeniorCitizen":    senior,
        "Partner":          yes_no(0.48),
        "Dependents":       yes_no(0.30),
        "tenure":           tenure,
        "PhoneService":     yes_no(0.90),
        "MultipleLines":    rng.choice(["No", "Yes", "No phone service"], size=n, p=[0.42, 0.42, 0.16]),
        "InternetService":  rng.choice(["DSL", "Fiber optic", "No"], size=n, p=[0.34, 0.44, 0.22]),
        "OnlineSecurity":   rng.choice(["No", "Yes", "No internet service"], size=n, p=[0.50, 0.28, 0.22]),
        "OnlineBackup":     rng.choice(["No", "Yes", "No internet service"], size=n, p=[0.44, 0.34, 0.22]),
        "DeviceProtection": rng.choice(["No", "Yes", "No internet service"], size=n, p=[0.44, 0.34, 0.22]),
        "TechSupport":      rng.choice(["No", "Yes", "No internet service"], size=n, p=[0.50, 0.28, 0.22]),
        "StreamingTV":      rng.choice(["No", "Yes", "No internet service"], size=n, p=[0.40, 0.38, 0.22]),
        "StreamingMovies":  rng.choice(["No", "Yes", "No internet service"], size=n, p=[0.40, 0.38, 0.22]),
        "Contract":         rng.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.21, 0.24]),
        "PaperlessBilling": yes_no(0.59),
        "PaymentMethod":    rng.choice(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            size=n, p=[0.34, 0.23, 0.22, 0.21],
        ),
        "MonthlyCharges":   monthly,
        "TotalCharges":     total.astype(str),
        "Churn":            np.where(churn, "Yes", "No"),
    })
    df.to_csv(RAW_PATH, index=False)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean, encode, and engineer features."""
    df = df.copy()

    # TotalCharges is stored as string with blanks for tenure==0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Target
    df[TARGET_COL] = (df["Churn"] == "Yes").astype(int)

    # Binary
    df["gender_Male"] = (df["gender"] == "Male").astype(int)
    for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        df[col] = (df[col] == "Yes").astype(int)

    # Engineered
    df["num_services"] = (
        (df["MultipleLines"] == "Yes").astype(int)
        + (df["OnlineSecurity"] == "Yes").astype(int)
        + (df["OnlineBackup"] == "Yes").astype(int)
        + (df["DeviceProtection"] == "Yes").astype(int)
        + (df["TechSupport"] == "Yes").astype(int)
        + (df["StreamingTV"] == "Yes").astype(int)
        + (df["StreamingMovies"] == "Yes").astype(int)
    )
    df["charges_per_month_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)

    # One-hot categoricals
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)

    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return all feature columns (excludes ID and target)."""
    exclude = {"customerID", "gender", "Churn", TARGET_COL}
    return [c for c in df.columns if c not in exclude]


def make_temporal_cohorts(
    df: pd.DataFrame,
    n_cohorts: int = 6,
    drift_start: int = 3,
    seed: int = 42,
) -> list[pd.DataFrame]:
    """
    Simulate temporal production cohorts with injected feature drift.

    Cohorts 0..(drift_start-1): stable, drawn from original distribution.
    Cohorts drift_start..n_cohorts-1: drift injected —
      - MonthlyCharges shifts upward (price inflation)
      - tenure distribution becomes shorter (newer customer base)
      - num_services erodes slightly

    Returns list of DataFrames, one per cohort, each ~300 rows.
    """
    rng = np.random.default_rng(seed)
    cohort_size = max(200, len(df) // (n_cohorts * 2))
    cohorts = []

    for i in range(n_cohorts):
        idx = rng.choice(len(df), size=cohort_size, replace=False)
        chunk = df.iloc[idx].copy().reset_index(drop=True)

        if i >= drift_start:
            severity = (i - drift_start + 1) / (n_cohorts - drift_start)
            # MonthlyCharges: shift up by up to 30%
            chunk["MonthlyCharges"] = chunk["MonthlyCharges"] * (1 + 0.3 * severity)
            chunk["TotalCharges"]   = chunk["TotalCharges"]   * (1 + 0.1 * severity)
            chunk["charges_per_month_tenure"] = (
                chunk["MonthlyCharges"] / (chunk["tenure"] + 1)
            )
            # tenure: compress toward low-tenure customers
            chunk["tenure"] = (chunk["tenure"] * (1 - 0.4 * severity)).clip(0).astype(int)
            # num_services: slight reduction
            drop_mask = rng.random(len(chunk)) < 0.2 * severity
            chunk.loc[drop_mask, "num_services"] = (
                chunk.loc[drop_mask, "num_services"] - 1
            ).clip(lower=0)

        chunk["cohort"] = i
        cohorts.append(chunk)

    return cohorts


def load_pipeline(
    n_cohorts: int = 6,
    drift_start: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.DataFrame], list[str]]:
    """
    Full loading pipeline.

    Returns
    -------
    train_df : training set (cohort 0 only)
    test_df  : held-out test set from cohort 0
    cohorts  : list of all cohort DataFrames (including train)
    features : list of feature column names
    """
    raw = download_data()
    processed = preprocess(raw)
    cohorts = make_temporal_cohorts(processed, n_cohorts=n_cohorts, drift_start=drift_start)

    features = get_feature_cols(cohorts[0])

    # Train/test from cohort 0 (80/20)
    base = cohorts[0]
    split = int(0.8 * len(base))
    train_df = base.iloc[:split].copy()
    test_df  = base.iloc[split:].copy()

    return train_df, test_df, cohorts, features
