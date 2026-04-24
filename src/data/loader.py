"""
Data loader for Cell2Cell Telecom Churn dataset.

Cell2Cell has ~71k customers and 58 features including rich behavioral signals
(minutes of use, data calls, equipment age) not present in IBM Telco.

Pipeline
--------
1. Download Cell2Cell CSV (fallback: synthetic replica)
2. Preprocess: clean, encode, engineer features
3. Generate Weibull event log — attach days_to_churn to each customer
4. Build horizon labels: churn_30d, churn_60d, churn_90d, churn_180d
5. Simulate temporal cohorts with injected drift for monitoring
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_PATH = DATA_DIR / "cell2cell.csv"

# Kaggle dataset slug
KAGGLE_DATASET = "jpacse/datasets-for-churn-telecom"

TARGET = "churn"
HORIZONS = [30, 60, 90, 180]

# ── Feature groups ─────────────────────────────────────────────────────────────

NUMERIC_COLS = [
    "months",           # tenure
    "uniqsubs",         # unique subscribers on account
    "actvsubs",         # active subscribers
    "revenue",          # monthly revenue
    "mou",              # minutes of use
    "recchrge",         # recurring charges
    "directas",         # directory assistance calls
    "overage",          # overage charges
    "roam",             # roaming calls
    "changem",          # change in MOU month over month
    "changer",          # change in revenue month over month
    "dropvce",          # dropped voice calls
    "blckvce",          # blocked voice calls
    "unansvce",         # unanswered voice calls
    "custcare",         # customer care calls
    "threeway",         # three-way calls
    "mourec",           # MOU to/from customer care
    "outcalls",         # outbound calls
    "incalls",          # inbound calls
    "peakvce",          # peak voice calls
    "offpeakvce",       # off-peak voice calls
    "dropblk",          # dropped + blocked calls
    "callfwdv",         # call forwarding
    "callwait",         # call waiting
    "churn",            # raw churn column (use churndep as target)
    "eqpdays",          # equipment age in days
    "age1",             # age of primary account holder
    "age2",             # age of secondary account holder
    "phones",           # number of phones on account
    "models",           # number of handset models
    "hnd_price",        # handset price
    "asl_flag",         # account spending limit flag (numeric after encoding)
    "mailorder",        # mail order purchase flag
    "webcap",           # web capable handset flag
    "truck",            # truck flag (occupation proxy)
    "rv",               # rv flag (occupation proxy)
    "ownrent",          # own/rent flag
    "lor",              # length of residence
    "income",           # income band
    "numbcars",         # number of cars
    "forgntvl",         # foreign travel flag
]

CATEGORICAL_COLS = [
    "crclscod",         # credit class code
    "asl_flag",         # account spending limit
    "area",             # service area
    "refurb_new",       # refurbished vs new handset
    "hnd_webcap",       # web capability category
    "marital",          # marital status
    "infobase",         # infobase flag
    "prizm_social_one", # prizm social category
    "dualband",         # dual band capability
    "mtrcycle",         # motorcycle flag
    "mailres",          # mail response flag
    "mailord",          # mail order flag
    "travel",           # travel flag
    "pcown",            # PC ownership
    "creditcd",         # credit card flag
    "newcelly",         # new cell user (year)
    "newcelln",         # new cell user (n)
    "occup",            # occupation
    "hhstatin",         # household status
    "dwlltype",         # dwelling type
    "dwllsize",         # dwelling size
    "ethnic",           # ethnicity
    "kid0_2",           # kids 0-2
    "kid3_5",           # kids 3-5
    "kid6_10",          # kids 6-10
    "kid11_15",         # kids 11-15
    "kid16_17",         # kids 16-17
]


# ── Download ───────────────────────────────────────────────────────────────────

def download_data(force: bool = False) -> pd.DataFrame:
    """
    Try to load Cell2Cell from disk, then Kaggle, then fall back to synthetic.
    Checks both data/cell2celltrain.csv (user-provided) and data/cell2cell.csv.
    """
    # Check user-provided filename first
    alt = DATA_DIR / "cell2celltrain.csv"
    if alt.exists() and not force:
        print(f"  Loading Cell2Cell from {alt} …")
        return pd.read_csv(alt, low_memory=False)

    if RAW_PATH.exists() and not force:
        print(f"  Loading Cell2Cell from {RAW_PATH} …")
        return pd.read_csv(RAW_PATH, low_memory=False)

    # Try Kaggle download
    try:
        import kaggle
        print("  Downloading Cell2Cell from Kaggle …")
        kaggle.api.dataset_download_files(
            KAGGLE_DATASET,
            path=str(DATA_DIR),
            unzip=True,
        )
        # Find the CSV
        csvs = list(DATA_DIR.glob("*.csv"))
        cell2cell_csv = next(
            (f for f in csvs if "cell2cell" in f.name.lower()), csvs[0] if csvs else None
        )
        if cell2cell_csv and cell2cell_csv != RAW_PATH:
            cell2cell_csv.rename(RAW_PATH)
        if RAW_PATH.exists():
            return pd.read_csv(RAW_PATH, low_memory=False)
    except Exception as e:
        print(f"  Kaggle download failed ({e})")

    print("  Generating synthetic Cell2Cell replica …")
    df = _generate_synthetic(n=71047, seed=42)
    df.to_csv(RAW_PATH, index=False)
    return df


def _generate_synthetic(n: int = 71047, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic replica of Cell2Cell schema with realistic marginal distributions.
    Richer feature space than IBM Telco — usage behavior, equipment, demographics.
    """
    rng = np.random.default_rng(seed)

    months    = rng.integers(1, 72, size=n)
    revenue   = rng.normal(58, 22, n).clip(5, 200).round(2)
    mou       = rng.exponential(450, n).clip(0, 3000).round()
    eqpdays   = rng.integers(1, 800, size=n)
    age1      = rng.integers(18, 80, size=n)
    hnd_price = rng.choice([49, 99, 149, 199, 249, 299, 399], size=n,
                           p=[0.05, 0.15, 0.25, 0.25, 0.15, 0.10, 0.05])

    overage   = rng.exponential(8, n).clip(0, 100).round(2)
    custcare  = rng.poisson(1.5, n).clip(0, 20)
    dropvce   = rng.poisson(2, n).clip(0, 30)

    # Churn probability — non-linear, with interactions
    churn_logit = (
        -1.5
        + 1.2  * (months < 6).astype(float)
        + 0.8  * (months < 12).astype(float)
        - 0.6  * (months > 36).astype(float)
        + 0.7  * (revenue > 80).astype(float)
        + 0.5  * (mou < 100).astype(float)
        - 0.4  * (mou > 600).astype(float)
        + 0.6  * (eqpdays > 500).astype(float)
        + 0.4  * (custcare > 3).astype(float)
        + 0.3  * (dropvce > 5).astype(float)
        + 0.4  * (overage > 20).astype(float)
        # interaction: high-rev + low-usage is churn risk
        + 0.5  * ((revenue > 70) & (mou < 200)).astype(float)
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churndep   = rng.binomial(1, churn_prob)

    df = pd.DataFrame({
        "customerid":   [f"C2C-{i:06d}" for i in range(n)],
        "churndep":     churndep,
        "months":       months,
        "uniqsubs":     rng.integers(1, 5, size=n),
        "actvsubs":     rng.integers(1, 4, size=n),
        "revenue":      revenue,
        "mou":          mou,
        "recchrge":     (revenue * rng.uniform(0.7, 1.0, n)).round(2),
        "directas":     rng.poisson(0.3, n),
        "overage":      overage,
        "roam":         rng.exponential(2, n).clip(0).round(2),
        "changem":      rng.normal(0, 50, n).round(2),
        "changer":      rng.normal(0, 8, n).round(2),
        "dropvce":      dropvce,
        "blckvce":      rng.poisson(1, n),
        "unansvce":     rng.poisson(3, n),
        "custcare":     custcare,
        "threeway":     rng.poisson(0.5, n),
        "mourec":       rng.exponential(20, n).clip(0).round(),
        "outcalls":     rng.poisson(80, n),
        "incalls":      rng.poisson(70, n),
        "peakvce":      rng.poisson(60, n),
        "offpeakvce":   rng.poisson(40, n),
        "dropblk":      rng.poisson(3, n),
        "callfwdv":     rng.poisson(0.2, n),
        "callwait":     rng.poisson(1, n),
        "churn":        churndep,
        "eqpdays":      eqpdays,
        "age1":         age1,
        "age2":         rng.integers(18, 80, size=n),
        "phones":       rng.integers(1, 5, size=n),
        "models":       rng.integers(1, 4, size=n),
        "hnd_price":    hnd_price,
        "lor":          rng.integers(1, 30, size=n),
        "income":       rng.integers(1, 8, size=n),
        "numbcars":     rng.integers(0, 4, size=n),
        "forgntvl":     rng.binomial(1, 0.15, n),
        "mailorder":    rng.binomial(1, 0.30, n),
        "webcap":       rng.binomial(1, 0.55, n),
        "truck":        rng.binomial(1, 0.08, n),
        "rv":           rng.binomial(1, 0.06, n),
        "ownrent":      rng.binomial(1, 0.65, n),
        "crclscod":     rng.choice(["A", "B", "C", "D", "E"], size=n,
                                   p=[0.25, 0.30, 0.25, 0.15, 0.05]),
        "asl_flag":     rng.choice(["Y", "N"], size=n, p=[0.15, 0.85]),
        "area":         rng.choice(["NORTHWEST", "SOUTHWEST", "NORTHEAST",
                                    "SOUTHEAST", "MIDWEST"], size=n),
        "refurb_new":   rng.choice(["R", "N"], size=n, p=[0.20, 0.80]),
        "hnd_webcap":   rng.choice(["WC", "WCMB", "NWC"], size=n,
                                   p=[0.35, 0.20, 0.45]),
        "marital":      rng.choice(["S", "M", "U"], size=n,
                                   p=[0.35, 0.50, 0.15]),
        "creditcd":     rng.choice(["Y", "N"], size=n, p=[0.70, 0.30]),
        "occup":        rng.choice(
            ["P", "C", "S", "M", "H", "R", "U"], size=n,
            p=[0.15, 0.20, 0.15, 0.15, 0.15, 0.10, 0.10],
        ),
        "dwlltype":     rng.choice(["H", "M", "C", "U"], size=n,
                                   p=[0.55, 0.15, 0.20, 0.10]),
        "ethnic":       rng.choice(["N", "S", "P", "O"], size=n,
                                   p=[0.70, 0.10, 0.10, 0.10]),
    })
    return df


# ── Preprocessing ──────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean, encode, and engineer features from Cell2Cell schema.
    Handles both the synthetic schema (lowercase short names) and the
    real Cell2Cell dataset schema (CamelCase long names).
    Returns a fully numeric DataFrame ready for modeling.
    """
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()

    # ── Normalise real Cell2Cell column names → internal names ────────────────
    # The published dataset uses CamelCase long names; map them to the short
    # names the rest of the pipeline expects.
    RENAME = {
        "churnlabel":               "churndep",
        "monthsinservice":          "months",
        "uniquesubs":               "uniqsubs",
        "activesubs":               "actvsubs",
        "monthlyrevenue":           "revenue",
        "monthlyminutes":           "mou",
        "totalrecurringcharge":     "recchrge",
        "directorassistedcalls":    "directas",
        "overageminutes":           "overage",
        "roamingcalls":             "roam",
        "percchangeminutes":        "changem",
        "percchangerevenues":       "changer",
        "droppedcalls":             "dropvce",
        "blockedcalls":             "blckvce",
        "unansweredcalls":          "unansvce",
        "customercarecalls":        "custcare",
        "threewaycalls":            "threeway",
        "receivedcalls":            "mourec",
        "outboundcalls":            "outcalls",
        "inboundcalls":             "incalls",
        "peakcallsinout":           "peakvce",
        "offpeakcallsinout":        "offpeakvce",
        "droppedblockedcalls":      "dropblk",
        "callforwardingcalls":      "callfwdv",
        "callwaitingcalls":         "callwait",
        "currentequipmentdays":     "eqpdays",
        "agehh1":                   "age1",
        "agehh2":                   "age2",
        "handsets":                 "phones",
        "handsetmodels":            "models",
        "handsetprice":             "hnd_price",
        "handsetrefurbished":       "refurb_new",
        "handsetwebcapable":        "webcap",
        "truckowner":               "truck",
        "rvowner":                  "rv",
        "homeownership":            "ownrent",
        "retentioncalls":           "retcalls",
        "retentionoffersaccepted":  "retaccpt",
        "newcellphoneuser":         "newcelly",
        "notnewcellphoneuser":      "newcelln",
        "referralsmadebysubscriber":"numbcars",   # closest proxy
        "incomegroup":              "income",
        "ownscomputer":             "pcown",
        "hascreditcard":            "creditcd",
        "buysviamailorder":         "mailorder",
        "respondstomail offers":    "mailres",
        "respondstomailoffers":     "mailres",
        "optoutmailings":           "mailord",
        "nonustravel":              "forgntvl",
        "ownsMotorcycle":           "mtrcycle",
        "ownsmotorcycle":           "mtrcycle",
        "adjustmentstocreditrating":"creditrating",
        "madecalltoretentionteam":  "madecall",
        "creditrating":             "crclscod",
        "prizmcode":                "prizm_social_one",
        "occupation":               "occup",
        "maritalstatus":            "marital",
        "servicearea":              "area",
        "childreninh h":            "kid0_2",
        "childreninhh":             "kid0_2",
        "numberof references":      "lor",
        "numberofreferences":       "lor",
    }
    df = df.rename(columns={k: v for k, v in RENAME.items() if k in df.columns})

    # Drop identifier
    df = df.drop(columns=["customerid"], errors="ignore")

    # Numeric coercion
    for col in ["revenue", "mou", "recchrge", "overage", "roam",
                "hnd_price", "eqpdays", "age1", "age2", "lor", "income"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill numeric nulls with median
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Encode binary flags
    for col in ["asl_flag", "refurb_new", "creditcd", "mailorder",
                "webcap", "truck", "rv", "ownrent", "forgntvl",
                "kid0_2", "mailres", "mailord", "pcown",
                "newcelly", "newcelln", "mtrcycle", "madecall"]:
        if col in df.columns and df[col].dtype == object:
            pos = df[col].mode()[0]
            df[col] = (df[col] == pos).astype(int)

    # One-hot encode categoricals that remain
    cat_cols = [c for c in ["crclscod", "area", "hnd_webcap", "marital",
                             "creditcd", "occup", "dwlltype", "ethnic",
                             "refurb_new", "prizm_social_one"]
                if c in df.columns and df[c].dtype == object]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    # ── Feature engineering ───────────────────────────────────────
    # Revenue efficiency: how much revenue per minute of use
    df["revenue_per_mou"] = df["revenue"] / (df["mou"] + 1)

    # Call quality: drop rate as fraction of total voice
    total_voice = df.get("peakvce", 0) + df.get("offpeakvce", 0) + 1
    df["drop_rate"] = df.get("dropvce", pd.Series(0, index=df.index)) / total_voice

    # Customer care intensity: calls per month of tenure
    df["custcare_rate"] = df["custcare"] / (df["months"] + 1)

    # Equipment age relative to tenure (equipment older than tenure = upgrade lag)
    df["eqp_age_ratio"] = df["eqpdays"] / (df["months"] * 30 + 1)

    # Usage trend: change in MOU normalized by current MOU
    if "changem" in df.columns:
        df["mou_trend"] = df["changem"] / (df["mou"] + 1)

    # Overage intensity: overage as fraction of base charge
    if "recchrge" in df.columns:
        df["overage_rate"] = df["overage"] / (df["recchrge"] + 1)

    # Ensure target is int — real dataset uses "Yes"/"No" or 1/0
    if df[TARGET].dtype == object:
        df[TARGET] = df[TARGET].str.strip().str.lower().map(
            {"yes": 1, "true": 1, "1": 1, "no": 0, "false": 0, "0": 0}
        ).fillna(0).astype(int)
    else:
        df[TARGET] = df[TARGET].fillna(0).astype(int)

    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return all feature columns, excluding target and horizon label columns."""
    exclude = {TARGET, "customerid", "churn"}
    exclude |= {f"churn_{h}d" for h in HORIZONS}
    exclude |= {"days_to_churn", "event_observed", "cohort"}
    return [c for c in df.columns if c not in exclude
            and not c.startswith("churn_")]


# ── Weibull event log ──────────────────────────────────────────────────────────

def generate_weibull_event_log(
    df: pd.DataFrame,
    shape: float = 1.5,
    max_days: int = 730,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Attach a simulated days_to_churn to each customer using a Weibull
    accelerated failure time model.

    The Weibull distribution is the standard model for time-to-event data.
    Shape > 1 means churn risk *increases* over time (aging / contract-expiry
    effect), which is realistic for telecom.

    The scale parameter is personalised per customer: high-risk customers
    get a lower scale (shorter expected lifetime). Risk is driven by the
    same features that appear in the data, anchored to the observed
    churndep label so the event log is consistent with known outcomes.

    Censoring is preserved: churndep=0 customers may have days_to_churn
    beyond the observation window — they haven't churned yet.

    Parameters
    ----------
    df        : preprocessed DataFrame with churndep column
    shape     : Weibull shape k. k>1 → increasing hazard (aging effect)
    max_days  : observation window in days (customers censored beyond this)
    seed      : random seed

    Returns
    -------
    df with added columns:
      days_to_churn : simulated time to churn event (int)
      event_observed: 1 if churn occurred within window, 0 if censored
      churn_Xd      : binary label for each horizon in HORIZONS
    """
    rng = np.random.default_rng(seed)
    df  = df.copy()

    # ── Risk score from observed features ────────────────────────
    # Normalise each term to [0,1] before combining
    def _pct(s, q):
        return (s > s.quantile(q)).astype(float)

    risk = (
        0.25 * (df["months"] < 6).astype(float)          # new = risky
      + 0.15 * (df["months"] < 12).astype(float)
      + 0.20 * _pct(df["revenue"], 0.75)                  # high billing
      + 0.15 * _pct(df["eqpdays"], 0.75)                  # old equipment
      + 0.10 * (df["mou"] < df["mou"].quantile(0.25)).astype(float)   # low usage
      + 0.10 * _pct(df["custcare"], 0.75)                 # frequent support
      + 0.05 * df[TARGET].astype(float)                   # anchor to label
    ).clip(0.05, 0.95)

    # ── Weibull sampling via inverse CDF ──────────────────────────
    # T = scale * (-ln(U))^(1/k),  U ~ Uniform(0,1)
    # scale personalised: high risk → small scale → short lifetime
    scale = max_days * (1.0 - 0.75 * risk)
    u     = rng.uniform(1e-6, 1 - 1e-6, size=len(df))
    days_to_churn = scale * (-np.log(u)) ** (1.0 / shape)
    days_to_churn = days_to_churn.clip(1, max_days * 2).round().astype(int)

    # ── Enforce consistency with observed churndep label ─────────
    # Known churners must have days_to_churn <= max_days
    churned_mask = df[TARGET] == 1
    days_to_churn = np.where(
        churned_mask & (days_to_churn > max_days),
        rng.integers(1, max_days, size=len(df)),
        days_to_churn,
    )
    # Known non-churners: push a fraction beyond max_days (censored)
    not_churned_mask = df[TARGET] == 0
    censor_mask = not_churned_mask & (days_to_churn <= max_days)
    days_to_churn = np.where(
        censor_mask,
        rng.integers(max_days + 1, max_days * 2, size=len(df)),
        days_to_churn,
    )

    df["days_to_churn"]  = days_to_churn
    df["event_observed"] = (df["days_to_churn"] <= max_days).astype(int)

    # ── Horizon labels ────────────────────────────────────────────
    for h in HORIZONS:
        df[f"churn_{h}d"] = (df["days_to_churn"] <= h).astype(int)

    return df


# ── Temporal cohorts ───────────────────────────────────────────────────────────

def make_temporal_cohorts(
    df: pd.DataFrame,
    n_cohorts: int = 6,
    drift_start: int = 3,
    cohort_size: int = 1200,
    seed: int = 42,
) -> list[pd.DataFrame]:
    """
    Simulate temporal production cohorts with injected feature drift.

    Cohorts 0..(drift_start-1): stable, drawn from original distribution.
    Cohorts drift_start..n_cohorts-1: drift injected:
      - revenue shifts upward (price increase)
      - mou decreases (usage erosion)
      - eqpdays increases (aging equipment base)
      - custcare increases (service degradation signal)
    """
    rng = np.random.default_rng(seed)
    cohorts = []

    for i in range(n_cohorts):
        idx   = rng.choice(len(df), size=min(cohort_size, len(df)), replace=False)
        chunk = df.iloc[idx].copy().reset_index(drop=True)

        if i >= drift_start:
            severity = (i - drift_start + 1) / max(n_cohorts - drift_start, 1)

            # Revenue: price inflation (+25% at full drift)
            chunk["revenue"]   = chunk["revenue"]   * (1 + 0.25 * severity)
            chunk["recchrge"]  = chunk["recchrge"]  * (1 + 0.20 * severity)
            chunk["overage"]   = chunk["overage"]   * (1 + 0.30 * severity)

            # MOU: usage erosion (−30% at full drift)
            chunk["mou"]       = (chunk["mou"] * (1 - 0.30 * severity)).clip(0)

            # Equipment: aging (+40% eqpdays)
            chunk["eqpdays"]   = (chunk["eqpdays"] * (1 + 0.40 * severity)).clip(0)

            # Customer care: service degradation
            chunk["custcare"]  = (chunk["custcare"] + severity * 2).round()

            # Re-derive engineered features after drift
            chunk["revenue_per_mou"] = chunk["revenue"] / (chunk["mou"] + 1)
            chunk["custcare_rate"]   = chunk["custcare"] / (chunk["months"] + 1)
            chunk["eqp_age_ratio"]   = chunk["eqpdays"] / (chunk["months"] * 30 + 1)
            if "overage_rate" in chunk.columns:
                chunk["overage_rate"] = chunk["overage"] / (chunk["recchrge"] + 1)

        chunk["cohort"] = i
        cohorts.append(chunk)

    return cohorts


# ── Full pipeline ──────────────────────────────────────────────────────────────

def load_pipeline(
    n_cohorts:   int = 6,
    drift_start: int = 3,
    horizon:     int = 90,
) -> tuple:
    """
    Full loading pipeline.

    Returns
    -------
    train_df  : training set (cohort 0, 80%)
    test_df   : held-out test set (cohort 0, 20%)
    cohorts   : all cohort DataFrames (with churn_score added later)
    features  : feature column names
    horizon   : the primary horizon used for the XGB classifier
    """
    print("  Downloading / loading Cell2Cell data …")
    raw = download_data()

    print("  Preprocessing …")
    processed = preprocess(raw)

    print("  Generating Weibull event log …")
    processed = generate_weibull_event_log(processed)

    churn_rate = processed[TARGET].mean()
    h_rates = {h: processed[f"churn_{h}d"].mean() for h in HORIZONS}
    print(f"  Churn rate (snapshot): {churn_rate:.1%}")
    for h, r in h_rates.items():
        print(f"  Churn rate ({h:>3}d horizon): {r:.1%}")

    print("  Building temporal cohorts …")
    cohorts  = make_temporal_cohorts(processed, n_cohorts=n_cohorts,
                                     drift_start=drift_start)
    features = get_feature_cols(cohorts[0])

    # Train/test split from cohort 0
    base  = cohorts[0]
    split = int(0.8 * len(base))
    train_df = base.iloc[:split].copy()
    test_df  = base.iloc[split:].copy()

    return train_df, test_df, cohorts, features, horizon