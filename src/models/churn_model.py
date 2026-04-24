"""
Churn scoring models.

Two model types:
  1. XGBHorizonClassifier  — binary classifier for a fixed horizon label
     (e.g. "will this customer churn within 90 days?")
     Fixes the calibration data-leak from the original project.

  2. SurvivalModel wrapper  — see survival.py for the lifelines AFT model

Fixes vs original project
--------------------------
* Calibration leak fixed: base XGB is fit on a dedicated training split;
  isotonic calibration is fit on a separate held-out calibration set.
  CalibratedClassifierCV is NOT used as a refitting wrapper here.
* SMOTE applied only within the training split, never touching calibration
  or test data.
* Training on 71k rows — small-N SMOTE concerns no longer apply, so SMOTE
  is optional (controlled by use_smote flag).
* LogisticRegression baseline included for comparison.
* Threshold optimization via F-beta / cost-weighted sweep.
"""

import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import shap
import joblib
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    brier_score_loss, classification_report,
    roc_curve, precision_recall_curve, f1_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET    = "churndep"
HORIZONS  = [30, 60, 90, 180]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _horizon_target(horizon: int) -> str:
    return f"churn_{horizon}d"


def _get_Xy(df: pd.DataFrame, features: list[str], target_col: str):
    X = df[features].fillna(0)
    y = df[target_col]
    return X, y


# ── Logistic Regression baseline ──────────────────────────────────────────────

def train_baseline(
    train_df:   pd.DataFrame,
    features:   list[str],
    horizon:    int = 90,
) -> Pipeline:
    """Scaled LogisticRegression baseline for comparison."""
    target = _horizon_target(horizon)
    X, y   = _get_Xy(train_df, features, target)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42, n_jobs=-1,
        )),
    ])
    pipe.fit(X, y)
    return pipe


# ── XGBoost horizon classifier ─────────────────────────────────────────────────

def _build_xgb(scale_pos_weight: float = 1.0) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,           # shallower trees → less memorisation
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,  # more feature dropout → more regularisation
        min_child_weight=10,   # require more samples per leaf
        gamma=1.0,             # min split gain → prunes trivial splits
        reg_alpha=0.1,         # L1 regularisation
        reg_lambda=2.0,        # L2 regularisation
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )


def train(
    train_df:        pd.DataFrame,
    features:        list[str],
    horizon:         int   = 90,
    use_smote:       bool  = False,
    experiment_name: str   = "churn-monitoring",
    run_name:        str   = "xgb_horizon",
) -> tuple:
    """
    Train XGBoost classifier for a fixed horizon label.

    Calibration is done correctly:
      1. Split train_df into model_train (70%) and calib_holdout (30%)
      2. Fit XGBoost on model_train (with optional SMOTE)
      3. Fit isotonic calibration on calib_holdout predictions only
         — the base model has never seen calib_holdout during training
      4. Cross-validation is on model_train only

    Returns
    -------
    model       : fitted XGBClassifier (uncalibrated)
    calibrated  : isotonic-calibrated wrapper
    baseline    : LogisticRegression baseline
    shap_values : SHAP values on model_train X
    explainer   : shap.TreeExplainer
    metrics     : dict of CV + train metrics
    """
    target = _horizon_target(horizon)
    X_all, y_all = _get_Xy(train_df, features, target)

    # ── 1. Split into model-train and calibration holdout ─────────
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_all, y_all, test_size=0.25, stratify=y_all, random_state=42
    )

    # ── 2. Optional SMOTE on model-train only ─────────────────────
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    if use_smote:
        smote   = SMOTE(random_state=42, k_neighbors=5)
        X_fit, y_fit = smote.fit_resample(X_train, y_train)
    else:
        X_fit, y_fit = X_train, y_train

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"{run_name}_{horizon}d"):

        # ── 3. Cross-validate on model-train ─────────────────────
        xgb_cv = _build_xgb(scale_pos_weight=pos_weight if not use_smote else 1.0)
        cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            xgb_cv, X_fit, y_fit, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        print(f"  [{horizon}d] CV AUROC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # ── 4. Fit final XGB on full model-train ──────────────────
        model = _build_xgb(scale_pos_weight=pos_weight if not use_smote else 1.0)
        model.fit(X_fit, y_fit)

        # ── 5. Calibrate on held-out calibration set ──────────────
        # Predict raw probabilities on calib set (model never saw this data).
        #
        # Platt scaling (logistic regression on raw scores) is more robust
        # than isotonic regression at ~13% minority rate — isotonic needs
        # many calibration samples to fit a stable monotone function.
        from sklearn.linear_model import LogisticRegression as _LR

        raw_calib_proba = model.predict_proba(X_calib)[:, 1].reshape(-1, 1)
        platt = _LR(C=1.0, random_state=42, max_iter=1000)
        platt.fit(raw_calib_proba, y_calib)
        # Ensure class ordering: platt.classes_[1] should be the positive class (1)
        # If LR fitted in reverse order (can happen with small calibration sets),
        # flip the wrapper so predict_proba[:,1] always means P(churn=1)
        pos_col = int(np.where(platt.classes_ == 1)[0][0]) if 1 in platt.classes_ else 1
        calibrated = _PlattWrapper(model, platt, pos_col=pos_col)

        # ── 6. Baseline ───────────────────────────────────────────
        baseline = train_baseline(
            train_df.__class__(
                np.hstack([X_train, y_train.values.reshape(-1, 1)]),
                columns=list(X_train.columns) + [target],
            ) if False else
            pd.concat([X_train, y_train], axis=1),
            features, horizon,
        )

        # ── 7. SHAP on model-train ────────────────────────────────
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_fit)

        # ── 8. Metrics ────────────────────────────────────────────
        # Use raw model scores for train AUROC — this is the honest measure
        # of in-sample discrimination before calibration is applied.
        # Calibrated scores are reserved for the test set evaluation.
        raw_train_proba = model.predict_proba(X_fit)[:, 1]
        metrics = {
            "horizon":       horizon,
            "train_auroc":   float(roc_auc_score(y_fit, raw_train_proba)),
            "cv_auroc_mean": float(cv_scores.mean()),
            "cv_auroc_std":  float(cv_scores.std()),
            "use_smote":     use_smote,
            "n_train":       len(X_fit),
            "n_calib":       len(X_calib),
            "pos_rate":      float(y_fit.mean()),
        }

        mlflow.log_params({
            "horizon": horizon, "use_smote": use_smote,
            "n_estimators": 400, "max_depth": 6, "learning_rate": 0.04,
        })
        mlflow.log_metrics({k: v for k, v in metrics.items()
                            if isinstance(v, float)})

        # Save artifacts
        joblib.dump(model,      MODEL_DIR / f"xgb_{horizon}d.pkl")
        joblib.dump(calibrated, MODEL_DIR / f"xgb_{horizon}d_calibrated.pkl")

    return model, calibrated, baseline, shap_values, explainer, metrics


# ── Isotonic calibration wrapper ───────────────────────────────────────────────

class _PlattWrapper:
    """
    Platt scaling: fit a logistic regression on raw XGB scores from a
    held-out calibration set. More robust than isotonic regression at
    ~13% minority rate where isotonic needs large N to fit stably.
    pos_col tracks which column of platt.predict_proba() is P(churn=1).
    """
    def __init__(self, base_model, platt_lr, pos_col: int = 1):
        self.base_model = base_model
        self.platt      = platt_lr
        self.pos_col    = pos_col   # column index for positive class in Platt output

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1].reshape(-1, 1)
        cal = self.platt.predict_proba(raw)[:, self.pos_col]
        return np.column_stack([1 - cal, cal])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


class _IsotonicWrapper:
    """Kept for backward compatibility."""
    def __init__(self, base_model, iso_regressor):
        self.base_model = base_model
        self.iso        = iso_regressor

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1]
        cal = self.iso.predict(raw)
        return np.column_stack([1 - cal, cal])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(
    model,
    test_df:  pd.DataFrame,
    features: list[str],
    horizon:  int   = 90,
    fnr_cost: float = 10.0,
    fpr_cost: float = 1.0,
) -> dict:
    """
    Full evaluation on a held-out test set.

    Also computes the cost-optimal decision threshold:
      cost(threshold) = FN_rate * fnr_cost + FP_rate * fpr_cost
    where fnr_cost = cost of missing a churner (lost customer LTV)
          fpr_cost = cost of unnecessary retention offer

    Default ratio 10:1 reflects typical telecom economics.
    """
    target = _horizon_target(horizon)
    X, y   = _get_Xy(test_df, features, target)

    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    fpr_arr, tpr_arr, roc_thresh = roc_curve(y, proba)
    prec_arr, rec_arr, pr_thresh = precision_recall_curve(y, proba)

    # Cost-optimal threshold — sweep over actual score percentiles so the
    # search space adapts to the calibrated score range (which may be
    # compressed into [0.05, 0.25] after Platt scaling on ~12% churn rate)
    score_min = float(np.percentile(proba, 1))
    score_max = float(np.percentile(proba, 99))
    candidates = np.linspace(score_min, score_max, 100)
    # Always include 0.50 as a reference point even if out of range
    candidates = np.unique(np.append(candidates, [0.50]))

    best_thresh, best_cost = float(np.median(candidates)), np.inf
    for t in candidates:
        p      = (proba >= t).astype(int)
        fn_r   = ((p == 0) & (y == 1)).sum() / max((y == 1).sum(), 1)
        fp_r   = ((p == 1) & (y == 0)).sum() / max((y == 0).sum(), 1)
        cost   = fn_r * fnr_cost + fp_r * fpr_cost
        if cost < best_cost:
            best_cost, best_thresh = cost, float(t)

    metrics = {
        "horizon":         horizon,
        "auroc":           float(roc_auc_score(y, proba)),
        "auprc":           float(average_precision_score(y, proba)),
        "brier":           float(brier_score_loss(y, proba)),
        "f1":              float(f1_score(y, pred)),
        "fpr":             fpr_arr,
        "tpr":             tpr_arr,
        "precision":       prec_arr,
        "recall":          rec_arr,
        "y_true":          y.values,
        "y_score":         proba,
        "report":          classification_report(y, pred, output_dict=True),
        "optimal_threshold": best_thresh,
        "optimal_cost":    best_cost,
    }
    return metrics


def evaluate_baseline(
    baseline,
    test_df:  pd.DataFrame,
    features: list[str],
    horizon:  int = 90,
) -> dict:
    target = _horizon_target(horizon)
    X, y   = _get_Xy(test_df, features, target)
    proba  = baseline.predict_proba(X)[:, 1]
    return {
        "auroc": float(roc_auc_score(y, proba)),
        "auprc": float(average_precision_score(y, proba)),
        "brier": float(brier_score_loss(y, proba)),
    }


# ── Cohort scoring ─────────────────────────────────────────────────────────────

def score_cohorts(
    model:    object,
    cohorts:  list[pd.DataFrame],
    features: list[str],
    horizon:  int = 90,
) -> list[pd.DataFrame]:
    """Score each cohort and attach churn_score column."""
    target  = _horizon_target(horizon)
    scored  = []
    for i, cohort in enumerate(cohorts):
        c = cohort.copy()
        X = c[features].fillna(0)
        c["churn_score"] = model.predict_proba(X)[:, 1]
        try:
            auroc = roc_auc_score(c[target], c["churn_score"])
            print(f"  Cohort {i}: n={len(c):5,} | "
                  f"churn_rate={c[target].mean():.3f} | "
                  f"AUROC={auroc:.4f}")
        except Exception:
            print(f"  Cohort {i}: n={len(c):5,} | AUROC=N/A")
        scored.append(c)
    return scored