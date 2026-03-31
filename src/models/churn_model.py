"""
Churn scoring model.

XGBoost classifier with:
  - SMOTE oversampling for class imbalance
  - Platt scaling (isotonic regression) for calibration
  - SHAP-based feature importance
  - MLflow experiment tracking
"""

import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import shap
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    brier_score_loss, classification_report,
    roc_curve, precision_recall_curve,
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Churn"


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(
    n_estimators: int = 300,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    scale_pos_weight: float = 2.5,
    random_state: int = 42,
) -> ImbPipeline:
    """
    XGBoost classifier with SMOTE oversampling in an imbalanced-learn pipeline.
    Calibration is applied separately after fitting.
    """
    xgb = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=random_state,
        verbosity=0,
        n_jobs=-1,
    )
    pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=random_state, k_neighbors=5)),
        ("clf",   xgb),
    ])
    return pipeline


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    train_df: pd.DataFrame,
    features: list[str],
    experiment_name: str = "churn-monitoring",
    run_name: str = "baseline",
) -> tuple:
    """
    Train model, calibrate, log to MLflow.

    Returns
    -------
    model       : fitted ImbPipeline (uncalibrated XGB)
    calibrated  : CalibratedClassifierCV wrapper
    shap_values : np.ndarray of SHAP values on training set
    explainer   : shap.Explainer object
    metrics     : dict of training metrics
    """
    X = train_df[features].fillna(0)
    y = train_df[TARGET]

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        # Cross-validated AUROC
        model = build_model()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Score on raw XGB (no SMOTE in CV to avoid leakage debate — note in README)
        xgb_only = build_model()
        cv_scores = cross_val_score(
            xgb_only, X, y, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        print(f"  CV AUROC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Fit on full training set
        model.fit(X, y)

        # Calibrate with isotonic regression on SMOTE-resampled training data
        X_smoted, y_smoted = model.named_steps["smote"].fit_resample(X, y)
        from sklearn.base import clone
        base_clf = clone(model.named_steps["clf"])
        base_clf.fit(X_smoted, y_smoted)
        calibrated = CalibratedClassifierCV(
            base_clf, method="isotonic", cv=5
        )
        calibrated.fit(X_smoted, y_smoted)

        # SHAP on raw XGB
        explainer = shap.TreeExplainer(model.named_steps["clf"])
        shap_values = explainer.shap_values(X_smoted)

        # Training metrics
        proba = calibrated.predict_proba(X_smoted)[:, 1]
        metrics = {
            "train_auroc":    float(roc_auc_score(y_smoted, proba)),
            "train_auprc":    float(average_precision_score(y_smoted, proba)),
            "train_brier":    float(brier_score_loss(y_smoted, proba)),
            "cv_auroc_mean":  float(cv_scores.mean()),
            "cv_auroc_std":   float(cv_scores.std()),
        }

        mlflow.log_params({
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
        })
        mlflow.log_metrics(metrics)

        # Save model
        import joblib
        joblib.dump(model,      MODEL_DIR / "xgb_pipeline.pkl")
        joblib.dump(calibrated, MODEL_DIR / "xgb_calibrated.pkl")

    return model, calibrated, shap_values, explainer, metrics


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    model,
    test_df: pd.DataFrame,
    features: list[str],
    threshold: float = 0.5,
) -> dict:
    """Full evaluation on a held-out test set."""
    X = test_df[features].fillna(0)
    y = test_df[TARGET]

    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= threshold).astype(int)

    fpr, tpr, roc_thresholds = roc_curve(y, proba)
    prec, rec, pr_thresholds  = precision_recall_curve(y, proba)

    metrics = {
        "auroc":          float(roc_auc_score(y, proba)),
        "auprc":          float(average_precision_score(y, proba)),
        "brier":          float(brier_score_loss(y, proba)),
        "fpr":            fpr,
        "tpr":            tpr,
        "precision":      prec,
        "recall":         rec,
        "y_true":         y.values,
        "y_score":        proba,
        "report":         classification_report(y, pred, output_dict=True),
    }
    return metrics


# ── Cohort scoring ────────────────────────────────────────────────────────────

def score_cohorts(
    model,
    cohorts: list[pd.DataFrame],
    features: list[str],
) -> list[pd.DataFrame]:
    """Return each cohort augmented with 'churn_score' column."""
    scored = []
    for i, cohort in enumerate(cohorts):
        c = cohort.copy()
        X = c[features].fillna(0)
        c["churn_score"] = model.predict_proba(X)[:, 1]
        scored.append(c)
        auroc = roc_auc_score(c[TARGET], c["churn_score"])
        print(f"  Cohort {i}: n={len(c):4d} | churn_rate={c[TARGET].mean():.3f} | AUROC={auroc:.4f}")
    return scored
