"""
Test suite — 24 unit and integration tests.

Run:  pytest tests/ -v
"""

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_df():
    """Minimal synthetic churn dataframe (processed schema)."""
    from src.data.loader import _generate_synthetic, preprocess
    raw = _generate_synthetic(n=400, seed=99)
    return preprocess(raw)


@pytest.fixture(scope="module")
def features(small_df):
    from src.data.loader import get_feature_cols
    return get_feature_cols(small_df)


@pytest.fixture(scope="module")
def trained(small_df, features):
    """Train a small model for downstream tests (no MLflow)."""
    import mlflow, os
    mlflow.set_tracking_uri("file:///tmp/mlflow_test")
    from src.models.churn_model import train
    split = int(0.8 * len(small_df))
    train_df = small_df.iloc[:split]
    return train(train_df, features, experiment_name="test", run_name="test")


# ── Data loading ──────────────────────────────────────────────────────────────

class TestDataLoader:
    def test_synthetic_shape(self, small_df):
        assert len(small_df) == 400
        assert "Churn" in small_df.columns

    def test_no_nulls_in_numeric(self, small_df):
        for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
            assert small_df[col].isna().sum() == 0, f"{col} has nulls"

    def test_target_binary(self, small_df):
        assert set(small_df["Churn"].unique()).issubset({0, 1})

    def test_engineered_features_present(self, small_df):
        for col in ["num_services", "charges_per_month_tenure"]:
            assert col in small_df.columns

    def test_num_services_non_negative(self, small_df):
        assert (small_df["num_services"] >= 0).all()

    def test_cohort_creation(self, small_df):
        from src.data.loader import make_temporal_cohorts
        cohorts = make_temporal_cohorts(small_df, n_cohorts=4, drift_start=2)
        assert len(cohorts) == 4
        for i, c in enumerate(cohorts):
            assert "cohort" in c.columns
            assert c["cohort"].iloc[0] == i

    def test_drift_increases_charges(self, small_df):
        from src.data.loader import make_temporal_cohorts
        cohorts = make_temporal_cohorts(small_df, n_cohorts=4, drift_start=2)
        mean_0 = cohorts[0]["MonthlyCharges"].mean()
        mean_3 = cohorts[3]["MonthlyCharges"].mean()
        assert mean_3 > mean_0, "Drift should increase MonthlyCharges"

    def test_feature_cols_excludes_target(self, features):
        assert "Churn" not in features
        assert "customerID" not in features


# ── Model ─────────────────────────────────────────────────────────────────────

class TestChurnModel:
    def test_train_returns_five_items(self, trained):
        assert len(trained) == 5

    def test_calibrated_predicts_proba(self, trained, small_df, features):
        _, calibrated, _, _, _ = trained
        X = small_df[features].fillna(0).head(10)
        proba = calibrated.predict_proba(X)
        assert proba.shape == (10, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_proba_in_unit_interval(self, trained, small_df, features):
        _, calibrated, _, _, _ = trained
        X = small_df[features].fillna(0)
        proba = calibrated.predict_proba(X)[:, 1]
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_train_auroc_reasonable(self, trained):
        _, _, _, _, metrics = trained
        assert metrics["train_auroc"] > 0.60

    def test_cv_auroc_present(self, trained):
        _, _, _, _, metrics = trained
        assert "cv_auroc_mean" in metrics
        assert 0 < metrics["cv_auroc_mean"] < 1

    def test_shap_values_shape(self, trained, small_df, features):
        _, _, shap_values, _, _ = trained
        assert shap_values.ndim == 2
        # SHAP values have one column per feature
        assert shap_values.shape[1] == len(features)

    def test_evaluate_returns_metrics(self, trained, small_df, features):
        _, calibrated, _, _, _ = trained
        from src.models.churn_model import evaluate
        split = int(0.8 * len(small_df))
        test_df = small_df.iloc[split:]
        m = evaluate(calibrated, test_df, features)
        for key in ["auroc", "auprc", "brier", "fpr", "tpr"]:
            assert key in m

    def test_score_cohorts_adds_column(self, trained, small_df, features):
        _, calibrated, _, _, _ = trained
        from src.data.loader import make_temporal_cohorts
        from src.models.churn_model import score_cohorts
        cohorts = make_temporal_cohorts(small_df, n_cohorts=2, drift_start=1)
        scored = score_cohorts(calibrated, cohorts, features)
        assert "churn_score" in scored[0].columns
        assert "churn_score" in scored[1].columns


# ── Drift detection ───────────────────────────────────────────────────────────

class TestDrift:
    def test_psi_identical_distributions(self):
        from src.monitoring.drift import psi
        x = np.random.default_rng(0).normal(0, 1, 1000)
        assert psi(x, x) < 0.01

    def test_psi_shifted_distribution(self):
        from src.monitoring.drift import psi
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 1000)
        y = rng.normal(2, 1, 1000)
        assert psi(x, y) > 0.20

    def test_psi_label_stable(self):
        from src.monitoring.drift import psi_label
        assert psi_label(0.05) == "stable"

    def test_psi_label_warning(self):
        from src.monitoring.drift import psi_label
        assert psi_label(0.15) == "warning"

    def test_psi_label_retrain(self):
        from src.monitoring.drift import psi_label
        assert psi_label(0.25) == "retrain"

    def test_feature_drift_returns_results(self, small_df, features):
        from src.monitoring.drift import detect_feature_drift
        results = detect_feature_drift(small_df.head(200), small_df.tail(200), features[:5])
        assert len(results) == 5
        for r in results:
            assert hasattr(r, "psi")
            assert r.psi >= 0

    def test_cohort_reports_length(self, trained, small_df, features):
        _, calibrated, _, _, _ = trained
        from src.data.loader import make_temporal_cohorts
        from src.models.churn_model import score_cohorts
        from src.monitoring.drift import build_cohort_reports
        cohorts = make_temporal_cohorts(small_df, n_cohorts=4, drift_start=2)
        scored = score_cohorts(calibrated, cohorts, features)
        reports = build_cohort_reports(scored[0], scored, features)
        assert len(reports) == 4

    def test_retrain_triggered_on_drifted_cohort(self, trained, small_df, features):
        _, calibrated, _, _, _ = trained
        from src.data.loader import make_temporal_cohorts
        from src.models.churn_model import score_cohorts
        from src.monitoring.drift import build_cohort_reports
        cohorts = make_temporal_cohorts(small_df, n_cohorts=6, drift_start=2)
        scored = score_cohorts(calibrated, cohorts, features)
        reports = build_cohort_reports(scored[0], scored, features)
        statuses = [r.retrain_triggered for r in reports]
        # At least one late cohort should trigger retraining
        assert any(statuses[3:]), "Late drifted cohorts should trigger retraining"

    def test_reports_to_dataframe_cols(self, trained, small_df, features):
        _, calibrated, _, _, _ = trained
        from src.data.loader import make_temporal_cohorts
        from src.models.churn_model import score_cohorts
        from src.monitoring.drift import build_cohort_reports, reports_to_dataframe
        cohorts = make_temporal_cohorts(small_df, n_cohorts=3, drift_start=1)
        scored = score_cohorts(calibrated, cohorts, features)
        reports = build_cohort_reports(scored[0], scored, features)
        df = reports_to_dataframe(reports)
        for col in ["cohort", "auroc", "score_psi", "retrain_triggered"]:
            assert col in df.columns
