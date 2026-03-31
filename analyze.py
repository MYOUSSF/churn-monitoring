"""
End-to-end churn monitoring pipeline.

Usage
-----
python analyze.py                          # default settings
python analyze.py --n-cohorts 8            # more cohorts
python analyze.py --drift-start 2          # earlier drift injection
python analyze.py --auroc-threshold 0.72   # stricter retrain trigger
python analyze.py --skip-plots             # skip visualisations
"""

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

# Ensure src/ is on the path when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args():
    p = argparse.ArgumentParser(description="Churn monitoring pipeline")
    p.add_argument("--n-cohorts",        type=int,   default=6,    help="Number of production cohorts")
    p.add_argument("--drift-start",      type=int,   default=3,    help="Cohort index at which drift is injected")
    p.add_argument("--auroc-threshold",  type=float, default=0.70, help="AUROC below which retraining is triggered")
    p.add_argument("--alpha",            type=float, default=0.05, help="Feature drift significance level")
    p.add_argument("--output-dir",       type=str,   default="results", help="Directory for CSVs and plots")
    p.add_argument("--skip-plots",       action="store_true",       help="Skip plot generation")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Churn Scoring + Drift Monitoring Pipeline")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────
    print("\n[1/5] Loading and preprocessing data …")
    from src.data.loader import load_pipeline
    train_df, test_df, cohorts, features = load_pipeline(
        n_cohorts=args.n_cohorts,
        drift_start=args.drift_start,
    )
    print(f"  Training set : {len(train_df):,} rows")
    print(f"  Test set     : {len(test_df):,} rows")
    print(f"  Cohorts      : {len(cohorts)} (drift from cohort {args.drift_start})")
    print(f"  Features     : {len(features)}")
    churn_rate = (train_df["Churn"] == 1).mean() if (train_df["Churn"] == 1).any() else (train_df["Churn"] == "Yes").mean()
    print(f"  Train churn rate: {churn_rate:.1%}")

    # ── 2. Train model ────────────────────────────────────────────
    print("\n[2/5] Training XGBoost + calibration …")
    from src.models.churn_model import train, evaluate, score_cohorts

    model, calibrated, shap_values, explainer, train_metrics = train(
        train_df, features,
        experiment_name="churn-monitoring",
        run_name=f"xgb_cohorts{args.n_cohorts}_drift{args.drift_start}",
    )
    print(f"  Train AUROC : {train_metrics['train_auroc']:.4f}")
    print(f"  CV AUROC    : {train_metrics['cv_auroc_mean']:.4f} ± {train_metrics['cv_auroc_std']:.4f}")

    # ── 3. Evaluate on test set ───────────────────────────────────
    print("\n[3/5] Evaluating on held-out test set …")
    test_metrics = evaluate(calibrated, test_df, features)
    print(f"  Test AUROC  : {test_metrics['auroc']:.4f}")
    print(f"  Test AUPRC  : {test_metrics['auprc']:.4f}")
    print(f"  Test Brier  : {test_metrics['brier']:.4f}")

    # ── 4. Score all cohorts + monitor drift ──────────────────────
    print("\n[4/5] Scoring cohorts and detecting drift …")
    scored_cohorts = score_cohorts(calibrated, cohorts, features)

    from src.monitoring.drift import build_cohort_reports, reports_to_dataframe
    reports = build_cohort_reports(
        reference_cohort=scored_cohorts[0],
        scored_cohorts=scored_cohorts,
        features=features,
        auroc_threshold=args.auroc_threshold,
    )
    report_df = reports_to_dataframe(reports)

    print("\n  Monitoring summary:")
    print(report_df[[
        "cohort", "churn_rate", "mean_score", "score_psi",
        "score_status", "auroc", "n_drifted_features", "retrain_triggered"
    ]].to_string(index=False))

    # Save report
    report_path = output_dir / "monitoring_report.csv"
    report_df.to_csv(report_path, index=False)
    print(f"\n  Report saved → {report_path}")

    # Retrain summary
    retrain_cohorts = report_df[report_df["retrain_triggered"]]["cohort"].tolist()
    if retrain_cohorts:
        print(f"\n  ⚠  Retrain triggered at cohorts: {retrain_cohorts}")
    else:
        print("\n  ✓  Model stable — no retrain triggered")

    # ── 5. Plots ──────────────────────────────────────────────────
    if not args.skip_plots:
        print("\n[5/5] Generating visualisations …")
        from src.evaluation.plots import generate_all
        generate_all(
            train_df=train_df,
            test_df=test_df,
            uncalibrated_model=model,
            calibrated_model=calibrated,
            shap_values=shap_values,
            explainer=explainer,
            test_metrics=test_metrics,
            report_df=report_df,
            scored_cohorts=scored_cohorts,
            features=features,
        )
    else:
        print("\n[5/5] Skipping plots (--skip-plots)")

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print(f"  Results → {output_dir.resolve()}")
    print(f"  MLflow  → run `mlflow ui` and open http://localhost:5000")
    print("=" * 60)


if __name__ == "__main__":
    main()
