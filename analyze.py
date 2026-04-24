"""
End-to-end churn monitoring pipeline — Cell2Cell edition.

What's new vs the original project
------------------------------------
Dataset   : Cell2Cell (71k customers, 58 features) replaces IBM Telco
Horizons  : Weibull event log → churn_30d/60d/90d/180d labels
Survival  : WeibullAFT survival model with proper censoring handling
XGB fix   : Calibration data-leak fixed — isotonic on dedicated holdout
Baseline  : LogisticRegression baseline for honest comparison
Threshold : Cost-weighted threshold optimization (FN/FP cost ratio)
Business  : Dollar-denominated impact — savings vs no-model baseline
Stream    : --stream flag scores cohort-by-cohort in real time (live demo)
Dashboard : Streamlit app (dashboard.py) for interactive exploration

Usage
-----
python analyze.py                          # default: 90d horizon
python analyze.py --horizon 30             # 30-day horizon
python analyze.py --stream                 # live cohort-by-cohort simulation
python analyze.py --stream --delay 1.5     # faster stream
python analyze.py --n-cohorts 8
python analyze.py --drift-start 2
python analyze.py --auroc-threshold 0.72
python analyze.py --skip-plots
python analyze.py --skip-survival          # skip lifelines (faster)
"""

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args():
    p = argparse.ArgumentParser(description="Cell2Cell churn monitoring pipeline")
    p.add_argument("--horizon",          type=int,   default=90,
                   choices=[30, 60, 90, 180], help="Primary prediction horizon (days)")
    p.add_argument("--n-cohorts",        type=int,   default=6)
    p.add_argument("--drift-start",      type=int,   default=3)
    p.add_argument("--auroc-threshold",  type=float, default=0.70)
    p.add_argument("--alpha",            type=float, default=0.05)
    p.add_argument("--use-smote",        action="store_true")
    p.add_argument("--output-dir",       type=str,   default="results")
    p.add_argument("--skip-plots",       action="store_true")
    p.add_argument("--skip-survival",    action="store_true")
    # ── New flags ──────────────────────────────────────────────────
    p.add_argument("--stream",           action="store_true",
                   help="Stream cohort-by-cohort monitoring to terminal in real time")
    p.add_argument("--delay",            type=float, default=2.5,
                   help="Seconds between cohort arrivals in stream mode")
    p.add_argument("--ltv",              type=float, default=1_200,
                   help="Customer LTV in $ (for business impact layer)")
    p.add_argument("--offer-cost",       type=float, default=75,
                   help="Retention offer cost in $ (for business impact layer)")
    p.add_argument("--monthly-at-risk",  type=int,   default=5_000,
                   help="Customers scored per month (for $ projections)")
    return p.parse_args()


def main():
    args       = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  Cell2Cell Churn Scoring + Drift Monitoring Pipeline")
    print("=" * 65)

    # ── 1. Load data ──────────────────────────────────────────────
    print(f"\n[1/7] Loading data (horizon={args.horizon}d) …")
    from src.data.loader import load_pipeline, HORIZONS
    train_df, test_df, cohorts, features, horizon = load_pipeline(
        n_cohorts=args.n_cohorts,
        drift_start=args.drift_start,
        horizon=args.horizon,
    )
    target     = f"churn_{horizon}d"
    churn_rate = train_df[target].mean()
    print(f"  Training set : {len(train_df):,} rows")
    print(f"  Test set     : {len(test_df):,} rows")
    print(f"  Cohorts      : {len(cohorts)} (drift from cohort {args.drift_start})")
    print(f"  Features     : {len(features)}")
    print(f"  Train churn rate ({horizon}d): {churn_rate:.1%}")

    # ── 2. Train XGBoost + baseline ───────────────────────────────
    print(f"\n[2/7] Training XGBoost ({horizon}d horizon) + LR baseline …")
    from src.models.churn_model import train, evaluate, evaluate_baseline, score_cohorts

    model, calibrated, baseline, shap_values, explainer, train_metrics = train(
        train_df, features,
        horizon=horizon,
        use_smote=args.use_smote,
        experiment_name="churn-monitoring-cell2cell",
        run_name=f"xgb_{horizon}d_cohorts{args.n_cohorts}",
    )
    print(f"  Train AUROC : {train_metrics['train_auroc']:.4f}")
    print(f"  CV AUROC    : {train_metrics['cv_auroc_mean']:.4f}"
          f" ± {train_metrics['cv_auroc_std']:.4f}")

    # ── 3. Evaluate on test set ───────────────────────────────────
    print(f"\n[3/7] Evaluating on held-out test set …")
    test_metrics     = evaluate(calibrated, test_df, features, horizon=horizon)
    baseline_metrics = evaluate_baseline(baseline, test_df, features, horizon=horizon)

    X_test = test_df[features].fillna(0)
    y_test = test_df[target]
    baseline_metrics["y_true"]  = y_test.values
    baseline_metrics["y_score"] = baseline.predict_proba(X_test)[:, 1]

    print(f"  XGBoost  — AUROC: {test_metrics['auroc']:.4f} | "
          f"AUPRC: {test_metrics['auprc']:.4f} | "
          f"Brier: {test_metrics['brier']:.4f}")
    print(f"  Baseline — AUROC: {baseline_metrics['auroc']:.4f} | "
          f"AUPRC: {baseline_metrics['auprc']:.4f}")
    print(f"  Optimal threshold (cost-weighted): {test_metrics['optimal_threshold']:.2f}")

    # ── 4. Business impact ────────────────────────────────────────
    print(f"\n[4/7] Computing business impact …")
    from src.business.business_metrics import (
        BusinessImpactCalculator, track_business_impact_over_cohorts
    )
    calc   = BusinessImpactCalculator(
        ltv=args.ltv,
        offer_cost=args.offer_cost,
        monthly_at_risk=args.monthly_at_risk,
    )
    biz_report = calc.full_report(test_metrics["y_true"], test_metrics["y_score"])
    print(biz_report.summary())

    # ── 5. Survival model ─────────────────────────────────────────
    aft           = None
    surv_features = None
    surv_metrics  = {}

    if not args.skip_survival:
        print(f"\n[5/7] Fitting WeibullAFT survival model …")
        try:
            from src.models.survival import train_survival, evaluate_survival
            aft, surv_features, surv_train_metrics = train_survival(train_df, features)
            surv_metrics, surv_preds = evaluate_survival(aft, test_df, surv_features)
            print(f"  Concordance index : {surv_metrics['concordance_index']:.4f}")
            for h in HORIZONS:
                k = f"auroc_{h}d"
                if k in surv_metrics:
                    print(f"  Survival AUROC {h:>3}d : {surv_metrics[k]:.4f}")
        except ImportError:
            print("  lifelines not installed — skipping (pip install lifelines)")
    else:
        print("\n[5/7] Skipping survival model (--skip-survival)")

    # ── 6. Score cohorts + monitor drift ─────────────────────────
    print(f"\n[6/7] Scoring cohorts and detecting drift …")
    scored_cohorts = score_cohorts(calibrated, cohorts, features, horizon=horizon)

    from src.monitoring.drift import build_cohort_reports, reports_to_dataframe
    reports = build_cohort_reports(
        reference_cohort=scored_cohorts[0],
        scored_cohorts=scored_cohorts,
        features=features,
        auroc_threshold=args.auroc_threshold,
        horizon=horizon,
    )
    report_df = reports_to_dataframe(reports)

    # Business impact per cohort
    biz_cohort_df = track_business_impact_over_cohorts(
        scored_cohorts, features, horizon=horizon,
        ltv=args.ltv, offer_cost=args.offer_cost,
        monthly_at_risk=args.monthly_at_risk,
    )

    print("\n  Monitoring summary:")
    print(report_df[[
        "cohort", "churn_rate", "mean_score", "score_psi",
        "score_status", "auroc", "n_drifted_features", "retrain_triggered"
    ]].to_string(index=False))

    if not biz_cohort_df.empty:
        print("\n  Business impact per cohort:")
        print(biz_cohort_df[[
            "cohort", "optimal_threshold", "optimal_recall",
            "monthly_savings", "churners_caught"
        ]].to_string(index=False))

    # Save outputs
    report_df.to_csv(output_dir / "monitoring_report.csv", index=False)
    biz_cohort_df.to_csv(output_dir / "business_impact.csv", index=False)
    biz_report.threshold_sweep.to_csv(output_dir / "threshold_sweep.csv", index=False)
    print(f"\n  Reports saved → {output_dir}")

    retrain_cohorts = report_df[report_df["retrain_triggered"]]["cohort"].tolist()
    if retrain_cohorts:
        print(f"\n  ⚠  Retrain triggered at cohorts: {retrain_cohorts}")
    else:
        print("\n  ✓  Model stable — no retrain triggered")

    # ── 7. Plots or Stream ────────────────────────────────────────
    if args.stream:
        print(f"\n[7/7] Launching real-time stream simulation …")
        from src.monitoring.stream import run_stream
        run_stream(
            model=calibrated,
            scored_cohorts=scored_cohorts,
            report_df=report_df,
            features=features,
            horizon=horizon,
            delay=args.delay,
            ltv=args.ltv,
            offer_cost=args.offer_cost,
            monthly_at_risk=args.monthly_at_risk,
        )
    elif not args.skip_plots:
        print(f"\n[7/7] Generating visualisations …")
        from src.evaluation.plots import generate_all
        generate_all(
            train_df=train_df,
            test_df=test_df,
            raw_model=model,
            calibrated_model=calibrated,
            baseline=baseline,
            baseline_metrics=baseline_metrics,
            shap_values=shap_values,
            explainer=explainer,
            test_metrics=test_metrics,
            report_df=report_df,
            scored_cohorts=scored_cohorts,
            features=features,
            horizon=horizon,
            aft=aft,
            surv_features=surv_features,
        )
    else:
        print("\n[7/7] Skipping plots (--skip-plots)")

    print("\n" + "=" * 65)
    print("  Pipeline complete.")
    print(f"  Results  → {output_dir.resolve()}")
    print(f"  MLflow   → mlflow ui  (http://localhost:5000)")
    print(f"  Dashboard → streamlit run dashboard.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
