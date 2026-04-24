# 📓 Notebooks

Four notebooks covering the full project end-to-end.  
Run them in order, or jump to any topic independently.

## Sequence

| Notebook | Topic | Key outputs |
|---|---|---|
| `01_eda.ipynb` | Exploratory Data Analysis | Feature distributions, correlations, Weibull event log construction |
| `02_modeling.ipynb` | Model Training | XGBoost vs LR baseline, calibration fix, SHAP, survival model |
| `03_monitoring.ipynb` | Drift Detection | PSI, KS tests, AUROC degradation, retrain trigger, interactive threshold |
| `04_business_impact.ipynb` | Business Impact | Dollar savings, threshold sweep, sensitivity analysis, ROI of retraining |

## Setup

```bash
# From the project root
pip install -r requirements.txt
pip install jupyter

cd notebooks/
jupyter notebook
```

Or open in VS Code with the Jupyter extension.

## What each notebook covers

### 01 — EDA
- Dataset shape, dtypes, missingness audit
- Class balance and churn rate
- Behavioral feature distributions by churn status (MOU, revenue, equipment age)
- Pearson correlation heatmap — what actually predicts churn
- Weibull event log construction — how horizon labels are built
- Weibull hazard function shapes (why k=1.5 is the right choice)

### 02 — Modeling
- Logistic Regression baseline (always establish a baseline first)
- Calibration deep-dive: the original data leak vs the correct holdout approach
- XGBoost training with 5-fold CV
- ROC and Precision-Recall curves (XGB vs baseline)
- Calibration reliability diagram
- Cost-weighted threshold optimisation
- SHAP global importance + waterfall for a high-risk customer
- WeibullAFT survival model: fit, evaluate, survival curves by risk tertile
- Per-customer comparison: XGB vs AFT at all four horizons

### 03 — Monitoring
- Score 6 production cohorts (drift injected from cohort 3)
- Score distribution shift — side-by-side histograms
- PSI calculation and status labelling (stable / warning / retrain)
- KS test + feature-level PSI for 10 behavioral features
- Feature drift heatmap (features × cohorts)
- AUROC degradation chart with retrain threshold line
- Full monitoring report with colour-coded status table
- **Interactive cell**: change PSI/AUROC thresholds and re-run to see trigger fire/clear

### 04 — Business Impact
- LTV / offer cost framework — the 16:1 asymmetry
- Full threshold sweep — savings, precision, recall, F1
- Cost breakdown pie chart at optimal threshold
- Sensitivity analysis heatmap — LTV × offer cost → savings
- Savings erosion under drift — monthly loss per cohort
- ROI of retraining — break-even in days
- Value waterfall — from no-model baseline to optimised system
- Break-even AUROC analysis — minimum model quality to justify deployment

## Tips

- **All notebooks are self-contained** — each reloads data from scratch
- Data auto-downloads on first run (Kaggle API or synthetic fallback)
- MLflow experiments are logged to `../mlruns/` — run `mlflow ui` to explore
- For live interaction without re-running cells: `streamlit run ../dashboard.py`
