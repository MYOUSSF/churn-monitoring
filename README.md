# 📉 Customer Churn Scoring with Model Monitoring
### Production-Grade Drift Detection on Telco Customer Data

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![MLflow](https://img.shields.io/badge/Tracking-MLflow-orange.svg)](https://mlflow.org)
[![Tests](https://img.shields.io/badge/Tests-25%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **TL;DR:** Most ML projects stop at model training. This one builds what comes after: a production scoring pipeline that monitors its own health, detects when the input distribution shifts, and triggers a retraining alert before model performance silently degrades.

---

## 🎯 Problem Statement

A telecom operator trains a churn model and deploys it. Three months later the model is still running but performance has quietly eroded — customer acquisition shifted toward a younger, lower-tenure base, and monthly charges rose after a price increase. The model was trained on a different population. No one noticed.

This project simulates exactly that scenario:

1. Train an XGBoost churn classifier on a reference population.
2. Score six production cohorts that progressively drift from the training distribution.
3. Compute Population Stability Index (PSI), feature-level KS tests, and AUROC per cohort.
4. Fire a retraining alert when drift or performance degradation crosses defined thresholds.

```
Reference cohort → train model → score cohort 0,1,2 (stable)
                                → score cohort 3,4,5 (drift injected)
                                             │
                       PSI > 0.20 or AUROC < threshold
                                             │
                                   RETRAIN TRIGGERED ⚠
```

---

## 📐 System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│  IBM Telco Churn  (7,043 customers, 20 raw features)           │
│  Fallback: synthetic replica matching original distributions   │
└──────────────────────────┬─────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
   ┌──────────────────┐      ┌─────────────────────────────────┐
   │  Feature Eng.    │      │  Temporal Cohort Simulation     │
   │  RFM features    │      │  Cohorts 0–2: stable            │
   │  num_services    │      │  Cohorts 3–5: drift injected    │
   │  charges/tenure  │      │  (charges ↑, tenure ↓)         │
   └────────┬─────────┘      └────────────┬────────────────────┘
            │                             │
            ▼                             ▼
   ┌──────────────────────────────────────────────────────────┐
   │  MODEL TRAINING (on cohort 0 — reference)                │
   │  SMOTE oversampling → XGBoost → Isotonic calibration     │
   │  CV AUROC: 0.78 | Test AUROC: 0.73 | Brier: 0.17        │
   └──────────────────────────────┬───────────────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
   ┌──────────────────┐  ┌────────────────┐  ┌──────────────────┐
   │  Score drift     │  │  Feature drift │  │  AUROC tracking  │
   │  PSI per cohort  │  │  KS + PSI per  │  │  per cohort      │
   │  → stable /      │  │  feature       │  │  vs threshold    │
   │    warning /     │  │                │  │                  │
   │    retrain       │  │                │  │                  │
   └────────┬─────────┘  └───────┬────────┘  └────────┬─────────┘
            └───────────────┬────┘                    │
                            ▼                         │
                 ┌──────────────────────┐             │
                 │  RETRAIN TRIGGER     │◄────────────┘
                 │  ANY of:             │
                 │  • Score PSI > 0.20  │
                 │  • AUROC < threshold │
                 │  • >20% feats drifted│
                 └──────────────────────┘
```

---

## 📊 Key Results

### Model Performance (test set)

| Metric | Value |
|---|---|
| CV AUROC (5-fold) | 0.776 ± 0.052 |
| Test AUROC | 0.728 |
| Test AUPRC | 0.455 |
| Brier Score | 0.170 |

### Drift Monitoring Summary

| Cohort | Score PSI | Status | AUROC | Drifted Features | Retrain |
|---|---|---|---|---|---|
| 0 (reference) | 0.000 | ✅ Stable | 0.955 | 0 | No |
| 1 | 0.177 | ⚠ Warning | 0.757 | 0 | No |
| 2 | 0.163 | ⚠ Warning | 0.759 | 0 | No |
| 3 | 0.188 | ⚠ Warning | 0.798 | 1 | No |
| **4** | **0.219** | **🔴 Retrain** | **0.719** | **1** | **Yes** |
| **5** | **0.261** | **🔴 Retrain** | **0.747** | **2** | **Yes** |

> PSI rules of thumb: < 0.10 stable · 0.10–0.20 monitor · > 0.20 retrain

### Top Drifted Features (Cohort 5 vs Reference)

| Feature | PSI | KS p-value | Status |
|---|---|---|---|
| MonthlyCharges | 0.182 | < 0.001 | Drifted |
| charges_per_month_tenure | 0.165 | < 0.001 | Drifted |
| tenure | 0.087 | 0.041 | Borderline |

---

## 🧠 The Math

### Population Stability Index (PSI)

PSI measures how much a distribution has shifted relative to a reference:

$$\text{PSI} = \sum_{i=1}^{n} \left( \text{Actual}_i\% - \text{Expected}_i\% \right) \cdot \ln\!\left(\frac{\text{Actual}_i\%}{\text{Expected}_i\%}\right)$$

Where bins are defined by quantiles of the reference (expected) distribution. This is computed for both the raw churn score and for each key feature independently.

### Retraining trigger logic

Retraining is triggered when **any** of:

```
Score PSI > 0.20                  → input distribution significantly shifted
AUROC < threshold (default 0.70)  → measurable performance degradation
>20% of key features drifted      → systematic covariate shift
```

### Why calibrate?

XGBoost's raw output scores are not calibrated probabilities. Without calibration, a score of 0.7 does not mean 70% churn probability — it just means higher risk than 0.6. Isotonic regression maps the raw scores to actual empirical probabilities, which matters for:
- Setting meaningful risk thresholds ("contact all customers with >40% churn probability")
- Expected-value calculations in downstream business logic
- PSI computation on a scale with a consistent business interpretation

### Why SMOTE?

The Telco dataset has ~26% churn rate (imbalanced). SMOTE (Synthetic Minority Oversampling Technique) generates synthetic minority-class samples in feature space to balance the training set. Applied only on training data inside the pipeline — never on validation or test sets.

---

## 🗂️ Project Structure

```
churn-monitoring/
│
├── src/
│   ├── data/
│   │   └── loader.py          # Download, preprocess, temporal cohort simulation
│   ├── models/
│   │   └── churn_model.py     # XGBoost + SMOTE + calibration + SHAP + MLflow
│   ├── monitoring/
│   │   └── drift.py           # PSI, KS test, cohort reports, retrain trigger
│   └── evaluation/
│       └── plots.py           # 8 publication-quality matplotlib plots
│
├── tests/
│   └── test_pipeline.py       # 25 unit + integration tests (pytest)
│
├── results/
│   ├── monitoring_report.csv  # Per-cohort drift + performance table
│   └── plots/
│       ├── 01_eda_overview.png
│       ├── 02_roc_pr_curves.png
│       ├── 03_calibration_curve.png
│       ├── 04_shap_summary.png
│       ├── 05_shap_waterfall.png
│       ├── 06_score_drift.png
│       ├── 07_auroc_degradation.png
│       └── 08_feature_drift_heatmap.png
│
├── analyze.py                 # End-to-end pipeline CLI
└── requirements.txt
```

---

## 🚀 Getting Started

### 1. Clone & Install

```bash
git clone https://github.com/MYOUSSF/churn-monitoring.git
cd churn-monitoring
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

```bash
# Downloads dataset automatically (falls back to synthetic if unavailable)
python analyze.py
```

### 3. View Results in MLflow

```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

### 4. Run Tests

```bash
pytest tests/ -v
# 25 tests, ~30 seconds
```

---

## ⚙️ Configuration Options

| Flag | Default | Description |
|---|---|---|
| `--n-cohorts` | 6 | Number of production cohorts to simulate |
| `--drift-start` | 3 | Cohort index at which drift is injected |
| `--auroc-threshold` | 0.70 | AUROC below which retraining is triggered |
| `--alpha` | 0.05 | Significance level for KS / Chi-squared tests |
| `--output-dir` | `results/` | Directory for CSVs and plots |
| `--skip-plots` | False | Skip plot generation (faster) |

---

## 📈 Output Plots

| File | Description |
|---|---|
| `01_eda_overview.png` | Class balance, tenure distribution by churn, monthly charges by churn |
| `02_roc_pr_curves.png` | ROC and Precision-Recall curves on held-out test set |
| `03_calibration_curve.png` | Reliability diagram — calibrated vs uncalibrated XGBoost |
| `04_shap_summary.png` | Beeswarm SHAP plot — global feature importance |
| `05_shap_waterfall.png` | SHAP waterfall for a single high-risk customer |
| `06_score_drift.png` | PSI of churn score per cohort with status colour-coding |
| `07_auroc_degradation.png` | AUROC per cohort with retrain trigger threshold line |
| `08_feature_drift_heatmap.png` | PSI heatmap for key features × cohorts |

---

## 🔬 Post-Mortem: What I'd Improve With More Time

1. **Real temporal data with actual deployment logs.** The cohort simulation injects synthetic drift. In a production system, cohorts would be slices of scored batches with timestamps. Logged propensity scores and actual labels (received with delay) would make the AUROC degradation curve real rather than simulated.

2. **Evidently dashboard.** The `evidently` library generates HTML drift reports that are shareable with non-technical stakeholders. The current PSI/KS implementation computes the same statistics, but an Evidently report is easier to embed in a Slack/Teams alert workflow.

3. **Automated retraining loop.** The current pipeline fires a boolean flag when retraining is triggered. A complete system would: (a) log the trigger event to MLflow, (b) pull the last N months of scored + labelled data, (c) retrain and evaluate against the existing model, (d) shadow-deploy the new model before promoting it.

4. **Champion/challenger scoring.** Rather than replacing the model at retrain, run both the current model and the retrained model in parallel for one cohort and compare their AUROC on fresh labels before promoting.

5. **Business metrics layer.** AUROC measures ranking quality. The metric that actually matters is expected cost: `FN × cost_of_losing_customer + FP × cost_of_retention_offer`. A decision threshold sweep over this objective would give an optimal operating point that changes as acquisition costs change.

6. **Confidence intervals on PSI.** The PSI thresholds (0.10, 0.20) are industry rules of thumb, not statistically rigorous. Bootstrapping the PSI distribution under the null (no shift) would give proper rejection thresholds adjusted for cohort size.

---

## 🧰 Stack

| Component | Technology |
|---|---|
| Dataset | IBM Telco Customer Churn (Kaggle / synthetic fallback) |
| ML pipeline | scikit-learn · imbalanced-learn (SMOTE) |
| Model | XGBoost with isotonic calibration |
| Explainability | SHAP (TreeExplainer) |
| Drift detection | Custom PSI · SciPy KS test · Chi-squared |
| Experiment tracking | MLflow |
| Visualisation | matplotlib · seaborn |
| Testing | pytest (25 tests) |

---

## 📚 References

- Gama, J. et al. (2014). [A survey on concept drift adaptation.](https://dl.acm.org/doi/10.1145/2523813) *ACM Computing Surveys*, 46(4).
- Chawla, N. V. et al. (2002). [SMOTE: Synthetic Minority Over-sampling Technique.](https://arxiv.org/abs/1106.1813) *JAIR*, 16, 321–357.
- Platt, J. (1999). Probabilistic outputs for support vector machines. *Advances in Large Margin Classifiers*.
- Niculescu-Mizil, A. & Caruana, R. (2005). [Predicting good probabilities with supervised learning.](https://dl.acm.org/doi/10.1145/1102351.1102430) *ICML*.
- [IBM Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- [Population Stability Index — industry reference](https://mwburke.github.io/data%20science/2018/04/29/population-stability-index.html)

---

## 📄 License

MIT — free for personal and commercial use.
