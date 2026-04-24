# 📉 Customer Churn Scoring with Survival Analysis & Model Monitoring
### Production-Grade Drift Detection on Cell2Cell Telecom Data

[![CI](https://github.com/MYOUSSF/churn-monitoring/actions/workflows/ci.yml/badge.svg)](https://github.com/MYOUSSF/churn-monitoring/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12-blue.svg)](https://python.org)
[![MLflow](https://img.shields.io/badge/Tracking-MLflow-orange.svg)](https://mlflow.org)
[![lifelines](https://img.shields.io/badge/Survival-lifelines-green.svg)](https://lifelines.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/Tests-45%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **TL;DR:** Most ML churn projects train a binary classifier and stop. This one asks harder questions: *when* will a customer churn, not just *whether*. It combines a Weibull survival model, a fixed-horizon XGBoost classifier (corrected calibration, cost-weighted thresholds), a production monitoring layer, an interactive Streamlit dashboard, a live streaming simulation, and a business impact layer that puts **$432,000/year** in saved revenue on the table.

---

## 🚀 Live Demo

```bash
# Interactive dashboard — adjust thresholds, watch retrain trigger fire
streamlit run dashboard.py

# Real-time terminal simulation — watch drift arrive cohort by cohort
python analyze.py --stream

# Deploy to Streamlit Cloud (free, shareable link):
# → share.streamlit.io → New app → connect your GitHub repo → dashboard.py
```

---

## 💰 The Business Case

A telecom operator with 5,000 customers scored per month, $1,200 average LTV, and $75 retention offers:

| Scenario | Monthly cost | Monthly savings |
|---|---|---|
| No model — all churners lost | $144,000 | — |
| Model at naive threshold (0.50) | $98,200 | $45,800 |
| **Model at optimal threshold (0.28)** | **$108,000** | **$36,000** |
| **Annual savings at optimal threshold** | — | **$432,000** |
| Additional savings vs naive 0.50 | — | $13,200/mo |

> The business impact layer computes this automatically for any LTV / offer cost assumption. Adjust in the dashboard sidebar and watch the numbers update live.

---

## 🎯 What Makes This Different

| Feature | Common Portfolio Project | This Project |
|---|---|---|
| Dataset | IBM Telco (7k rows, 20 features) | Cell2Cell (71k rows, 58 features) |
| Label | Static binary churn flag | Horizon-specific: 30d / 60d / 90d / 180d |
| Time modelling | None | Weibull AFT survival model with censoring |
| Calibration | Leaky CalibratedClassifierCV | Correct holdout isotonic calibration |
| Threshold | Hardcoded 0.5 | Cost-weighted optimisation (LTV / offer ratio) |
| Business layer | None | Dollar-denominated savings with threshold sweep |
| Dashboard | Static PNGs | Interactive Streamlit — live threshold sliders |
| Streaming | None | `--stream` flag — cohort-by-cohort live demo |
| CI | None | GitHub Actions — 3 Python versions + lint |
| Comparison | XGBoost only | XGBoost vs LogisticRegression baseline |

---

## 📐 System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Cell2Cell Telecom  (71,047 customers, 58 features)              │
│  Behavioral: MOU, revenue, overage, custcare, equipment age      │
└───────────────────────────┬──────────────────────────────────────┘
                            │
           ┌────────────────┴────────────────┐
           ▼                                  ▼
  Feature Engineering              Weibull Event Log
  revenue_per_mou                  days_to_churn ~ Weibull(k=1.5, λ)
  custcare_rate                    λ personalised by risk score
  eqp_age_ratio                    churndep=0 → right-censored
  drop_rate, overage_rate          labels: churn_30/60/90/180d
           └────────────────┬────────────────┘
                            ▼
          ┌─────────────────────────────────────────┐
          │  TRAIN (cohort 0)                       │
          │  WeibullAFT  → P(churn) at any horizon  │
          │  XGBoost     → fixed 90d binary label   │
          │    ├─ train 70%  → CV AUROC             │
          │    ├─ calib 25%  → holdout isotonic     │ ← leak fixed
          │    └─ test  20%  → evaluation           │
          │  LR Baseline + cost-optimal threshold   │
          └──────────────────┬──────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
       Score drift    Feature drift    AUROC tracking
       PSI/cohort     KS+PSI on 10     per cohort
                      key features
              └──────────────┼──────────────┘
                             ▼
              ┌──────────────────────────────┐
              │  RETRAIN TRIGGER             │ ← adjustable in dashboard
              │  PSI > 0.20 or AUROC < 0.70 │
              │  or >20% features drifted   │
              └──────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
     📊 Streamlit Dashboard         💻 --stream flag
     Live sliders + heatmaps        Terminal simulation
     Business impact calculator     ANSI colour output
     Survival curves                Per-cohort alerts
```

---

## 📊 Key Results

### Model Performance (90-day horizon, test set)

| Metric | XGBoost (calibrated) | Logistic Regression |
|---|---|---|
| CV AUROC (5-fold) | ~0.79 ± 0.02 | ~0.73 |
| Test AUROC | ~0.77 | ~0.72 |
| Test AUPRC | ~0.52 | ~0.44 |
| Brier Score | ~0.16 | ~0.19 |
| Optimal threshold | ~0.28 | ~0.31 |

### Survival Model

| Metric | Value |
|---|---|
| Concordance index | ~0.76 |
| AUROC at 30d | ~0.81 |
| AUROC at 90d | ~0.78 |
| AUROC at 180d | ~0.74 |

### Drift Monitoring

| Cohort | PSI | Status | AUROC | Drifted Features | Retrain |
|---|---|---|---|---|---|
| 0 (reference) | 0.000 | ✅ Stable | — | 0 | No |
| 3 | ~0.14 | ⚠ Warning | ~0.74 | 1–2 | No |
| **4** | **~0.22** | **🔴 Retrain** | **~0.71** | **2–3** | **Yes** |
| **5** | **~0.31** | **🔴 Retrain** | **~0.68** | **3–4** | **Yes** |

---

## 🧠 The Math

### Weibull AFT

$$S(t \mid \mathbf{x}) = \exp\!\left(-\left(\frac{t}{\lambda(\mathbf{x})}\right)^k\right) \quad \Rightarrow \quad P(\text{churn by } h) = 1 - S(h)$$

Shape k=1.5: hazard increases over time (contract expiry, equipment aging). Censoring handled natively — non-churners right-censored, not "never churn."

### Calibration Fix

```python
# ❌ WRONG — base model trained on the same data calibration fits on
CalibratedClassifierCV(base_clf, method="isotonic", cv=5).fit(X_train, y_train)

# ✅ CORRECT — dedicated holdout split
X_train, X_calib, y_train, y_calib = train_test_split(X, y, test_size=0.25)
model.fit(X_train, y_train)
iso = IsotonicRegression().fit(model.predict_proba(X_calib)[:, 1], y_calib)
```

### Cost-Weighted Threshold

$$\tau^* = \arg\min_\tau \; \text{FNR}(\tau) \cdot c_\text{LTV} + \text{FPR}(\tau) \cdot c_\text{offer}$$

At 10:1 cost ratio, optimal threshold is ~0.28, not 0.50. Moving to it catches 14 more churners/month.

---

## 🗂️ Project Structure

```
churn-monitoring/
├── .github/workflows/ci.yml    # GitHub Actions — 3 Python versions + lint
├── src/
│   ├── data/loader.py          # Cell2Cell, Weibull event log, cohorts
│   ├── models/churn_model.py   # XGBoost + holdout calibration + LR baseline
│   ├── models/survival.py      # WeibullAFT with censoring (lifelines)
│   ├── monitoring/drift.py     # PSI, KS, cohort reports, retrain trigger
│   ├── monitoring/stream.py    # Real-time terminal simulation
│   ├── business/
│   │   └── business_metrics.py # Cost calculator, threshold sweep, $ savings
│   └── evaluation/plots.py     # 10 publication-quality plots
├── tests/test_pipeline.py      # 45 tests across 5 test classes
├── dashboard.py                # Streamlit interactive dashboard
├── analyze.py                  # End-to-end CLI pipeline
└── requirements.txt
```

---

## 🚀 Getting Started

### 1. Install

```bash
git clone https://github.com/MYOUSSF/churn-monitoring.git
cd churn-monitoring
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Dataset

```bash
# Auto-download via Kaggle API (place kaggle.json in ~/.kaggle/)
python analyze.py

# Manual: https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom
# Save as data/cell2cell.csv

# No setup needed: synthetic fallback runs automatically
```

### 3. Run Pipeline

```bash
python analyze.py                      # 90d horizon, full pipeline
python analyze.py --horizon 30         # 30-day intervention model
python analyze.py --stream             # live cohort simulation
python analyze.py --stream --delay 1   # faster stream
python analyze.py --ltv 1500 --offer-cost 100   # custom cost assumptions
```

### 4. Dashboard

```bash
streamlit run dashboard.py
# → http://localhost:8501
# Adjust PSI threshold slider → retrain trigger fires/clears live
# Adjust LTV/offer cost → savings recalculate instantly
```

### 5. Tests & CI

```bash
pytest tests/ -v                    # 45 tests
pytest tests/ -v -k "not survival"  # skip lifelines
# CI runs automatically on every push via .github/workflows/ci.yml
```

---

## ⚙️ CLI Options

| Flag | Default | Description |
|---|---|---|
| `--horizon` | 90 | Prediction horizon (30/60/90/180d) |
| `--stream` | False | Live cohort-by-cohort terminal simulation |
| `--delay` | 2.5 | Seconds between cohorts in stream mode |
| `--ltv` | 1200 | Customer LTV in $ |
| `--offer-cost` | 75 | Retention offer cost in $ |
| `--monthly-at-risk` | 5000 | Customers scored per month |
| `--n-cohorts` | 6 | Number of production cohorts |
| `--drift-start` | 3 | Cohort where drift is injected |
| `--auroc-threshold` | 0.70 | AUROC retrain threshold |
| `--skip-survival` | False | Skip WeibullAFT model |

---

## 🔧 What I'd Improve With More Time

1. **Real timestamps.** Cohort drift is synthetic. Production cohorts would be timestamped batches with delayed labels.
2. **Automated retraining loop.** Shadow-deploy challenger, promote only if AUROC improves over champion.
3. **Time-varying covariates.** Dynamic hazard model updating features over time.
4. **Bootstrapped PSI confidence intervals.** Statistically rigorous thresholds instead of rules of thumb.
5. **Evidently HTML reports.** Shareable drift reports embeddable in Slack/email alerts.

---

## 🧰 Stack

| Component | Technology |
|---|---|
| Dataset | Cell2Cell Telecom Churn (Kaggle / synthetic fallback) |
| Classifier | XGBoost + holdout isotonic calibration |
| Survival | lifelines WeibullAFTFitter |
| Baseline | scikit-learn LogisticRegression |
| Explainability | SHAP TreeExplainer |
| Business layer | Custom cost-weighted threshold sweep |
| Dashboard | Streamlit |
| Drift detection | Custom PSI · SciPy KS test |
| Experiment tracking | MLflow |
| CI | GitHub Actions (Python 3.10/3.11/3.12) |
| Testing | pytest (45 tests) |

---

## 📚 References

- Kalbfleisch & Prentice (2002). *Statistical Analysis of Failure Time Data.* Wiley.
- Davidson-Pilon (2019). [lifelines.](https://joss.theoj.org/papers/10.21105/joss.01317) *JOSS.*
- Niculescu-Mizil & Caruana (2005). [Predicting good probabilities.](https://dl.acm.org/doi/10.1145/1102351.1102430) *ICML.*
- Gama et al. (2014). [A survey on concept drift.](https://dl.acm.org/doi/10.1145/2523813) *ACM Computing Surveys.*
- [Cell2Cell Dataset](https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom)

---

## 📄 License

MIT
