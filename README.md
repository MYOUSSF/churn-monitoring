# 📉 Customer Churn Scoring with Survival Analysis & Model Monitoring
### Production-Grade ML Pipeline on Cell2Cell Telecom Data

[![CI](https://github.com/MYOUSSF/churn-monitoring/actions/workflows/ci.yml/badge.svg)](https://github.com/MYOUSSF/churn-monitoring/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12-blue.svg)](https://python.org)
[![MLflow](https://img.shields.io/badge/Tracking-MLflow-orange.svg)](https://mlflow.org)
[![lifelines](https://img.shields.io/badge/Survival-lifelines-green.svg)](https://lifelines.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/Tests-45%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **TL;DR:** Most ML churn projects stop at model training. This one builds what comes after — a production scoring pipeline that asks *when* a customer will churn, not just *whether*, monitors its own health across cohorts, detects distribution shift before performance silently degrades, and translates model outputs into dollar-denominated business decisions. Deployable live at [churn-monitoring.streamlit](https://churn-monitoring.streamlit.app).

---

## 🚀 Quick Start

```bash
# Run the full pipeline (CLI)
python analyze.py

# Launch the interactive dashboard
streamlit run dashboard.py

# Live cohort-by-cohort terminal simulation
python analyze.py --stream
```

---

## 💰 The Business Case

Cell2Cell telecom, 5,000 customers scored per month, LTV = $1,200, retention offer = $75:

| Scenario | Monthly cost | Monthly savings |
|---|---|---|
| No model — all churners lost | ~$688,000 | — |
| Model at naive threshold (0.50) | — | $0 (catches nobody) |
| **Model at cost-optimal threshold** | — | **~$325,000–$460,000** |
| **Annual savings** | — | **~$3.9M–$5.5M** |

> The cost-optimal threshold (~0.08–0.12 for Platt-calibrated scores at 11.5% churn rate) is computed automatically. The threshold sweep adapts to the actual score distribution rather than a hardcoded 0.1–0.9 range — which matters because calibrated scores are compressed toward the base rate.

The campaign budget simulator in the dashboard lets you set a monthly offer budget and shows the efficient frontier of recall vs cost — reframing the model as a resource allocation tool rather than a ranking exercise.

---

## 🎯 Design Decisions Worth Discussing

**Why Cell2Cell over IBM Telco?**
Cell2Cell has 10× the rows (71k vs 7k), 3× the features, and real behavioral signals that IBM Telco lacks: minutes of use, equipment age in days, customer care call history, retention contact flags, overage charges. SHAP feature importance shows genuinely competing signals rather than tenure dominating everything.

**Why a Weibull event log instead of the raw snapshot label?**
The dataset provides a single binary `Churn` flag — a snapshot in time. It tells us *that* a customer churned, not *when*. The Weibull event log attaches a `days_to_churn` to each customer drawn from a customer-specific Weibull distribution, from which we derive horizon-specific labels: `churn_30d`, `churn_60d`, `churn_90d`, `churn_180d`. This enables the survival model and makes the XGB classifier target a specific intervention window rather than an ambiguous historical label.

**Why Platt scaling instead of isotonic regression?**
At ~11.5% minority rate, isotonic regression needs many calibration samples to fit a stable monotone function. Platt scaling (logistic regression on held-out raw scores) is more robust at moderate imbalance and correctly compresses scores toward the base rate. The resulting calibrated probabilities are lower than 0.50 for most customers — this is correct, not a bug.

**Why train on the full dataset instead of one cohort?**
Training on cohort 0 alone gives 960 training rows — 1.7% of a 71k dataset. The pipeline uses a proper 80/20 stratified split across the full dataset (40k train, 10k test). Cohorts are used for monitoring simulation only.

**The calibration data-leak (and how it's fixed):**
```python
# ❌ Wrong — CalibratedClassifierCV refits the base model on the same data
CalibratedClassifierCV(base_clf, method="isotonic", cv=5).fit(X_train, y_train)

# ✅ Correct — dedicated holdout; base model never sees calibration data
X_train, X_calib, y_train, y_calib = train_test_split(X, y, test_size=0.25)
model.fit(X_train, y_train)
platt = LogisticRegression().fit(model.predict_proba(X_calib)[:, 1], y_calib)
```

---

## 📐 System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Cell2Cell Telecom  (71,047 customers, 58 raw features)          │
│  File: data/cell2celltrain.csv                                   │
└───────────────────────────┬──────────────────────────────────────┘
                            │
         ┌──────────────────┴──────────────────┐
         ▼                                      ▼
  Preprocessing                        Weibull Event Log
  Column rename (CamelCase → short)    days_to_churn ~ Weibull(k=1.5, λᵢ)
  Label encode high-cardinality cols   λᵢ driven by churndep label + noise
  Feature engineering:                 churndep=0 → right-censored
    retention_contact                  labels: churn_30d/60d/90d/180d
    retention_accept_rate
    outbound_ratio
    unanswered_rate
    revenue_declining
    revenue_per_mou, drop_rate
    custcare_rate, eqp_age_ratio
         └──────────────────┬──────────────────┘
                            ▼
        ┌───────────────────────────────────────────────┐
        │  TRAIN — full dataset, stratified 80/20 split  │
        │  40,837 train  |  10,210 test  |  61 features  │
        │                                               │
        │  WeibullAFT (lifelines)                       │
        │  → P(churn) at any horizon from one model     │
        │  → Handles censoring natively                 │
        │  → Concordance index ~0.58                    │
        │                                               │
        │  XGBoost  (fixed 90d horizon)                 │
        │  → max_depth=4, gamma=1, L1+L2 regularised   │
        │  → 5-fold CV on model-train split             │
        │  → Platt calibration on 25% holdout           │
        │  → CV AUROC: 0.62 | Test AUROC: 0.64         │
        │  → LogisticRegression baseline: 0.59 AUROC    │
        │  → Adaptive cost-weighted threshold sweep     │
        └──────────────────┬────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   Score drift      Feature drift     AUROC tracking
   PSI per cohort   KS + PSI on       per cohort vs
                    10 key features   threshold
          └────────────────┼────────────────┘
                           ▼
            ┌──────────────────────────────┐
            │  RETRAIN TRIGGER             │
            │  ANY of:                     │
            │  · Score PSI > 0.20          │
            │  · AUROC < threshold         │
            │  · >20% features drifted     │
            └──────────────────────────────┘
                           │
           ┌───────────────┴───────────────┐
           ▼                               ▼
  📊 Streamlit Dashboard          💻 Terminal stream
  · Live cohort simulation        · --stream flag
  · Customer SHAP explorer        · Per-cohort alerts
  · Survival curve explorer       · ANSI colour output
  · Campaign budget simulator
  · Threshold / drift sliders
```

---

## 📊 Results

### Model Performance (90-day horizon, real Cell2Cell data)

| Metric | XGBoost (Platt-calibrated) | Logistic Regression |
|---|---|---|
| CV AUROC (5-fold) | 0.623 ± 0.005 | — |
| Test AUROC | 0.636 | 0.594 |
| Test AUPRC | 0.182 | 0.157 |
| Brier Score | 0.099 | — |
| Train AUROC | 0.804 | — |
| Train / CV gap | 0.18 | — |

**On the train/CV gap:** a gap of 0.18 is acceptable but worth noting. The AUROC ceiling is partly architectural — the Weibull labels are generated from a noisy risk score intentionally decoupled from the training features to prevent memorisation. On a dataset with ground-truth time-to-churn labels the gap would close and AUROC would be higher.

### Survival Model (WeibullAFT)

| Metric | Value |
|---|---|
| Concordance index | 0.583 |
| AUROC at 30d | 0.581 |
| AUROC at 90d | 0.579 |
| AUROC at 180d | 0.595 |

The survival model's primary value is not AUROC on a single horizon — it's the ability to read off P(churn by t) for any t from one fitted model, with censoring handled correctly. The concordance of 0.58 reflects the noisy Weibull label construction rather than model weakness.

### Drift Monitoring (6 cohorts, drift from cohort 3)

| Cohort | Score PSI | Status | AUROC | Drifted Features | Retrain |
|---|---|---|---|---|---|
| 0 (reference) | 0.000 | ✅ Stable | 0.757 | 0 | No |
| 1 | 0.023 | ✅ Stable | 0.710 | 0 | No |
| 2 | 0.017 | ✅ Stable | 0.742 | 0 | No |
| 3 | 0.046 | ✅ Stable | 0.697 | **4** | **Yes** |
| 4 | 0.056 | ✅ Stable | 0.711 | **6** | **Yes** |
| 5 | 0.081 | ✅ Stable | 0.694 | **7** | **Yes** |

Cohorts 3–5 trigger retrain via the **feature drift count rule** (>20% of monitored features drifted), not PSI. Score PSI stays in the warning-or-stable range because the model still produces reasonable score distributions even as the underlying features shift — the feature-level KS tests catch the covariate shift earlier than score PSI does. This is the correct behaviour.

---

## 🧠 The Math

### Weibull AFT

$$\ln(T_i) = \mathbf{x}_i^\top \boldsymbol{\beta} + \sigma\varepsilon_i \quad \Rightarrow \quad S(t \mid \mathbf{x}) = \exp\!\left(-\left(\frac{t}{\lambda(\mathbf{x})}\right)^k\right)$$

Shape k=1.5: hazard increases over time — realistic for telecom where contract expiry and equipment aging drive churn. Churn probability at horizon h: **1 − S(h)**. Non-churners are right-censored (they haven't churned *yet*, not *never*).

### Cost-Weighted Threshold

$$\tau^* = \arg\min_\tau \; \text{FNR}(\tau) \cdot c_\text{LTV} + \text{FPR}(\tau) \cdot c_\text{offer}$$

At a 16:1 cost ratio (LTV \$1,200 / offer \$75), the optimal threshold is substantially below 0.50. The sweep searches the actual score distribution percentiles — not a fixed [0.1, 0.9] range — because Platt-calibrated scores at 11.5% churn rate are compressed into [0.05, 0.25].

### PSI

$$\text{PSI} = \sum_{i=1}^{n} \left( A_i\% - E_i\% \right) \cdot \ln\!\left(\frac{A_i\%}{E_i\%}\right)$$

< 0.10 stable · 0.10–0.20 monitor · > 0.20 retrain. Computed on both the score distribution and 10 individual features per cohort.

---

## 🗂️ Project Structure

```
churn-monitoring/
│
├── .github/
│   └── workflows/ci.yml          # GitHub Actions — Python 3.10/11/12 + ruff lint
│
├── src/
│   ├── data/
│   │   └── loader.py             # Cell2Cell ingest, preprocessing, Weibull event log,
│   │                             # horizon labels, temporal cohort simulation
│   ├── models/
│   │   ├── churn_model.py        # XGBoost + Platt calibration + LR baseline
│   │   └── survival.py           # WeibullAFT (lifelines) with censoring
│   ├── monitoring/
│   │   ├── drift.py              # PSI, KS test, cohort reports, retrain trigger
│   │   └── stream.py             # Real-time ANSI terminal simulation
│   ├── business/
│   │   └── business_metrics.py   # Cost calculator, adaptive threshold sweep, $ savings
│   └── evaluation/
│       └── plots.py              # 10 publication-quality static plots
│
├── notebooks/
│   ├── 01_eda.ipynb              # EDA, new Cell2Cell signals, Weibull construction
│   ├── 02_modeling.ipynb         # XGBoost, calibration, SHAP, survival model
│   ├── 03_monitoring.ipynb       # PSI, feature drift, AUROC degradation, interactive
│   └── 04_business_impact.ipynb  # Threshold sweep, sensitivity, ROI of retraining
│
├── tests/
│   └── test_pipeline.py          # 45 unit + integration tests (5 test classes)
│
├── data/
│   └── cell2celltrain.csv        # Cell2Cell dataset (add to repo for cloud deploy)
│
├── dashboard.py                  # Streamlit dashboard (6 tabs)
├── analyze.py                    # End-to-end CLI pipeline
└── requirements.txt              # Pinned deployment dependencies
```

---

## 📊 Dashboard Tabs

| Tab | What it does |
|---|---|
| 🔴 **Drift Monitor** | Monitoring table, PSI chart, AUROC chart, feature heatmap, **▶ live simulation button** |
| 💰 **Business Impact** | Threshold sweep, savings erosion by cohort, **campaign budget simulator** |
| 🔍 **Customer Explorer** | Top-200 highest-risk table, per-customer **SHAP waterfall**, population percentile comparison |
| 📈 **Survival Curves** | Side-by-side **personalised survival curves** with feature sliders, churn probability at each horizon |
| 📊 **Model Performance** | ROC, PR, calibration curves — XGBoost vs LR baseline |
| 🔬 **Feature Analysis** | Distribution by churn, correlation heatmap, score distribution by cohort |

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

Place `cell2celltrain.csv` in the `data/` directory.  
Download from: [kaggle.com/datasets/jpacse/datasets-for-churn-telecom](https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom)

The pipeline falls back to a synthetic replica if the file is absent — all tabs work, results use generated data.

### 3. Run the Pipeline

```bash
python analyze.py                           # 90d horizon, full pipeline
python analyze.py --horizon 30              # 30-day early-intervention model
python analyze.py --stream                  # live cohort-by-cohort terminal demo
python analyze.py --stream --delay 1.0      # faster stream
python analyze.py --ltv 1500 --offer-cost 100  # custom cost assumptions
python analyze.py --skip-survival           # skip WeibullAFT (faster)
```

### 4. Dashboard

```bash
streamlit run dashboard.py
# → http://localhost:8501
```

Enable the survival model tab via the **"Load survival model"** checkbox in the sidebar (+30s on first load, cached thereafter).

### 5. Notebooks

```bash
pip install jupyter
cd notebooks && jupyter notebook
# Run in order: 01 → 02 → 03 → 04
# Each notebook is self-contained and reloads data from scratch
```

### 6. Tests

```bash
pytest tests/ -v                       # 45 tests, ~50s
pytest tests/ -v -k "not survival"     # skip lifelines tests
```

### 7. MLflow

```bash
mlflow ui --port 5000
# → http://localhost:5000
```

---

## ⚙️ CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--horizon` | 90 | Prediction horizon in days (30/60/90/180) |
| `--stream` | False | Real-time cohort-by-cohort terminal simulation |
| `--delay` | 2.5 | Seconds between cohorts in stream mode |
| `--ltv` | 1200 | Customer LTV in $ |
| `--offer-cost` | 75 | Retention offer cost in $ |
| `--monthly-at-risk` | 5000 | Customers scored per month |
| `--n-cohorts` | 6 | Production cohorts to simulate |
| `--drift-start` | 3 | Cohort where drift is injected |
| `--auroc-threshold` | 0.70 | AUROC retrain threshold |
| `--skip-survival` | False | Skip WeibullAFT model |
| `--skip-plots` | False | Skip static plot generation |

---

## 🔬 Known Limitations & What I'd Improve

**Weibull label quality.** The `days_to_churn` values are synthetic — the dataset only provides a snapshot churn flag, not event timestamps. The Weibull risk score is intentionally decoupled from the training features to prevent label memorisation (which caused train AUROC ~0.95, CV AUROC ~0.62 before the fix). A dataset with real time-to-churn data would produce sharper models and more meaningful survival curves.

**AUROC ceiling.** Test AUROC of 0.64 reflects the Weibull label noise rather than the model's capacity. On the snapshot `churndep` label (no survival layer) the same model scores ~0.73 — the gap is the label construction overhead.

**Cohort drift is synthetic.** In production, cohorts would be timestamped scoring batches with delayed ground-truth labels. The drift injection (revenue +25%, MOU -30%, equipment aging +40%) is realistic but simulated.

**Survival model concordance.** The WeibullAFT concordance of 0.58 reflects the noisy Weibull labels. The model's value is the any-horizon probability interface and correct censoring treatment — not outperforming XGB on a single horizon.

**What a complete production system would add:** automated retraining loop with champion/challenger deployment, Evidently HTML reports for stakeholder sharing, bootstrapped PSI confidence intervals (the 0.10/0.20 thresholds are industry rules of thumb, not statistically derived), and time-varying covariates in the survival model.

---

## 🧰 Stack

| Component | Technology |
|---|---|
| Dataset | Cell2Cell Telecom Churn (71k customers, 58 features) |
| Classifier | XGBoost — max_depth=4, L1+L2 regularised, Platt calibration |
| Survival model | lifelines WeibullAFTFitter with right-censoring |
| Baseline | scikit-learn LogisticRegression (scaled) |
| Explainability | SHAP TreeExplainer — global + per-customer waterfall |
| Drift detection | Custom PSI · SciPy KS test · Chi-squared |
| Business layer | Adaptive cost-weighted threshold sweep · campaign simulator |
| Dashboard | Streamlit — 6 tabs, live simulation, survival explorer |
| Experiment tracking | MLflow |
| CI | GitHub Actions (Python 3.10/3.11/3.12 + ruff lint) |
| Testing | pytest · 45 tests · 5 test classes |

---

## 📚 References

- Kalbfleisch, J. & Prentice, R. (2002). *The Statistical Analysis of Failure Time Data.* Wiley.
- Davidson-Pilon, C. (2019). [lifelines: survival analysis in Python.](https://joss.theoj.org/papers/10.21105/joss.01317) *JOSS.*
- Niculescu-Mizil, A. & Caruana, R. (2005). [Predicting good probabilities with supervised learning.](https://dl.acm.org/doi/10.1145/1102351.1102430) *ICML.*
- Gama, J. et al. (2014). [A survey on concept drift adaptation.](https://dl.acm.org/doi/10.1145/2523813) *ACM Computing Surveys.*
- [Cell2Cell Telecom Dataset](https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom)
- [Population Stability Index reference](https://mwburke.github.io/data%20science/2018/04/29/population-stability-index.html)

---

## 📄 License

MIT