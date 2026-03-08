# 🧪 CUPED Variance Reduction for A/B Testing
### Cutting Required Sample Size on Real Bank Marketing Campaign Data

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![MLflow](https://img.shields.io/badge/Tracking-MLflow-orange.svg)](https://mlflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **TL;DR:** Standard A/B tests waste sample size. CUPED removes variance explained by pre-experiment user behaviour, shrinking confidence intervals and cutting the users needed to reach statistical significance — without touching the unbiasedness of your estimate.

---

## 🎯 Problem Statement

Running A/B tests in e-commerce is expensive: every day in experiment is a day you can't ship. The standard two-sample t-test ignores everything you know about users *before* the experiment starts — their past purchase history, engagement, tenure. CUPED exploits that signal.

This project implements CUPED on the UCI Bank Marketing dataset (45K real campaign records), simulates an A/B experiment with a known ground truth lift, and rigorously demonstrates the variance reduction benefit across three methods.

```
Standard A/B:   Compare Y_treatment vs Y_control directly
CUPED A/B:      Compare (Y - θX)_treatment vs (Y - θX)_control
                where X = pre-experiment covariate, θ estimated on control only

Result:         Same unbiased ATE estimate. Smaller standard error. Narrower CI.
                Fewer users needed. Faster experiments.
```

---

## 📐 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  UCI Bank Marketing (45K rows)                                  │
│  ─────────────────────────────                                  │
│  Features: age, balance, duration, job, prior contacts, ...     │
│  Outcome:  term deposit subscription (11.5% base rate)          │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  A/B SIMULATION                                                 │
│  Hash-based 50/50 split + injected 3pp true lift                │
└──────────────────────┬──────────────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
┌─────────────────┐       ┌──────────────────────────────────────┐
│  RAW ANALYSIS   │       │  CUPED ANALYSIS                      │
│  Two-sample     │       │                                      │
│  t-test         │       │  1. Single-covariate CUPED           │
│  (baseline)     │       │     Y* = Y - θ(X - X̄)               │
│                 │       │     θ = Cov(Y,X)/Var(X) [ctrl only]  │
│                 │       │                                      │
│                 │       │  2. MultiCUPED (regression residual) │
│                 │       │     Y* = Y - ŷ_cv + Ȳ               │
│                 │       │     CV predictions, 5-fold           │
└────────┬────────┘       └───────────────┬──────────────────────┘
         │                                │
         └────────────┬───────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  EVALUATION                                                     │
│  • Variance reduction %          • Power curves                 │
│  • CI width comparison           • Sample size savings          │
│  • Bootstrap CIs                 • MLflow experiment tracking   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Key Results

### Variance Reduction

| Method | SE of ATE | Variance Reduction | CI Width |
|---|---|---|---|
| Raw (baseline) | 0.00412 | — | 0.01614 |
| **CUPED** | **0.00371** | **~18%** | **0.01453** |
| **MultiCUPED** | **0.00348** | **~29%** | **0.01363** |

### Sample Size Savings at MDE = 0.02pp, 80% power

| Method | N per arm | Savings vs. Raw |
|---|---|---|
| Raw | 8,340 | — |
| CUPED | 6,839 | **−18% (−1,501 users)** |
| MultiCUPED | 5,921 | **−29% (−2,419 users)** |

> *Results from full UCI Bank Marketing dataset (45,211 rows). Variance reduction depends on covariate-outcome correlation — 'previous' (prior campaign contacts) has ~0.42 correlation with subscription.*

---

## 🧠 The Math

### Classic CUPED

Given outcome $Y$ and a pre-experiment covariate $X$ (independent of treatment $T$):

$$Y^* = Y - \hat{\theta}(X - \bar{X}), \quad \hat{\theta} = \frac{\text{Cov}(Y, X)}{\text{Var}(X)}$$

**Unbiasedness:** Since $X \perp T$, adjusting does not bias the ATE:
$$\mathbb{E}[Y^*_{\text{treat}} - Y^*_{\text{ctrl}}] = \mathbb{E}[Y_{\text{treat}} - Y_{\text{ctrl}}]$$

**Variance reduction:**
$$\text{Var}(Y^*) = \text{Var}(Y)(1 - \rho^2_{XY})$$

A covariate with $\rho = 0.5$ removes **25% of variance**, meaning you need **25% fewer users** for the same power.

### MultiCUPED

With multiple covariates, fit a linear model $\hat{Y} = X\beta$ using cross-validated predictions (to prevent overfitting), then:

$$Y^* = Y - \hat{Y}_{CV} + \bar{Y}$$

The residuals have strictly lower variance when covariates explain outcome variation.

### Why estimate θ on control only?

Fitting $\theta$ on the full sample risks contamination if the treatment shifts $\text{Cov}(Y, X)$. Fitting on control preserves the independence assumption and gives a cleaner estimator.

---

## 🗂️ Project Structure

```
cuped-ab-testing/
│
├── src/
│   ├── data/
│   │   └── loader.py           # UCI dataset loading, preprocessing, A/B simulation
│   ├── models/
│   │   └── cuped.py            # CUPED, MultiCUPED, sample size, power, bootstrap
│   └── evaluation/
│       └── plots.py            # All matplotlib/seaborn visualisation
│
├── streamlit_app/
│   └── app.py                  # Interactive demo with Monte Carlo simulation
│
├── tests/
│   └── test_cuped.py           # 22 unit + integration tests
│
├── results/
│   └── plots/                  # Output charts (regenerate via analyze.py)
│
├── analyze.py                  # End-to-end pipeline CLI
└── requirements.txt
```

---

## 🚀 Getting Started

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/cuped-ab-testing.git
cd cuped-ab-testing
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Full Analysis

```bash
# Downloads dataset automatically (~5MB, no login needed)
python analyze.py --true-lift 0.03 --n-bootstrap 2000
```

### 3. View Results in MLflow

```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

### 4. Launch the Interactive Demo

```bash
# No dataset download needed — uses synthetic simulation
streamlit run streamlit_app/app.py
```

### 5. Run Tests

```bash
pytest tests/ -v
# 22 tests, ~10 seconds
```

---

## ⚙️ Configuration Options

| Flag | Default | Description |
|---|---|---|
| `--true-lift` | 0.03 | Injected treatment effect (3pp) |
| `--treatment-rate` | 0.5 | Fraction of users assigned to treatment |
| `--n-bootstrap` | 2000 | Bootstrap resamples for CI |
| `--alpha` | 0.05 | Significance level |
| `--output-dir` | `results/` | Where to save plots and CSVs |

---

## 📈 Output Plots

All saved to `results/plots/` and logged to MLflow:

| File | Description |
|---|---|
| `variance_reduction.png` | SE and variance reduction % by method |
| `confidence_intervals.png` | Forest plot: ATE + CI for all methods |
| `power_curves.png` | Power vs. MDE — raw vs. CUPED |
| `sample_size_savings.png` | N required per arm across MDE range |
| `covariate_correlations.png` | Feature correlation heatmap |
| `bootstrap_distribution.png` | Bootstrap ATE distribution with CI |

---

## 🔬 Post-Mortem: What I'd Improve With More Time

1. **Propensity-weighted CUPED:** In observational settings, treatment assignment isn't perfectly random. Combining CUPED with Inverse Propensity Weighting (IPW) gives a doubly robust estimator that handles both covariate imbalance and outcome variance.

2. **Nonlinear covariate adjustment:** Classic CUPED uses linear $\theta$. Replacing the linear model with a gradient boosted tree (XGBOOST residuals) typically captures more variance, especially when the covariate-outcome relationship is nonlinear.

3. **Stratified CUPED:** Fitting separate $\theta$ values per user segment (e.g., new vs. returning users) can improve variance reduction in heterogeneous populations.

4. **Sequential testing integration:** CUPED reduces variance but doesn't by itself enable early stopping. Combining it with Sequential Probability Ratio Tests (SPRT) or alpha-spending functions would allow peeking at results without inflating Type I error.

5. **Delta method for ratio metrics:** Subscription rate is a simple proportion. For ratio metrics (e.g., revenue per user = total revenue / n users), the variance formula is more complex — the delta method is needed before applying CUPED adjustment.

6. **A/A test validation:** Before running any A/B test, validate the pipeline with an A/A test (both arms get control). The p-value distribution should be uniform under the null. This is a standard sanity check that every experimentation platform should run.

---

## 🧰 Stack

| Component | Technology |
|---|---|
| Dataset | UCI Bank Marketing via `ucimlrepo` |
| Statistical analysis | scipy, statsmodels |
| ML covariate model | scikit-learn (LinearRegression + CV) |
| Experiment tracking | MLflow |
| Visualization | matplotlib, seaborn, plotly |
| Demo app | Streamlit |
| Testing | pytest (22 tests) |

---

## 📚 References

- Deng, A., Xu, Y., Kohavi, R., & Walker, T. (2013). **Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data.** WSDM 2013. — *Original CUPED paper*
- Kohavi, R., Tang, D., & Xu, Y. (2020). **Trustworthy Online Controlled Experiments.** Cambridge University Press.
- Booking.com Engineering. (2018). *How Booking.com increases the power of online experiments with CUPED.*
- [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing) — Moro et al. (2014)

---

## 📄 License

MIT — use freely for personal and commercial projects.
