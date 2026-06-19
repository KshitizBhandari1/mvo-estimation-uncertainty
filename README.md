# Mean-Variance Optimization under Estimation Uncertainty (S&P 500)

A walk-forward empirical study of classical Mean-Variance Optimization (MVO), focusing on how estimation uncertainty in expected returns and covariance matrices affects portfolio performance under realistic trading constraints.

The project is implemented as a full backtesting engine with annual rebalancing, transaction costs, and benchmark comparison against SPY and an equal-weight portfolio.

---

## Motivation

Mean-Variance Optimization is theoretically optimal but empirically unstable because it relies on **noisy inputs (expected returns and covariances)**.

In practice, portfolio-construction problems of this type are sensitive to small changes in inputs, often producing unstable weights and concentrated allocations even under simple long-only constraints.

This type of instability is not unique to optimization frameworks. In discretionary and systematic equity strategies (e.g., cross-sectional momentum screens), portfolio performance can be highly regime dependent, with periods of persistent performance followed by sharp reversals after market stress. This motivates examining whether similar sensitivity to estimation noise appears in classical optimization-based strategy.

From a modeling perspective:

* training data -> parameter estimation (μ, Σ)
* model -> optimizer
* output -> portfolio weights
* test -> realized out-of-sample performance of allocation

The goal of this project is to evaluate:

> how much of MVO instability comes from estimation uncertainty, and how much is mitigated by practical regularization techniques.

Regularization methods tested:

* Position constraints (max weight per asset)
* Covariance shrinkage (Ledoit-Wolf)
* Mean shrinkage toward cross-sectional average

---

## Methodology Overview

* Universe: Current S&P 500 constituents (survivorship bias applies)
* Rebalancing: Annual (end of year)
* Holding period: 1 year
* Optimization: Long-only maximum Sharpe ratio
* Benchmarks: SPY and equal-weight portfolio
* Transaction costs: 25 bps proportional to turnover

---

## Key Results Summary

### 1. Effect of Position Constraints (Primary Driver)

#### Sample μ + Sample Σ

| Max Weight | Sharpe | Final Wealth | Effective N |
| ---------- | ------ | ------------ | ----------- |
|        20% |  1.385 |      $ 17.92 |        10.3 |
|        10% |  1.478 |      $ 19.51 |        14.0 |
|         5% |  1.533 |      $ 18.78 |        23.7 |

**Key observation:**

* Reducing max weight from 20% → 5% without implementing any shrinkage

  * Results in higher realized out-of-sample Sharpe: **+10.7% (1.385 → 1.533)**
  * Diversification more than doubles (Eff N: **10.3 → 23.7**)

Main driver of performance stability is NOT statistical shrinkage, but **portfolio constraint structure**.

---

### 2. Covariance Shrinkage (Ledoit-Wolf)

At 20% max weight:

| Model    | Sharpe | Wealth | Eff N | Turnover |
| -------- | ------ | ------ | ----- | -------- |
| Sample Σ |  1.385 | $17.92 |  10.3 |   181.7% |
| Ledoit Σ |  1.430 | $19.62 |  12.5 |   181.2% |

At 10% max weight:

| Model    | Sharpe | Wealth | Eff N | Turnover |
| -------- | ------ | ------ | ----- | -------- |
| Sample Σ |  1.478 | $19.51 |  14.0 |   181.8% |
| Ledoit Σ |  1.500 | $20.83 |  15.2 |   181.6% |

At 5% max weight:

| Model    | Sharpe | Wealth | Eff N | Turnover |
| -------- | ------ | ------ | ----- | -------- |
| Sample Σ |  1.533 | $18.78 |  23.7 |   176.4% |
| Ledoit Σ |  1.534 | $19.26 |  24.2 |   176.5% |

**Key observations:**

* Sharpe improvement more noticeable at higher weight caps: **+0.022 at 10% cap, +0.045 at 20% cap**
* Sharpe impact at 5% cap: **+0.001, effectively neutral**
* Effective diversification improves modestly (**~+0.5 to ~+2 names depending on cap**)
* Turnover remains essentially unchanged (~180%)

> Thus, covariance shrinkage alone modestly improves **portfolio diversification**, but does not address turnover. Moreover, there is limited impact on Sharpe under tight constraints.

---

### 3. Mean Shrinkage (Cross-sectional regularization)

At 20% max weight:

| Model    | Sharpe | Wealth | Eff N | Turnover |
| -------- | ------ | ------ | ----- | -------- |
| Sample μ |  1.385 | $17.92 |  10.3 |   181.7% |
| Shrunk μ |  1.409 | $13.71 |  11.1 |   179.8% |

At 10% max weight:

| Model    | Sharpe | Wealth | Eff N | Turnover |
| -------- | ------ | ------ | ----- | -------- |
| Sample μ |  1.478 | $19.51 |  14.0 |   181.8% |
| Shrunk μ |  1.527 | $15.51 |  14.9 |   178.0% |

At 5% max weight:

| Model    | Sharpe | Wealth | Eff N | Turnover |
| -------- | ------ | ------ | ----- | -------- |
| Sample μ |  1.533 | $18.78 |  23.7 |   176.4% |
| Shrunk μ |  1.610 | $16.16 |  24.4 |   169.5% |

**Key observations:**

* Sharpe increases modestly but consistently:
  * 20% cap: +1.73% or +0.024
  * 10% cap: +3.32% or +0.049
  * 5% cap: +5.02% or +0.077
* Terminal wealth decreases across all cases
* Effective diversification increases but by less than 1 stock
* Greater reduction in turnover compared to covariance shrinkage, but still negligible for such a high turnover.

> Changes in expected return estimates (via shrinkage) have a stronger and more variable impact on risk-adjusted performance than covariance shrinkage, but introduces a systematic return compression effect by reducing exposure to high-performing assets.

* Shrinking μ improves risk-adjusted performance
* But reduces exposure to extreme return winners

---

### 4. Combined Effect (Best Configurations)

At 20% cap:

| Configuration         | Sharpe | Wealth | Eff N | Turnover |
| --------------------- | ------ | ------ | ----- | -------- |
| Sample μ + Sample Σ   |  1.385 | $17.92 |  10.3 |   181.7% |
| Sample μ + Ledoit Σ   |  1.430 | $19.62 |  12.5 |   181.2% |
| Shrunk μ + Sample Σ   |  1.409 | $13.71 |  11.1 |   179.8% |
| Shrunk μ + Ledoit Σ   |  1.475 | $15.67 |  14.1 |   178.8% |

At 10% cap:

| Configuration         | Sharpe | Wealth | Eff N | Turnover |
| --------------------- | ------ | ------ | ----- | -------- |
| Sample μ + Sample Σ   |  1.478 | $19.51 |  14.0 |   181.8% |
| Sample μ + Ledoit Σ   |  1.500 | **$20.83** |  15.2 |   181.6% |
| Shrunk μ + Sample Σ   |  1.527 | $15.51 |  14.9 |   178.0% |
| Shrunk μ + Ledoit Σ   |  1.546 | $16.45 |  16.7 |   177.7% |

At 5% cap:

| Configuration       | Sharpe | Wealth | Eff N | Turnover |
| ------------------- | ------ | ------ | ----- | -------- |
| Sample μ + Sample Σ |  1.533 | $18.78 |  23.7 |   176.4% |
| Sample μ + Ledoit Σ |  1.534 | $19.26 |  24.2 |   176.5% |
| Shrunk μ + Sample Σ |  **1.610** | $16.16 |  24.4 |   169.5% |
| Shrunk μ + Ledoit Σ |  1.604 | $16.36 |  25.6 |   169.5% |

**Key observation:**

* Best Sharpe: **1.610 (Shrunk μ + Sample Σ, 5% cap)**
* Best wealth: **$20.83 (Sample μ + Ledoit Σ, 10% cap)**

There is a persistent trade-off between:
* risk-adjusted returns (Sharpe)
* absolute performance (terminal wealth)

Moreover, best Sharpe is observed at 5% cap with shrunk mean and sample covariance, further supporting the fact that weight caps contribute to the strongest regularization.

---

### 5. Structural Insight: MVO behaves like a high-variance estimator

Across all configurations:

* Turnover remains extremely high: **~170-182% annually**
* Quite a few assets repeatedly hit max weight constraints
* Small changes in μ significantly change portfolio composition

Interpretation:

> MVO behaves like a high-variance estimator where expected returns play a disproportionate role relative to covariance structure under long-only constraints.

---

## Core Takeaways

1. **Position constraints are the strongest regularizer**

   * Moving from a loose 20% cap to a tighter 5% cap yields a **+7.3% to +14.3%** relative improvement in out-of-sample Sharpe ratio across all estimation frameworks.

2. **Mean estimation error dominates covariance error**

   * Regularizing expected returns via cross-sectional shrinkage systematically improves out-of-sample Sharpe ratios across all asset constraints. 
   * However, it forces a clear trade-off: shrinkage in expected returns reduces exposure to massive historical tail winners, reducing terminal wealth.

3. **Covariance shrinkage improves stability, not turnover**

   * Ledoit-Wolf shrinkage only offers slight absolute Sharpe gains from +0.001 (5% cap) to 0.045 (20% cap).
   * It expands effective diversification ($N$) by roughly 1 to 2 assets, but fails to noticeably change portfolio turnover or trading behavior.

4. **MVO exhibits classic overfitting properties without structural guardrails**

   * Across every configuration, annual turnover remains exceptionally high (~170-182%)
   * This proves that long-only mean-variance optimizers act like high-variance estimators, creating significant portfolio redesigns at each rebalance based on small shifts in underlying parameters.

---

## Limitations

* Survivorship bias (current S&P 500 constituents only)
* Annual estimation windows only
* No factor model for expected returns
* Simplified transaction cost model
* No regime-dependent macro features
