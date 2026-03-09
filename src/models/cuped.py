"""
CUPED: Controlled-experiment Using Pre-Experiment Data
======================================================
Original paper: Deng, Xu, Kohavi, Walker (2013) — Microsoft

Core idea:
  Instead of comparing raw metric Y between treatment and control,
  construct an adjusted metric Y* that has lower variance by removing
  the component explained by a pre-experiment covariate X:

      Y* = Y - θ * X,   where θ = Cov(Y, X) / Var(X)

  Because X is independent of treatment assignment (pre-experiment),
  E[Y*] = E[Y] — the adjustment is unbiased.
  But Var(Y*) < Var(Y) when X is correlated with Y.

  Lower variance → narrower confidence intervals → smaller required sample size
  to detect the same effect at the same power.

This module implements:
  1. Classic CUPED (OLS covariate adjustment)
  2. CUPED with multiple covariates (MultiCUPED via regression residuals)
  3. Variance reduction and sample size calculations
  4. Bootstrap confidence intervals for both raw and adjusted ATE
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Classic CUPED
# ─────────────────────────────────────────────────────────────────────────────

class CUPED:
    """
    Classic single-covariate CUPED variance reduction.

    Usage:
        cuped = CUPED()
        cuped.fit(X_pre, Y)          # estimate θ on control group (or full pre-period)
        Y_adj = cuped.transform(Y, X_pre)
        result = cuped.analyze(Y_adj, treatment)
    """

    def __init__(self):
        self.theta: Optional[float] = None
        self.x_mean: Optional[float] = None
        self.variance_reduction: Optional[float] = None

    def fit(self, X_pre: np.ndarray, Y: np.ndarray) -> "CUPED":
        """
        Estimate θ = Cov(Y, X) / Var(X).

        Best practice: fit on control group only to avoid contamination
        from treatment effect leaking into θ.
        """
        X_pre = np.asarray(X_pre, dtype=float)
        Y = np.asarray(Y, dtype=float)

        cov_matrix = np.cov(Y, X_pre)
        self.theta = cov_matrix[0, 1] / cov_matrix[1, 1]
        self.x_mean = X_pre.mean()

        # Theoretical variance reduction
        corr = np.corrcoef(Y, X_pre)[0, 1]
        self.variance_reduction = corr ** 2  # = 1 - Var(Y*)/Var(Y)

        logger.info(
            f"CUPED θ={self.theta:.4f} | "
            f"Corr(Y,X)={corr:.4f} | "
            f"Variance reduction: {self.variance_reduction:.1%}"
        )
        return self

    def transform(self, Y: np.ndarray, X_pre: np.ndarray) -> np.ndarray:
        """Apply CUPED adjustment: Y* = Y - θ * (X - E[X])"""
        if self.theta is None:
            raise RuntimeError("Must call fit() before transform().")
        Y = np.asarray(Y, dtype=float)
        X_pre = np.asarray(X_pre, dtype=float)
        return Y - self.theta * (X_pre - self.x_mean)

    def analyze(
        self,
        Y_adj: np.ndarray,
        treatment: np.ndarray,
        alpha: float = 0.05,
    ) -> dict:
        """
        Two-sample t-test on CUPED-adjusted metric.
        Returns ATE estimate, CI, p-value, and standard error.
        """
        Y_adj = np.asarray(Y_adj, dtype=float)
        treatment = np.asarray(treatment, dtype=int)

        treat = Y_adj[treatment == 1]
        ctrl = Y_adj[treatment == 0]

        ate = treat.mean() - ctrl.mean()
        se = np.sqrt(treat.var(ddof=1) / len(treat) + ctrl.var(ddof=1) / len(ctrl))
        t_stat, p_value = stats.ttest_ind(treat, ctrl, equal_var=False)
        df = len(treat) + len(ctrl) - 2
        t_crit = stats.t.ppf(1 - alpha / 2, df=df)
        ci_lower = ate - t_crit * se
        ci_upper = ate + t_crit * se

        return {
            "ate": ate,
            "se": se,
            "t_stat": t_stat,
            "p_value": p_value,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "significant": p_value < alpha,
            "alpha": alpha,
            "n_treat": len(treat),
            "n_ctrl": len(ctrl),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Covariate CUPED (regression residual method)
# ─────────────────────────────────────────────────────────────────────────────

class MultiCUPED:
    """
    CUPED with multiple pre-experiment covariates using OLS regression.

    Y* = Y - ŷ_regression + mean(Y)

    Fits a linear model predicting Y from covariates using CV predictions
    (to avoid overfitting bias), then uses the residuals as the adjusted metric.
    """

    def __init__(self, cv_folds: int = 5):
        self.cv_folds = cv_folds
        self.model = LinearRegression()
        self.y_mean: Optional[float] = None
        self.variance_reduction: Optional[float] = None
        self.feature_names: Optional[List[str]] = None

    def fit_transform(
        self,
        Y: np.ndarray,
        X_covariates: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Fit regression on covariates and return residual-adjusted Y.
        Uses cross-validated predictions to avoid in-sample bias.
        """
        Y = np.asarray(Y, dtype=float)
        X_covariates = np.asarray(X_covariates, dtype=float)
        self.feature_names = feature_names or [f"X{i}" for i in range(X_covariates.shape[1])]
        self.y_mean = Y.mean()

        # CV predictions to avoid overfitting
        y_hat = cross_val_predict(self.model, X_covariates, Y, cv=self.cv_folds)

        # Refit on full data for coefficient inspection
        self.model.fit(X_covariates, Y)

        Y_adj = Y - y_hat + self.y_mean
        self.variance_reduction = 1 - Y_adj.var() / Y.var()

        logger.info(
            f"MultiCUPED with {X_covariates.shape[1]} covariates | "
            f"Variance reduction: {self.variance_reduction:.1%}"
        )
        return Y_adj

    def get_covariate_importance(self) -> pd.DataFrame:
        """Return regression coefficients as feature importance proxy."""
        if not hasattr(self.model, "coef_"):
            raise RuntimeError("Must call fit_transform() first.")
        return pd.DataFrame({
            "feature": self.feature_names,
            "coefficient": self.model.coef_,
            "abs_coefficient": np.abs(self.model.coef_),
        }).sort_values("abs_coefficient", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# Raw (unadjusted) A/B analysis for comparison
# ─────────────────────────────────────────────────────────────────────────────

def analyze_raw(
    Y: np.ndarray,
    treatment: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """Standard two-sample t-test without any variance reduction."""
    Y = np.asarray(Y, dtype=float)
    treatment = np.asarray(treatment, dtype=int)

    treat = Y[treatment == 1]
    ctrl = Y[treatment == 0]

    ate = treat.mean() - ctrl.mean()
    se = np.sqrt(treat.var(ddof=1) / len(treat) + ctrl.var(ddof=1) / len(ctrl))
    t_stat, p_value = stats.ttest_ind(treat, ctrl, equal_var=False)
    df_deg = len(treat) + len(ctrl) - 2
    t_crit = stats.t.ppf(1 - alpha / 2, df=df_deg)
    ci_lower = ate - t_crit * se
    ci_upper = ate + t_crit * se

    return {
        "ate": ate,
        "se": se,
        "t_stat": t_stat,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant": p_value < alpha,
        "alpha": alpha,
        "n_treat": len(treat),
        "n_ctrl": len(ctrl),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sample Size & Power Analysis
# ─────────────────────────────────────────────────────────────────────────────

def required_sample_size(
    baseline_rate: float,
    mde: float,
    variance_reduction: float = 0.0,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict:
    """
    Calculate required sample size per arm for a two-proportion z-test.

    variance_reduction: fraction of variance removed by CUPED (0 to 1).
    Returns sample size with and without CUPED for direct comparison.
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    p1 = baseline_rate
    p2 = baseline_rate + mde
    pooled_var = p1 * (1 - p1) + p2 * (1 - p2)

    # Raw sample size (no variance reduction)
    n_raw = int(np.ceil(pooled_var * (z_alpha + z_beta) ** 2 / mde ** 2))

    # CUPED-adjusted sample size
    n_cuped = int(np.ceil(n_raw * (1 - variance_reduction)))

    return {
        "n_raw": n_raw,
        "n_cuped": n_cuped,
        "reduction_pct": variance_reduction * 100,
        "sample_savings": n_raw - n_cuped,
        "speedup_factor": n_raw / max(n_cuped, 1),
        "mde": mde,
        "baseline_rate": baseline_rate,
        "alpha": alpha,
        "power": power,
    }


def power_curve(
    baseline_rate: float,
    mde_range: np.ndarray,
    n_per_arm: int,
    variance_reduction: float = 0.0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Compute statistical power across a range of MDEs,
    with and without CUPED variance reduction.
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    records = []

    for mde in mde_range:
        p2 = baseline_rate + mde
        pooled_var_raw = baseline_rate * (1 - baseline_rate) + p2 * (1 - p2)
        pooled_var_cuped = pooled_var_raw * (1 - variance_reduction)

        se_raw = np.sqrt(pooled_var_raw / n_per_arm)
        se_cuped = np.sqrt(pooled_var_cuped / n_per_arm)

        power_raw = 1 - stats.norm.cdf(z_alpha - mde / se_raw) + stats.norm.cdf(-z_alpha - mde / se_raw)
        power_cuped = 1 - stats.norm.cdf(z_alpha - mde / se_cuped) + stats.norm.cdf(-z_alpha - mde / se_cuped)

        records.append({
            "mde": mde,
            "power_raw": float(np.clip(power_raw, 0, 1)),
            "power_cuped": float(np.clip(power_cuped, 0, 1)),
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap Confidence Intervals
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(
    Y: np.ndarray,
    treatment: np.ndarray,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> dict:
    """
    Compute bootstrap confidence interval for ATE.
    More robust than parametric CI when distribution is skewed.
    """
    rng = np.random.default_rng(random_state)
    Y = np.asarray(Y, dtype=float)
    treatment = np.asarray(treatment, dtype=int)

    boot_ates = []
    n = len(Y)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        Y_b, T_b = Y[idx], treatment[idx]
        ate_b = Y_b[T_b == 1].mean() - Y_b[T_b == 0].mean()
        boot_ates.append(ate_b)

    boot_ates = np.array(boot_ates)
    observed_ate = Y[treatment == 1].mean() - Y[treatment == 0].mean()

    return {
        "ate": observed_ate,
        "ci_lower": float(np.percentile(boot_ates, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(boot_ates, 100 * (1 - alpha / 2))),
        "se": float(boot_ates.std()),
        "n_bootstrap": n_bootstrap,
    }
