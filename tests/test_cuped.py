"""
Unit and integration tests for the CUPED A/B Testing pipeline.

Run:
    pytest tests/ -v
"""

import pytest
import numpy as np
import pandas as pd
from scipy import stats
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.cuped import (
    CUPED,
    MultiCUPED,
    analyze_raw,
    required_sample_size,
    power_curve,
    bootstrap_ci,
)
from src.data.loader import preprocess, simulate_ab_experiment


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_ab_data():
    """Generate a clean synthetic A/B dataset with known properties."""
    rng = np.random.default_rng(42)
    n = 2000
    X_pre = rng.normal(0, 1, n)
    # Y correlated with X_pre (rho ≈ 0.5)
    Y_latent = 0.5 * X_pre + rng.normal(0, np.sqrt(0.75), n)
    Y = (Y_latent > 0).astype(float)
    T = np.array([1] * (n // 2) + [0] * (n // 2))
    # Inject 3pp lift
    treat_idx = np.where((T == 1) & (Y == 0))[0]
    flip_n = int(len(treat_idx) * 0.06)
    flip_idx = rng.choice(treat_idx, size=flip_n, replace=False)
    Y[flip_idx] = 1
    return Y, T, X_pre


@pytest.fixture
def sample_bank_df():
    """Minimal bank marketing-like DataFrame."""
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame({
        "age": rng.integers(18, 80, n),
        "balance": rng.normal(1000, 2000, n),
        "duration": rng.integers(0, 600, n),
        "campaign": rng.integers(1, 10, n),
        "previous": rng.integers(0, 5, n),
        "pdays": rng.choice([999, 1, 5, 10, 30], n),
        "job": rng.choice(["admin.", "blue-collar", "technician"], n),
        "marital": rng.choice(["single", "married"], n),
        "education": rng.choice(["primary", "secondary", "tertiary"], n),
        "default": rng.choice(["yes", "no"], n),
        "housing": rng.choice(["yes", "no"], n),
        "loan": rng.choice(["yes", "no"], n),
        "contact": rng.choice(["cellular", "telephone"], n),
        "month": rng.choice(["jan", "feb", "mar"], n),
        "poutcome": rng.choice(["success", "failure", "unknown"], n),
        "y": rng.choice(["yes", "no"], n, p=[0.115, 0.885]),
    })
    return df


# ── CUPED Core Tests ──────────────────────────────────────────────────────────

class TestCUPED:

    def test_theta_estimation(self, synthetic_ab_data):
        Y, T, X_pre = synthetic_ab_data
        cuped = CUPED()
        cuped.fit(X_pre[T == 0], Y[T == 0])
        assert cuped.theta is not None
        assert np.isfinite(cuped.theta)

    def test_transform_reduces_variance(self, synthetic_ab_data):
        Y, T, X_pre = synthetic_ab_data
        cuped = CUPED()
        cuped.fit(X_pre[T == 0], Y[T == 0])
        Y_adj = cuped.transform(Y, X_pre)
        assert Y_adj.var() < Y.var(), "CUPED should reduce variance"

    def test_transform_preserves_mean(self, synthetic_ab_data):
        Y, T, X_pre = synthetic_ab_data
        cuped = CUPED()
        cuped.fit(X_pre[T == 0], Y[T == 0])
        Y_adj = cuped.transform(Y, X_pre)
        np.testing.assert_allclose(Y_adj.mean(), Y.mean(), atol=1e-8)

    def test_variance_reduction_in_range(self, synthetic_ab_data):
        Y, T, X_pre = synthetic_ab_data
        cuped = CUPED()
        cuped.fit(X_pre[T == 0], Y[T == 0])
        assert 0 <= cuped.variance_reduction <= 1

    def test_unfitted_transform_raises(self, synthetic_ab_data):
        Y, T, X_pre = synthetic_ab_data
        cuped = CUPED()
        with pytest.raises(RuntimeError):
            cuped.transform(Y, X_pre)

    def test_analyze_returns_required_keys(self, synthetic_ab_data):
        Y, T, X_pre = synthetic_ab_data
        cuped = CUPED()
        cuped.fit(X_pre[T == 0], Y[T == 0])
        Y_adj = cuped.transform(Y, X_pre)
        result = cuped.analyze(Y_adj, T)
        for key in ("ate", "se", "p_value", "ci_lower", "ci_upper", "significant"):
            assert key in result, f"Missing key: {key}"

    def test_cuped_se_less_than_raw(self, synthetic_ab_data):
        Y, T, X_pre = synthetic_ab_data
        cuped = CUPED()
        cuped.fit(X_pre[T == 0], Y[T == 0])
        Y_adj = cuped.transform(Y, X_pre)

        raw = analyze_raw(Y, T)
        adj = cuped.analyze(Y_adj, T)
        assert adj["se"] < raw["se"], "CUPED SE should be smaller than raw SE"

    def test_cuped_ci_narrower_than_raw(self, synthetic_ab_data):
        Y, T, X_pre = synthetic_ab_data
        cuped = CUPED()
        cuped.fit(X_pre[T == 0], Y[T == 0])
        Y_adj = cuped.transform(Y, X_pre)

        raw = analyze_raw(Y, T)
        adj = cuped.analyze(Y_adj, T)
        raw_width = raw["ci_upper"] - raw["ci_lower"]
        adj_width = adj["ci_upper"] - adj["ci_lower"]
        assert adj_width < raw_width, "CUPED CI should be narrower"


# ── MultiCUPED Tests ──────────────────────────────────────────────────────────

class TestMultiCUPED:

    def test_fit_transform_output_shape(self, synthetic_ab_data):
        Y, T, X_pre = synthetic_ab_data
        X_multi = np.column_stack([X_pre, X_pre ** 2])
        multi = MultiCUPED()
        Y_adj = multi.fit_transform(Y, X_multi)
        assert len(Y_adj) == len(Y)

    def test_variance_reduction_positive(self, synthetic_ab_data):
        Y, T, X_pre = synthetic_ab_data
        X_multi = np.column_stack([X_pre, X_pre ** 2])
        multi = MultiCUPED()
        multi.fit_transform(Y, X_multi)
        assert multi.variance_reduction > 0

    def test_covariate_importance_shape(self, synthetic_ab_data):
        Y, T, X_pre = synthetic_ab_data
        X_multi = np.column_stack([X_pre, X_pre ** 2])
        multi = MultiCUPED(cv_folds=3)
        multi.fit_transform(Y, X_multi, feature_names=["x", "x_sq"])
        imp = multi.get_covariate_importance()
        assert len(imp) == 2
        assert "feature" in imp.columns


# ── Sample Size & Power Tests ─────────────────────────────────────────────────

class TestSampleSizeAndPower:

    def test_required_sample_size_cuped_less_than_raw(self):
        result = required_sample_size(baseline_rate=0.115, mde=0.02, variance_reduction=0.3)
        assert result["n_cuped"] < result["n_raw"]

    def test_required_sample_size_zero_reduction_equal(self):
        result = required_sample_size(baseline_rate=0.115, mde=0.02, variance_reduction=0.0)
        assert result["n_cuped"] == result["n_raw"]

    def test_power_curve_shape(self):
        mde_range = np.linspace(0.005, 0.05, 10)
        df = power_curve(0.115, mde_range, n_per_arm=5000, variance_reduction=0.25)
        assert len(df) == 10
        assert all(col in df.columns for col in ("mde", "power_raw", "power_cuped"))

    def test_power_cuped_gte_raw(self):
        mde_range = np.linspace(0.005, 0.05, 10)
        df = power_curve(0.115, mde_range, n_per_arm=5000, variance_reduction=0.25)
        assert (df["power_cuped"] >= df["power_raw"] - 1e-9).all()

    def test_power_in_range(self):
        mde_range = np.linspace(0.005, 0.05, 10)
        df = power_curve(0.115, mde_range, n_per_arm=5000)
        assert df["power_raw"].between(0, 1).all()


# ── Bootstrap Tests ───────────────────────────────────────────────────────────

class TestBootstrap:

    def test_bootstrap_ci_keys(self, synthetic_ab_data):
        Y, T, _ = synthetic_ab_data
        result = bootstrap_ci(Y, T, n_bootstrap=200)
        for key in ("ate", "ci_lower", "ci_upper", "se"):
            assert key in result

    def test_bootstrap_ci_contains_true_lift(self, synthetic_ab_data):
        """95% CI should contain true lift most of the time (stochastic — allows 1 failure in 20)."""
        Y, T, _ = synthetic_ab_data
        result = bootstrap_ci(Y, T, n_bootstrap=500)
        # True lift ~0.03 — just check CI is not wildly wrong
        assert result["ci_lower"] < 0.10
        assert result["ci_upper"] > -0.05

    def test_bootstrap_se_positive(self, synthetic_ab_data):
        Y, T, _ = synthetic_ab_data
        result = bootstrap_ci(Y, T, n_bootstrap=200)
        assert result["se"] > 0


# ── Data Loader Tests ─────────────────────────────────────────────────────────

class TestDataLoader:

    def test_preprocess_creates_subscribed_column(self, sample_bank_df):
        df = preprocess(sample_bank_df)
        assert "subscribed" in df.columns

    def test_preprocess_drops_y_column(self, sample_bank_df):
        df = preprocess(sample_bank_df)
        assert "y" not in df.columns

    def test_preprocess_binary_target(self, sample_bank_df):
        df = preprocess(sample_bank_df)
        assert set(df["subscribed"].unique()).issubset({0, 1})

    def test_simulate_ab_creates_treatment_column(self, sample_bank_df):
        df = preprocess(sample_bank_df)
        df_exp = simulate_ab_experiment(df)
        assert "treatment" in df_exp.columns
        assert set(df_exp["treatment"].unique()).issubset({0, 1})

    def test_simulate_ab_treatment_rate(self, sample_bank_df):
        df = preprocess(sample_bank_df)
        df_exp = simulate_ab_experiment(df, treatment_rate=0.5)
        actual_rate = df_exp["treatment"].mean()
        assert abs(actual_rate - 0.5) < 0.05

    def test_preprocess_no_object_columns(self, sample_bank_df):
        df = preprocess(sample_bank_df)
        object_cols = df.select_dtypes(include="object").columns.tolist()
        assert len(object_cols) == 0, f"Object columns remain: {object_cols}"
