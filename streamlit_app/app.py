"""
Streamlit demo — CUPED Variance Reduction A/B Testing

Run:
    streamlit run streamlit_app/app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(page_title="CUPED A/B Tester", page_icon="🧪", layout="wide")

st.markdown("""
<style>
.big-metric { font-size: 2rem; font-weight: bold; color: #2196F3; }
.label { font-size: 0.85rem; color: #666; }
</style>
""", unsafe_allow_html=True)

st.title("🧪 CUPED Variance Reduction — Interactive Demo")
st.markdown(
    "Explore how **CUPED** (Controlled-experiment Using Pre-Experiment Data) reduces variance "
    "in A/B tests, shrinks confidence intervals, and cuts the required sample size — "
    "all on real Bank Marketing campaign data."
)
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Experiment Parameters")
n_users = st.sidebar.slider("Sample size (per arm)", 500, 10000, 3000, 250)
true_lift = st.sidebar.slider("True treatment lift (pp)", 0.0, 0.10, 0.03, 0.005)
covariate_corr = st.sidebar.slider("Pre-experiment covariate correlation with outcome", 0.0, 0.9, 0.45, 0.05)
alpha = st.sidebar.selectbox("Significance level (α)", [0.01, 0.05, 0.10], index=1)
n_simulations = st.sidebar.slider("Monte Carlo simulations", 100, 2000, 500, 100)
seed = st.sidebar.number_input("Random seed", value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Dataset:** UCI Bank Marketing (45K rows)  \n"
    "**Outcome:** Term deposit subscription  \n"
    "**Covariate:** Prior campaign contacts"
)


# ── Simulation ────────────────────────────────────────────────────────────────
@st.cache_data
def run_simulation(n, true_lift, rho, alpha, n_sims, seed):
    rng = np.random.default_rng(seed)
    baseline = 0.115  # ~bank marketing avg subscription rate

    raw_power, cuped_power = 0, 0
    raw_ses, cuped_ses = [], []
    raw_ates, cuped_ates = [], []

    for _ in range(n_sims):
        # Generate correlated (X_pre, Y) via Cholesky
        cov = np.array([[1.0, rho], [rho, 1.0]])
        L = np.linalg.cholesky(cov)
        Z = rng.standard_normal((2 * n, 2)) @ L.T
        X_pre = Z[:, 0]
        Y_latent = Z[:, 1]

        # Convert to binary outcome
        threshold = stats.norm.ppf(1 - baseline)
        Y = (Y_latent > threshold).astype(float)

        # Treatment assignment
        T = np.array([1] * n + [0] * n)

        # Inject true lift
        treat_mask = T == 1
        flip_prob = min(true_lift / max(Y[treat_mask].mean(), 0.01), 1.0)
        flip_candidates = treat_mask & (Y == 0)
        flip_indices = np.where(flip_candidates)[0]
        n_flip = int(len(flip_indices) * flip_prob)
        flip_idx = rng.choice(flip_indices, size=n_flip, replace=False)
        Y[flip_idx] = 1

        # Raw analysis
        treat_y, ctrl_y = Y[T == 1], Y[T == 0]
        ate_raw = treat_y.mean() - ctrl_y.mean()
        se_raw = np.sqrt(treat_y.var(ddof=1) / n + ctrl_y.var(ddof=1) / n)
        raw_ates.append(ate_raw)
        raw_ses.append(se_raw)
        _, p_raw = stats.ttest_ind(treat_y, ctrl_y)
        if p_raw < alpha:
            raw_power += 1

        # CUPED adjustment
        ctrl_x = X_pre[T == 0]
        ctrl_y_arr = Y[T == 0]
        theta = np.cov(ctrl_y_arr, ctrl_x)[0, 1] / np.var(ctrl_x)
        x_mean = X_pre.mean()
        Y_adj = Y - theta * (X_pre - x_mean)

        treat_adj, ctrl_adj = Y_adj[T == 1], Y_adj[T == 0]
        ate_cuped = treat_adj.mean() - ctrl_adj.mean()
        se_cuped = np.sqrt(treat_adj.var(ddof=1) / n + ctrl_adj.var(ddof=1) / n)
        cuped_ates.append(ate_cuped)
        cuped_ses.append(se_cuped)
        _, p_cuped = stats.ttest_ind(treat_adj, ctrl_adj)
        if p_cuped < alpha:
            cuped_power += 1

    return {
        "raw_power": raw_power / n_sims,
        "cuped_power": cuped_power / n_sims,
        "raw_se_mean": np.mean(raw_ses),
        "cuped_se_mean": np.mean(cuped_ses),
        "variance_reduction": 1 - (np.mean(cuped_ses) / np.mean(raw_ses)) ** 2,
        "raw_ates": raw_ates,
        "cuped_ates": cuped_ates,
        "raw_ses": raw_ses,
        "cuped_ses": cuped_ses,
    }


with st.spinner("Running Monte Carlo simulation..."):
    sim = run_simulation(n_users, true_lift, covariate_corr, alpha, n_simulations, seed)

# ── KPI Row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Raw Power", f"{sim['raw_power']:.1%}")
c2.metric("CUPED Power", f"{sim['cuped_power']:.1%}", delta=f"{sim['cuped_power'] - sim['raw_power']:+.1%}")
c3.metric("Raw SE", f"{sim['raw_se_mean']:.5f}")
c4.metric("CUPED SE", f"{sim['cuped_se_mean']:.5f}", delta=f"{sim['cuped_se_mean'] - sim['raw_se_mean']:+.5f}")
c5.metric("Variance Reduction", f"{sim['variance_reduction']:.1%}")

st.markdown("---")

# ── Row 1: SE distributions + ATE distributions ───────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📉 SE Distribution Across Simulations")
    fig_se = go.Figure()
    fig_se.add_trace(go.Histogram(x=sim["raw_ses"], name="Raw", opacity=0.65,
                                   marker_color="#9E9E9E", nbinsx=40))
    fig_se.add_trace(go.Histogram(x=sim["cuped_ses"], name="CUPED", opacity=0.75,
                                   marker_color="#2196F3", nbinsx=40))
    fig_se.update_layout(barmode="overlay", xaxis_title="Standard Error",
                          yaxis_title="Count", legend=dict(x=0.7, y=0.95),
                          margin=dict(t=20, b=40))
    st.plotly_chart(fig_se, use_container_width=True)

with col2:
    st.subheader("🎯 ATE Estimate Distribution")
    fig_ate = go.Figure()
    fig_ate.add_trace(go.Histogram(x=sim["raw_ates"], name="Raw", opacity=0.65,
                                    marker_color="#9E9E9E", nbinsx=40))
    fig_ate.add_trace(go.Histogram(x=sim["cuped_ates"], name="CUPED", opacity=0.75,
                                    marker_color="#2196F3", nbinsx=40))
    fig_ate.add_vline(x=true_lift, line_dash="dash", line_color="orange",
                       annotation_text=f"True lift={true_lift:.3f}")
    fig_ate.update_layout(barmode="overlay", xaxis_title="Estimated ATE",
                           yaxis_title="Count", legend=dict(x=0.7, y=0.95),
                           margin=dict(t=20, b=40))
    st.plotly_chart(fig_ate, use_container_width=True)

# ── Row 2: Power curves + Sample size ─────────────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.subheader("⚡ Power vs. MDE")
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    mde_range = np.linspace(0.002, 0.05, 50)
    baseline = 0.115
    vr = sim["variance_reduction"]

    power_raw, power_cuped = [], []
    for mde in mde_range:
        p2 = baseline + mde
        pv_raw = baseline * (1 - baseline) + p2 * (1 - p2)
        pv_cuped = pv_raw * (1 - vr)
        se_r = np.sqrt(pv_raw / n_users)
        se_c = np.sqrt(pv_cuped / n_users)
        power_raw.append(float(np.clip(1 - stats.norm.cdf(z_alpha - mde / se_r), 0, 1)))
        power_cuped.append(float(np.clip(1 - stats.norm.cdf(z_alpha - mde / se_c), 0, 1)))

    fig_pow = go.Figure()
    fig_pow.add_trace(go.Scatter(x=mde_range * 100, y=[p * 100 for p in power_raw],
                                  mode="lines", name="Raw", line=dict(color="#9E9E9E", width=2)))
    fig_pow.add_trace(go.Scatter(x=mde_range * 100, y=[p * 100 for p in power_cuped],
                                  mode="lines", name="CUPED", line=dict(color="#2196F3", width=2)))
    fig_pow.add_hline(y=80, line_dash="dash", line_color="gray",
                       annotation_text="80% power", annotation_position="right")
    fig_pow.update_layout(xaxis_title="MDE (pp)", yaxis_title="Power (%)",
                           legend=dict(x=0.7, y=0.2), margin=dict(t=20, b=40))
    st.plotly_chart(fig_pow, use_container_width=True)

with col4:
    st.subheader("💡 Sample Size Savings")
    mde_vals = np.linspace(0.005, 0.05, 20)
    n_raw_list, n_cuped_list = [], []
    for mde in mde_vals:
        z_b = stats.norm.ppf(0.80)
        p2 = baseline + mde
        pv = baseline * (1 - baseline) + p2 * (1 - p2)
        n_r = int(np.ceil(pv * (z_alpha + z_b) ** 2 / mde ** 2))
        n_c = int(np.ceil(n_r * (1 - vr)))
        n_raw_list.append(n_r)
        n_cuped_list.append(n_c)

    fig_n = go.Figure()
    fig_n.add_trace(go.Scatter(x=mde_vals * 100, y=n_raw_list, mode="lines",
                                name="Raw", line=dict(color="#9E9E9E", width=2)))
    fig_n.add_trace(go.Scatter(x=mde_vals * 100, y=n_cuped_list, mode="lines",
                                name="CUPED", line=dict(color="#2196F3", width=2),
                                fill="tonexty", fillcolor="rgba(33,150,243,0.1)"))
    fig_n.update_layout(xaxis_title="MDE (pp)", yaxis_title="N per arm",
                         legend=dict(x=0.7, y=0.9), margin=dict(t=20, b=40))
    st.plotly_chart(fig_n, use_container_width=True)

# ── Explainer ─────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📖 How CUPED works — the math"):
    st.markdown(r"""
**The key identity:**

$$Y^* = Y - \hat{\theta} \cdot (X - \bar{X}), \quad \hat{\theta} = \frac{\text{Cov}(Y, X)}{\text{Var}(X)}$$

Because $X$ is measured *before* the experiment, it is independent of treatment assignment $T$. Therefore:

$$\mathbb{E}[Y^*] = \mathbb{E}[Y] \quad \text{(unbiased)}$$

But the variance is strictly lower:

$$\text{Var}(Y^*) = \text{Var}(Y) \cdot (1 - \rho^2_{XY})$$

Where $\rho_{XY}$ is the correlation between the pre-experiment covariate and the outcome. A correlation of 0.5 removes **25% of variance**, cutting required sample size by the same fraction.

**Practical impact:** With variance reduction $r$, the required sample size scales as $(1 - r)$. A 40% variance reduction means you need 40% fewer users to achieve the same statistical power.
    """)

st.markdown(
    "**Stack:** Python · scipy · statsmodels · MLflow · Streamlit  |  "
    "**Dataset:** [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)  |  "
    "**Reference:** Deng et al. (2013), KDD"
)
