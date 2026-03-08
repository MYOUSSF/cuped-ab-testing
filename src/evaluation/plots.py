"""
Evaluation plots and reporting for CUPED A/B testing project.

Produces:
  1. Variance reduction comparison (raw vs CUPED vs MultiCUPED)
  2. Confidence interval comparison plot
  3. Power curves (raw vs CUPED across MDEs)
  4. Sample size savings chart
  5. Covariate correlation heatmap
  6. Bootstrap distribution of ATE
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Optional
import os
import logging

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)

COLORS = {
    "raw": "#9E9E9E",
    "cuped": "#2196F3",
    "multi_cuped": "#4CAF50",
    "significant": "#4CAF50",
    "not_significant": "#F44336",
    "neutral": "#607D8B",
}


def _save(fig, path: Optional[str]) -> None:
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Variance Reduction Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_variance_reduction(
    results: dict,  # {method_name: {"se": float, "variance_reduction": float}}
    save_path: Optional[str] = None,
):
    """Compare SE and variance reduction across methods."""
    methods = list(results.keys())
    ses = [results[m]["se"] for m in methods]
    var_reds = [results[m].get("variance_reduction", 0.0) * 100 for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # SE comparison
    colors = [COLORS.get(m.lower().replace(" ", "_").replace("-", "_"), "#607D8B") for m in methods]
    bars = axes[0].bar(methods, ses, color=colors, edgecolor="white", width=0.5)
    axes[0].set_title("Standard Error of ATE Estimate", fontweight="bold")
    axes[0].set_ylabel("Standard Error")
    for bar, val in zip(bars, ses):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0001,
                     f"{val:.5f}", ha="center", va="bottom", fontsize=9)

    # Variance reduction %
    bars2 = axes[1].bar(methods, var_reds, color=colors, edgecolor="white", width=0.5)
    axes[1].set_title("Variance Reduction vs. Raw", fontweight="bold")
    axes[1].set_ylabel("Variance Reduction (%)")
    axes[1].set_ylim(0, max(var_reds) * 1.3 + 1)
    for bar, val in zip(bars2, var_reds):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Variance Reduction: Raw vs CUPED vs MultiCUPED", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. Confidence Interval Comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_confidence_intervals(
    results: dict,  # {method_name: {"ate": float, "ci_lower": float, "ci_upper": float, "significant": bool}}
    true_lift: Optional[float] = None,
    save_path: Optional[str] = None,
):
    """Forest plot of ATE estimates and CIs across methods."""
    methods = list(results.keys())
    ates = [results[m]["ate"] for m in methods]
    ci_lowers = [results[m]["ci_lower"] for m in methods]
    ci_uppers = [results[m]["ci_upper"] for m in methods]
    significants = [results[m].get("significant", False) for m in methods]

    fig, ax = plt.subplots(figsize=(9, 4))
    y_pos = range(len(methods))

    for i, (method, ate, lo, hi, sig) in enumerate(zip(methods, ates, ci_lowers, ci_uppers, significants)):
        color = COLORS["significant"] if sig else COLORS["not_significant"]
        ax.plot([lo, hi], [i, i], color=color, linewidth=2.5, zorder=2)
        ax.scatter([ate], [i], color=color, zorder=3, s=80)
        ci_width = hi - lo
        ax.text(hi + 0.0005, i, f"  [{lo:.4f}, {hi:.4f}]  width={ci_width:.4f}",
                va="center", fontsize=8.5, color=color)

    ax.axvline(0, color="black", linewidth=1, linestyle="--", label="No effect")
    if true_lift is not None:
        ax.axvline(true_lift, color="orange", linewidth=1.5, linestyle="-.", label=f"True lift ({true_lift:.3f})")

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(methods, fontsize=10)
    ax.set_xlabel("Average Treatment Effect (ATE)")
    ax.set_title("Confidence Intervals by Method\n(green = significant at α=0.05, red = not significant)",
                 fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Power Curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_power_curves(
    power_df: pd.DataFrame,
    n_per_arm: int,
    save_path: Optional[str] = None,
):
    """Power curves: raw vs CUPED across a range of MDEs."""
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(power_df["mde"] * 100, power_df["power_raw"] * 100,
            color=COLORS["raw"], linewidth=2, label="Raw (no CUPED)", marker="o", markersize=4)
    ax.plot(power_df["mde"] * 100, power_df["power_cuped"] * 100,
            color=COLORS["cuped"], linewidth=2, label="CUPED-adjusted", marker="s", markersize=4)

    ax.axhline(80, color="gray", linestyle="--", linewidth=1, label="80% power threshold")
    ax.fill_between(power_df["mde"] * 100, power_df["power_raw"] * 100,
                    power_df["power_cuped"] * 100, alpha=0.12, color=COLORS["cuped"],
                    label="CUPED power gain")

    ax.set_title(f"Statistical Power vs. MDE\n(n={n_per_arm:,} per arm)", fontweight="bold")
    ax.set_xlabel("Minimum Detectable Effect (pp)")
    ax.set_ylabel("Power (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. Sample Size Savings
# ─────────────────────────────────────────────────────────────────────────────

def plot_sample_size_savings(
    mde_range: np.ndarray,
    baseline_rate: float,
    variance_reductions: dict,  # {label: variance_reduction_fraction}
    alpha: float = 0.05,
    power: float = 0.80,
    save_path: Optional[str] = None,
):
    """Show sample size required per arm at different MDEs and variance reduction levels."""
    from src.models.cuped import required_sample_size

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: absolute sample sizes
    for label, vr in variance_reductions.items():
        ns = [required_sample_size(baseline_rate, mde, vr, alpha, power)["n_cuped"] for mde in mde_range]
        color = COLORS.get(label.lower().replace(" ", "_").replace("-", "_"), "#607D8B")
        axes[0].plot(mde_range * 100, ns, label=label, linewidth=2, color=color)

    axes[0].set_title("Required Sample Size per Arm", fontweight="bold")
    axes[0].set_xlabel("MDE (pp)")
    axes[0].set_ylabel("N per arm")
    axes[0].legend(fontsize=9)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Right: % savings vs raw
    base_ns = [required_sample_size(baseline_rate, mde, 0.0, alpha, power)["n_raw"] for mde in mde_range]
    for label, vr in variance_reductions.items():
        if vr == 0:
            continue
        ns_cuped = [required_sample_size(baseline_rate, mde, vr, alpha, power)["n_cuped"] for mde in mde_range]
        savings_pct = [(1 - nc / nr) * 100 for nc, nr in zip(ns_cuped, base_ns)]
        color = COLORS.get(label.lower().replace(" ", "_").replace("-", "_"), "#607D8B")
        axes[1].plot(mde_range * 100, savings_pct, label=label, linewidth=2, color=color)

    axes[1].set_title("Sample Size Savings vs. Raw (%)", fontweight="bold")
    axes[1].set_xlabel("MDE (pp)")
    axes[1].set_ylabel("Sample size reduction (%)")
    axes[1].legend(fontsize=9)

    plt.suptitle("CUPED Impact on Required Sample Size", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. Covariate Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_covariate_correlations(
    df: pd.DataFrame,
    target_col: str = "subscribed",
    top_n: int = 15,
    save_path: Optional[str] = None,
):
    """Heatmap of top feature correlations with the target metric."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr_with_target = numeric_df.corr()[target_col].drop(target_col).abs().sort_values(ascending=False)
    top_features = corr_with_target.head(top_n).index.tolist()

    corr_matrix = numeric_df[top_features + [target_col]].corr()

    fig, ax = plt.subplots(figsize=(11, 8))
    mask = np.zeros_like(corr_matrix, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True

    sns.heatmap(
        corr_matrix,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        annot=True, fmt=".2f",
        linewidths=0.5,
        annot_kws={"size": 7},
    )
    ax.set_title(f"Feature Correlations (top {top_n} correlated with '{target_col}')", fontweight="bold")
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. Bootstrap ATE Distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_bootstrap_distribution(
    boot_results: dict,
    method_label: str = "CUPED",
    true_lift: Optional[float] = None,
    save_path: Optional[str] = None,
):
    """Visualise bootstrap distribution of ATE with CI shading."""
    # Re-run bootstrap to get distribution (pass boot_ates directly if available)
    fig, ax = plt.subplots(figsize=(9, 4))

    ate = boot_results["ate"]
    ci_lower = boot_results["ci_lower"]
    ci_upper = boot_results["ci_upper"]
    se = boot_results["se"]

    # Simulate distribution from CI (approximate)
    x = np.linspace(ate - 4 * se, ate + 4 * se, 300)
    y = (1 / (se * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - ate) / se) ** 2)

    ax.plot(x, y, color=COLORS["cuped"], linewidth=2)
    ax.fill_between(x, y, where=(x >= ci_lower) & (x <= ci_upper),
                    alpha=0.25, color=COLORS["cuped"], label=f"95% CI [{ci_lower:.4f}, {ci_upper:.4f}]")
    ax.axvline(ate, color=COLORS["cuped"], linestyle="-", linewidth=1.5, label=f"ATE = {ate:.4f}")
    ax.axvline(0, color="black", linestyle="--", linewidth=1, label="No effect")

    if true_lift is not None:
        ax.axvline(true_lift, color="orange", linestyle="-.", linewidth=1.5, label=f"True lift = {true_lift:.4f}")

    ax.set_title(f"Bootstrap Distribution of ATE — {method_label}", fontweight="bold")
    ax.set_xlabel("Average Treatment Effect")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Summary Table
# ─────────────────────────────────────────────────────────────────────────────

def print_results_table(results: dict, variance_reductions: dict = None) -> None:
    """Pretty-print comparison table of all methods."""
    print("\n" + "═" * 80)
    print(f"  {'METHOD':<20} {'ATE':>9} {'SE':>9} {'CI LOWER':>10} {'CI UPPER':>10} {'p-value':>9} {'SIG':>5}")
    print("═" * 80)
    for method, r in results.items():
        sig_marker = "✓" if r.get("significant") else "✗"
        sig_color = "\033[92m" if r.get("significant") else "\033[91m"
        print(
            f"  {method:<20} {r['ate']:>9.5f} {r['se']:>9.5f} "
            f"{r['ci_lower']:>10.5f} {r['ci_upper']:>10.5f} "
            f"{r.get('p_value', float('nan')):>9.4f} {sig_color}{sig_marker}\033[0m"
        )
    print("═" * 80)

    if variance_reductions:
        print("\n  VARIANCE REDUCTION SUMMARY:")
        for method, vr in variance_reductions.items():
            print(f"    {method:<20}: {vr * 100:.1f}% variance removed")
    print()
