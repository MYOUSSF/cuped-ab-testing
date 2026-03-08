"""
Main analysis pipeline for CUPED Variance Reduction on Bank Marketing data.

Run:
    python analyze.py --help

Example:
    python analyze.py --true-lift 0.03 --n-bootstrap 2000
"""

import argparse
import logging
import os
import sys
import numpy as np
import pandas as pd
import mlflow

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import (
    load_bank_marketing,
    preprocess,
    simulate_ab_experiment,
    get_pre_experiment_covariate,
)
from src.models.cuped import (
    CUPED,
    MultiCUPED,
    analyze_raw,
    required_sample_size,
    power_curve,
    bootstrap_ci,
)
from src.evaluation.plots import (
    plot_variance_reduction,
    plot_confidence_intervals,
    plot_power_curves,
    plot_sample_size_savings,
    plot_covariate_correlations,
    plot_bootstrap_distribution,
    print_results_table,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="CUPED A/B testing analysis pipeline.")
    parser.add_argument("--output-dir", default="results", help="Output directory for plots and results")
    parser.add_argument("--true-lift", type=float, default=0.03, help="Simulated true treatment lift")
    parser.add_argument("--treatment-rate", type=float, default=0.5, help="Fraction of users in treatment")
    parser.add_argument("--n-bootstrap", type=int, default=2000, help="Bootstrap iterations for CI")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    parser.add_argument("--experiment-name", default="cuped_ab_testing")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name="cuped_full_analysis"):
        mlflow.log_params(vars(args))

        # ── 1. Load & Preprocess ──────────────────────────────────────────────
        logger.info("Step 1/6: Loading and preprocessing data...")
        raw_df = load_bank_marketing()
        df_processed = preprocess(raw_df)

        # ── 2. Simulate A/B Experiment ────────────────────────────────────────
        logger.info("Step 2/6: Simulating A/B experiment...")
        df_exp = simulate_ab_experiment(
            df_processed,
            treatment_rate=args.treatment_rate,
            true_lift=args.true_lift,
        )

        Y = df_exp["subscribed"].values
        T = df_exp["treatment"].values
        baseline_cr = Y[T == 0].mean()
        logger.info(f"  Baseline conversion rate: {baseline_cr:.2%}")
        mlflow.log_metric("baseline_conversion_rate", baseline_cr)

        # ── 3. Covariate Correlation Analysis ─────────────────────────────────
        logger.info("Step 3/6: Analyzing covariate correlations...")
        plot_covariate_correlations(
            df_exp,
            target_col="subscribed",
            save_path=os.path.join(plots_dir, "covariate_correlations.png"),
        )

        # ── 4. CUPED Analysis ─────────────────────────────────────────────────
        logger.info("Step 4/6: Running CUPED analysis...")

        # Single-covariate CUPED (pre-experiment contact count)
        X_pre = raw_df["previous"].values  # raw covariate before one-hot encoding

        cuped = CUPED()
        ctrl_mask = T == 0
        cuped.fit(X_pre[ctrl_mask], Y[ctrl_mask])  # fit on control only
        Y_cuped = cuped.transform(Y, X_pre)

        # Multi-covariate CUPED
        covariate_cols = [c for c in df_exp.columns if c not in ("subscribed", "treatment")]
        X_multi = df_exp[covariate_cols].values
        multi_cuped = MultiCUPED(cv_folds=5)
        Y_multi_cuped = multi_cuped.fit_transform(Y, X_multi, feature_names=covariate_cols)

        # ── 5. Statistical Analysis & Comparisons ─────────────────────────────
        logger.info("Step 5/6: Statistical analysis and comparisons...")

        result_raw = analyze_raw(Y, T, alpha=args.alpha)
        result_cuped = CUPED().fit(X_pre[ctrl_mask], Y[ctrl_mask]).analyze(
            CUPED().fit(X_pre[ctrl_mask], Y[ctrl_mask]).transform(Y, X_pre), T, alpha=args.alpha
        )
        # Re-fit cleanly for reporting
        cuped2 = CUPED()
        cuped2.fit(X_pre[ctrl_mask], Y[ctrl_mask])
        Y_cuped2 = cuped2.transform(Y, X_pre)
        result_cuped = cuped2.analyze(Y_cuped2, T, alpha=args.alpha)

        result_multi = analyze_raw(Y_multi_cuped, T, alpha=args.alpha)

        all_results = {
            "Raw": result_raw,
            "CUPED": result_cuped,
            "MultiCUPED": result_multi,
        }

        variance_reductions = {
            "Raw": 0.0,
            "CUPED": cuped2.variance_reduction,
            "MultiCUPED": multi_cuped.variance_reduction,
        }

        # Augment results with variance reduction for plotting
        for method, vr in variance_reductions.items():
            all_results[method]["variance_reduction"] = vr

        print_results_table(all_results, variance_reductions)

        # Log to MLflow
        for method, r in all_results.items():
            mlflow.log_metrics({
                f"{method}_ate": r["ate"],
                f"{method}_se": r["se"],
                f"{method}_p_value": r.get("p_value", float("nan")),
                f"{method}_variance_reduction": r.get("variance_reduction", 0.0),
            })

        # ── 6. Generate All Plots ──────────────────────────────────────────────
        logger.info("Step 6/6: Generating plots...")

        plot_variance_reduction(
            all_results,
            save_path=os.path.join(plots_dir, "variance_reduction.png"),
        )

        plot_confidence_intervals(
            all_results,
            true_lift=args.true_lift,
            save_path=os.path.join(plots_dir, "confidence_intervals.png"),
        )

        # Power curves
        mde_range = np.linspace(0.005, 0.05, 30)
        n_per_arm = int(len(df_exp) * args.treatment_rate)
        pc_df = power_curve(baseline_cr, mde_range, n_per_arm, cuped2.variance_reduction, args.alpha)
        plot_power_curves(
            pc_df,
            n_per_arm=n_per_arm,
            save_path=os.path.join(plots_dir, "power_curves.png"),
        )

        # Sample size savings
        plot_sample_size_savings(
            mde_range=mde_range,
            baseline_rate=baseline_cr,
            variance_reductions={
                "Raw": 0.0,
                "CUPED": cuped2.variance_reduction,
                "MultiCUPED": multi_cuped.variance_reduction,
            },
            save_path=os.path.join(plots_dir, "sample_size_savings.png"),
        )

        # Bootstrap CI for CUPED-adjusted metric
        boot_result = bootstrap_ci(Y_cuped2, T, n_bootstrap=args.n_bootstrap)
        plot_bootstrap_distribution(
            boot_result,
            method_label="CUPED",
            true_lift=args.true_lift,
            save_path=os.path.join(plots_dir, "bootstrap_distribution.png"),
        )

        # Save summary CSV
        summary_rows = []
        for method, r in all_results.items():
            summary_rows.append({
                "method": method,
                "ate": r["ate"],
                "se": r["se"],
                "ci_lower": r["ci_lower"],
                "ci_upper": r["ci_upper"],
                "p_value": r.get("p_value", float("nan")),
                "significant": r.get("significant", False),
                "variance_reduction": r.get("variance_reduction", 0.0),
            })

        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(args.output_dir, "results_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        mlflow.log_artifact(summary_path)
        mlflow.log_artifacts(plots_dir, artifact_path="plots")

        logger.info(f"\n✅ Analysis complete.")
        logger.info(f"   Results saved to: {args.output_dir}/")
        logger.info(f"   MLflow UI: mlflow ui --port 5000")

        return summary_df


if __name__ == "__main__":
    main()
