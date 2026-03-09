"""
Data loading and preprocessing for the UCI Bank Marketing dataset.

Dataset: UCI Bank Marketing (id=222)
  - 45,211 rows, 17 features
  - Real Portuguese bank telemarketing campaign data
  - Target: did the client subscribe to a term deposit? (y: yes/no)

Fetch without login:
    from ucimlrepo import fetch_ucirepo
    bank = fetch_ucirepo(id=222)
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_bank_marketing(cache_path: str = "data/bank_marketing.parquet") -> pd.DataFrame:
    """
    Load UCI Bank Marketing dataset.
    Caches to parquet after first download to avoid repeated network calls.
    """
    if os.path.exists(cache_path):
        logger.info(f"Loading from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    logger.info("Downloading UCI Bank Marketing dataset...")
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=222)
        X = dataset.data.features
        y = dataset.data.targets
        df = pd.concat([X, y], axis=1)
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch dataset: {e}\n"
            "Ensure you have internet access and ucimlrepo installed:\n"
            "  pip install ucimlrepo"
        )

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path, index=False)
    logger.info(f"Dataset cached to {cache_path} ({len(df):,} rows)")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and encode the raw Bank Marketing DataFrame.

    Key transformations:
      - Encode binary target (y: yes/no → 1/0)
      - One-hot encode categoricals
      - Fill missing 'pdays' sentinel (-1 means not contacted)
      - Create 'previously_contacted' flag
    """
    df = df.copy()

    # Target encoding
    df["subscribed"] = (df["y"] == "yes").astype(int)
    df = df.drop(columns=["y"])

    # pdays == 999 means "not previously contacted" in the dataset
    df["previously_contacted"] = (df["pdays"] != 999).astype(int)
    df["pdays"] = df["pdays"].replace(999, 0)

    # One-hot encode categorical columns
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    logger.info(f"Preprocessed: {df.shape[0]:,} rows × {df.shape[1]} columns | "
                f"Subscription rate: {df['subscribed'].mean():.2%}")
    return df


def simulate_ab_experiment(
    df: pd.DataFrame,
    treatment_rate: float = 0.5,
    true_lift: float = 0.03,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Simulate an A/B experiment on top of the Bank Marketing data.

    Treatment definition:
      - treatment=1: client received the new campaign script (B variant)
      - treatment=0: client received the standard script (A variant)

    The 'subscribed' outcome is adjusted for treated users by adding
    a simulated lift, mimicking a real uplift from the new script.

    This is the standard approach when you have observational data
    but want to study A/B testing methodology — clearly documented.
    """
    rng = np.random.default_rng(random_state)
    df = df.copy()

    # Random assignment (mimics hash-based bucketing)
    df["treatment"] = rng.binomial(1, treatment_rate, len(df))

    # Inject a small true lift for treated users to make the experiment detectable
    treated_mask = df["treatment"] == 1
    flip_prob = true_lift / max(df.loc[treated_mask, "subscribed"].mean(), 0.01)
    flip_mask = treated_mask & (df["subscribed"] == 0) & (rng.random(len(df)) < flip_prob)
    df.loc[flip_mask, "subscribed"] = 1

    n_treat = treated_mask.sum()
    n_ctrl = (~treated_mask).sum()
    cr_treat = df.loc[treated_mask, "subscribed"].mean()
    cr_ctrl = df.loc[~treated_mask, "subscribed"].mean()

    logger.info(
        f"Simulated A/B split — Treatment: {n_treat:,} | Control: {n_ctrl:,}\n"
        f"  Control CR: {cr_ctrl:.2%} | Treatment CR: {cr_treat:.2%} | "
        f"Observed lift: {cr_treat - cr_ctrl:+.2%}"
    )
    return df


def get_pre_experiment_covariate(df: pd.DataFrame) -> pd.Series:
    """
    Return the pre-experiment covariate used for CUPED adjustment.

    We use 'previous' (number of contacts in prior campaign) as a
    proxy for pre-experiment engagement — a strong predictor of
    subscription that is independent of current treatment assignment.

    In a real system this would be the metric value from a prior period
    (e.g., revenue in the 30 days before the experiment started).
    """
    if "previous" not in df.columns:
        raise KeyError("Column 'previous' not found. Ensure raw data is loaded before one-hot encoding.")
    return df["previous"].copy()


def train_test_split_temporal(
    df: pd.DataFrame,
    test_fraction: float = 0.3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split preserving row order (temporal proxy — later records = test).
    Avoids data leakage in time-series-adjacent business data.
    """
    split_idx = int(len(df) * (1 - test_fraction))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
