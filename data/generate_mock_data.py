"""
Gigit — Mock Gig Worker Data Generator
=============================================
Generates 500 rows of synthetic gig worker financial data for model training.

Each row represents a gig worker with behavioral financial metrics and a
binary default-risk label that is *correlated* with the features so the
downstream model can learn meaningful patterns.

Usage:
    python data/generate_mock_data.py
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_WORKERS = 500
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "mock_gig_worker_data.csv")

# Use a time-based seed so each run produces different data.
# The seed is printed so you can reproduce a specific run if needed.
SEED = int.from_bytes(os.urandom(4), "big")
np.random.seed(SEED)


def generate_worker_ids(n: int) -> list[str]:
    """Create worker IDs in the format GIG_001 … GIG_500."""
    return [f"GIG_{i:03d}" for i in range(1, n + 1)]


def generate_features(n: int) -> pd.DataFrame:
    """
    Generate the four core behavioral features with realistic distributions.

    Feature ranges / distributions:
      • average_monthly_income  — ₹8 000 – ₹80 000  (log-normal)
      • income_volatility_%     — 5 % – 60 %         (beta-ish)
      • expense_to_income_ratio — 0.30 – 1.10        (normal, clipped)
      • work_tenure_months      — 1 – 120             (uniform-ish)
    """
    # --- Average monthly income (log-normal for right-skew) ----------------
    avg_income = np.random.lognormal(mean=10.0, sigma=0.5, size=n)
    avg_income = np.clip(avg_income, 8_000, 80_000).round(2)

    # --- Income volatility (%) — higher = riskier --------------------------
    income_vol = np.random.beta(a=2, b=5, size=n) * 60  # 0-60 %
    income_vol = np.clip(income_vol, 5, 60).round(2)

    # --- Expense-to-income ratio — >1.0 means spending > earning -----------
    expense_ratio = np.random.normal(loc=0.65, scale=0.15, size=n)
    expense_ratio = np.clip(expense_ratio, 0.30, 1.10).round(4)

    # --- Work tenure (months) — longer tenure = lower risk ------------------
    tenure = np.random.randint(1, 121, size=n)

    return pd.DataFrame({
        "average_monthly_income": avg_income,
        "income_volatility_percentage": income_vol,
        "expense_to_income_ratio": expense_ratio,
        "work_tenure_months": tenure,
    })


def generate_default_labels(df: pd.DataFrame) -> np.ndarray:
    """
    Create a *correlated* binary default label using a logistic model.

    The probability of default increases when:
      • income is low
      • income volatility is high
      • expense ratio is high
      • tenure is short

    This ensures the training data has learnable signal.
    """
    # Normalise features to [0, 1] for the logistic combination
    inc_norm = (df["average_monthly_income"] - 8_000) / (80_000 - 8_000)
    vol_norm = df["income_volatility_percentage"] / 60.0
    exp_norm = (df["expense_to_income_ratio"] - 0.30) / 0.80
    ten_norm = df["work_tenure_months"] / 120.0

    # Linear combination — positive = riskier
    z = (
        -2.0 * inc_norm      # higher income → lower risk
        + 2.5 * vol_norm     # higher volatility → higher risk
        + 2.0 * exp_norm     # higher expense ratio → higher risk
        - 1.5 * ten_norm     # longer tenure → lower risk
        + np.random.normal(0, 0.4, size=len(df))  # noise
    )

    # Sigmoid → probability, then sample binary label
    prob = 1 / (1 + np.exp(-z))
    labels = (np.random.rand(len(df)) < prob).astype(int)
    return labels


def main() -> None:
    """Entry-point: build the DataFrame and write to CSV."""
    print("[*] Generating mock gig worker data...")
    print(f"[*] Random seed: {SEED}")

    ids = generate_worker_ids(NUM_WORKERS)
    features = generate_features(NUM_WORKERS)
    features.insert(0, "worker_id", ids)
    features["historical_default_risk"] = generate_default_labels(features)

    features.to_csv(OUTPUT_PATH, index=False)
    print(f"[OK] Saved {NUM_WORKERS} rows -> {OUTPUT_PATH}")
    print(features.head(10).to_string(index=False))
    print(f"\nDefault rate: {features['historical_default_risk'].mean():.1%}")


if __name__ == "__main__":
    main()
