"""
Gigit — Feature Engineering Utilities
============================================
Helper functions that extract the four core behavioral metrics for a given
worker from the mock CSV, ready for model inference.

Metrics:
  1. Average Monthly Income
  2. Income Volatility (%)
  3. Expense-to-Income Ratio
  4. Work Tenure (months)
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Default path to the generated CSV
# ---------------------------------------------------------------------------
DEFAULT_CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "mock_gig_worker_data.csv",
)

# Ordered list of feature columns expected by the model
FEATURE_COLUMNS = [
    "average_monthly_income",
    "income_volatility_percentage",
    "expense_to_income_ratio",
    "work_tenure_months",
]


def load_data(csv_path: str | None = None) -> pd.DataFrame:
    """
    Load the gig worker dataset from a CSV file.

    Parameters
    ----------
    csv_path : str, optional
        Path to the CSV.  Falls back to the default generated data file.

    Returns
    -------
    pd.DataFrame
    """
    path = csv_path or DEFAULT_CSV_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Data file not found at '{path}'.  "
            "Run  python data/generate_mock_data.py  first."
        )
    return pd.read_csv(path)


def get_worker_features(worker_id: str, df: pd.DataFrame) -> np.ndarray:
    """
    Extract the 4 behavioral features for a single worker as a NumPy array.

    Parameters
    ----------
    worker_id : str
        e.g. ``"GIG_042"``
    df : pd.DataFrame
        The full dataset (as returned by :func:`load_data`).

    Returns
    -------
    np.ndarray
        Shape ``(1, 4)`` — ready for model input after scaling.

    Raises
    ------
    ValueError
        If the worker_id is not found in the dataset.
    """
    row = df.loc[df["worker_id"] == worker_id]
    if row.empty:
        raise ValueError(
            f"Worker '{worker_id}' not found.  "
            f"Valid IDs range from GIG_001 to GIG_{len(df):03d}."
        )
    features = row[FEATURE_COLUMNS].values.astype(np.float32)
    return features  # shape (1, 4)


def get_worker_summary(worker_id: str, df: pd.DataFrame) -> dict:
    """
    Return a human-readable dictionary of a worker's metrics.

    Useful for the Streamlit dashboard to display alongside the risk score.

    Parameters
    ----------
    worker_id : str
    df : pd.DataFrame

    Returns
    -------
    dict
        Keys: ``worker_id``, ``average_monthly_income``,
        ``income_volatility_percentage``, ``expense_to_income_ratio``,
        ``work_tenure_months``, ``historical_default_risk``.
    """
    row = df.loc[df["worker_id"] == worker_id]
    if row.empty:
        raise ValueError(
            f"Worker '{worker_id}' not found.  "
            f"Valid IDs range from GIG_001 to GIG_{len(df):03d}."
        )
    return row.iloc[0].to_dict()


def get_all_worker_ids(df: pd.DataFrame) -> list[str]:
    """Return a sorted list of all worker IDs in the dataset."""
    return sorted(df["worker_id"].unique().tolist())
