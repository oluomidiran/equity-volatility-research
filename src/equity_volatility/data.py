"""Data acquisition with local snapshot fallback for deterministic reproduction."""
from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

MULTI_ASSET_URL = "https://raw.githubusercontent.com/plotly/datasets/master/stockdata.csv"
APPLE_URL = "https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv"

ASSETS = ["MSFT", "IBM", "SBUX", "AAPL"]


def _load(url: str, snapshot: str) -> pd.DataFrame:
    """Prefer the local snapshot; fetch over HTTP and cache it if absent."""
    path = DATA_DIR / snapshot
    if path.exists():
        return pd.read_csv(path, index_col="Date", parse_dates=True)

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text), index_col="Date", parse_dates=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path)
    return frame


def load_multi_asset() -> pd.DataFrame:
    """MSFT, IBM, SBUX, AAPL closes, 2007-01-03 to 2016-03-01."""
    return _load(MULTI_ASSET_URL, "stockdata.csv")[ASSETS]


def load_apple_adjusted() -> pd.Series:
    """Apple adjusted closes, 2015-02-17 onward. Adjusted for splits and dividends."""
    return _load(APPLE_URL, "finance-charts-apple.csv")["AAPL.Adjusted"]


def generate_synthetic(seed: int = 7) -> pd.DataFrame:
    """Two simulated GBM paths, 522 business days.

    SYNTHETIC DATA. Column labels are illustrative and do not represent the
    historical prices of any listed company. Used only for the introductory
    risk-adjusted comparison; no empirical conclusion depends on it.
    """
    rng = np.random.default_rng(None)  # unused; kept explicit for clarity
    np.random.seed(seed)
    days = pd.bdate_range("2024-01-02", "2025-12-31")
    n = len(days)

    def path(start: float, mu: float, sigma: float) -> np.ndarray:
        steps = np.random.normal(mu / 252, sigma / np.sqrt(252), n)
        return start * np.exp(np.cumsum(steps))

    return pd.DataFrame(
        {
            "HIGH_VOL": path(185, 0.18, 0.26).round(2),
            "LOW_VOL": path(60, 0.06, 0.14).round(2),
        },
        index=pd.Index(days, name="Date"),
    )
