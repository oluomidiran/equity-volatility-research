"""Realized volatility measurement, persistence analysis, and forecast targets."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
DEFAULT_WINDOW = 21
RISKMETRICS_LAMBDA = 0.94


def rolling_volatility(returns, window: int = DEFAULT_WINDOW):
    """Annualized trailing realized volatility over a rolling window."""
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)


def squared_return_volatility(returns, window: int):
    """Annualized volatility from mean squared returns, assuming zero mean.

    Justified empirically at daily frequency: the squared mean daily return is
    smaller than the daily variance by roughly two orders of magnitude. The
    assumption would not hold at monthly or annual frequency.
    """
    return (returns**2).rolling(window).mean() ** 0.5 * np.sqrt(TRADING_DAYS)


def forward_volatility(returns, window: int = DEFAULT_WINDOW):
    """Realized volatility over the `window` days FOLLOWING each date.

    Built by reversing the series, applying a trailing window, reversing back,
    then shifting so the value on date t covers days t+1 through t+window.
    This is the evaluation target: it shares no observations with any forecast
    formed from information available at time t.
    """
    reversed_std = returns[::-1].rolling(window).std()[::-1]
    return reversed_std.shift(-1) * np.sqrt(TRADING_DAYS)


def overlapping_target(returns, window: int = DEFAULT_WINDOW):
    """Next day's TRAILING volatility. Retained only to demonstrate distortion.

    Consecutive trailing windows share window-1 of their observations, so a
    random-walk forecast is scored largely against itself.
    """
    return rolling_volatility(returns, window).shift(-1)


def ewma_volatility(returns, lam: float = RISKMETRICS_LAMBDA, adjust: bool = False):
    """RiskMetrics EWMA: v[t+1] = lam * v[t] + (1 - lam) * r[t]^2.

    Effective memory is 1 / (1 - lam) days; at lam = 0.94 that is about 16.7.
    """
    alpha = 1 - lam
    return np.sqrt((returns**2).ewm(alpha=alpha, adjust=adjust).mean() * TRADING_DAYS)


def ewma_std_volatility(returns, lam: float = RISKMETRICS_LAMBDA, adjust: bool = False):
    """Exponentially weighted standard deviation, which estimates the mean."""
    return returns.ewm(alpha=1 - lam, adjust=adjust).std() * np.sqrt(TRADING_DAYS)


def ewma_weights(lam: float = RISKMETRICS_LAMBDA, n: int = 60) -> pd.Series:
    """Weight assigned to each past observation: (1 - lam) * lam^k."""
    k = np.arange(n)
    return pd.Series((1 - lam) * lam**k, index=k, name="weight")


def autocorrelation_table(returns, lags=(1, 21, 63)) -> pd.DataFrame:
    """Persistence of returns, absolute returns, and rolling volatility.

    Volatility autocorrelation at lag 1 is mechanically inflated because
    consecutive windows overlap. Only lags at or beyond the window length
    support inference.
    """
    vol = rolling_volatility(returns).dropna()
    out = {
        "returns_lag1": returns.apply(lambda s: s.autocorr(1)),
        "abs_returns_lag1": returns.abs().apply(lambda s: s.autocorr(1)),
    }
    for lag in lags:
        out[f"volatility_lag{lag}"] = vol.apply(lambda s, l=lag: s.autocorr(l))
    return pd.DataFrame(out)


def volatility_range_table(returns) -> pd.DataFrame:
    """Minimum, mean, maximum, and ratio of rolling volatility per asset."""
    vol = rolling_volatility(returns)
    table = pd.DataFrame(
        {"min": vol.min(), "mean": vol.mean(), "max": vol.max()}
    )
    table["max_over_min"] = table["max"] / table["min"]
    table["max_date"] = [vol[c].idxmax() for c in vol.columns]
    return table
