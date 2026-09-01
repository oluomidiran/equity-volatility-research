"""Return computation, annualization, and risk-adjusted performance."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def daily_returns(prices):
    """Simple daily returns. The first observation is undefined and dropped."""
    return prices.pct_change().dropna()


def annualize_arithmetic(returns):
    """Mean daily return scaled linearly. Describes a typical day."""
    return returns.mean() * TRADING_DAYS


def annualize_geometric(returns):
    """Compound annual growth rate. Describes what happened to the money."""
    n = returns.count()
    return (1 + returns).prod() ** (TRADING_DAYS / n) - 1


def annualize_volatility(returns):
    """Daily volatility scaled by sqrt(252).

    Variance is additive over independent periods, so volatility -- its square
    root -- scales with the square root of time.
    """
    return returns.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(returns, risk_free_rate: float = 0.0):
    """Excess annualized return per unit of annualized volatility."""
    return (annualize_geometric(returns) - risk_free_rate) / annualize_volatility(returns)


def volatility_drag(returns):
    """Approximate gap between arithmetic and geometric annual return: sigma^2 / 2."""
    return annualize_volatility(returns) ** 2 / 2


def total_return(returns):
    """Compounded total return over the full sample."""
    return (1 + returns).prod() - 1


def growth_curve(returns):
    """Cumulative wealth path starting from 1.0."""
    return (1 + returns).cumprod()


def summary_table(returns) -> pd.DataFrame:
    """Per-asset return and risk summary."""
    return pd.DataFrame(
        {
            "total_return": total_return(returns),
            "ann_return_arithmetic": annualize_arithmetic(returns),
            "ann_return_geometric": annualize_geometric(returns),
            "ann_volatility": annualize_volatility(returns),
            "sharpe_rf0": sharpe_ratio(returns, 0.0),
            "sharpe_rf4": sharpe_ratio(returns, 0.04),
        }
    )
