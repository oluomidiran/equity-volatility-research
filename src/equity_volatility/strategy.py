"""Moving-average strategy backtesting with execution-timing and cost controls."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .returns import total_return

DEFAULT_WINDOW = 20
DEFAULT_COST = 0.001  # 10 basis points per position change


def moving_average(prices, window: int = DEFAULT_WINDOW):
    """Trailing simple moving average. First `window - 1` values are undefined."""
    return prices.rolling(window).mean()


def momentum_signal(prices, window: int = DEFAULT_WINDOW):
    """1 when price exceeds its trailing moving average, else 0."""
    return (prices > moving_average(prices, window)).astype(int)


def reversion_signal(prices, window: int = DEFAULT_WINDOW):
    """1 when price is below its trailing moving average, else 0."""
    return (prices < moving_average(prices, window)).astype(int)


def to_position(signal):
    """Lag the signal one period.

    A signal computed from today's close cannot earn today's return, because
    the close is not observable until the session ends. This lag is the single
    control preventing look-ahead bias.
    """
    return signal.shift(1)


def count_trades(position):
    """A trade occurs whenever the held position changes."""
    return position.diff().abs()


def strategy_returns(position, returns, cost_per_trade: float = 0.0):
    """Strategy return net of transaction costs charged on position changes."""
    trades = count_trades(position)
    return position * returns - cost_per_trade * trades


def backtest(prices, returns, window: int = DEFAULT_WINDOW,
             cost_per_trade: float = DEFAULT_COST, reversion: bool = False) -> dict:
    """Run a single backtest and return headline figures."""
    signal = reversion_signal(prices, window) if reversion else momentum_signal(prices, window)
    position = to_position(signal)
    gross = strategy_returns(position, returns, 0.0)
    net = strategy_returns(position, returns, cost_per_trade)
    trades = count_trades(position).sum()
    return {
        "trades": int(trades) if np.isscalar(trades) else trades.astype(int),
        "gross_total": total_return(gross.dropna()),
        "net_total": total_return(net.dropna()),
        "buyhold_total": total_return(returns) - cost_per_trade,
        "gross_series": gross,
        "net_series": net,
    }


def cost_sensitivity(prices, returns, cost_grid, window: int = DEFAULT_WINDOW) -> pd.DataFrame:
    """Net strategy return across a grid of assumed costs per trade.

    Two break-even points are distinguished: the benchmark crossover, where the
    strategy stops beating cost-adjusted buy-and-hold, and the zero-return
    point, where its own compounded return reaches zero.
    """
    position = to_position(momentum_signal(prices, window))
    rows = []
    for cost in cost_grid:
        net = total_return(strategy_returns(position, returns, cost).dropna())
        rows.append(
            {
                "cost_bps": cost * 10_000,
                "net_return": net,
                "buyhold_return": total_return(returns) - cost,
                "beats_benchmark": net > (total_return(returns) - cost),
            }
        )
    return pd.DataFrame(rows)


def regime_performance(position, returns, periods: dict, cost_per_trade: float = DEFAULT_COST):
    """Strategy versus buy-and-hold within named date ranges.

    NOTE: regime boundaries here are descriptive and were selected after
    observing the sample. They are not an out-of-sample partition.
    """
    net = strategy_returns(position, returns, cost_per_trade)
    out = {}
    for label, (start, end) in periods.items():
        out[label] = pd.DataFrame(
            {
                "strategy": total_return(net.loc[start:end].dropna()),
                "buyhold": total_return(returns.loc[start:end]),
            }
        )
    return out
