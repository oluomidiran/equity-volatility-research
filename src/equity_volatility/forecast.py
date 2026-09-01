"""Volatility forecasting models, purged validation, and RMSE scoring."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import lstsq as scipy_lstsq

from .volatility import (
    ewma_std_volatility,
    ewma_volatility,
    forward_volatility,
    rolling_volatility,
    squared_return_volatility,
)

HAR_HORIZONS = (1, 5, 22)
FORECAST_HORIZON = 21


def rmse(forecast, actual) -> float:
    """Root mean squared error, aligning inputs BY DATE rather than position.

    Constructing a frame from two Series matches them on their index, so the
    two need not share a length or a date range. Only dates present in both
    survive the dropna.
    """
    paired = pd.DataFrame({"forecast": forecast, "actual": actual}).dropna()
    return float(np.sqrt(((paired["forecast"] - paired["actual"]) ** 2).mean()))


def baseline_forecasts(returns) -> dict:
    """Random walk, expanding historical mean, and three EWMA variants."""
    vol = rolling_volatility(returns)
    return {
        "random_walk": vol,
        "historical_mean": vol.expanding().mean(),
        "ewma_squared_returns": ewma_volatility(returns, adjust=False),
        "ewma_std_adjust_false": ewma_std_volatility(returns, adjust=False),
        "ewma_std_adjust_true": ewma_std_volatility(returns, adjust=True),
    }


def har_features(returns, horizons=HAR_HORIZONS) -> pd.DataFrame:
    """Volatility measured at each horizon, under the zero-mean assumption."""
    return pd.DataFrame(
        {f"v{h}": squared_return_volatility(returns, h) for h in horizons}
    )


def build_dataset(returns, horizons=HAR_HORIZONS,
                  forecast_horizon: int = FORECAST_HORIZON) -> pd.DataFrame:
    """Features plus forward-volatility target, aligned and complete-case."""
    frame = har_features(returns, horizons)
    frame["target"] = forward_volatility(returns, forecast_horizon)
    return frame.dropna()


def purged_split(data, train_fraction: float = 0.70,
                 forecast_horizon: int = FORECAST_HORIZON):
    """Chronological split with the boundary purged.

    Each target spans `forecast_horizon` forward days, so the final
    `forecast_horizon` training rows have targets built from test-period
    returns. Those rows are removed from training. The test set is unchanged.

    Returns (train, purged, test).
    """
    split = int(len(data) * train_fraction)
    train = data.iloc[: split - forecast_horizon]
    purged = data.iloc[split - forecast_horizon : split]
    test = data.iloc[split:]
    return train, purged, test


def _design_matrix(frame, feature_names):
    """Feature matrix with a leading column of ones for the intercept."""
    return np.column_stack([np.ones(len(frame)), frame[feature_names].to_numpy()])


def fit_har(train, feature_names=None):
    """Least-squares fit. Verified across NumPy and SciPy implementations."""
    feature_names = feature_names or [c for c in train.columns if c != "target"]
    X = _design_matrix(train, feature_names)
    y = train["target"].to_numpy()

    numpy_betas, *_ = np.linalg.lstsq(X, y, rcond=None)
    scipy_betas, *_ = scipy_lstsq(X, y, cond=None)
    np.testing.assert_allclose(numpy_betas, scipy_betas, rtol=1e-10, atol=1e-12)

    return {
        "betas": numpy_betas,
        "scipy_betas": scipy_betas,
        "max_solver_difference": float(np.max(np.abs(numpy_betas - scipy_betas))),
        "feature_names": feature_names,
    }


def predict_har(fit, test):
    """Apply fitted coefficients to held-out features."""
    X = _design_matrix(test, fit["feature_names"])
    return pd.Series(X @ fit["betas"], index=test.index, name="har_forecast")


def evaluate(forecasts: dict, actual) -> pd.Series:
    """RMSE for each named forecast against a common target."""
    return pd.Series({name: rmse(f, actual) for name, f in forecasts.items()})
