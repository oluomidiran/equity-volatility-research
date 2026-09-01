"""Research-integrity tests.

Each test asserts one methodological control that a reviewer would otherwise
have to take on trust. These convert the corrections documented in the report
into executable checks.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from equity_volatility import data, forecast, returns as ret, strategy, volatility as vol


class TestExecutionTiming(unittest.TestCase):
    """The position must lag the signal by exactly one period."""

    def test_position_lags_signal(self):
        signal = pd.Series([0, 1, 1, 0, 1], index=pd.bdate_range("2024-01-01", periods=5))
        position = strategy.to_position(signal)
        self.assertTrue(np.isnan(position.iloc[0]))
        for i in range(1, len(signal)):
            self.assertEqual(position.iloc[i], signal.iloc[i - 1])

    def test_lookahead_inflates_result(self):
        """The unlagged variant must overstate performance; this is the bug."""
        prices = data.load_apple_adjusted()
        r = ret.daily_returns(prices)
        signal = strategy.momentum_signal(prices)
        unlagged = ret.total_return((signal * r).dropna())
        lagged = ret.total_return((strategy.to_position(signal) * r).dropna())
        self.assertGreater(unlagged, lagged * 3)


class TestForecastTarget(unittest.TestCase):
    """The forward target must use only returns AFTER the forecast date."""

    def test_forward_target_uses_only_future(self):
        r = pd.Series(np.arange(1.0, 41.0), index=pd.bdate_range("2024-01-01", periods=40))
        fwd = vol.forward_volatility(r, window=5)
        t = 10
        expected = r.iloc[t + 1 : t + 6].std() * np.sqrt(252)
        self.assertAlmostEqual(fwd.iloc[t], expected, places=10)

    def test_forward_target_tail_is_undefined(self):
        r = pd.Series(np.random.randn(40), index=pd.bdate_range("2024-01-01", periods=40))
        self.assertTrue(vol.forward_volatility(r, window=5).iloc[-1:].isna().all())

    def test_overlapping_target_is_not_used_for_headline(self):
        """Overlapping and forward targets must differ materially."""
        r = ret.daily_returns(data.load_multi_asset())["AAPL"]
        overlap = vol.overlapping_target(r)
        fwd = vol.forward_volatility(r)
        rw = vol.rolling_volatility(r)
        self.assertLess(forecast.rmse(rw, overlap), forecast.rmse(rw, fwd) / 2)


class TestPurge(unittest.TestCase):
    """Purging must remove every training label reaching into the test period."""

    def test_purge_removes_exactly_horizon_rows(self):
        idx = pd.bdate_range("2020-01-01", periods=1000)
        frame = pd.DataFrame({"v1": 1.0, "v5": 1.0, "v22": 1.0, "target": 1.0}, index=idx)
        train, purged, test = forecast.purged_split(frame, 0.70, 21)
        self.assertEqual(len(purged), 21)
        self.assertEqual(len(train) + len(purged) + len(test), len(frame))

    def test_no_training_label_reaches_test_period(self):
        r = ret.daily_returns(data.load_multi_asset())["AAPL"]
        dataset = forecast.build_dataset(r)
        train, _, test = forecast.purged_split(dataset)
        gap_days = (test.index.min() - train.index.max()).days
        self.assertGreaterEqual(gap_days, 21)

    def test_test_set_is_not_shortened(self):
        """Purging removes training rows only; the test set stays intact."""
        r = ret.daily_returns(data.load_multi_asset())["AAPL"]
        dataset = forecast.build_dataset(r)
        _, _, purged_test = forecast.purged_split(dataset)
        split = int(len(dataset) * 0.70)
        self.assertEqual(len(purged_test), len(dataset.iloc[split:]))


class TestRMSEAlignment(unittest.TestCase):
    """RMSE must align by date, never by row position."""

    def test_aligns_on_index_not_position(self):
        idx = pd.bdate_range("2024-01-01", periods=10)
        actual = pd.Series(np.arange(10.0), index=idx)
        shorter = pd.Series(np.arange(5.0, 10.0), index=idx[5:])
        self.assertAlmostEqual(forecast.rmse(shorter, actual), 0.0, places=12)

    def test_misaligned_dates_would_not_score_zero(self):
        idx = pd.bdate_range("2024-01-01", periods=10)
        actual = pd.Series(np.arange(10.0), index=idx)
        misaligned = pd.Series(np.arange(5.0, 10.0), index=idx[:5])
        self.assertGreater(forecast.rmse(misaligned, actual), 0.0)


class TestSolverAgreement(unittest.TestCase):
    """NumPy and SciPy least squares must return identical coefficients."""

    def test_numpy_scipy_agree(self):
        r = ret.daily_returns(data.load_multi_asset())["AAPL"]
        train, _, _ = forecast.purged_split(forecast.build_dataset(r))
        fit = forecast.fit_har(train)
        self.assertLess(fit["max_solver_difference"], 1e-10)


class TestSyntheticReproducibility(unittest.TestCase):
    """The synthetic foundation must reproduce exactly from its seed."""

    def test_seed_reproduces(self):
        a = data.generate_synthetic(seed=7)
        b = data.generate_synthetic(seed=7)
        pd.testing.assert_frame_equal(a, b)

    def test_known_metrics(self):
        r = ret.daily_returns(data.generate_synthetic(seed=7))
        self.assertAlmostEqual(ret.annualize_volatility(r)["HIGH_VOL"], 0.2556, places=3)
        self.assertAlmostEqual(ret.annualize_volatility(r)["LOW_VOL"], 0.1314, places=3)


class TestAnnualization(unittest.TestCase):
    """Variance scales by 252; volatility by its square root."""

    def test_volatility_scaling(self):
        r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 2000))
        self.assertAlmostEqual(ret.annualize_volatility(r), r.std() * np.sqrt(252), places=12)

    def test_geometric_below_arithmetic_when_volatile(self):
        """Volatility drag: compounded growth trails the arithmetic mean."""
        r = ret.daily_returns(data.load_apple_adjusted())
        self.assertLess(ret.annualize_geometric(r), ret.annualize_arithmetic(r))


if __name__ == "__main__":
    unittest.main(verbosity=2)
