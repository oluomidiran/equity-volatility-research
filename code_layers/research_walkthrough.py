"""
EQUITY VOLATILITY RESEARCH PIPELINE — SESSION-BY-SESSION WALKTHROUGH
====================================================================

The analysis in the order it was built, from first principles.

`src/equity_volatility/` refactors this same methodology into reusable, tested
modules. This file keeps the linear structure so the reasoning can be read in
sequence, which makes it the better entry point for understanding *how* the
research was developed. Both produce identical numbers.

Sessions
--------
  1  Returns, annualization, volatility, Sharpe               synthetic data
  2  Live data over HTTP; adjusted closes; volatility drag    real AAPL
  3  Moving-average rule; look-ahead bias; mean reversion
  4  Transaction costs and cost sensitivity
  5  Cross-stock robustness and market regime analysis        four equities
  6  Rolling realized volatility
  7  Persistence: direction versus magnitude
  8  Forecast models and evaluation-target design
  9  HAR regression with a purged chronological split

Data note
---------
Session 1 uses reproducible synthetic prices generated from a fixed random
seed. Its columns are labelled HIGH_VOL and LOW_VOL because they are simulated
paths, not the prices of any listed company. Sessions 2-9 use real public
market data from the Plotly datasets repository.

Run:  python code_layers/research_walkthrough.py
"""

import numpy as np
import pandas as pd
import requests
from io import StringIO
from scipy.linalg import lstsq as scipy_lstsq

# A named constant rather than a bare 252 scattered through the file. It states
# the reason for the number (trading days in a year, not calendar days) and
# guarantees every calculation uses the same value.
TRADING_DAYS = 252


def rmse(forecast, actual):
    """Root mean squared error, aligning inputs BY DATE rather than by position.

    Building a DataFrame from two Series makes pandas match them on their
    index. The two series therefore do not need the same length or the same
    date range: only dates present in both survive `.dropna()`. This is what
    lets a full-length forecast be scored against a short test target without
    slicing anything by hand, and it is why positional `.tail()` is unnecessary.
    """
    paired = pd.DataFrame({"forecast": forecast, "actual": actual}).dropna()
    return np.sqrt(((paired["forecast"] - paired["actual"]) ** 2).mean())


# =============================================================================
# SESSION 1 — RETURNS, ANNUALIZATION, VOLATILITY, SHARPE
# -----------------------------------------------------------------------------
# Question: given two price series, which was the better investment?
#
# Built on simulated data so the arithmetic can be checked against known
# parameters. The point of the session is that the higher-return asset is not
# necessarily the better one once risk is priced in.
# =============================================================================
np.random.seed(7)                    # fixes the random draw so this is reproducible
days = pd.bdate_range("2024-01-02", "2025-12-31")     # business days only


def gbm_path(start_price, annual_drift, annual_vol, n_days):
    """Generate a geometric Brownian motion price path.

    Daily log-steps are drawn from a normal distribution, scaled to daily
    units: drift divides by 252, volatility divides by sqrt(252). Taking the
    cumulative sum and exponentiating turns those steps into a price path.
    """
    steps = np.random.normal(annual_drift / TRADING_DAYS,
                             annual_vol / np.sqrt(TRADING_DAYS),
                             n_days)
    return start_price * np.exp(np.cumsum(steps))


synthetic_prices = pd.DataFrame({
    "HIGH_VOL": gbm_path(185, 0.18, 0.26, len(days)).round(2),
    "LOW_VOL": gbm_path(60, 0.06, 0.14, len(days)).round(2),
}, index=days)

# The first row is dropped because there is no prior price to compare against.
# This is the smallest instance of a rule that recurs throughout the project:
# a value cannot be computed from data that does not exist yet.
synthetic_returns = synthetic_prices.pct_change().dropna()

# Returns are scaled by 252 because expected returns add across periods.
# Volatility is scaled by sqrt(252) because variance adds across independent
# periods, and volatility is the square root of variance.
synthetic_ann_return = synthetic_returns.mean() * TRADING_DAYS
synthetic_ann_vol = synthetic_returns.std() * np.sqrt(TRADING_DAYS)
synthetic_sharpe = synthetic_ann_return / synthetic_ann_vol

print("=" * 70)
print("SESSION 1 — synthetic two-asset comparison")
print("=" * 70)
print(pd.DataFrame({
    "ann return %": (synthetic_ann_return * 100).round(2),
    "ann vol %": (synthetic_ann_vol * 100).round(2),
    "Sharpe": synthetic_sharpe.round(2),
}).to_string())
print("\nThe higher-volatility asset lost money while carrying twice the risk.")
print("Return alone would not have revealed that; the Sharpe ratio does.")


# =============================================================================
# SESSION 2 — LIVE DATA, ADJUSTED CLOSES, AND VOLATILITY DRAG
# -----------------------------------------------------------------------------
# Question: do the two ways of annualizing a return agree, and if not, why?
#
# Adjusted closes correct historical prices for stock splits and dividends.
# A two-for-one split halves the raw closing price overnight while leaving
# shareholder wealth unchanged; computing returns from raw closes would record
# a 50% loss on a day nobody lost anything.
# =============================================================================
apple_url = ("https://raw.githubusercontent.com/plotly/datasets/master/"
             "finance-charts-apple.csv")
apple_response = requests.get(apple_url, timeout=30)
apple_response.raise_for_status()      # stops the script if the server refused

print("\n" + "=" * 70)
print("SESSION 2 — live data over HTTP")
print("=" * 70)
print("HTTP status:", apple_response.status_code)     # 200 means the request succeeded

# `response.text` is the body of the reply as one long string. `StringIO` wraps
# it so `read_csv` treats it as though it were a file on disk.
apple_data = pd.read_csv(StringIO(apple_response.text),
                         index_col="Date", parse_dates=True)
apple_prices = apple_data["AAPL.Adjusted"]           # adjusted, not raw close
apple_returns = apple_prices.pct_change().dropna()

# Two annualization conventions that disagree by a predictable amount.
# Arithmetic averages the daily returns and scales linearly: it describes a
# typical day. Geometric compounds the actual path: it describes what happened
# to the money. Money multiplies rather than adds, so geometric is the honest
# figure whenever a realized outcome is being reported.
apple_arithmetic_ann = apple_returns.mean() * TRADING_DAYS
apple_geometric_ann = ((1 + apple_returns).prod()
                       ** (TRADING_DAYS / apple_returns.count()) - 1)
apple_ann_vol = apple_returns.std() * np.sqrt(TRADING_DAYS)

# The gap between them is volatility drag, approximated by variance / 2.
volatility_drag = apple_ann_vol ** 2 / 2
observed_gap = apple_arithmetic_ann - apple_geometric_ann

print(f"total return            {((1 + apple_returns).prod() - 1) * 100:6.2f}%")
print(f"arithmetic annualized   {apple_arithmetic_ann * 100:6.2f}%   a typical day")
print(f"geometric annualized    {apple_geometric_ann * 100:6.2f}%   what happened to the money")
print(f"annualized volatility   {apple_ann_vol * 100:6.2f}%")
print(f"volatility drag         {volatility_drag * 100:6.2f}%   "
      f"approximates the observed {observed_gap * 100:.2f}pp gap")

risk_free_rate = 0.04                                 # 4% annual
print(f"\nSharpe (rf = 0)         {apple_geometric_ann / apple_ann_vol:6.2f}")
print(f"Sharpe (rf = 4%)        "
      f"{(apple_geometric_ann - risk_free_rate) / apple_ann_vol:6.2f}")


# =============================================================================
# SESSION 3 — TRADING RULE, LOOK-AHEAD BIAS, AND THE OPPOSITE RULE
# -----------------------------------------------------------------------------
# Question: does a 20-day moving-average rule beat simply holding the stock?
#
# The critical control is execution timing. A signal computed from today's
# closing price cannot earn today's return, because the close is not
# observable until the session ends. Both the momentum rule and its opposite
# are tested, since they are the two competing schools: momentum buys what is
# rising, mean reversion buys what has fallen.
# =============================================================================
moving_average_20 = apple_prices.rolling(20).mean()   # trailing 20-day average

# A signal of 1 means hold the stock, 0 means hold cash.
momentum_signal = (apple_prices > moving_average_20).astype(int)
reversion_signal = (apple_prices < moving_average_20).astype(int)

# INVALID: pairs today's signal with today's return. Up-days push the price
# above the average, which sets the signal to 1, which then captures that same
# up-day. The selection happens after the outcome is already known.
unlagged_total = (1 + (momentum_signal * apple_returns)).prod() - 1

# VALID: `.shift(1)` moves each signal down one row, so the decision made at
# yesterday's close governs today's position. This single line is the
# look-ahead control.
positions = momentum_signal.shift(1)
momentum_total = (1 + (positions * apple_returns)).prod() - 1

reversion_positions = reversion_signal.shift(1)
reversion_total = (1 + (reversion_positions * apple_returns)).prod() - 1

buyhold_gross = (1 + apple_returns).prod() - 1

print("\n" + "=" * 70)
print("SESSION 3 — trading rule and look-ahead bias")
print("=" * 70)
print(f"unlagged (INVALID)      {unlagged_total * 100:8.2f}%   acts on information not yet available")
print(f"one-day lag (valid)     {momentum_total * 100:8.2f}%")
print(f"mean reversion          {reversion_total * 100:8.2f}%   the opposite rule")
print(f"buy and hold            {buyhold_gross * 100:8.2f}%")
print(f"\nLook-ahead overstated the result by "
      f"{(unlagged_total - momentum_total) * 100:.0f} percentage points.")
print("A backtest that looks exceptional should be assumed broken until checked.")


# =============================================================================
# SESSION 4 — TRANSACTION COSTS AND COST SENSITIVITY
# -----------------------------------------------------------------------------
# Question: does the strategy survive the cost of trading?
#
# A trade occurs whenever the held position changes. Buy-and-hold is charged
# one entry cost so the comparison is fair: it also has to be purchased once.
# =============================================================================
cost_per_trade = 0.001               # 10 basis points = 0.10%, covering
                                     # commission, half the bid-ask spread,
                                     # and slippage combined

# Absolute value, so that entering and exiting each count as one trade.
trades = positions.diff().abs()

net_returns = (positions * apple_returns) - (cost_per_trade * trades)
net_total = (1 + net_returns).prod() - 1
buyhold_total = (1 + apple_returns).prod() - 1 - cost_per_trade   # one entry cost

print("\n" + "=" * 70)
print("SESSION 4 — transaction costs")
print("=" * 70)
print(f"trades                  {int(trades.sum()):8d}")
print(f"gross                   {momentum_total * 100:8.2f}%")
print(f"net of 10bp             {net_total * 100:8.2f}%")
print(f"buy and hold (1 trade)  {buyhold_total * 100:8.2f}%")
print(f"cost destroyed          {(momentum_total - net_total) * 100:8.2f} percentage points")
print("\nThat exceeds the naive 49 x 0.10% = 4.9%, because money paid in fees")
print("early is money that cannot compound afterwards.")

# The conclusion rests on a cost assumption that was stipulated rather than
# measured, so the honest response is to test how much that assumption matters.
# Two break-even points are distinguished because they answer different
# questions: one is about beating the benchmark, the other about making money
# at all.
print("\ncost sensitivity, 0 to 100 basis points:")
print(f"  {'bp':>4} {'net %':>9} {'buy&hold %':>11}  beats benchmark?")

benchmark_crossover_bp = None        # where it stops beating buy-and-hold
zero_return_bp = None                # where its own return reaches zero

for basis_points in range(0, 101):
    cost = basis_points / 10_000     # 1 basis point = 0.0001
    net = (1 + ((positions * apple_returns) - cost * trades)).prod() - 1
    benchmark = (1 + apple_returns).prod() - 1 - cost

    if benchmark_crossover_bp is None and net <= benchmark:
        benchmark_crossover_bp = basis_points
    if zero_return_bp is None and net <= 0:
        zero_return_bp = basis_points

    # Print only the originally reported grid; the loop itself runs at
    # one-basis-point resolution so the crossover points are located exactly.
    if basis_points in (0, 10, 20, 30, 50, 75, 100):
        verdict = "yes" if net > benchmark else "no"
        print(f"  {basis_points:>4} {net * 100:>9.2f} {benchmark * 100:>11.2f}  {verdict}")

print(f"\nbenchmark crossover  ~{benchmark_crossover_bp} bp   stops beating buy-and-hold")
print(f"zero-return point    ~{zero_return_bp} bp   stops making money at all")
print("The strategy is viable only under execution costs below roughly 32 bp.")
print("Its edge is a function of execution quality as much as of signal quality.")


# =============================================================================
# SESSION 5 — CROSS-STOCK ROBUSTNESS AND MARKET REGIME ANALYSIS
# -----------------------------------------------------------------------------
# Question: was the Apple result a general property, or a feature of one stock
# over one window?
#
# Four equities across nine years, including the 2008 financial crisis. This is
# where the result either generalizes or does not.
# =============================================================================
multi_url = "https://raw.githubusercontent.com/plotly/datasets/master/stockdata.csv"
multi_response = requests.get(multi_url, timeout=30)
multi_response.raise_for_status()
multi_data = pd.read_csv(StringIO(multi_response.text),
                         index_col="Date", parse_dates=True)

# A DataFrame rather than a Series, so every operation below applies to all
# four columns at once.
prices = multi_data[["MSFT", "IBM", "SBUX", "AAPL"]]
returns = prices.pct_change().dropna()

moving_average = prices.rolling(20).mean()
signal = (prices > moving_average).astype(int)
positions_multi = signal.shift(1)                     # same one-day lag
trades_multi = positions_multi.diff().abs()

gross_multi = (1 + (positions_multi * returns)).prod() - 1
net_returns_multi = (positions_multi * returns) - cost_per_trade * trades_multi
net_multi = (1 + net_returns_multi).prod() - 1
buyhold_multi = (1 + returns).prod() - 1 - cost_per_trade

print("\n" + "=" * 70)
print("SESSION 5 — cross-stock robustness, 2007-2016")
print("=" * 70)
cross_stock = pd.DataFrame({
    "trades": trades_multi.sum().astype(int),
    "gross %": (gross_multi * 100).round(1),
    "net %": (net_multi * 100).round(1),
    "buy&hold %": (buyhold_multi * 100).round(1),
})
cross_stock["beats?"] = np.where(net_multi > buyhold_multi, "yes", "no")
print(cross_stock.to_string())

print("\nThe rule underperformed buy-and-hold on all four names and lost money")
print("outright on three. This is a null result, and it is reported rather than")
print("discarded: the Apple figure came from a window that happened to suit it.")

# Regime analysis explains *why* it failed, which is more useful than the fact
# that it failed. The sample is split at the March 2009 market trough.
print("\nmarket regime analysis:")
for label, start, end in [("CRISIS    Oct 2007 - Mar 2009", "2007-10-01", "2009-03-09"),
                          ("RECOVERY  Mar 2009 - Mar 2016", "2009-03-10", "2016-03-01")]:
    # `.loc[start:end]` slices by date label, inclusive of both endpoints.
    strategy_pct = ((1 + net_returns_multi.loc[start:end]).prod() - 1) * 100
    buyhold_pct = ((1 + returns.loc[start:end]).prod() - 1) * 100
    print(f"\n  {label}")
    for ticker in returns.columns:
        flag = "  <- strategy wins" if strategy_pct[ticker] > buyhold_pct[ticker] else ""
        print(f"    {ticker:5} strategy {strategy_pct[ticker]:>8.1f}%   "
              f"buy&hold {buyhold_pct[ticker]:>8.1f}%{flag}")

print("\nThe rule moved to cash as prices broke below their averages, which")
print("protected capital during the crash. It then re-entered only after prices")
print("had recovered above the average, missing the sharpest rebounds. That is")
print("whipsaw, and over this sample the trade-off was unfavourable.")
print("\nNOTE: these regime boundaries were chosen after observing the sample.")
print("They are a diagnostic, not an out-of-sample partition.")


# =============================================================================
# SESSION 6 — ROLLING REALIZED VOLATILITY
# -----------------------------------------------------------------------------
# Question: is a single volatility figure meaningful?
#
# One full-sample number averages calm and crisis into a value that describes
# no month that actually occurred. A rolling window replaces it with one
# estimate per day.
# =============================================================================
# 21 trading days is approximately one calendar month. `.std()` uses the sample
# convention with n-1 in the denominator, which is correct here because each
# window is a sample used to estimate an unobservable quantity, not a
# population.
rolling_vol = returns.rolling(21).std() * np.sqrt(TRADING_DAYS)

print("\n" + "=" * 70)
print("SESSION 6 — rolling 21-day realized volatility, annualized")
print("=" * 70)
volatility_range = pd.DataFrame({
    "min %": (rolling_vol.min() * 100).round(1),
    "mean %": (rolling_vol.mean() * 100).round(1),
    "max %": (rolling_vol.max() * 100).round(1),
})
volatility_range["max/min"] = (rolling_vol.max() / rolling_vol.min()).round(1)
volatility_range["max on"] = [rolling_vol[col].idxmax().date()
                              for col in rolling_vol.columns]
print(volatility_range.to_string())

print("\nEvery asset was at least ten times more volatile in its worst month")
print("than in its calmest, and all four peaked within six weeks of each other")
print("in late 2008. Volatility is not a constant to be estimated once.")


# =============================================================================
# SESSION 7 — PERSISTENCE: DIRECTION VERSUS MAGNITUDE
# -----------------------------------------------------------------------------
# Question: does the past predict the future, and if so, which part of it?
#
# Autocorrelation is the correlation of a series with a delayed copy of itself.
# The lag is how many periods back the copy is taken from.
# =============================================================================
volatility = rolling_vol.dropna()

print("\n" + "=" * 70)
print("SESSION 7 — autocorrelation")
print("=" * 70)
# Each lambda receives one column at a time and returns its autocorrelation.
print(pd.DataFrame({
    # Direction: does yesterday's move predict today's?
    "returns lag1": returns.apply(lambda series: series.autocorr(1)),
    # Magnitude: does the SIZE of yesterday's move predict today's size?
    "|returns| lag1": returns.abs().apply(lambda series: series.autocorr(1)),
    # Lag 1 on volatility is inflated by construction: today's 21-day window and
    # yesterday's share 20 of their 21 observations, so they are nearly the same
    # calculation. Reported for transparency, not for inference.
    "vol lag1": volatility.apply(lambda series: series.autocorr(1)),
    # Lag 21: one full month apart, so the two windows share no observations.
    "vol lag21": volatility.apply(lambda series: series.autocorr(21)),
    # Lag 63: one quarter apart. Persistence at this distance is the real finding.
    "vol lag63": volatility.apply(lambda series: series.autocorr(63)),
}).round(3).to_string())

print("\nReturn autocorrelation is indistinguishable from zero: direction is not")
print("forecastable from recent history. Absolute returns and volatility remain")
print("persistent even at non-overlapping lags.")
print("\nThat asymmetry is the economic basis for volatility trading. Directional")
print("patterns get competed away, because everyone can trade them. Volatility")
print("clustering survives, because knowing the market will be violent does not")
print("tell you which side to take.")


# =============================================================================
# SESSION 8 — FORECASTING, AND WHY THE EVALUATION TARGET DECIDES THE WINNER
# -----------------------------------------------------------------------------
# Question: which model predicts volatility best?
#
# The answer turns out to depend entirely on what the models are asked to
# predict. Same data, same models, opposite conclusion.
# =============================================================================
aapl_returns = returns["AAPL"]

# TARGET A (flawed): the next day's TRAILING 21-day window. Today's window and
# tomorrow's share 20 of 21 observations, so a random-walk forecast is being
# graded largely against itself. Retained to quantify the distortion.
# `.shift(-1)` moves values UP one row, pulling tomorrow onto today's line.
target_trailing = volatility["AAPL"].shift(-1)

# TARGET B (honest): realized volatility over the NEXT 21 days. Reversing the
# series, taking a trailing window, and reversing back relocates the window
# from "the last 21 days" to "the next 21 days". The final `.shift(-1)` starts
# it tomorrow, so today's own return is excluded. This shares no observations
# with any forecast built from information available today.
target_forward = (aapl_returns[::-1].rolling(21).std()[::-1].shift(-1)
                  * np.sqrt(TRADING_DAYS))

# --- the five forecasts ------------------------------------------------------
# 1. Random walk: tomorrow equals today. A benchmark rather than a model. Any
#    proposed method must beat it to justify its complexity.
forecast_random_walk = rolling_vol["AAPL"]

# 2. Historical mean: the expanding average of everything observed so far.
#    `.expanding()` grows the window each day and never sees the future, unlike
#    a plain `.mean()` which would average the entire dataset including days
#    that have not happened yet.
forecast_historical_mean = rolling_vol["AAPL"].expanding().mean()

# 3-5. EWMA variants. lambda = 0.94 is the RiskMetrics daily convention, so
#      alpha = 1 - lambda = 0.06. Weights decay geometrically: today gets 6%,
#      yesterday 5.6%, and so on, giving an effective memory of about 17 days.
#
# (a) squared returns, assuming the daily mean return is zero. Justified because
#     the squared mean is roughly 300 times smaller than the daily variance.
forecast_ewma_zero_mean = np.sqrt(
    (aapl_returns ** 2).ewm(alpha=0.06, adjust=False).mean() * TRADING_DAYS)

# (b) weighted standard deviation, recursive form. This estimates the mean
#     rather than assuming it, and `adjust=False` applies the pure recursion so
#     the most recent observations dominate.
forecast_ewma_recursive = (aapl_returns.ewm(alpha=0.06, adjust=False).std()
                           * np.sqrt(TRADING_DAYS))

# (c) the same, bias-adjusted. `adjust=True` normalises by the sum of weights
#     available so far, which changes the early values while there is little
#     history. The two converge after roughly fifty observations.
forecast_ewma_bias_adjusted = (aapl_returns.ewm(alpha=0.06, adjust=True).std()
                               * np.sqrt(TRADING_DAYS))

print("\n" + "=" * 70)
print("SESSION 8 — forecast RMSE (%) under both evaluation targets")
print("=" * 70)
print(f"  {'model':30} {'trailing':>10} {'forward':>10}")
for name, forecast in [
    ("random walk", forecast_random_walk),
    ("historical mean", forecast_historical_mean),
    ("EWMA squared returns", forecast_ewma_zero_mean),
    ("EWMA std, adjust=False", forecast_ewma_recursive),
    ("EWMA std, adjust=True", forecast_ewma_bias_adjusted),
]:
    print(f"  {name:30} {rmse(forecast, target_trailing) * 100:>10.2f} "
          f"{rmse(forecast, target_forward) * 100:>10.2f}")

print("\nUnder the trailing target the random walk appears best by a wide margin.")
print("Under the forward target the ranking reverses. The data, the models and")
print("the code are identical; only the answer key changed.")
print("\nThe zero-mean variant edges out both variants that estimate the mean.")
print("The true daily mean is small enough that estimating it introduces more")
print("noise than the bias it removes: a bias-variance trade-off.")


# =============================================================================
# SESSION 9 — HAR REGRESSION WITH A PURGED CHRONOLOGICAL SPLIT
# -----------------------------------------------------------------------------
# Question: can volatility measured at several horizons beat a single-parameter
# decay scheme?
#
# This is supervised learning applied to time-series data: features, a target,
# a training partition and a held-out test partition. The ordering of the data
# imposes two constraints that a standard setup would violate, and both are
# handled explicitly below.
# =============================================================================
# FEATURES: volatility measured over one day, one week, and one month, under
# the same zero-mean assumption validated in Session 8. Naming them by horizon
# rather than by window length keeps the code in the same language as the
# model's coefficients: beta_daily, beta_weekly, beta_monthly.
vol_daily = (aapl_returns ** 2).rolling(1).mean() ** 0.5 * np.sqrt(TRADING_DAYS)
vol_weekly = (aapl_returns ** 2).rolling(5).mean() ** 0.5 * np.sqrt(TRADING_DAYS)
vol_monthly = (aapl_returns ** 2).rolling(22).mean() ** 0.5 * np.sqrt(TRADING_DAYS)

# Assembling features and target into one DataFrame aligns them by date, and a
# single `.dropna()` then removes any row where any column is missing. Each
# feature has a different warm-up length (vol_daily none, vol_weekly four,
# vol_monthly twenty-one), so aligning before dropping keeps every surviving
# feature vector matched to the correct target.
dataset = pd.DataFrame({
    "vol_daily": vol_daily,
    "vol_weekly": vol_weekly,
    "vol_monthly": vol_monthly,
    "target": target_forward,
}).dropna()

# CONSTRAINT 1: observations are ordered, so they cannot be shuffled into
# random folds. The split is chronological: the first 70% trains, the last 30%
# tests. `.iloc` selects by integer position even though the index holds dates.
FORECAST_HORIZON = 21
split = int(len(dataset) * 0.70)

# CONSTRAINT 2: each target is built from the 21 days FOLLOWING its date. The
# final 21 training rows therefore have targets constructed from returns inside
# the test period. Those rows are removed before fitting. This is purging, and
# the test set is left untouched.
train_unpurged = dataset.iloc[:split]                              # for comparison
train = dataset.iloc[:split - FORECAST_HORIZON]                    # purged
purged_boundary = dataset.iloc[split - FORECAST_HORIZON:split]     # discarded
test = dataset.iloc[split:]                                        # unchanged

print("\n" + "=" * 70)
print("SESSION 9 — HAR with a purged chronological split")
print("=" * 70)
print(f"train {len(train)} | purged {len(purged_boundary)} | test {len(test)}")
print(f"train ends {train.index.max().date()} | "
      f"test starts {test.index.min().date()}")

FEATURES = ["vol_daily", "vol_weekly", "vol_monthly"]

# `np.column_stack` joins arrays side by side as columns. Prepending a column
# of ones lets the regression fit an intercept: the first fitted coefficient
# becomes beta_0, because that column is constant.
design_matrix_train = np.column_stack([np.ones(len(train)), train[FEATURES]])
design_matrix_unpurged = np.column_stack([np.ones(len(train_unpurged)),
                                          train_unpurged[FEATURES]])

# `lstsq` finds the coefficient vector minimising total squared error in
# X @ beta ~ y. It returns several outputs; `*_` absorbs the ones not needed.
# `rcond` (NumPy) and `cond` (SciPy) control when very small singular values
# are treated as zero — the same setting under two different parameter names.
numpy_betas, *_ = np.linalg.lstsq(design_matrix_train, train["target"], rcond=None)
scipy_betas, *_ = scipy_lstsq(design_matrix_train, train["target"], cond=None)
betas_unpurged, *_ = np.linalg.lstsq(design_matrix_unpurged,
                                     train_unpurged["target"], rcond=None)

# Both libraries solve the same problem. Asserting they agree guards against a
# silent numerical failure in either implementation.
np.testing.assert_allclose(numpy_betas, scipy_betas, rtol=1e-10, atol=1e-12)

print("\ncoefficients [intercept, beta_daily, beta_weekly, beta_monthly]")
print("  NumPy (purged):  ", numpy_betas.round(4))
print("  SciPy (purged):  ", scipy_betas.round(4))
print("  NumPy (unpurged):", betas_unpurged.round(4))
print("  max difference NumPy vs SciPy:", np.max(np.abs(numpy_betas - scipy_betas)))

print("\nThe monthly coefficient is roughly eighteen times the daily one. Least")
print("squares allocates weight by signal quality, and a single squared return")
print("is an extremely noisy variance estimate; a 22-day average is not.")

# PREDICTION: `@` is matrix multiplication, computing
#   beta_0 + beta_d * vol_daily + beta_w * vol_weekly + beta_m * vol_monthly
# for every test row at once. Wrapping the result in a Series with the test
# index keeps each forecast attached to its date, which is what lets the rmse
# function align it correctly against the target.
design_matrix_test = np.column_stack([np.ones(len(test)), test[FEATURES]])
har_forecast = pd.Series(design_matrix_test @ numpy_betas, index=test.index)
har_forecast_unpurged = pd.Series(design_matrix_test @ betas_unpurged,
                                  index=test.index)

har_rmse = rmse(har_forecast, test["target"])
har_rmse_unpurged = rmse(har_forecast_unpurged, test["target"])
ewma_rmse = rmse(forecast_ewma_zero_mean, test["target"])
random_walk_rmse = rmse(forecast_random_walk, test["target"])

print("\nout-of-sample RMSE (%), identical held-out dates:")
print(f"  HAR (purged)       {har_rmse * 100:6.2f}   reported result")
print(f"  EWMA               {ewma_rmse * 100:6.2f}")
print(f"  random walk        {random_walk_rmse * 100:6.2f}")
print(f"\n  HAR (unpurged)     {har_rmse_unpurged * 100:6.2f}   flattered by target leakage")
print(f"  cost of purging    {(har_rmse - har_rmse_unpurged) * 100:6.2f} percentage points")

print("\nPurging made the result slightly worse, and that is the evidence the")
print("leakage was real. A correction that degrades a number is a correction")
print("worth trusting; one that improves it deserves scrutiny.")

print(f"\nHAR improves on EWMA by "
      f"{(ewma_rmse - har_rmse) / ewma_rmse * 100:.2f}%")
print(f"HAR improves on the random walk by "
      f"{(random_walk_rmse - har_rmse) / random_walk_rmse * 100:.2f}%")
print("\nThe improvement is modest. Volatility forecasting at a one-month")
print("horizon is genuinely difficult, and the honest comparison is HAR against")
print("EWMA on the same held-out dates, not against Session 8's full-period")
print("figures, which cover a different and more volatile sample.")
