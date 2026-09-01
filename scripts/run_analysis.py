"""Run the complete equity volatility research pipeline.

Usage:  PYTHONPATH=src python scripts/run_analysis.py
Writes: results/project_results.json, results/tables/*.csv, results/figures/*.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from equity_volatility import data, forecast, plots, returns as ret, strategy, volatility as vol

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
for d in (TABLES, FIGURES):
    d.mkdir(parents=True, exist_ok=True)

results: dict = {}


def save_table(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(TABLES / f"{name}.csv")


def main() -> None:
    # ----- 1. Synthetic foundation ------------------------------------------
    synth = data.generate_synthetic(seed=7)
    synth_returns = ret.daily_returns(synth)
    synth_summary = ret.summary_table(synth_returns)
    save_table(synth_summary, "01_synthetic_summary")
    results["synthetic"] = {
        "note": "SYNTHETIC DATA. Illustrative only; no conclusion depends on it.",
        "summary": synth_summary.round(6).to_dict(),
    }

    # ----- 2. Apple: returns, drag, strategy --------------------------------
    apple = data.load_apple_adjusted()
    apple_ret = ret.daily_returns(apple)
    results["apple_returns"] = {
        "total_return": ret.total_return(apple_ret),
        "ann_arithmetic": ret.annualize_arithmetic(apple_ret),
        "ann_geometric": ret.annualize_geometric(apple_ret),
        "ann_volatility": ret.annualize_volatility(apple_ret),
        "volatility_drag": ret.volatility_drag(apple_ret),
        "sharpe_rf0": ret.sharpe_ratio(apple_ret, 0.0),
        "sharpe_rf4": ret.sharpe_ratio(apple_ret, 0.04),
    }

    signal = strategy.momentum_signal(apple)
    position = strategy.to_position(signal)
    unlagged = ret.total_return((signal * apple_ret).dropna())
    bt = strategy.backtest(apple, apple_ret)
    mr = strategy.backtest(apple, apple_ret, reversion=True)

    results["apple_strategy"] = {
        "unlagged_lookahead_total": unlagged,
        "lagged_gross_total": bt["gross_total"],
        "lagged_net_total": bt["net_total"],
        "mean_reversion_net_total": mr["net_total"],
        "buyhold_total": bt["buyhold_total"],
        "trades": bt["trades"],
        "lookahead_overstatement_pp": (unlagged - bt["gross_total"]) * 100,
    }

    plots.strategy_growth(
        {
            "Buy and hold": ret.growth_curve(apple_ret),
            "Momentum (gross)": ret.growth_curve(bt["gross_series"].dropna()),
            "Momentum (net 10bp)": ret.growth_curve(bt["net_series"].dropna()),
            "Mean reversion": ret.growth_curve(mr["net_series"].dropna()),
        },
        FIGURES,
    )

    # ----- 3. Cost sensitivity ----------------------------------------------
    grid = np.array([0, 5, 10, 20, 30, 40, 50, 75, 100]) / 10_000
    fine = np.arange(0, 101) / 10_000
    cost_table = strategy.cost_sensitivity(apple, apple_ret, grid)
    save_table(cost_table, "02_cost_sensitivity")

    fine_table = strategy.cost_sensitivity(apple, apple_ret, fine)
    crossover = fine_table.loc[~fine_table["beats_benchmark"], "cost_bps"].min()
    zero_point = fine_table.loc[fine_table["net_return"] <= 0, "cost_bps"].min()
    results["cost_sensitivity"] = {
        "benchmark_crossover_bps": float(crossover),
        "zero_return_bps": float(zero_point),
        "grid": cost_table.round(6).to_dict(orient="records"),
    }
    plots.cost_sensitivity(cost_table, crossover, zero_point,
                           ret.total_return(apple_ret), FIGURES)

    # ----- 4. Cross-stock and regimes ---------------------------------------
    prices = data.load_multi_asset()
    rets = ret.daily_returns(prices)
    multi = strategy.backtest(prices, rets)
    cross = pd.DataFrame({
        "trades": strategy.count_trades(strategy.to_position(
            strategy.momentum_signal(prices))).sum().astype(int),
        "gross_total": multi["gross_total"],
        "net_total": multi["net_total"],
        "buyhold_total": multi["buyhold_total"],
    })
    cross["net_beats_benchmark"] = cross["net_total"] > cross["buyhold_total"]
    save_table(cross, "03_cross_stock")
    results["cross_stock"] = cross.round(6).to_dict()
    plots.cross_stock(cross, FIGURES)

    regimes = strategy.regime_performance(
        strategy.to_position(strategy.momentum_signal(prices)), rets,
        {"crisis_2007_2009": ("2007-10-01", "2009-03-09"),
         "recovery_2009_2016": ("2009-03-10", "2016-03-01")},
    )
    for label, frame in regimes.items():
        save_table(frame, f"04_regime_{label}")
    results["regimes"] = {k: v.round(6).to_dict() for k, v in regimes.items()}

    # ----- 5. Volatility structure ------------------------------------------
    vol_range = vol.volatility_range_table(rets)
    save_table(vol_range, "05_volatility_range")
    results["volatility_range"] = vol_range.assign(
        max_date=vol_range["max_date"].astype(str)).round(6).to_dict()
    plots.rolling_volatility(vol.rolling_volatility(rets).dropna(), FIGURES)

    acf = vol.autocorrelation_table(rets)
    save_table(acf, "06_autocorrelation")
    results["autocorrelation"] = acf.round(6).to_dict()
    plots.autocorrelation(acf, FIGURES)

    plots.ewma_weights(vol.ewma_weights(), FIGURES)

    # ----- 6. Forecast comparison, both targets -----------------------------
    R = rets["AAPL"]
    baselines = forecast.baseline_forecasts(R)
    overlapping = forecast.evaluate(baselines, vol.overlapping_target(R))
    forward = forecast.evaluate(baselines, vol.forward_volatility(R))
    target_table = pd.DataFrame({"overlapping_target": overlapping,
                                 "forward_target": forward})
    save_table(target_table, "07_target_integrity")
    results["target_integrity"] = target_table.round(6).to_dict()
    plots.target_integrity(overlapping, forward, FIGURES)

    # ----- 7. HAR with purged split -----------------------------------------
    dataset = forecast.build_dataset(R)
    train, purged, test = forecast.purged_split(dataset)
    fit = forecast.fit_har(train)
    har_pred = forecast.predict_har(fit, test)

    scores = {
        "HAR (purged)": forecast.rmse(har_pred, test["target"]),
        "EWMA (λ=0.94)": forecast.rmse(baselines["ewma_squared_returns"], test["target"]),
        "Random walk": forecast.rmse(baselines["random_walk"], test["target"]),
    }

    # unpurged comparison, retained to quantify the leakage
    split = int(len(dataset) * 0.70)
    unpurged_fit = forecast.fit_har(dataset.iloc[:split])
    unpurged_rmse = forecast.rmse(forecast.predict_har(unpurged_fit, test), test["target"])

    results["har"] = {
        "n_train": len(train), "n_purged": len(purged), "n_test": len(test),
        "train_end": str(train.index.max().date()),
        "test_start": str(test.index.min().date()),
        "test_end": str(test.index.max().date()),
        "coefficients": dict(zip(["intercept", "beta_daily", "beta_weekly", "beta_monthly"],
                                 fit["betas"].round(6).tolist())),
        "max_solver_difference": fit["max_solver_difference"],
        "rmse": {k: round(v, 6) for k, v in scores.items()},
        "rmse_unpurged": round(unpurged_rmse, 6),
        "leakage_effect_pp": round((scores["HAR (purged)"] - unpurged_rmse) * 100, 4),
        "improvement_vs_ewma_pct": round(
            (scores["EWMA (λ=0.94)"] - scores["HAR (purged)"]) / scores["EWMA (λ=0.94)"] * 100, 3),
        "improvement_vs_rw_pct": round(
            (scores["Random walk"] - scores["HAR (purged)"]) / scores["Random walk"] * 100, 3),
    }
    save_table(pd.DataFrame({"rmse": scores}), "08_out_of_sample")
    plots.out_of_sample(scores, FIGURES)
    plots.purge_diagram(len(train), len(purged), len(test), FIGURES)

    # ----- 8. Persist --------------------------------------------------------
    with open(RESULTS / "project_results.json", "w") as fh:
        json.dump(results, fh, indent=2, default=str)

    print("HAR purged RMSE      :", round(scores["HAR (purged)"] * 100, 2), "%")
    print("EWMA same test dates :", round(scores["EWMA (λ=0.94)"] * 100, 2), "%")
    print("Random walk          :", round(scores["Random walk"] * 100, 2), "%")
    print("improvement vs EWMA  :", results["har"]["improvement_vs_ewma_pct"], "%")
    print("figures + tables written to", RESULTS)


if __name__ == "__main__":
    main()
