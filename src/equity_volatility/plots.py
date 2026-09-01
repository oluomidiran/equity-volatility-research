"""Figure generation with a consistent visual identity."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NAVY = "#1F3864"
BLUE = "#2E75B6"
TEAL = "#17A2A2"
CORAL = "#E4572E"
AMBER = "#F0A202"
GRAY = "#7A828E"
LIGHT = "#E8ECF1"

PALETTE = [BLUE, CORAL, TEAL, AMBER]
CRISIS = ("2007-10-01", "2009-03-09")


def _style(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, fontsize=12, fontweight="bold", color=NAVY, pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9.5, color="#3A3A3A")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.5, color="#3A3A3A")
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#C8CDD4")
    ax.tick_params(labelsize=9, colors="#3A3A3A")


def _save(fig, outdir, name):
    path = Path(outdir) / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def strategy_growth(curves: dict, outdir, name="01_strategy_growth"):
    """Cumulative wealth for each strategy variant against buy-and-hold."""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    styles = {"Buy and hold": (GRAY, "--", 1.8), "Momentum (gross)": (BLUE, "-", 1.8),
              "Momentum (net 10bp)": (TEAL, "-", 1.8), "Mean reversion": (CORAL, "-", 1.4)}
    for label, series in curves.items():
        color, ls, lw = styles.get(label, (NAVY, "-", 1.5))
        ax.plot(series.index, series.values, label=label, color=color, linestyle=ls, linewidth=lw)
    ax.axhline(1.0, color="#B0B6BE", linewidth=0.8, zorder=0)
    _style(ax, "Apple: strategy growth under look-ahead-safe execution",
           ylabel="Growth of $1")
    ax.legend(frameon=False, fontsize=9, ncol=2)
    return _save(fig, outdir, name)


def cost_sensitivity(table, crossover_bps, zero_bps, buyhold_return,
                     outdir, name="02_cost_sensitivity"):
    """Net return against assumed cost, with both break-even points marked."""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(table["cost_bps"], table["net_return"] * 100, color=BLUE,
            marker="o", markersize=4.5, linewidth=1.8, label="Strategy, net")
    ax.axhline(buyhold_return * 100, color=GRAY, linestyle="--", linewidth=1.4,
               label=f"Buy and hold ({buyhold_return*100:.1f}%)")
    ax.axhline(0, color="#B0B6BE", linewidth=0.9)
    ax.axvline(crossover_bps, color=AMBER, linestyle=":", linewidth=1.6,
               label=f"Benchmark crossover ≈ {crossover_bps:.0f} bp")
    ax.axvline(zero_bps, color=CORAL, linestyle=":", linewidth=1.6,
               label=f"Zero-return point ≈ {zero_bps:.0f} bp")
    _style(ax, "Transaction-cost sensitivity: two distinct break-even points",
           "Cost per position change (basis points)", "Compounded total return (%)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    return _save(fig, outdir, name)


def cross_stock(table, outdir, name="03_cross_stock"):
    """Gross, net, and benchmark returns across the four-equity universe."""
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(table))
    w = 0.26
    ax.bar(x - w, table["gross_total"] * 100, w, label="Strategy, gross", color=BLUE)
    ax.bar(x, table["net_total"] * 100, w, label="Strategy, net", color=TEAL)
    ax.bar(x + w, table["buyhold_total"] * 100, w, label="Buy and hold", color=GRAY)
    ax.axhline(0, color="#8A9099", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(table.index)
    _style(ax, "Cross-stock robustness, 2007–2016: the rule did not generalize",
           ylabel="Compounded total return (%)")
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, outdir, name)


def rolling_volatility(vol, outdir, name="04_rolling_volatility"):
    """Annualized 21-day rolling volatility with the crisis window shaded."""
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    for color, col in zip(PALETTE, vol.columns):
        ax.plot(vol.index, vol[col] * 100, label=col, color=color, linewidth=1.0, alpha=0.85)
    ax.axvspan(pd.Timestamp(CRISIS[0]), pd.Timestamp(CRISIS[1]),
               color=CORAL, alpha=0.10, label="2008 crisis")
    _style(ax, "Annualized 21-day rolling volatility: a moving target, not a constant",
           ylabel="Volatility (%)")
    ax.legend(frameon=False, fontsize=8.5, ncol=5)
    return _save(fig, outdir, name)


def autocorrelation(table, outdir, name="05_autocorrelation"):
    """Direction versus magnitude persistence."""
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    cols = ["returns_lag1", "abs_returns_lag1", "volatility_lag21", "volatility_lag63"]
    labels = ["Returns\n(lag 1)", "|Returns|\n(lag 1)", "Volatility\n(lag 21)", "Volatility\n(lag 63)"]
    x = np.arange(len(cols))
    w = 0.2
    for i, (color, asset) in enumerate(zip(PALETTE, table.index)):
        ax.bar(x + (i - 1.5) * w, table.loc[asset, cols].values, w, label=asset, color=color)
    ax.axhline(0, color="#8A9099", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    _style(ax, "Direction is not forecastable; magnitude is",
           ylabel="Autocorrelation")
    ax.legend(frameon=False, fontsize=9, ncol=4)
    return _save(fig, outdir, name)


def ewma_weights(weights, outdir, name="06_ewma_weights"):
    """Exponential weight decay and cumulative share."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.bar(weights.index[:40], weights.values[:40] * 100, color=BLUE, width=0.8)
    _style(a1, "EWMA weight by age (λ = 0.94)", "Days ago", "Weight (%)")
    cum = weights.cumsum() / weights.sum()
    a2.plot(cum.index, cum.values * 100, color=TEAL, linewidth=2)
    a2.axvline(16.7, color=CORAL, linestyle=":", linewidth=1.5,
               label="Effective memory ≈ 16.7 days")
    _style(a2, "Cumulative weight", "Days included", "Cumulative weight (%)")
    a2.legend(frameon=False, fontsize=9)
    return _save(fig, outdir, name)


def target_integrity(overlapping, forward, outdir, name="07_target_integrity"):
    """The evaluation target reverses the model ranking."""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    labels = [n.replace("_", " ") for n in overlapping.index]
    x = np.arange(len(labels))
    w = 0.36
    ax.bar(x - w / 2, overlapping.values * 100, w, label="Overlapping target (flawed)", color=GRAY)
    ax.bar(x + w / 2, forward.values * 100, w, label="Forward target (honest)", color=BLUE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=8.5)
    _style(ax, "Forecast ranking depends entirely on target construction",
           ylabel="RMSE (percentage points)")
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, outdir, name)


def out_of_sample(scores: dict, outdir, name="08_out_of_sample"):
    """Held-out RMSE across models on identical test dates."""
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    names = list(scores.keys())
    vals = [scores[k] * 100 for k in names]
    colors = [TEAL if "HAR" in n else BLUE if "EWMA" in n else GRAY for n in names]
    bars = ax.bar(names, vals, color=colors, width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.12, f"{v:.2f}",
                ha="center", fontsize=9.5, fontweight="bold", color=NAVY)
    _style(ax, "Out-of-sample forecast accuracy, purged split (lower is better)",
           ylabel="RMSE (percentage points)")
    ax.set_ylim(0, max(vals) * 1.18)
    plt.xticks(rotation=12, ha="right", fontsize=9)
    return _save(fig, outdir, name)


def purge_diagram(n_train, n_purge, n_test, outdir, name="09_purge_diagram"):
    """Why the train/test boundary requires a purge."""
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.barh(0, n_train, color=TEAL, label=f"Training ({n_train})")
    ax.barh(0, n_purge, left=n_train, color=CORAL, label=f"Purged ({n_purge})")
    ax.barh(0, n_test, left=n_train + n_purge, color=BLUE, label=f"Test ({n_test})")
    ax.set_yticks([])
    ax.set_ylim(-0.6, 0.6)
    ax.set_xlabel("Chronologically ordered observations", fontsize=9.5, color="#3A3A3A", labelpad=8)
    ax.set_title("Purged split: removing labels whose 21-day targets reach into the test period",
                 fontsize=11.5, fontweight="bold", color=NAVY, pad=32)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, 1.02))
    path = Path(outdir) / f"{name}.png"
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path
