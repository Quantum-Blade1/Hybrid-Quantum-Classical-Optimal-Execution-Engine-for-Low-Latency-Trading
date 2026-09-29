"""Matplotlib figures for the experiment scripts (kept out of the qexec library)."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qexec.analysis.shortfall import ISComponents
from qexec.analysis.walk_forward import WindowResult
from qexec.execution.strategies.qubo import StrategyComparison

IS_COMPONENTS = ("Delay Cost", "Market Impact", "Timing Risk", "Opportunity Cost")


def plot_is_breakdown(strategies: Mapping[str, ISComponents], path: Path) -> Path:
    """Stacked bars of the shortfall components per strategy, total labelled on top."""
    labels = list(strategies)
    colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(labels))
    for component, color in zip(IS_COMPONENTS, colors, strict=True):
        values = np.array([s.to_dict()[component] for s in strategies.values()])
        ax.bar(labels, values, bottom=bottom, label=component, color=color, width=0.5)
        bottom += values
    for i, total in enumerate(bottom):
        ax.text(i, total, f"${total:,.0f}", ha="center", va="bottom")
    ax.set_title("Implementation Shortfall Decomposition")
    ax.set_ylabel("Cost ($)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_walk_forward(results: Sequence[WindowResult], path: Path) -> Path:
    """Cumulative shortfall of the three walk-forward strategies."""
    windows = [r.window_id for r in results]
    fig, ax = plt.subplots(figsize=(10, 6))
    series = [
        ("Static VWAP", [r.shortfall_static for r in results], "--"),
        ("Adaptive VWAP", [r.shortfall_adaptive for r in results], "-"),
        ("Hybrid (SA-QUBO)", [r.shortfall_hybrid for r in results], "-"),
    ]
    for label, values, style in series:
        ax.plot(windows, np.cumsum(values), style, label=label)
    ax.set_title("Walk-Forward Analysis: Cumulative Implementation Shortfall")
    ax.set_xlabel("Rolling Window")
    ax.set_ylabel("Cumulative Cost ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_strategy_comparison(
    comparison: StrategyComparison, prices: Sequence[float], path: Path
) -> Path:
    """Prices with average fills, total cost, slippage vs VWAP and cost breakdown."""
    reports = comparison.reports
    names = list(reports)
    colors = ["green", "orange", "red"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(prices, label="Market Price", color="blue", alpha=0.7)
    ax.axhline(
        comparison.vwap_report.benchmark_vwap, color="gray", linestyle="--", label="VWAP Benchmark"
    )
    for (name, report), color in zip(reports.items(), colors, strict=True):
        avg = report.average_execution_price
        ax.axhline(avg, color=color, alpha=0.8, label=f"{name} Avg: ${avg:.2f}")
    ax.set_xlabel("Minute")
    ax.set_ylabel("Price ($)")
    ax.set_title("Execution Price Comparison")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    costs = [r.total_cost for r in reports.values()]
    bars = ax.bar(names, costs, color=colors, alpha=0.7)
    ax.bar_label(bars, labels=[f"${c:.2f}" for c in costs])
    ax.set_ylabel("Total Cost ($)")
    ax.set_title("Total Execution Cost")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 0]
    slippages = [r.slippage_vs_vwap_bps for r in reports.values()]
    bars = ax.bar(names, slippages, color=colors, alpha=0.7)
    ax.bar_label(bars, labels=[f"{s:+.2f}" for s in slippages])
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Slippage (bps)")
    ax.set_title("Slippage vs VWAP Benchmark")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 1]
    x = np.arange(len(names))
    width = 0.25
    ax.bar(x - width / 2, [r.spread_cost for r in reports.values()], width, label="Spread Cost")
    ax.bar(x + width / 2, [r.impact_cost for r in reports.values()], width, label="Impact Cost")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Cost ($)")
    ax.set_title("Cost Breakdown by Component")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
