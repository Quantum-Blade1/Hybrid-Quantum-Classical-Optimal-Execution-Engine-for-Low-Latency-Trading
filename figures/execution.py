import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from figures.common import STRATEGY_COLORS, Results, ci_errorbars, metric, new_figure


def fig19_execution_schedule_comparison(res: Results) -> Figure:
    s = res.table("ac_frontier", "schedules")
    fig, ax = new_figure(figsize=(3.5, 2.8))
    ax.step(s["step"], s["twap"], where="mid", label="TWAP", color="#9E9E9E")
    ax.step(s["step"], s["vwap_expected_profile"], where="mid", label="VWAP", color="#2196F3")
    ax.step(
        s["step"],
        s["almgren_chriss_1e-4"],
        where="mid",
        label="AC ($\\lambda$=1e-4)",
        color="#4CAF50",
    )
    ax.step(
        s["step"], s["qubo_raw"], where="mid", label="QUBO (as solved)", color="#D32F2F", ls="--"
    )
    ax.step(s["step"], s["qubo_repaired"], where="mid", label="QUBO (repaired)", color="#D32F2F")
    ax.set_xlabel("Time slice")
    ax.set_ylabel("Shares")
    ax.set_title(f"Schedules, {round(s['twap'].sum()):,} shares")
    ax.legend(fontsize=6)
    fig.tight_layout()
    return fig


def fig20_almgren_chriss_frontier(res: Results) -> Figure:
    f = res.table("ac_frontier", "frontier")
    fig, ax = new_figure(figsize=(3.5, 2.8))
    sd = np.sqrt(f["variance"])
    sc = ax.scatter(
        sd,
        f["expected_cost"],
        c=np.log10(f["risk_aversion"]),
        cmap="RdYlBu_r",
        s=30,
        edgecolors="black",
        linewidths=0.5,
    )
    ax.plot(sd, f["expected_cost"], "k-", alpha=0.3, linewidth=0.5)
    ax.set_xlabel(r"$\sqrt{V[C]}$ ($)")
    ax.set_ylabel("E[C] ($)")
    ax.set_title("Almgren-Chriss efficient frontier")
    fig.colorbar(sc, ax=ax, label=r"$\log_{10}\lambda$", shrink=0.9)
    fig.tight_layout()
    return fig


def _paired_panel(ax, paired: pd.DataFrame, baseline: str, order: list[str]) -> None:
    d = paired[(paired["baseline"] == baseline) & (paired["metric"] == "shortfall_bps")]
    d = d.set_index("strategy").reindex([s for s in order if s in set(d["strategy"])])
    y = np.arange(len(d))
    err = [
        (d["mean_diff"] - d["ci_low"]).clip(lower=0),
        (d["ci_high"] - d["mean_diff"]).clip(lower=0),
    ]
    ax.errorbar(d["mean_diff"], y, xerr=err, fmt="o", color="black", capsize=3)
    for yi, (_name, row) in zip(y, d.iterrows(), strict=True):
        ax.text(row["ci_high"], yi, f"  p={row['wilcoxon_p']:.2g}", va="center", fontsize=6)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(d.index)
    ax.set_xlabel(f"Shortfall minus {baseline} (bps)")


def fig21_walk_forward_shortfall(res: Results) -> Figure:
    summary = metric(res.table("walk_forward", "summary"), "shortfall_bps")
    paired = res.table("walk_forward", "paired")
    order = ["TWAP", "Static", "Adaptive", "Hybrid"]
    s = summary.set_index("strategy").loc[order]
    fig, (ax1, ax2) = new_figure(1, 2, figsize=(7, 2.8))
    ax1.bar(
        order,
        s["mean"],
        yerr=ci_errorbars(s),
        capsize=3,
        alpha=0.75,
        color=[STRATEGY_COLORS[o] for o in order],
    )
    ax1.set_ylabel("Mean shortfall (bps)")
    ax1.set_title(f"Walk-forward, {int(s['count'].iloc[0])} seeds")
    _paired_panel(ax2, paired, "TWAP", order)
    ax2.set_title("Paired vs TWAP (95% CI, Wilcoxon p)")
    fig.tight_layout()
    return fig


def fig22_is_decomposition(res: Results) -> Figure:
    summary = res.table("is_comparison", "summary")
    paired = res.table("is_comparison", "paired")
    order = ["TWAP", "VWAP", "SA-QUBO", "Hybrid"]
    parts = [
        ("spread_cost_bps", "Half spread", "#ff9999"),
        ("impact_cost_bps", "Impact", "#66b3ff"),
        ("opportunity_cost_bps", "Opportunity", "#ffcc99"),
    ]
    fig, (ax1, ax2) = new_figure(1, 2, figsize=(7, 2.8))
    x = np.arange(len(order))
    bottom = np.zeros(len(order))
    for column, label, color in parts:
        values = metric(summary, column).set_index("strategy").loc[order, "mean"].to_numpy()
        ax1.bar(x, values, bottom=bottom, color=color, label=label, width=0.6)
        bottom += values
    timing = metric(summary, "timing_cost_bps").set_index("strategy").loc[order]
    ax1.errorbar(
        x,
        timing["mean"],
        yerr=ci_errorbars(timing),
        fmt="D",
        color="#2E7D32",
        capsize=2,
        label="Timing (mean, CI)",
    )
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(order)
    ax1.set_ylabel("bps of arrival notional")
    ax1.set_title(f"Shortfall components ({int(timing['count'].iloc[0])} seeds)")
    ax1.legend(fontsize=6)
    _paired_panel(ax2, paired, "TWAP", order)
    ax2.set_title("Total shortfall, paired vs TWAP")
    fig.tight_layout()
    return fig


def fig23_stress_test_results(res: Results) -> Figure:
    summary = res.table("stress_test", "summary")
    paired = res.table("stress_test", "paired")
    scenarios = list(dict.fromkeys(summary["scenario"]))
    strategies = ["VWAP", "SA-QUBO", "Hybrid"]
    fig, (ax1, ax2) = new_figure(1, 2, figsize=(7, 3))
    x = np.arange(len(scenarios))
    width = 0.8 / len(strategies)
    d = paired[(paired["baseline"] == "TWAP") & (paired["metric"] == "shortfall_bps")]
    for k, strategy in enumerate(strategies):
        rows = d[d["strategy"] == strategy].set_index("scenario").loc[scenarios]
        err = [
            (rows["mean_diff"] - rows["ci_low"]).clip(lower=0),
            (rows["ci_high"] - rows["mean_diff"]).clip(lower=0),
        ]
        ax1.bar(
            x + k * width,
            rows["mean_diff"],
            width,
            yerr=err,
            capsize=2,
            alpha=0.8,
            color=STRATEGY_COLORS[strategy],
            label=strategy,
        )
    ax1.axhline(0, color="black", linewidth=0.6)
    ax1.set_xticks(x + width * (len(strategies) - 1) / 2)
    ax1.set_xticklabels([s.replace(" ", "\n") for s in scenarios], fontsize=7)
    ax1.set_ylabel("Shortfall minus TWAP (bps)")
    ax1.set_title("Paired vs TWAP (95% CI)")
    ax1.legend(fontsize=6)
    fill = metric(summary, "fill_rate")
    all_strategies = ["TWAP", *strategies]
    width = 0.8 / len(all_strategies)
    for k, strategy in enumerate(all_strategies):
        rows = fill[fill["strategy"] == strategy].set_index("scenario").loc[scenarios]
        ax2.bar(
            x + k * width,
            rows["mean"],
            width,
            alpha=0.8,
            color=STRATEGY_COLORS[strategy],
            label=strategy,
        )
    ax2.set_xticks(x + width * (len(all_strategies) - 1) / 2)
    ax2.set_xticklabels([s.replace(" ", "\n") for s in scenarios], fontsize=7)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Fill rate")
    ax2.set_title(f"Fill rate ({int(fill['count'].iloc[0])} seeds)")
    fig.tight_layout()
    return fig


def fig24_latency_distribution(res: Results) -> Figure:
    samples = res.table("latency", "samples")
    env = res.manifest("latency")["environment"]
    machine = env.get("cpu_model") or env.get("processor") or env.get("machine")
    fig, (ax1, ax2) = new_figure(1, 2, figsize=(7, 2.8))
    series = [
        ("runtime", "fast_path", "runtime tick work", "#1976D2"),
        ("pipeline", "fast_path", "pipeline tick work", "#D32F2F"),
        ("runtime", "tick_lateness", "runtime tick lateness", "#7B1FA2"),
    ]
    selected = samples[samples["component"].isin({c for _, c, _, _ in series})]
    positive = selected["duration_us"][selected["duration_us"] > 0]
    bins = np.logspace(np.log10(max(positive.min(), 0.5)), np.log10(positive.max()), 60)
    for source, component, name, color in series:
        d = samples[(samples["source"] == source) & (samples["component"] == component)]
        d = d["duration_us"]
        if d.empty:
            continue
        label = f"{name}: median {d.median():.0f}, p99 {d.quantile(0.99):.0f} $\\mu$s"
        ax1.hist(d.clip(lower=0.5), bins=bins, alpha=0.55, color=color, label=label)
    ax1.set_xscale("log")
    ax1.set_xlabel("$\\mu$s")
    ax1.set_ylabel("Ticks")
    ax1.set_title("Fast path (CPython)")
    ax1.legend(fontsize=6)
    slow = samples[samples["component"] == "slow_path_optimize"]
    data = [slow[slow["source"] == s]["duration_us"] / 1000 for s in ("runtime", "pipeline")]
    ax2.boxplot(data, widths=0.5)
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(["runtime SA", "pipeline SA"])
    ax2.set_yscale("log")
    ax2.set_ylabel("Slow-path solve (ms)")
    ax2.set_title("Slow path")
    fig.suptitle(f"Measured on {machine}, Python {env['python']}", fontsize=8)
    fig.tight_layout()
    return fig
