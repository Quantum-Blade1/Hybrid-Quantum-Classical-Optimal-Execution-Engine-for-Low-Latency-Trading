"""Phase 7 figures: real-data execution (results/real_data_dev, results/real_data_test) and
the recovered IBM hardware counts (results/hardware).

These inputs can be legitimately absent (no downloaded data, no recovered IBM counts);
their registry entries are `optional` and are skipped, not failed, when absent.
"""

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from figures.common import SOLVER_COLORS, Results, new_figure

REAL_COLORS = {
    "TWAP": "#9E9E9E",
    "VWAP": "#2196F3",
    "AC": "#4CAF50",
    "QUBO": "#FF9800",
    "Hybrid": "#D32F2F",
    "DP": "#7B1FA2",
}
COMPONENTS = ("spread_cost_bps", "impact_cost_bps", "timing_cost_bps", "opportunity_cost_bps")
SENSITIVITY = ("eval_x0.5", "primary", "eval_x2", "both_x0.5", "both_x2")


def _forest(comparisons: pd.DataFrame, title: str) -> Figure:
    df = comparisons[comparisons["variant"] == "primary"].reset_index(drop=True)
    symbols = list(dict.fromkeys(df["symbol"]))
    fig, axes = new_figure(1, len(symbols), figsize=(3.6 * len(symbols), 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, symbol in zip(axes, symbols, strict=True):
        g = df[df["symbol"] == symbol].reset_index(drop=True)
        y = np.arange(len(g))[::-1]
        err = [g["mean_diff_bps"] - g["ci_low"], g["ci_high"] - g["mean_diff_bps"]]
        colors = [REAL_COLORS.get(s, "k") for s in g["strategy"]]
        ax.errorbar(g["mean_diff_bps"], y, xerr=err, fmt="none", ecolor="k", lw=0.8, capsize=2)
        ax.scatter(g["mean_diff_bps"], y, c=colors, zorder=3, s=14)
        ax.axvline(0, color="k", lw=0.6)
        for yi, (_, row) in zip(y, g.iterrows(), strict=True):
            ax.annotate(
                f"p$_{{Holm}}$={row['p_holm']:.2g}",
                (row["ci_high"], yi),
                xytext=(3, -2),
                textcoords="offset points",
                fontsize=6,
            )
        ax.set_yticks(y, [f"{s} - {b}" for s, b in zip(g["strategy"], g["baseline"], strict=True)])
        ax.set_xlabel("Shortfall difference (bps), 95% CI")
        ax.set_title(symbol)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def fig_real_primary_comparisons(res: Results) -> Figure:
    return _forest(
        res.table("real_data_test", "comparisons"),
        "Held-out test days: paired shortfall difference per window (negative = cheaper)",
    )


def fig_real_dev_comparisons(res: Results) -> Figure:
    return _forest(
        res.table("real_data_dev", "comparisons"),
        "Development days (in-sample): paired shortfall difference per window",
    )


def fig_real_sensitivity(res: Results) -> Figure:
    df = res.table("real_data_test", "comparisons")
    variants = [v for v in SENSITIVITY if v in set(df["variant"])]
    symbols = list(dict.fromkeys(df["symbol"]))
    fig, axes = new_figure(1, len(symbols), figsize=(3.6 * len(symbols), 2.8), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(variants))
    for ax, symbol in zip(axes, symbols, strict=True):
        g = df[df["symbol"] == symbol]
        for (strategy, baseline), pair in g.groupby(["strategy", "baseline"], sort=False):
            h = pair.set_index("variant").reindex(variants)
            style = "o-" if strategy == "QUBO" else "s--"
            ax.plot(x, h["mean_diff_bps"], style, label=f"{strategy} - {baseline}", ms=3)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x, variants, rotation=30, fontsize=7)
        ax.set_title(symbol)
        ax.set_ylabel("Mean shortfall difference (bps)")
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle("Impact sensitivity (evaluator x, optimiser x)", fontsize=10)
    fig.tight_layout()
    return fig


def fig_real_cost_components(res: Results) -> Figure:
    df = res.table("real_data_test", "strategy_summary")
    df = df[(df["variant"] == "primary") & df["strategy"].isin(list(REAL_COLORS))]
    symbols = list(dict.fromkeys(df["symbol"]))
    fig, axes = new_figure(1, len(symbols), figsize=(3.6 * len(symbols), 2.8))
    axes = np.atleast_1d(axes)
    hatches = ("", "//", "..", "xx")
    for ax, symbol in zip(axes, symbols, strict=True):
        g = df[df["symbol"] == symbol].reset_index(drop=True)
        x = np.arange(len(g))
        pos, neg = np.zeros(len(g)), np.zeros(len(g))
        for comp, hatch in zip(COMPONENTS, hatches, strict=True):
            v = g[f"{comp}_mean"].to_numpy()
            base = np.where(v >= 0, pos, neg)
            ax.bar(
                x,
                v,
                bottom=base,
                hatch=hatch,
                color="white",
                edgecolor="k",
                lw=0.5,
                label=comp.replace("_cost_bps", ""),
            )
            pos, neg = pos + np.clip(v, 0, None), neg + np.clip(v, None, 0)
        ax.plot(x, g["shortfall_bps_mean"], "D", color="#D32F2F", ms=4, label="total")
        ax.set_xticks(x, g["strategy"], fontsize=7)
        ax.set_title(symbol)
        ax.set_ylabel("Mean cost (bps)")
    axes[0].legend(fontsize=6)
    fig.suptitle("Held-out cost components (primary)", fontsize=10)
    fig.tight_layout()
    return fig


def fig_real_impact_calibration(res: Results) -> Figure:
    df = res.table("real_data_dev", "impact_bins")
    symbols = list(dict.fromkeys(df["symbol"]))
    fig, axes = new_figure(1, len(symbols), figsize=(3.4 * len(symbols), 2.6))
    axes = np.atleast_1d(axes)
    for ax, symbol in zip(axes, symbols, strict=True):
        g = df[df["symbol"] == symbol]
        beta = float(g["beta_bps"].iloc[0])
        ax.plot(g["x_mean"], g["return_mean_bps"], "o", ms=3, color="#2196F3", label="binned")
        xs = np.linspace(g["x_mean"].min(), g["x_mean"].max(), 50)
        ax.plot(xs, beta * xs, "-", color="#D32F2F", label=f"OLS $\\beta$={beta:.2f} bps")
        ax.set_xlabel("Signed volume / expected volume")
        ax.set_ylabel("1-min return (bps)")
        ax.set_title(f"{symbol} (development days)")
        ax.legend(fontsize=6)
    fig.tight_layout()
    return fig


def fig_real_qubo_ac_gap(res: Results) -> Figure:
    df = res.table("real_data_test", "gaps")
    df = df[df["variant"] == "primary"]
    fig, ax = new_figure(figsize=(4.2, 2.8))
    labels, qubo, dp = [], [], []
    for (symbol, size, horizon), g in df.groupby(["symbol", "size_pct", "horizon"]):
        labels.append(f"{symbol[:4]} {size}% {horizon}m")
        qubo.append((g["objective_QUBO"] - g["objective_AC"]).mean())
        dp.append((g["objective_DP"] - g["objective_AC"]).mean())
    x = np.arange(len(labels))
    ax.bar(x - 0.2, qubo, 0.4, color=REAL_COLORS["QUBO"], label="QUBO (SA) - AC")
    ax.bar(x + 0.2, dp, 0.4, color=REAL_COLORS["DP"], label="integer optimum (DP) - AC")
    ax.set_xticks(x, labels, rotation=40, fontsize=6)
    ax.set_ylabel("Model objective gap (bps)")
    ax.set_title("QUBO vs discretized AC in the cost model")
    ax.legend(fontsize=6)
    fig.tight_layout()
    return fig


def fig_hw_ibm_success_prob(res: Results) -> Figure:
    s = res.table("hardware", "summary")
    note = str(res.manifest("hardware").get("data", ""))
    fig, ax = new_figure(figsize=(3.6, 2.6))
    ax.semilogy(s["n"], s["success_probability"], "o-", color="#D32F2F", label="IBM (final jobs)")
    ax.semilogy(
        s["n"],
        s["uniform_success_probability"],
        "--",
        color=SOLVER_COLORS["Random"],
        label="uniform random",
    )
    for solver in ("qaoa_ideal", "qaoa_noisy"):
        col = f"{solver}_p1_success_probability"
        if col in s:
            key = "QAOA_Ideal" if solver == "qaoa_ideal" else "QAOA_Noisy"
            ax.semilogy(s["n"], s[col], "s:", color=SOLVER_COLORS[key], label=f"Aer {key[5:]} p=1")
    ax.set_xlabel("n (qubits)")
    ax.set_ylabel("P(optimal bitstring)")
    ax.set_title(
        f"Success probability{' [' + note + ']' if 'SYNTHETIC' in note else ''}", fontsize=8
    )
    ax.legend(fontsize=6)
    fig.tight_layout()
    return fig


def fig_hw_ibm_approx_ratio(res: Results) -> Figure:
    jobs = res.table("hardware", "jobs")
    fig, ax = new_figure(figsize=(3.6, 2.6))
    for role, marker in (("optimization", "."), ("final", "o")):
        g = jobs[jobs["role"] == role]
        ax.plot(g["n"], g["approx_ratio_mean"], marker, ls="none", label=f"IBM {role} jobs")
    u = jobs.groupby("n")["uniform_approx_ratio_mean"].first()
    ax.plot(u.index, u.values, "k--", label="uniform random")
    ax.set_xlabel("n (qubits)")
    ax.set_ylabel("Approximation ratio of mean energy")
    ax.set_title("Recomputed from raw counts")
    ax.legend(fontsize=6)
    fig.tight_layout()
    return fig
