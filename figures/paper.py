"""Figures of the IEEE TQE manuscript (paper/ieee/main.tex), sized for IEEE columns.

Single-column figures are 3.5 in wide and double-column figures 7.16 in wide, drawn at
their final size with 7-8 pt text, so nothing is scaled down in the PDF. Series are told
apart by marker, line style and gray level as well as colour, so every figure reads in
grayscale. Like the other figure modules, these read numbers only from `results/`.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from figures.common import Results, metric

COLUMN_IN = 3.5
DOUBLE_IN = 7.16

IEEE_RC: dict[str, Any] = {
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "legend.frameon": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.4,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.5,
    "errorbar.capsize": 1.5,
    "pdf.fonttype": 42,
}

# Gray levels chosen to stay distinct when printed in grayscale.
GRAY = {"black": "0.0", "dark": "0.3", "mid": "0.5", "light": "0.72"}
STRATEGY_STYLE: dict[str, dict[str, Any]] = {
    "TWAP": {"color": GRAY["light"], "marker": "v", "hatch": ""},
    "VWAP": {"color": GRAY["mid"], "marker": "^", "hatch": "..."},
    "AC": {"color": GRAY["dark"], "marker": "D", "hatch": "///"},
    "QUBO": {"color": GRAY["black"], "marker": "o", "hatch": "xxx"},
    "Hybrid": {"color": GRAY["black"], "marker": "s", "hatch": "\\\\\\"},
}
SOLVER_STYLE: dict[str, dict[str, Any]] = {
    "QAOA_Ideal": {"color": "0.0", "marker": "o", "ls": "-", "label": "QAOA (ideal)"},
    "QAOA_Noisy": {"color": "0.45", "marker": "s", "ls": "--", "label": "QAOA (noisy)"},
    "Uniform": {"color": "0.65", "marker": "x", "ls": ":", "label": "Uniform (same shots)"},
    "SA": {"color": "0.25", "marker": "^", "ls": "-.", "label": "SA (16 restarts)"},
    "SA_1": {"color": "0.55", "marker": "v", "ls": "--", "label": "SA (1 restart)"},
    "Greedy": {"color": "0.7", "marker": "D", "ls": ":", "label": "Greedy"},
}
SYMBOLS = ("BTCUSDT", "LINKUSDT")


def _figure(width: float, height: float, *args: Any, **kwargs: Any) -> tuple[Figure, Any]:
    plt.rcParams.update(IEEE_RC)
    fig, axes = plt.subplots(*args, figsize=(width, height), **kwargs)
    return fig, axes


def _panel_label(ax: Any, text: str) -> None:
    ax.text(-0.02, 1.02, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=8)


# --------------------------------------------------------------------------------------
# Architecture (illustrative)
# --------------------------------------------------------------------------------------


def _box(ax: Any, xy: tuple[float, float], w: float, h: float, *, text: str, fill: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            xy,
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.015",
            linewidth=0.7,
            edgecolor="0.0",
            facecolor=fill,
        )
    )
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=6.3)


def _arrow(ax: Any, start: tuple[float, float], end: tuple[float, float], **kw: Any) -> None:
    style = {"arrowstyle": "-|>", "mutation_scale": 7, "linewidth": 0.7, "color": "0.0"}
    style.update(kw)
    ax.add_patch(FancyArrowPatch(start, end, **style))


def fig_paper_architecture(_: Results) -> Figure:
    """Latency-decoupled hybrid: fast tick loop, slow optimizer thread, one policy slot."""
    fig, ax = _figure(COLUMN_IN, 2.35)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.97, "Slow path (optimizer thread)", fontsize=6.8, weight="bold", va="top")
    ax.text(0.02, 0.40, "Fast path (tick thread)", fontsize=6.8, weight="bold", va="top")
    _box(
        ax,
        (0.02, 0.62),
        0.28,
        0.26,
        text="Cost model\n(remaining order,\nmarket state)",
        fill="0.93",
    )
    _box(
        ax,
        (0.36, 0.62),
        0.28,
        0.26,
        text="Execution QUBO\n$\\min_z\\, z^{\\top}Qz$",
        fill="0.93",
    )
    _box(
        ax,
        (0.70, 0.62),
        0.28,
        0.26,
        text="Solver: SA\n(exact, QAOA offline);\nuniform on failure",
        fill="0.93",
    )
    _box(ax, (0.62, 0.43), 0.36, 0.12, text="Latest-value policy slot", fill="0.80")
    _box(ax, (0.02, 0.06), 0.22, 0.26, text="Market data\n(bars / ticks)", fill="1.0")
    _box(
        ax,
        (0.30, 0.06),
        0.30,
        0.26,
        text="Tick loop: poll slot\n(never waits), re-plan\nremaining shares",
        fill="1.0",
    )
    _box(
        ax,
        (0.66, 0.06),
        0.32,
        0.26,
        text="Child order\n(TWAP fallback if\nno policy yet)",
        fill="1.0",
    )
    _arrow(ax, (0.30, 0.75), (0.36, 0.75))
    _arrow(ax, (0.64, 0.75), (0.70, 0.75))
    _arrow(ax, (0.84, 0.62), (0.84, 0.55))
    _arrow(ax, (0.62, 0.49), (0.45, 0.32), linestyle="--")
    _arrow(ax, (0.24, 0.19), (0.30, 0.19))
    _arrow(ax, (0.60, 0.19), (0.66, 0.19))
    _arrow(ax, (0.13, 0.32), (0.13, 0.62), linestyle=":")
    ax.text(0.145, 0.47, "state", fontsize=5.8, va="center")
    ax.text(0.855, 0.585, "publish", fontsize=5.8, va="center")
    ax.text(0.55, 0.38, "poll (non-blocking)", fontsize=5.8, va="center")
    fig.tight_layout(pad=0.1)
    return fig


# --------------------------------------------------------------------------------------
# Real data (held-out test days)
# --------------------------------------------------------------------------------------

_PRIMARY_ORDER = [
    ("QUBO", "TWAP"),
    ("QUBO", "VWAP"),
    ("QUBO", "AC"),
    ("Hybrid", "TWAP"),
    ("Hybrid", "VWAP"),
    ("Hybrid", "AC"),
]


def fig_paper_primary(res: Results) -> Figure:
    """Forest plot of the 12 pre-registered held-out comparisons with the ±0.5 bps band."""
    comp = res.table("real_data_test", "comparisons")
    comp = comp[comp["variant"] == "primary"]
    fig, axes = _figure(COLUMN_IN, 3.1, 2, 1)
    for ax, symbol in zip(axes, SYMBOLS, strict=True):
        d = comp[comp["symbol"] == symbol].set_index(["strategy", "baseline"])
        ax.axvspan(-0.5, 0.5, color="0.9", zorder=0, lw=0)
        ax.axvline(0, color="0.0", lw=0.6)
        for i, key in enumerate(_PRIMARY_ORDER):
            if key not in d.index:
                continue
            row = d.loc[key]
            y = len(_PRIMARY_ORDER) - 1 - i
            st = STRATEGY_STYLE[key[0]]
            ax.errorbar(
                row["mean_diff_bps"],
                y,
                xerr=[
                    [row["mean_diff_bps"] - row["ci_low"]],
                    [row["ci_high"] - row["mean_diff_bps"]],
                ],
                fmt=st["marker"],
                color="0.0",
                mfc="0.0" if key[0] == "QUBO" else "white",
                ms=4,
                lw=0.8,
            )
            ax.text(
                1.01,
                y,
                f"$p_{{\\mathrm{{Holm}}}}$={row['p_holm']:.2f}",
                transform=ax.get_yaxis_transform(),
                fontsize=6,
                va="center",
            )
        ax.set_yticks(range(len(_PRIMARY_ORDER)))
        ax.set_yticklabels([f"{a} $-$ {b}" for a, b in reversed(_PRIMARY_ORDER)])
        ax.set_title(symbol, loc="left", fontsize=7.5)
        ax.grid(axis="y", visible=False)
    axes[-1].set_xlabel("Shortfall difference (bps): mean, 95% CI; negative = strategy cheaper")
    fig.tight_layout(pad=0.2, h_pad=0.6)
    return fig


def fig_paper_qubo_ac_gap(res: Results) -> Figure:
    """Model-objective excess over discretized AC per order cell (TWAP, VWAP, QUBO)."""
    gaps = res.table("real_data_test", "gaps")
    gaps = gaps[gaps["variant"] == "primary"].copy()
    for name in ("TWAP", "VWAP", "QUBO", "DP"):
        gaps[name] = gaps[f"objective_{name}"] - gaps["objective_AC"]
    cells = gaps.groupby(["symbol", "size_pct", "horizon"])[["TWAP", "VWAP", "QUBO"]].mean()
    fig, ax = _figure(COLUMN_IN, 2.0)
    x = np.arange(len(cells))
    width = 0.27
    floor = 1e-5
    for k, name in enumerate(("TWAP", "VWAP", "QUBO")):
        st = STRATEGY_STYLE[name]
        vals = np.maximum(cells[name].to_numpy(), floor)
        ax.bar(
            x + (k - 1) * width,
            vals,
            width,
            color=st["color"],
            edgecolor="0.0",
            linewidth=0.4,
            hatch=st["hatch"],
            label=f"{name} $-$ AC",
        )
    labels = [f"{p:g}%\n{h} min" for _, p, h in cells.index]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5.8)
    symbols = [s for s, _, _ in cells.index]
    for symbol in dict.fromkeys(symbols):
        idx = [i for i, s in enumerate(symbols) if s == symbol]
        ax.text(np.mean(idx), 0.3, symbol, ha="center", va="center", fontsize=6.5)
        if idx[0] > 0:
            ax.axvline(idx[0] - 0.5, color="0.0", lw=0.5)
    ax.set_yscale("log")
    ax.set_ylim(floor, 1)
    ax.set_ylabel("Model cost above AC (bps)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=6)
    ax.grid(axis="x", visible=False)
    fig.tight_layout(pad=0.2)
    return fig


def fig_paper_signal_vs_noise(res: Results) -> Figure:
    """What a schedule can change (model costs) against what it cannot (price noise)."""
    summary = res.table("real_data_test", "strategy_summary")
    summary = summary[summary["variant"] == "primary"]
    comp = res.table("real_data_test", "comparisons")
    comp = comp[comp["variant"] == "primary"]
    gaps = res.table("real_data_test", "gaps")
    gaps = gaps[gaps["variant"] == "primary"]
    rows = []
    for symbol in SYMBOLS:
        s = summary[summary["symbol"] == symbol]
        s = s[s["strategy"].isin(STRATEGY_STYLE)]
        c = comp[comp["symbol"] == symbol]
        g = gaps[gaps["symbol"] == symbol]
        rows.append(
            {
                "symbol": symbol,
                "TWAP model cost above AC": float((g["objective_TWAP"] - g["objective_AC"]).mean()),
                "Realized spread + impact": float(
                    (s["spread_cost_bps_mean"] + s["impact_cost_bps_mean"]).mean()
                ),
                "Largest |mean paired difference|": float(c["mean_diff_bps"].abs().max()),
                "Mean 95% CI half-width": float(((c["ci_high"] - c["ci_low"]) / 2).mean()),
                "Per-order std of shortfall": float(s["shortfall_std"].mean()),
            }
        )
    df = pd.DataFrame(rows).set_index("symbol")
    fig, ax = _figure(COLUMN_IN, 2.5)
    shades = ["0.95", "0.8", "0.6", "0.4", "0.15"]
    hatches = ["///", "", "...", "", ""]
    width = 0.16
    x = np.arange(len(df))
    for k, col in enumerate(df.columns):
        ax.bar(
            x + (k - 2) * width,
            df[col],
            width,
            color=shades[k],
            edgecolor="0.0",
            linewidth=0.4,
            hatch=hatches[k],
            label=col,
        )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(df.index)
    ax.set_ylabel("bps (log scale)")
    ax.legend(fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.grid(axis="x", visible=False)
    fig.tight_layout(pad=0.2)
    return fig


# --------------------------------------------------------------------------------------
# Synthetic strategy experiments
# --------------------------------------------------------------------------------------


def fig_paper_synthetic(res: Results) -> Figure:
    """Paired shortfall difference vs TWAP for SA-QUBO and Hybrid, every synthetic setting."""
    rows: list[tuple[str, str, pd.Series]] = []

    def add(label: str, paired: pd.DataFrame, **where: Any) -> None:
        d = paired[(paired["baseline"] == "TWAP") & (paired["metric"] == "shortfall_bps")]
        for column, value in where.items():
            d = d[d[column] == value]
        for strategy in ("SA-QUBO", "Hybrid"):
            r = d[d["strategy"] == strategy]
            if len(r):
                rows.append((label, strategy, r.iloc[0]))

    add("1 h, 100k sh.", res.table("is_comparison", "paired"))
    day = res.table("strategy_comparison", "paired")
    for frac in sorted(day["order_fraction_of_adv"].unique()):
        add(f"1 day, {100 * frac:g}% ADV", day, order_fraction_of_adv=frac)
    stress = res.table("stress_test", "paired")
    for scenario in stress["scenario"].unique():
        add(str(scenario), stress, scenario=scenario)
    add("Walk-forward", res.table("walk_forward", "paired"))

    labels = list(dict.fromkeys(label for label, _, _ in rows))
    fig, ax = _figure(COLUMN_IN, 2.9)
    ax.axvline(0, color="0.0", lw=0.6)
    for label, strategy, r in rows:
        y = len(labels) - 1 - labels.index(label) + (0.15 if strategy == "SA-QUBO" else -0.15)
        ax.errorbar(
            r["mean_diff"],
            y,
            xerr=[[r["mean_diff"] - r["ci_low"]], [r["ci_high"] - r["mean_diff"]]],
            fmt="o" if strategy == "SA-QUBO" else "s",
            color="0.0",
            mfc="0.0" if strategy == "SA-QUBO" else "white",
            ms=3.5,
            lw=0.8,
            label=strategy,
        )
    handles, names = ax.get_legend_handles_labels()
    unique = dict(zip(names, handles, strict=True))
    ax.legend(unique.values(), unique.keys(), loc="lower left", fontsize=6)
    ax.set_xscale("symlog", linthresh=10)
    ticks = [-100, -10, 0, 10, 100]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(list(reversed(labels)))
    ax.set_xlabel("Shortfall $-$ TWAP (bps), mean and 95% CI (symlog)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(pad=0.2)
    return fig


# --------------------------------------------------------------------------------------
# Solvers and QAOA
# --------------------------------------------------------------------------------------


def fig_paper_solver_success(res: Results) -> Figure:
    """Fraction of seeds at the exact optimum vs n (binary execution encoding and toy)."""
    summary = res.table("solver_benchmark", "summary")
    families = [f for f in ("slice", "toy") if f in set(summary["family"])]
    titles = {"slice": "Binary execution QUBO", "toy": "Toy hardware QUBO"}
    fig, axes = _figure(COLUMN_IN, 1.75, 1, len(families), sharey=True, squeeze=False)
    for ax, family in zip(axes[0], families, strict=True):
        for solver in ("SA", "SA_1", "Greedy"):
            d = metric(summary, "optimal", family=family, solver=solver).sort_values("n")
            if d.empty:
                continue
            st = SOLVER_STYLE[solver]
            ax.plot(
                d["n"],
                d["mean"],
                ls=st["ls"],
                marker=st["marker"],
                color=st["color"],
                label=st["label"],
            )
        ax.set_title(titles[family], fontsize=7)
        ax.set_xlabel("QUBO variables $n$")
        ax.set_ylim(-0.05, 1.05)
    axes[0][0].set_ylabel("Share of seeds optimal")
    axes[0][0].legend(fontsize=5.8, loc="lower left")
    fig.tight_layout(pad=0.2, w_pad=0.4)
    return fig


def fig_paper_qaoa(res: Results) -> Figure:
    """QAOA vs same-budget uniform sampling on the binary execution QUBO (slice family)."""
    runs = res.table("qaoa_benchmark", "runs")
    family = "slice" if "slice" in set(runs["family"]) else "toy"
    runs = runs[runs["family"] == family]
    q = runs[runs["solver"].isin(["QAOA_Ideal", "QAOA_Noisy"])]
    sa = runs[runs["solver"] == "SA"]
    fig, axes = _figure(DOUBLE_IN, 1.9, 1, 3)
    ideal = q[q["solver"] == "QAOA_Ideal"]
    uniform = ideal.groupby("n")[
        ["random_success_probability", "random_approx_ratio_mean", "random_optimal_found"]
    ].mean()
    panels = (
        (
            "success_probability",
            "random_success_probability",
            "$P_{\\mathrm{opt}}$ (final distribution)",
        ),
        (
            "approx_ratio_mean",
            "random_approx_ratio_mean",
            "Ratio of mean energy $r(\\langle E\\rangle)$",
        ),
        ("optimal_found", "random_optimal_found", "Share of runs: best sample optimal"),
    )
    for k, (ax, (col, ucol, ylabel)) in enumerate(zip(axes, panels, strict=True)):
        for solver in ("QAOA_Ideal", "QAOA_Noisy"):
            d = q[q["solver"] == solver].groupby("n")[col].mean()
            st = SOLVER_STYLE[solver]
            ax.plot(
                d.index,
                d.to_numpy(dtype=float),
                ls=st["ls"],
                marker=st["marker"],
                color=st["color"],
                label=st["label"],
            )
        st = SOLVER_STYLE["Uniform"]
        ax.plot(
            uniform.index,
            uniform[ucol].to_numpy(dtype=float),
            ls=st["ls"],
            marker=st["marker"],
            color=st["color"],
            label=st["label"],
        )
        if col == "optimal_found" and not sa.empty:
            d = sa.groupby("n")["optimal_found"].mean()
            st = SOLVER_STYLE["SA"]
            ax.plot(
                d.index,
                d.to_numpy(dtype=float),
                ls=st["ls"],
                marker=st["marker"],
                color=st["color"],
                label=st["label"],
            )
        ax.set_xlabel("Qubits $n$")
        ax.set_ylabel(ylabel)
        _panel_label(ax, f"({'abc'[k]})")
    axes[0].set_yscale("log")
    axes[2].set_ylim(0, 1.05)
    axes[2].legend(fontsize=6, loc="lower left")
    fig.tight_layout(pad=0.2, w_pad=1.0)
    return fig


# --------------------------------------------------------------------------------------
# Latency
# --------------------------------------------------------------------------------------


def fig_paper_latency(res: Results) -> Figure:
    """Empirical CDFs of fast-path work, tick lateness, policy propagation and SA solves."""
    samples = res.table("latency", "samples")
    series = (
        ("runtime", "fast_path", "Runtime tick work", "-", "0.0"),
        ("pipeline", "fast_path", "Pipeline tick work", "--", "0.3"),
        ("runtime", "policy_propagation", "Policy publish $\\to$ apply", "-.", "0.5"),
        ("runtime", "tick_lateness", "Tick lateness (5 ms ticks)", ":", "0.0"),
        ("runtime", "slow_path_optimize", "Slow-path SA solve", "-", "0.65"),
    )
    fig, ax = _figure(COLUMN_IN, 2.4)
    for source, component, label, ls, color in series:
        v = np.sort(
            samples[(samples["source"] == source) & (samples["component"] == component)][
                "duration_us"
            ].to_numpy(dtype=float)
        )
        if v.size == 0:
            continue
        ax.plot(v, np.arange(1, v.size + 1) / v.size, ls=ls, color=color, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Duration ($\\mu$s, log scale)")
    ax.set_ylabel("Empirical CDF")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2)
    fig.tight_layout(pad=0.2)
    return fig
