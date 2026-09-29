"""Microstructure and regime figures (fig10-12, fig15-17) from results/microstructure/ and
results/regime/."""

import matplotlib.patches as mpatches
from matplotlib.figure import Figure

from figures.common import Results, ci_errorbars, metric, new_figure

STRESS = (200, 400)
VPIN_THRESHOLD = 0.7
REGIME_COLORS = {"low": "#4CAF50", "normal": "#2196F3", "high": "#FF9800", "extreme": "#D32F2F"}


def _stress(ax, label: str | None = None) -> None:
    ax.axvspan(*STRESS, alpha=0.12, color="red", label=label)


def fig10_kyle_lambda(res: Results) -> Figure:
    s = res.table("microstructure", "series")
    ratio = metric(res.table("microstructure", "summary"), "kyle_lambda_ratio").iloc[0]
    fig, (ax1, ax2) = new_figure(2, 1, figsize=(3.5, 4), sharex=True)
    ax1.plot(s["tick"], s["price"], color="#1565C0", linewidth=0.8)
    _stress(ax1, "5x volatility")
    ax1.set_ylabel("Price ($)")
    ax1.set_title("Kyle's lambda (rolling OLS, 50 ticks)")
    ax1.legend(fontsize=7)
    ax2.plot(s["tick"], s["kyle_lambda"], color="#D32F2F", linewidth=0.8)
    _stress(ax2)
    ax2.set_xlabel("Tick")
    ax2.set_ylabel(r"$\hat\lambda_K$")
    ax2.set_title(
        f"Stress/calm ratio {ratio['mean']:.1f} [{ratio['ci_low']:.1f}, {ratio['ci_high']:.1f}]"
        f" ({int(ratio['count'])} seeds)",
        fontsize=8,
    )
    fig.tight_layout()
    return fig


def fig11_vpin_estimation(res: Results) -> Figure:
    s = res.table("microstructure", "series")
    fig, ax = new_figure(figsize=(3.5, 2.8))
    ax.plot(s["tick"], s["vpin"], color="#7B1FA2", linewidth=0.8, label="VPIN (tick rule)")
    ax.axhline(VPIN_THRESHOLD, color="red", linestyle="--", linewidth=0.8, label="0.7")
    _stress(ax, "High-vol block")
    ax.fill_between(
        s["tick"],
        s["vpin"],
        VPIN_THRESHOLD,
        where=s["vpin"] > VPIN_THRESHOLD,
        color="red",
        alpha=0.2,
    )
    ax.set_xlabel("Tick")
    ax.set_ylabel("VPIN")
    ax.set_title("Volume-synchronised PIN")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def fig12_microstructure_dashboard(res: Results) -> Figure:
    s = res.table("microstructure", "series")
    fig, axes = new_figure(2, 2, figsize=(7, 5), sharex=True)
    panels = [
        (axes[0, 0], "analyzer_kyle_lambda", r"Kyle's $\lambda$", "Price impact", "#D32F2F"),
        (axes[0, 1], "analyzer_vpin", "VPIN", "Order-flow toxicity", "#7B1FA2"),
        (axes[1, 0], "adverse_selection_cost", "AS cost", "Adverse selection", "#F57C00"),
        (axes[1, 1], "spread_bps", "Spread (bps)", "Bid-ask spread", "#00796B"),
    ]
    for ax, column, ylabel, title, color in panels:
        ax.plot(s["tick"], s[column], color=color, linewidth=0.8)
        _stress(ax)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    axes[0, 1].axhline(VPIN_THRESHOLD, color="red", linestyle="--", linewidth=0.6)
    for ax in axes[1]:
        ax.set_xlabel("Tick")
    fig.suptitle("Microstructure estimates (synthetic ticks)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def fig15_lambda_adaptation(res: Results) -> Figure:
    s = res.table("regime", "series")
    fig, (ax1, ax2) = new_figure(2, 1, figsize=(3.5, 4), sharex=True)
    ax1.plot(s["tick"], s["price"], color="#1565C0", linewidth=0.8)
    _stress(ax1, "High vol")
    ax1.set_ylabel("Price ($)")
    ax1.set_title("Adaptive risk aversion")
    ax1.legend(fontsize=7)
    ax2.plot(s["tick"], s["lambda"], color="#D32F2F", linewidth=0.8)
    for tick, regime in zip(s["tick"], s["regime"], strict=True):
        ax2.axvspan(tick - 1, tick, alpha=0.15, color=REGIME_COLORS.get(regime, "#999"), lw=0)
    ax2.set_xlabel("Tick")
    ax2.set_ylabel(r"$\lambda$ / $\lambda_{base}$")
    patches = [mpatches.Patch(color=c, alpha=0.3, label=r) for r, c in REGIME_COLORS.items()]
    ax2.legend(handles=patches, fontsize=6, ncol=2)
    fig.tight_layout()
    return fig


def fig16_volatility_regime_detection(res: Results) -> Figure:
    s = res.table("regime", "series")
    fig, (ax1, ax2) = new_figure(2, 1, figsize=(3.5, 4), sharex=True)
    ax1.plot(
        s["tick"], s["fast_vol"], color="#D32F2F", linewidth=0.8, label=r"Fast ($\alpha$=0.06)"
    )
    ax1.plot(
        s["tick"], s["slow_vol"], color="#1976D2", linewidth=0.8, label=r"Slow ($\alpha$=0.01)"
    )
    _stress(ax1)
    ax1.set_ylabel("EWMA volatility")
    ax1.set_title("Dual-timescale EWMA volatility")
    ax1.legend(fontsize=6)
    ax2.plot(s["tick"], s["vol_ratio"], color="#7B1FA2", linewidth=0.8)
    for level, color in ((0.7, "green"), (1.3, "#FF9800"), (2.0, "red")):
        ax2.axhline(level, color=color, linestyle="--", linewidth=0.6)
    _stress(ax2)
    ax2.set_xlabel("Tick")
    ax2.set_ylabel("Fast / slow")
    ax2.set_title("Regime signal (thresholds 0.7, 1.3, 2.0)")
    fig.tight_layout()
    return fig


def fig17_regime_distribution(res: Results) -> Figure:
    summary = res.table("regime", "summary")
    series = res.table("regime", "series_1000")
    regimes = list(REGIME_COLORS)
    fractions = summary.set_index("metric").loc[[f"frac_{r}" for r in regimes]]
    fig, (ax1, ax2) = new_figure(1, 2, figsize=(7, 2.8))
    ax1.bar(
        [r.capitalize() for r in regimes],
        fractions["mean"],
        yerr=ci_errorbars(fractions),
        color=list(REGIME_COLORS.values()),
        alpha=0.7,
        edgecolor="black",
        capsize=2,
    )
    ax1.set_ylabel("Fraction of ticks")
    ax1.set_title(f"Regime distribution ({int(fractions['count'].iloc[0])} seeds, 95% CI)")
    present = [r for r in regimes if (series["regime"] == r).any()]
    bp = ax2.boxplot(
        [series[series["regime"] == r]["lambda"] for r in present], patch_artist=True, widths=0.5
    )
    for patch, r in zip(bp["boxes"], present, strict=True):
        patch.set_facecolor(REGIME_COLORS[r])
        patch.set_alpha(0.6)
    ax2.set_xticks(range(1, len(present) + 1))
    ax2.set_xticklabels([r.capitalize() for r in present])
    ax2.set_ylabel(r"$\lambda$ / $\lambda_{base}$")
    ax2.set_title("Regime-conditioned $\\lambda$ (one seed)")
    fig.tight_layout()
    return fig
