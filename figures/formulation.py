import numpy as np
from matplotlib.figure import Figure

from figures.common import Results, ci_errorbars, metric, new_figure

TERM_LABELS = {
    "impact_cost": "Market\nImpact",
    "timing_cost": "Timing\nRisk",
    "transaction_cost": "Transaction\nCost",
    "adverse_selection_cost": "Adverse\nSelection",
    "inventory_risk_cost": "Inventory\nRisk",
    "information_leakage_cost": "Info\nLeakage",
}
TERM_COLORS = ["#1976D2", "#388E3C", "#F57C00", "#D32F2F", "#7B1FA2", "#00796B"]
VENUE_COLORS = {"Lit": "#D32F2F", "Dark": "#4CAF50", "ECN": "#2196F3"}


def fig03_hft_qubo_cost_decomposition(res: Results) -> Figure:
    summary = res.table("formulation", "cost_breakdown_summary")
    terms = list(TERM_LABELS)
    cost = metric(summary, "cost").set_index("term").loc[terms]
    share = metric(summary, "share_of_abs_total").set_index("term").loc[terms]
    n = int(cost["count"].iloc[0])
    fig, (ax1, ax2) = new_figure(1, 2, figsize=(7, 3))
    x = np.arange(len(terms))
    ax1.bar(x, cost["mean"], color=TERM_COLORS, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax1.errorbar(x, cost["mean"], yerr=ci_errorbars(cost), fmt="none", ecolor="black", capsize=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels([TERM_LABELS[t] for t in terms], fontsize=7)
    ax1.set_ylabel("Weighted cost term")
    ax1.set_title(f"HFT QUBO cost terms (SA, {n} seeds)")
    ax2.pie(
        share["mean"],
        autopct="%1.0f%%",
        colors=TERM_COLORS,
        pctdistance=0.75,
        startangle=90,
        textprops={"fontsize": 7},
    )
    ax2.legend(
        [TERM_LABELS[t].replace("\n", " ") for t in terms],
        loc="center left",
        bbox_to_anchor=(0.9, 0.5),
        fontsize=6,
    )
    ax2.set_title("Share of |cost|")
    fig.tight_layout()
    return fig


def fig04_qubo_ising_conversion(res: Results) -> Figure:
    df = res.table("formulation", "ising_check")
    fig, (ax1, ax2) = new_figure(1, 2, figsize=(7, 2.8))
    ax1.semilogy(df["n"], df["max_abs_error"] + 1e-18, "o-", color="#D32F2F", label="max |abs|")
    ax1.semilogy(df["n"], df["max_rel_error"] + 1e-18, "s--", color="#1976D2", label="max rel.")
    ax1.axhline(1e-10, color="green", linestyle=":", label="$10^{-10}$")
    ax1.set_xlabel("Problem size n")
    ax1.set_ylabel("QUBO vs Ising error")
    ax1.set_title(f"Equivalence ({int(df['samples'].iloc[0])} random x per n)")
    ax1.legend(fontsize=7)
    ax2.bar(df["n"], df["hamiltonian_terms"], color="#1565C0", alpha=0.7, edgecolor="black")
    ax2.plot(df["n"], df["max_terms"], "r--", label="$n + n(n-1)/2$")
    ax2.set_xlabel("Problem size n")
    ax2.set_ylabel("Hamiltonian terms")
    ax2.set_title("Ising term count (dense random Q)")
    ax2.legend(fontsize=7)
    fig.tight_layout()
    return fig


def fig18_qubo_param_sensitivity(res: Results) -> Figure:
    df = res.table("formulation", "sensitivity")
    fig, axes = new_figure(1, 2, figsize=(7, 2.8))
    panels = [
        (
            "impact_weight_multiplier",
            "Impact-weight multiplier $\\lambda$ (w$_I$ = 0.25$\\lambda$)",
        ),
        ("vpin", "VPIN"),
    ]
    for ax, (parameter, label), color in zip(axes, panels, ["#D32F2F", "#7B1FA2"], strict=True):
        g = df[df["parameter"] == parameter].groupby("value")["energy"]
        ax.errorbar(
            g.mean().index, g.mean(), yerr=g.std(), fmt="o-", color=color, capsize=2, markersize=3
        )
        ax.set_xlabel(label)
        ax.set_ylabel("SA energy (mean $\\pm$ sd over seeds)")
    axes[0].set_title("Cost vs impact weight")
    axes[1].set_title("Cost vs VPIN")
    fig.tight_layout()
    return fig


def fig27_venue_routing(res: Results) -> Figure:
    df = res.table("formulation", "venue_routing")
    first = df[df["seed"] == df["seed"].min()]
    ticks = int(df["tick"].max()) + 1
    fig, (ax1, ax2) = new_figure(1, 2, figsize=(7, 2.8))
    bottom = np.zeros(ticks)
    for venue, color in VENUE_COLORS.items():
        per_tick = first[first["venue"] == venue].groupby("tick")["shares"].sum()
        values = per_tick.reindex(range(ticks), fill_value=0).to_numpy(dtype=float)
        ax1.bar(range(ticks), values, bottom=bottom, color=color, alpha=0.7, label=venue)
        bottom += values
    ax1.set_xlabel("Tick slice")
    ax1.set_ylabel("Shares")
    ax1.set_title("Venue routing (one seed, VPIN = 0.4)")
    ax1.legend(fontsize=7)
    per_seed = df.groupby(["seed", "venue"])["shares"].sum().unstack(fill_value=0)
    fractions = per_seed.div(per_seed.sum(axis=1), axis=0).mean()
    venues = [v for v in VENUE_COLORS if v in fractions]
    ax2.pie(
        [fractions[v] for v in venues],
        labels=venues,
        autopct="%1.0f%%",
        colors=[VENUE_COLORS[v] for v in venues],
        textprops={"fontsize": 8},
    )
    ax2.set_title(f"Mean venue split ({len(per_seed)} seeds)")
    fig.tight_layout()
    return fig


def fig28_qaoa_circuit_depth(res: Results) -> Figure:
    df = res.table("formulation", "circuit_scaling")
    fig, (ax1, ax2) = new_figure(1, 2, figsize=(7, 2.8))
    markers = {1: "o", 2: "s", 3: "^"}
    for p, g in df.groupby("p"):
        m = markers.get(int(p), "o")
        ax1.plot(g["n"], g["logical_depth"], f"{m}--", label=f"p={p} logical")
        ax1.plot(g["n"], g["transpiled_depth"], f"{m}-", label=f"p={p} transpiled")
        ax2.plot(g["n"], g["transpiled_cx"], f"{m}-", label=f"p={p}")
    ax1.set_xlabel("Qubits")
    ax1.set_ylabel("Circuit depth")
    ax1.set_title("QAOA depth (basis cx, rz, sx, x)")
    ax1.legend(fontsize=6, ncol=2)
    ax2.set_xlabel("Qubits")
    ax2.set_ylabel("CX count (transpiled)")
    ax2.set_title("Two-qubit gate count")
    ax2.legend(fontsize=7)
    fig.tight_layout()
    return fig
