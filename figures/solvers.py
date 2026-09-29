import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from figures.common import SOLVER_COLORS, Results, ci_errorbars, metric, new_figure

QAOA = ("QAOA_Ideal", "QAOA_Noisy")
P_STYLES = {1: "o-", 2: "s--", 3: "^:"}


def fig05_sa_convergence(res: Results) -> Figure:
    hist = res.table("solver_benchmark", "sa_convergence_history")
    final = res.table("solver_benchmark", "sa_convergence_final")
    e_min = final["min_energy"].iloc[0]
    fig, (ax1, ax2) = new_figure(1, 2, figsize=(7, 2.8))
    for sweeps, g in hist.groupby("num_sweeps"):
        ax1.plot(g["iteration"], g["best_energy"] - e_min + 1, label=f"{sweeps} requested")
    ax1.set_yscale("log")
    ax1.set_xlabel("Sweep")
    ax1.set_ylabel("Best $E - E_{min} + 1$")
    ax1.set_title("SA convergence (12-var execution QUBO)")
    ax1.legend(fontsize=6)
    g = final.groupby("num_sweeps")
    ax2.plot(g["optimal"].mean().index, g["optimal"].mean(), "o-", color="#D32F2F")
    ax2.set_xscale("log")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel("Requested sweeps")
    ax2.set_ylabel("Fraction of seeds at optimum")
    honoured = bool((final["sweeps_run"] == final["num_sweeps"]).all())
    ax2.set_title("All requested sweeps run" if honoured else "Sweeps cut short")
    fig.tight_layout()
    return fig


def _by_n(summary: pd.DataFrame, name: str, **where: object) -> pd.DataFrame:
    return metric(summary, name, **where).sort_values("n")


def fig06_solver_comparison(res: Results) -> Figure:
    s = res.table("solver_benchmark", "summary")
    fig, (ax1, ax2) = new_figure(1, 2, figsize=(7, 2.8))
    for family, style in (("random", "-"), ("execution", "--")):
        for solver, marker in (("SA", "s"), ("Greedy", "^")):
            d = _by_n(s, "approx_ratio", family=family, solver=solver)
            ax1.errorbar(
                d["n"],
                d["mean"],
                yerr=ci_errorbars(d),
                fmt=marker + style,
                color=SOLVER_COLORS[solver],
                capsize=2,
                label=f"{solver} ({family})",
            )
    ax1.axhline(1.0, color="green", linestyle=":", alpha=0.6)
    ax1.set_xlabel("QUBO variables")
    ax1.set_ylabel("Approximation ratio")
    ax1.set_title("Solution quality (mean, 95% CI)")
    ax1.legend(fontsize=6)
    for solver, marker in (("BruteForce", "o"), ("SA", "s"), ("Greedy", "^")):
        d = _by_n(s, "time_s", family="random", solver=solver)
        ax2.semilogy(d["n"], d["mean"], marker + "-", color=SOLVER_COLORS[solver], label=solver)
    ax2.set_xlabel("QUBO variables")
    ax2.set_ylabel("Solve time (s)")
    ax2.set_title("Wall time (random QUBOs)")
    ax2.legend(fontsize=7)
    fig.tight_layout()
    return fig


def fig09_solution_quality_scaling(res: Results) -> Figure:
    s = res.table("solver_benchmark", "summary")
    fig, (ax1, ax2) = new_figure(1, 2, figsize=(7, 2.8))
    families = (
        ("toy", "#4CAF50"),
        ("execution", "#D32F2F"),
        ("slice", "#FF9800"),
        ("random", "#1976D2"),
    )
    for family, color in families:
        for solver, style in (("SA", "o-"), ("SA_1", "x:")):
            d = _by_n(s, "optimal", family=family, solver=solver)
            if d.empty:
                continue
            label = family if solver == "SA" else None
            ax1.plot(d["n"], d["mean"], style, color=color, label=label)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlabel("QUBO variables")
    ax1.set_ylabel("Fraction of seeds at optimum")
    ax1.set_title("SA success (solid: restarts, dotted: 1 replica)")
    ax1.legend(fontsize=6)
    for family, color in (("execution", "#D32F2F"), ("slice", "#FF9800")):
        d = _by_n(s, "fill_rate", family=family, solver="SA")
        if d.empty:
            continue
        ax2.errorbar(
            d["n"], d["mean"], yerr=ci_errorbars(d), fmt="s-", color=color, capsize=2, label=family
        )
    ax2.legend(fontsize=7)
    ax2.axhline(1.0, color="green", linestyle=":", alpha=0.6)
    ax2.set_xlabel("QUBO variables")
    ax2.set_ylabel("Selected / ordered shares")
    ax2.set_title("SA fill of the order (before repair)")
    fig.tight_layout()
    return fig


def _qaoa_runs(res: Results, family: str) -> pd.DataFrame:
    runs = res.table("qaoa_benchmark", "runs")
    return runs[runs["family"] == family]


def fig07_qaoa_vs_sa(res: Results) -> Figure:
    runs = _qaoa_runs(res, "fig07")
    ideal = runs[runs["solver"] == "QAOA_Ideal"]
    sa = runs[runs["solver"] == "SA"]
    depths = sorted(ideal["p"].unique())
    labels = ["SA"] + [f"QAOA p={p}" for p in depths] + ["Random"]
    found = [sa["optimal_found"].mean()] + [
        ideal[ideal["p"] == p]["optimal_found"].mean() for p in depths
    ]
    found.append(ideal["random_optimal_found"].mean())
    p_opt = [np.nan] + [ideal[ideal["p"] == p]["success_probability"].mean() for p in depths]
    p_opt.append(ideal["random_success_probability"].mean())
    times = [sa["time_s"].mean()] + [ideal[ideal["p"] == p]["time_s"].mean() for p in depths]
    colors = [SOLVER_COLORS["SA"]] + [SOLVER_COLORS["QAOA_Ideal"]] * len(depths)
    fig, (ax1, ax2, ax3) = new_figure(1, 3, figsize=(7.5, 2.6))
    ax1.bar(labels, found, color=[*colors, SOLVER_COLORS["Random"]], alpha=0.75)
    ax1.set_ylabel("Runs finding the optimum")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Best-of-shots")
    ax2.bar(labels[1:], p_opt[1:], color=[*colors[1:], SOLVER_COLORS["Random"]], alpha=0.75)
    ax2.set_ylabel("$P_{opt}$")
    ax2.set_title("Probability on optimum")
    ax3.bar(labels[:-1], times, color=colors, alpha=0.75)
    ax3.set_yscale("log")
    ax3.set_ylabel("Wall time (s)")
    ax3.set_title("Mean solve time")
    for ax in (ax1, ax2, ax3):
        ax.tick_params(axis="x", rotation=45, labelsize=6)
    fig.suptitle(f"12-variable execution QUBO, {sa['seed'].nunique()} seeds", fontsize=9)
    fig.tight_layout()
    return fig


def fig08_qaoa_landscape(res: Results) -> Figure:
    grid = res.table("qaoa_landscape", "landscape")
    summary = res.json("qaoa_landscape", "summary")
    table = grid.pivot(index="beta", columns="gamma", values="expected_energy")
    fig, ax = new_figure(figsize=(3.5, 3.0))
    im = ax.imshow(
        table.to_numpy(),
        extent=[table.columns.min(), table.columns.max(), table.index.min(), table.index.max()],
        aspect="auto",
        cmap="RdYlBu_r",
        origin="lower",
    )
    best = summary["best_grid_point"]
    ax.plot(best["gamma"], best["beta"], "w*", markersize=10, markeredgecolor="black")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\beta$")
    ax.set_title(rf"Exact $\langle H\rangle$, $p=1$, n={summary['n']}")
    fig.colorbar(im, ax=ax, label=r"$\langle H \rangle$", shrink=0.9)
    fig.tight_layout()
    return fig


def _mean_by(runs: pd.DataFrame, column: str, keys: list[str]) -> pd.DataFrame:
    return runs.groupby(keys, as_index=False)[column].mean()


def fig_hw_approx_ratio_vs_size(res: Results) -> Figure:
    runs = _qaoa_runs(res, "toy")
    fig, (ax1, ax2) = new_figure(1, 2, figsize=(7, 2.8))
    for solver in QAOA:
        for p, g in _mean_by(
            runs[runs["solver"] == solver], "approx_ratio_mean", ["p", "n"]
        ).groupby("p"):
            ax1.plot(
                g["n"],
                g["approx_ratio_mean"],
                P_STYLES.get(int(p), "o-"),
                color=SOLVER_COLORS[solver],
                label=f"{solver.removeprefix('QAOA_')} p={p}",
            )
    base = _mean_by(runs[runs["solver"] == "QAOA_Ideal"], "random_approx_ratio_mean", ["n"])
    ax1.plot(base["n"], base["random_approx_ratio_mean"], "x-", color="#757575", label="Uniform")
    ax1.set_xlabel("Qubits")
    ax1.set_ylabel(r"Ratio of $\langle H \rangle$")
    ax1.set_title("Mean-energy quality")
    ax1.legend(fontsize=5, ncol=2)
    for solver in ("SA", *QAOA):
        g = _mean_by(runs[runs["solver"] == solver], "approx_ratio_best", ["n"])
        ax2.plot(g["n"], g["approx_ratio_best"], "o-", color=SOLVER_COLORS[solver], label=solver)
    rb = _mean_by(runs[runs["solver"] == "QAOA_Ideal"], "random_approx_ratio_best", ["n"])
    ax2.plot(rb["n"], rb["random_approx_ratio_best"], "x-", color="#757575", label="Uniform")
    ax2.set_xlabel("Qubits")
    ax2.set_ylabel("Ratio of best sample")
    ax2.set_title("Best-of-shots (same shot budget)")
    ax2.legend(fontsize=6)
    fig.tight_layout()
    return fig


def fig_hw_solve_time_scaling(res: Results) -> Figure:
    runs = _qaoa_runs(res, "toy")
    fig, ax = new_figure(figsize=(3.5, 2.8))
    for solver in ("SA", *QAOA):
        g = _mean_by(runs[runs["solver"] == solver], "time_s", ["n"])
        ax.semilogy(g["n"], g["time_s"], "o-", color=SOLVER_COLORS[solver], label=solver)
    ax.set_xlabel("Qubits")
    ax.set_ylabel("Wall time (s), mean over p and seeds")
    ax.set_title("Solve time (simulators)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def fig_hw_energy_distribution(res: Results) -> Figure:
    runs = _qaoa_runs(res, "toy")
    sizes = sorted(runs["n"].unique())
    fig, axes = new_figure(1, len(sizes), figsize=(7, 2.6), squeeze=False)
    for ax, n in zip(axes[0], sizes, strict=True):
        d = runs[runs["n"] == n]
        solvers = [s for s in ("SA", *QAOA) if (d["solver"] == s).any()]
        data = [d[d["solver"] == s]["approx_ratio_mean"].dropna() for s in solvers if s != "SA"]
        names = [s.removeprefix("QAOA_") for s in solvers if s != "SA"]
        data.append(d[d["solver"] == "QAOA_Ideal"]["random_approx_ratio_mean"])
        names.append("Unif.")
        ax.boxplot(data, widths=0.6)
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels(names, rotation=45, fontsize=6)
        ax.set_title(f"n={n}", fontsize=9)
    axes[0][0].set_ylabel(r"Ratio of $\langle H \rangle$")
    fig.suptitle("Distribution of mean-energy quality over seeds and depths", fontsize=9)
    fig.tight_layout()
    return fig


def fig_hw_depth_effect(res: Results) -> Figure:
    runs = _qaoa_runs(res, "toy")
    fig, axes = new_figure(1, 2, figsize=(7, 2.8))
    for ax, solver in zip(axes, QAOA, strict=True):
        d = _mean_by(runs[runs["solver"] == solver], "approx_ratio_mean", ["n", "p"])
        for n, g in d.groupby("n"):
            ax.plot(g["p"], g["approx_ratio_mean"], "o-", label=f"n={n}")
        ax.set_xlabel("QAOA depth p")
        ax.set_ylabel(r"Ratio of $\langle H \rangle$")
        ax.set_title(solver.replace("_", " "))
        ax.legend(fontsize=6)
    fig.tight_layout()
    return fig


def _success_heatmap(res: Results, solver: str) -> Figure:
    runs = _qaoa_runs(res, "toy")
    d = runs[runs["solver"] == solver]
    p_opt = d.pivot_table(index="p", columns="n", values="success_probability", aggfunc="mean")
    base = d.pivot_table(index="p", columns="n", values="random_success_probability")
    fig, ax = new_figure(figsize=(3.8, 2.8))
    im = ax.imshow(p_opt.to_numpy(), aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_xticks(range(len(p_opt.columns)))
    ax.set_xticklabels(p_opt.columns)
    ax.set_yticks(range(len(p_opt.index)))
    ax.set_yticklabels([f"p={p}" for p in p_opt.index])
    ax.set_xlabel("Qubits")
    ax.set_title(f"$P_{{opt}}$ ({solver.removeprefix('QAOA_')}); (x uniform)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    for i in range(p_opt.shape[0]):
        for j in range(p_opt.shape[1]):
            value, ref = p_opt.iloc[i, j], base.iloc[i, j]
            ax.text(
                j, i, f"{value:.3f}\n({value / ref:.1f}x)", ha="center", va="center", fontsize=6
            )
    fig.tight_layout()
    return fig


def fig_hw_success_prob_ideal(res: Results) -> Figure:
    return _success_heatmap(res, "QAOA_Ideal")


def fig_hw_success_prob_noisy(res: Results) -> Figure:
    return _success_heatmap(res, "QAOA_Noisy")


def fig_hw_noise_degradation(res: Results) -> Figure:
    runs = _qaoa_runs(res, "toy")
    keys = ["n", "p"]
    ideal = runs[runs["solver"] == "QAOA_Ideal"].groupby(keys)
    noisy = runs[runs["solver"] == "QAOA_Noisy"].groupby(keys)
    p_ratio = (noisy["success_probability"].mean() / ideal["success_probability"].mean()).dropna()
    h_ratio = (noisy["approx_ratio_mean"].mean() / ideal["approx_ratio_mean"].mean()).dropna()
    fig, (ax1, ax2) = new_figure(1, 2, figsize=(7, 2.8))
    for ax, ratio, label in (
        (ax1, p_ratio, "$P_{opt}$ noisy / ideal"),
        (ax2, h_ratio, r"$\langle H\rangle$-ratio noisy / ideal"),
    ):
        table = ratio.unstack("p")
        width = 0.8 / max(1, table.shape[1])
        for k, p in enumerate(table.columns):
            ax.bar(np.arange(len(table)) + k * width, table[p], width, label=f"p={p}", alpha=0.8)
        ax.set_xticks(np.arange(len(table)) + width * (table.shape[1] - 1) / 2)
        ax.set_xticklabels(table.index)
        ax.axhline(1.0, color="black", linewidth=0.6)
        ax.set_xlabel("Qubits")
        ax.set_ylabel(label)
        ax.legend(fontsize=7)
    ax1.set_title("Success probability under noise")
    ax2.set_title("Mean-energy quality under noise")
    fig.tight_layout()
    return fig


def fig_hw_count_distribution(res: Results) -> Figure:
    counts = res.table("qaoa_benchmark", "top_counts")
    p = int(counts["p"].min())
    fig, axes = new_figure(1, 2, figsize=(7, 2.8))
    for ax, solver in zip(axes, QAOA, strict=True):
        d = counts[(counts["solver"] == solver) & (counts["p"] == p)]
        colors = ["#4CAF50" if o else SOLVER_COLORS[solver] for o in d["optimal"]]
        ax.bar(range(len(d)), d["count"], color=colors, alpha=0.8, edgecolor="black", lw=0.3)
        ax.set_xticks(range(len(d)))
        ax.set_xticklabels(d["bitstring"].astype(str).str.zfill(4), rotation=90, fontsize=6)
        ax.set_ylabel("Counts (final shots)")
        ax.set_title(f"{solver.replace('_', ' ')}, n=4, p={p} (green = optimal)")
    fig.tight_layout()
    return fig
