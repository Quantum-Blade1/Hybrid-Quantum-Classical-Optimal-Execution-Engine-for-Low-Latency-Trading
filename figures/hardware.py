"""IBM ibm_fez figures (results/hardware/), at IEEE column width and legible in grayscale.

Every number comes from the tables written by `experiments.hardware_analysis`: per final
job (`final_tests.csv`), per n (`summary.csv`, which carries the Aer p=1 references) and
per job (`jobs.csv`, for the COBYLA trajectories). Hardware runs are drawn individually,
never pooled; simulator references show the mean and the range over seeds.
"""

from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from figures.common import Results
from figures.paper import COLUMN_IN, _figure, _panel_label

RUN_MARKERS = ("o", "s", "^", "D", "v")
RUN_LINES = ("-", "--", "-.", ":", (0, (5, 1)))
SIM_STYLE: dict[str, dict[str, Any]] = {
    "qaoa_ideal": {"marker": "o", "mfc": "white", "color": "0.35", "label": "Aer ideal, $p=1$"},
    "qaoa_noisy": {"marker": "s", "mfc": "white", "color": "0.6", "label": "Aer noisy, $p=1$"},
}


def _legend_below(ax: Any) -> None:
    ax.legend(fontsize=5.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13))


def _title_note(res: Results) -> str:
    note = str(res.manifest("hardware").get("data", ""))
    return f" [{note}]" if "SYNTHETIC" in note.upper() else ""


def _sim_points(ax: Any, summary: pd.DataFrame, metric: str, x: np.ndarray, scale: Any) -> None:
    """Mean over seeds with a min-max bar, for each Aer solver present in the summary."""
    for k, (solver, style) in enumerate(SIM_STYLE.items()):
        col = f"{solver}_p1_{metric}"
        if col not in summary:
            continue
        mean, lo, hi = (scale(summary[c]) for c in (col, f"{col}_min", f"{col}_max"))
        xs = x + 0.22 + 0.12 * k
        ax.errorbar(
            xs,
            mean,
            yerr=[mean - lo, hi - mean],
            fmt=style["marker"],
            mfc=style["mfc"],
            color=style["color"],
            ecolor=style["color"],
            elinewidth=0.7,
            label=style["label"] + " (seed range)",
        )


def _run_points(ax: Any, tests: pd.DataFrame, x_of: dict[int, int], y: str, err: tuple) -> None:
    for run, g in tests.groupby("run"):
        k = int(run) - 1
        xs = np.array([x_of[int(n)] for n in g["n"]]) - 0.2 + 0.1 * k
        ax.errorbar(
            xs,
            g[y],
            yerr=[g[y] - g[err[0]], g[err[1]] - g[y]],
            fmt=RUN_MARKERS[k % len(RUN_MARKERS)],
            color="0.0",
            ms=3.5,
            elinewidth=0.8,
            label=f"ibm_fez run {int(run)}",
        )


def fig_hw_ibm_success_prob(res: Results) -> Figure:
    """P(optimum) of each final job relative to uniform sampling (Wilson 95% CI)."""
    tests = res.table("hardware", "final_tests")
    summary = res.table("hardware", "summary").sort_values("n").reset_index(drop=True)
    sizes = [int(n) for n in summary["n"]]
    x_of = {n: i for i, n in enumerate(sizes)}
    uniform = summary.set_index("n")["uniform_success_probability"]
    t = tests.assign(
        rel=tests["success_probability"] / tests["n"].map(uniform),
        rel_lo=tests["success_ci_low"] / tests["n"].map(uniform),
        rel_hi=tests["success_ci_high"] / tests["n"].map(uniform),
    )
    fig, ax = _figure(COLUMN_IN, 2.35)
    _run_points(ax, t, x_of, "rel", ("rel_lo", "rel_hi"))
    u = summary["uniform_success_probability"].to_numpy()
    _sim_points(ax, summary, "success_probability", np.arange(len(sizes)), lambda s: s / u)
    ax.axhline(1.0, color="0.5", ls=":", lw=0.8, label="uniform sampling")
    ax.set_yscale("log")
    ax.set_xticks(range(len(sizes)), [f"$n={n}$" for n in sizes])
    ax.set_xlim(-0.5, len(sizes) - 0.3)
    ax.set_ylabel(r"$P_{\mathrm{opt}}$ / uniform $2^{-n}$")
    _legend_below(ax)
    ax.set_title("Final jobs (10,000 shots)" + _title_note(res))
    fig.tight_layout()
    return fig


def fig_hw_ibm_approx_ratio(res: Results) -> Figure:
    """Mean-energy approximation ratio of each final job (shot bootstrap 95% CI) with the
    exact uniform value per n and the Aer p=1 references."""
    tests = res.table("hardware", "final_tests")
    summary = res.table("hardware", "summary").sort_values("n").reset_index(drop=True)
    sizes = [int(n) for n in summary["n"]]
    x_of = {n: i for i, n in enumerate(sizes)}
    t = tests.assign(
        lo=tests["ratio_advantage_ci_low"] + tests["uniform_approx_ratio_mean"],
        hi=tests["ratio_advantage_ci_high"] + tests["uniform_approx_ratio_mean"],
    )
    fig, ax = _figure(COLUMN_IN, 2.35)
    _run_points(ax, t, x_of, "approx_ratio_mean", ("lo", "hi"))
    _sim_points(ax, summary, "approx_ratio_mean", np.arange(len(sizes)), lambda s: s.to_numpy())
    for i, r in summary.iterrows():
        ax.hlines(
            r["uniform_approx_ratio_mean"],
            i - 0.35,
            i + 0.5,
            color="0.5",
            ls=":",
            lw=0.9,
            label="uniform sampling (exact)" if i == 0 else None,
        )
    ax.set_xticks(range(len(sizes)), [f"$n={n}$" for n in sizes])
    ax.set_xlim(-0.5, len(sizes) - 0.3)
    ax.set_ylabel("Approx. ratio of mean energy")
    _legend_below(ax)
    ax.set_title("Final jobs (10,000 shots)" + _title_note(res))
    fig.tight_layout()
    return fig


def fig_hw_ibm_trajectory(res: Results) -> Figure:
    """Mean-energy ratio of every COBYLA-loop job by position in its run, per n; the final
    job of each run is the filled marker at the end. Incomplete runs are drawn in gray."""
    jobs = res.table("hardware", "jobs")
    sizes = sorted(int(n) for n in jobs["n"].unique())
    fig, axes = _figure(COLUMN_IN, 2.2, 1, len(sizes), sharey=False, squeeze=False)
    for ax, n, label in zip(axes[0], sizes, "abcdefgh", strict=False):
        g = jobs[jobs["n"] == n]
        for run, group in g.groupby("run"):
            k = int(run) - 1
            r = group.sort_values("iteration")
            loop = r[r["role"] == "optimization"]
            complete = bool(r["run_complete"].iloc[0])
            color = "0.0" if complete else "0.75"
            ax.plot(
                loop["iteration"],
                loop["approx_ratio_mean"],
                ls=RUN_LINES[k % len(RUN_LINES)] if complete else "-",
                color=color,
                lw=0.8 if complete else 0.6,
                label=f"run {int(run)}" + ("" if complete else " (no final job)"),
            )
            final = r[r["role"] == "final"]
            if len(final):
                ax.plot(
                    final["iteration"],
                    final["approx_ratio_mean"],
                    RUN_MARKERS[k % len(RUN_MARKERS)],
                    color="0.0",
                    ms=3.5,
                )
        ax.axhline(
            g["uniform_approx_ratio_mean"].iloc[0], color="0.5", ls=":", lw=0.8, label="uniform"
        )
        ax.set_xlabel("COBYLA evaluation (job in run)")
        ax.set_title(f"$n={n}$" + _title_note(res))
        _panel_label(ax, f"({label})")
    axes[0][0].set_ylabel("Approx. ratio of mean energy")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.tight_layout(w_pad=0.6, rect=(0, 0.1, 1, 1))
    fig.legend(handles, labels, fontsize=5.5, ncol=5, loc="lower center", handlelength=2.2)
    return fig
