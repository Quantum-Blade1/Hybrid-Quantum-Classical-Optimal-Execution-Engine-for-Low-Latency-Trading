"""QAOA (ideal Aer) vs simulated annealing on a 12-variable ExecutionQUBO (4 slices x 3 levels).

Reports energy, excess energy E - E_opt, approximation ratio (E_max - E)/(E_max - E_opt),
optimality gap, time and the rate of reaching the exact optimum, and writes a plot. The
equality penalty makes E_max - E_opt ~ 6e7, so the ratio and relative gap of any
near-feasible solution round to 1 and 0; the excess energy and hit rate discriminate.

Usage:
    python experiments/qaoa_vs_sa.py [--runs 5] [--seed 42] [--output-dir results]
"""

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from qexec.optimization.qubo import ExecutionQUBO, QUBOConfig
from qexec.optimization.solvers.annealing import SimulatedAnnealingSolver
from qexec.optimization.solvers.metrics import (
    EnergyBounds,
    approximation_ratio,
    energy_bounds,
    optimality_gap,
)
from qexec.optimization.solvers.qaoa import QAOASolver

OPTIMUM_TOL = 1e-6


@dataclass(frozen=True)
class SolverRuns:
    name: str
    energies: list[float]
    times: list[float]

    def success_rate(self, bounds: EnergyBounds) -> float:
        hits = [abs(e - bounds.min_energy) < OPTIMUM_TOL for e in self.energies]
        return sum(hits) / len(hits)

    def mean_ratio(self, bounds: EnergyBounds) -> float:
        return float(np.mean([approximation_ratio(e, bounds) for e in self.energies]))

    def mean_gap(self, bounds: EnergyBounds) -> float:
        return float(np.mean([optimality_gap(e, bounds) for e in self.energies]))


def run_qaoa(Q: NDArray[np.float64], runs: int, seed: int, p: int, maxiter: int) -> SolverRuns:
    energies, times = [], []
    for i in range(runs):
        result = QAOASolver(p=p, shots=500, maxiter=maxiter, seed=seed + i).solve(Q)
        energies.append(result.energy)
        times.append(result.solve_time)
    return SolverRuns(f"QAOA p={p}", energies, times)


def run_sa(Q: NDArray[np.float64], runs: int, seed: int, sweeps: int) -> SolverRuns:
    energies, times = [], []
    for i in range(runs):
        start = perf_counter()
        result = SimulatedAnnealingSolver(num_sweeps=sweeps, seed=seed + i).solve(Q)
        times.append(perf_counter() - start)
        energies.append(result.energy)
    return SolverRuns(f"SA {sweeps} sweeps", energies, times)


def print_summary(solvers: list[SolverRuns], bounds: EnergyBounds) -> None:
    print(f"\nExact optimum {bounds.min_energy:.4f}, worst assignment {bounds.max_energy:.4f}")
    header = f"{'Metric':<22}" + "".join(f"{s.name:>18}" for s in solvers)
    print(header)
    print("-" * len(header))
    rows: list[tuple[str, Callable[[SolverRuns], str]]] = [
        ("Best energy", lambda s: f"{min(s.energies):.4f}"),
        ("Mean energy", lambda s: f"{np.mean(s.energies):.4f}"),
        ("Std energy", lambda s: f"{np.std(s.energies):.4f}"),
        ("Mean E - E_opt", lambda s: f"{np.mean(s.energies) - bounds.min_energy:.4f}"),
        ("Mean approx. ratio", lambda s: f"{s.mean_ratio(bounds):.8f}"),
        ("Mean optimality gap", lambda s: f"{s.mean_gap(bounds):.2%}"),
        ("Optimum reached", lambda s: f"{s.success_rate(bounds):.0%}"),
        ("Mean time (s)", lambda s: f"{np.mean(s.times):.4f}"),
    ]
    for label, cell in rows:
        print(f"{label:<22}" + "".join(f"{cell(s):>18}" for s in solvers))


def plot_comparison(solvers: list[SolverRuns], bounds: EnergyBounds, path: Path) -> None:
    names = [s.name for s in solvers]
    colors = ["tab:blue", "tab:orange"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].boxplot([s.energies for s in solvers], tick_labels=names)
    axes[0].axhline(bounds.min_energy, color="green", linestyle="--", label="Optimum")
    axes[0].set_ylabel("Energy")
    axes[0].set_title("Energy Distribution")
    axes[0].legend()
    axes[1].bar(names, [np.mean(s.times) for s in solvers], color=colors, alpha=0.7)
    axes[1].set_ylabel("Time (s)")
    axes[1].set_title("Mean Solve Time")
    axes[2].bar(names, [s.mean_ratio(bounds) for s in solvers], color=colors, alpha=0.7)
    axes[2].set_ylabel("Approximation Ratio")
    axes[2].set_title("Mean Approximation Ratio")
    axes[2].set_ylim(0, 1.05)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = QUBOConfig(
        total_shares=400,
        num_time_slices=4,
        num_venues=1,
        quantity_levels=[0, 100, 200],
        equality_penalty=100.0,
        capacity_penalty=50.0,
    )
    qubo = ExecutionQUBO(config)
    Q = qubo.build_qubo_matrix()
    bounds = energy_bounds(Q)
    print(f"{config.total_shares} shares, {config.num_time_slices} slices: {Q.shape[0]} variables")

    solvers = [
        run_qaoa(Q, args.runs, args.seed, p=2, maxiter=30),
        run_sa(Q, args.runs, args.seed, sweeps=500),
    ]
    print_summary(solvers, bounds)

    best = SimulatedAnnealingSolver(num_sweeps=1000, seed=args.seed).solve(Q)
    print("\nSchedule from SA (1000 sweeps):")
    print(qubo.interpret_solution(best.solution).to_string(index=False))

    path = args.output_dir / "qaoa_vs_sa_comparison.png"
    plot_comparison(solvers, bounds, path)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
