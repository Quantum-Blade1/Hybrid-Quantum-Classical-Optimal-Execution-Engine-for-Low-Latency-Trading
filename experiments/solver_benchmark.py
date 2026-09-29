"""Solver benchmark on random QUBOs: brute force, SA, fixed-angle QAOA and (if installed) Gurobi.

QAOA here uses fixed heuristic angles (gamma = 0.5 + 0.1p, beta = 0.3 + 0.05p) without
variational optimisation, sampled with shots (QASM) or read from the exact statevector.
Quality is the approximation ratio (E_max - E)/(E_max - E_opt) and the optimality gap
against exhaustive-search bounds.

Usage:
    python experiments/solver_benchmark.py [--runs 3] [--seed 42] [--output-dir results]
"""

import argparse
import importlib.util
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from numpy.typing import NDArray
from qiskit import transpile
from qiskit_aer import AerSimulator

from qexec.optimization.ising import build_qaoa_circuit_from_ising, qubo_to_ising
from qexec.optimization.solvers.annealing import SimulatedAnnealingSolver
from qexec.optimization.solvers.exact import BruteForceSolver
from qexec.optimization.solvers.metrics import (
    EnergyBounds,
    approximation_ratio,
    energy_bounds,
    optimality_gap,
)

Matrix = NDArray[np.float64]
Solution = tuple[NDArray[np.int_], float]
OPTIMUM_TOL = 1e-6
STATEVECTOR_MIN_PROBABILITY = 0.01


class Solver(ABC):
    name: str
    solver_type: str

    @abstractmethod
    def solve(self, Q: Matrix) -> Solution:
        """Return (solution, energy)."""


class ExactSolver(Solver):
    name = "BruteForce"
    solver_type = "classical"

    def solve(self, Q: Matrix) -> Solution:
        result = BruteForceSolver().solve(Q)
        return result.solution.astype(int), result.energy


class SABenchmarkSolver(Solver):
    """SA with one generator across runs, so repeated runs differ but are reproducible."""

    solver_type = "classical"

    def __init__(self, num_sweeps: int = 500, seed: int = 42) -> None:
        self.name = f"SA-{num_sweeps}"
        self._solver = SimulatedAnnealingSolver(num_sweeps=num_sweeps, seed=seed)

    def solve(self, Q: Matrix) -> Solution:
        result = self._solver.solve(Q)
        return result.solution.astype(int), result.energy


class GurobiSolver(Solver):
    name = "Gurobi"
    solver_type = "classical"

    @staticmethod
    def is_available() -> bool:
        return importlib.util.find_spec("gurobipy") is not None

    def solve(self, Q: Matrix) -> Solution:
        import gurobipy as gp

        n = Q.shape[0]
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            with gp.Model(env=env) as model:
                x = model.addVars(n, vtype=gp.GRB.BINARY, name="x")
                model.setObjective(
                    gp.quicksum(Q[i, j] * x[i] * x[j] for i in range(n) for j in range(n)),
                    gp.GRB.MINIMIZE,
                )
                model.optimize()
                solution = np.array([round(x[i].X) for i in range(n)])
                return solution, float(model.objVal)


def _fixed_angles(p: int) -> tuple[float, float]:
    return 0.5 + 0.1 * p, 0.3 + 0.05 * p


def _best_of(Q: Matrix, candidates: list[NDArray[np.int_]]) -> Solution:
    best_x, best_e = np.zeros(Q.shape[0], dtype=int), float("inf")
    for x in candidates:
        e = float(x @ Q @ x)
        if e < best_e:
            best_x, best_e = x, e
    return best_x, best_e


class QASMSimulatorSolver(Solver):
    """Lowest-energy bitstring among `shots` samples of the fixed-angle circuit.

    Each call uses a fresh simulator seed drawn from `seed`, so runs differ but repeat exactly.
    """

    solver_type = "quantum_sim"

    def __init__(self, shots: int = 4000, p: int = 1, seed: int = 42) -> None:
        self.name = f"QASM-p{p}"
        self.shots = shots
        self.p = p
        self.simulator = AerSimulator()
        self.rng = np.random.default_rng(seed)

    def solve(self, Q: Matrix) -> Solution:
        gamma, beta = _fixed_angles(self.p)
        qc = build_qaoa_circuit_from_ising(qubo_to_ising(Q), gamma, beta, self.p)
        result = self.simulator.run(
            transpile(qc, self.simulator),
            shots=self.shots,
            seed_simulator=int(self.rng.integers(2**31)),
        ).result()
        candidates = [np.array([int(b) for b in bs[::-1]]) for bs in result.get_counts()]
        return _best_of(Q, [x for x in candidates if len(x) == Q.shape[0]])


class StatevectorSolver(Solver):
    """Lowest-energy basis state with probability > 1% in the exact output state."""

    solver_type = "quantum_sim"

    def __init__(self, p: int = 1) -> None:
        self.name = f"Statevector-p{p}"
        self.p = p
        self.simulator = AerSimulator(method="statevector")

    def solve(self, Q: Matrix) -> Solution:
        n = Q.shape[0]
        gamma, beta = _fixed_angles(self.p)
        qc = build_qaoa_circuit_from_ising(qubo_to_ising(Q), gamma, beta, self.p)
        qc.remove_final_measurements()
        qc.save_statevector()
        state = self.simulator.run(transpile(qc, self.simulator)).result().get_statevector()
        probs = np.abs(np.asarray(state)) ** 2
        candidates = [
            np.array([(i >> j) & 1 for j in range(n)])
            for i in np.flatnonzero(probs > STATEVECTOR_MIN_PROBABILITY)
        ]
        return _best_of(Q, candidates)


@dataclass(frozen=True)
class BenchmarkResult:
    solver_name: str
    solver_type: str
    problem_size: int
    energy: float
    bounds: EnergyBounds
    solve_time: float
    num_runs: int
    success_rate: float
    all_energies: list[float] = field(default_factory=list)

    @property
    def approximation_ratio(self) -> float:
        return approximation_ratio(self.energy, self.bounds)

    @property
    def optimality_gap(self) -> float:
        return optimality_gap(self.energy, self.bounds)


def random_qubo(n: int, rng: np.random.Generator) -> Matrix:
    Q = rng.standard_normal((n, n))
    return (Q + Q.T) / 2


def run_solver(solver: Solver, Q: Matrix, bounds: EnergyBounds, num_runs: int) -> BenchmarkResult:
    """Best energy over `num_runs`, mean time, and the fraction of runs reaching the optimum."""
    energies, times = [], []
    for _ in range(num_runs):
        start = perf_counter()
        _, energy = solver.solve(Q)
        times.append(perf_counter() - start)
        energies.append(energy)
    hits = [abs(e - bounds.min_energy) < OPTIMUM_TOL for e in energies]
    return BenchmarkResult(
        solver_name=solver.name,
        solver_type=solver.solver_type,
        problem_size=Q.shape[0],
        energy=min(energies),
        bounds=bounds,
        solve_time=float(np.mean(times)),
        num_runs=num_runs,
        success_rate=sum(hits) / num_runs,
        all_energies=energies,
    )


def run_benchmark(
    solvers: list[Solver], sizes: list[int], num_runs: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for size in sizes:
        Q = random_qubo(size, rng)
        bounds = energy_bounds(Q)
        print(f"\nn = {size}: optimum {bounds.min_energy:.4f}, worst {bounds.max_energy:.4f}")
        for solver in solvers:
            r = run_solver(solver, Q, bounds, num_runs)
            print(
                f"  {r.solver_name:<16} ratio {r.approximation_ratio:.4f} "
                f"gap {r.optimality_gap:7.2%} ({r.solve_time:.3f}s)"
            )
            rows.append(
                {
                    "Solver": r.solver_name,
                    "Type": r.solver_type,
                    "Size": r.problem_size,
                    "Energy": r.energy,
                    "Optimal": bounds.min_energy,
                    "Approx. ratio": r.approximation_ratio,
                    "Gap (%)": 100 * r.optimality_gap,
                    "Time (s)": r.solve_time,
                    "Success Rate": r.success_rate,
                }
            )
    return pd.DataFrame(rows)


def _grouped_bars(ax: Axes, df: pd.DataFrame, column: str, scale: float = 1.0) -> None:
    solvers = list(df["Solver"].unique())
    sizes = sorted(df["Size"].unique())
    x = np.arange(len(solvers))
    width = 0.8 / len(sizes)
    for i, size in enumerate(sizes):
        subset = df[df["Size"] == size].set_index("Solver")[column]
        ax.bar(
            x + i * width, [scale * subset.get(s, 0.0) for s in solvers], width, label=f"n={size}"
        )
    ax.set_xticks(x + width * (len(sizes) - 1) / 2)
    ax.set_xticklabels(solvers, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")


def plot_report(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    _grouped_bars(axes[0, 0], df, "Approx. ratio")
    axes[0, 0].set(ylabel="Approximation ratio", ylim=(0, 1.05), title="Solution Quality")
    _grouped_bars(axes[0, 1], df, "Time (s)")
    axes[0, 1].set(ylabel="Time (s)", yscale="log", title="Time to Solution")
    _grouped_bars(axes[1, 0], df, "Success Rate", scale=100)
    axes[1, 0].set(ylabel="Runs reaching the optimum (%)", ylim=(0, 105), title="Reliability")

    summary = df.groupby("Solver", sort=False)[["Approx. ratio", "Time (s)", "Success Rate"]].mean()
    cells = [
        [str(name), f"{row.iloc[0]:.3f}", f"{row.iloc[1]:.4f}", f"{row.iloc[2]:.0%}"]
        for name, row in summary.iterrows()
    ]
    axes[1, 1].axis("off")
    table = axes[1, 1].table(
        cellText=cells,
        colLabels=["Solver", "Mean ratio", "Mean time (s)", "Optimum reached"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    axes[1, 1].set_title("Summary", pad=20)

    fig.suptitle("QUBO Solver Benchmark", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    solvers: list[Solver] = [
        ExactSolver(),
        SABenchmarkSolver(num_sweeps=500, seed=args.seed),
        QASMSimulatorSolver(shots=2000, p=1, seed=args.seed),
        QASMSimulatorSolver(shots=2000, p=2, seed=args.seed),
        StatevectorSolver(p=1),
    ]
    if GurobiSolver.is_available():
        solvers.insert(2, GurobiSolver())
    print(f"Gurobi: {'available' if GurobiSolver.is_available() else 'not installed'}")

    df = run_benchmark(solvers, args.sizes, args.runs, args.seed)
    print()
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    path = args.output_dir / "benchmark_report.png"
    plot_report(df, path)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
