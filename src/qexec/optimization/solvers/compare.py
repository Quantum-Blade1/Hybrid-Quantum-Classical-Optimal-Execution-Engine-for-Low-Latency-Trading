"""Run several QUBO solvers on the same instance and tabulate the results."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from qexec.optimization.solvers.annealing import SimulatedAnnealingSolver
from qexec.optimization.solvers.exact import BruteForceSolver
from qexec.optimization.solvers.greedy import GreedySolver
from qexec.optimization.solvers.result import QUBOResult, QUBOSolver

_BRUTE_FORCE_LIMIT = 20


def default_solvers(n: int, seed: int = 42) -> list[QUBOSolver]:
    """Brute force (when n <= 20), greedy and SA with 500 and 2000 sweeps."""
    solvers: list[QUBOSolver] = []
    if n <= _BRUTE_FORCE_LIMIT:
        solvers.append(BruteForceSolver())
    solvers += [
        GreedySolver(seed=seed),
        SimulatedAnnealingSolver(num_sweeps=500, seed=seed),
        SimulatedAnnealingSolver(num_sweeps=2000, seed=seed),
    ]
    return solvers


def compare_solvers(
    Q: NDArray[np.float64], solvers: Sequence[QUBOSolver] | None = None
) -> list[QUBOResult]:
    """Solve `Q` with each solver (default: `default_solvers`) in order."""
    if solvers is None:
        solvers = default_solvers(Q.shape[0])
    return [solver.solve(Q) for solver in solvers]


def format_comparison(results: Sequence[QUBOResult]) -> str:
    """Plain-text table of energy, evaluations and time, with the best solver last."""
    lines = [f"{'Solver':<25} {'Energy':>12} {'Evaluations':>12} {'Time (s)':>12}"]
    lines += [
        f"{r.solver_name:<25} {r.energy:>12.4f} {r.num_evaluations:>12,} {r.solve_time:>12.4f}"
        for r in results
    ]
    best = min(results, key=lambda r: r.energy)
    lines.append(f"Best: {best.solver_name} with energy {best.energy:.4f}")
    return "\n".join(lines)
