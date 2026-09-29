from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from qexec.optimization.solvers.annealing import flip_delta
from qexec.optimization.solvers.result import QUBOResult

_IMPROVEMENT_TOL = 1e-10


class GreedySolver:
    """Sweeps bits in order, flipping any bit that lowers the energy, until a local minimum."""

    def __init__(self, max_iterations: int = 1000, seed: int | None = None) -> None:
        self.max_iterations = max_iterations
        self.rng = np.random.default_rng(seed)

    def solve(
        self, Q: NDArray[np.float64], initial_solution: NDArray[np.int8] | None = None
    ) -> QUBOResult:
        n = Q.shape[0]
        start = perf_counter()
        x: NDArray[np.int8] = (
            initial_solution.astype(np.int8)
            if initial_solution is not None
            else np.asarray(self.rng.integers(0, 2, size=n, dtype=np.int8))
        )

        current_energy = float(x @ Q @ x)
        num_evaluations = 1
        sweeps = 0
        improved = True
        while improved and sweeps < self.max_iterations:
            sweeps += 1
            improved = False
            for i in range(n):
                delta = flip_delta(Q, x, i)
                num_evaluations += 1
                if delta < -_IMPROVEMENT_TOL:
                    x[i] = 1 - x[i]
                    current_energy += delta
                    improved = True

        return QUBOResult(
            solution=x,
            energy=current_energy,
            num_evaluations=num_evaluations,
            solve_time=perf_counter() - start,
            solver_name="Greedy",
            iterations=sweeps,
        )
