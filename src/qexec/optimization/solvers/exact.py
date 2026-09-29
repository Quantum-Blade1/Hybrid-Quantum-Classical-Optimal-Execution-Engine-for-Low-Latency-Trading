from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from qexec.optimization.solvers.result import QUBOResult


class BruteForceSolver:
    """Enumerates all 2^n assignments; O(n^2 2^n), practical for n <= ~20."""

    def __init__(self, max_variables: int = 25) -> None:
        self.max_variables = max_variables

    def solve(self, Q: NDArray[np.float64]) -> QUBOResult:
        n = Q.shape[0]
        if n > self.max_variables:
            raise ValueError(
                f"Problem size {n} exceeds max_variables={self.max_variables}. "
                "Use simulated annealing for larger problems."
            )

        start = perf_counter()
        best_solution = np.zeros(n, dtype=np.int8)
        best_energy = float("inf")
        total_solutions = 2**n
        for i in range(total_solutions):
            x = np.array([(i >> bit) & 1 for bit in range(n)], dtype=np.int8)
            energy = float(x @ Q @ x)
            if energy < best_energy:
                best_energy = energy
                best_solution = x

        return QUBOResult(
            solution=best_solution,
            energy=best_energy,
            num_evaluations=total_solutions,
            solve_time=perf_counter() - start,
            solver_name="BruteForce",
        )
