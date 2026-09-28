"""Simulated annealing QUBO solver with single-bit-flip Metropolis moves."""

import logging
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from qexec.optimization.solvers.result import QUBOResult

logger = logging.getLogger(__name__)


def flip_delta(Q: NDArray[np.float64], x: NDArray[np.int8], i: int) -> float:
    """Energy change of flipping bit i for symmetric Q: (1 - 2x_i)(Q_ii + 2 sum_{j!=i} Q_ij x_j)."""
    flip = 1 - 2 * x[i]
    return float(Q[i, i] * flip + 2 * (Q[i, :] @ x - Q[i, i] * x[i]) * flip)


class SimulatedAnnealingSolver:
    """Geometric cooling T_{k+1} = cooling_rate * T_k; one sweep visits every bit in random order.

    Stops after `num_sweeps` sweeps or once T <= final_temp.
    """

    def __init__(
        self,
        initial_temp: float = 10.0,
        final_temp: float = 0.01,
        cooling_rate: float = 0.95,
        num_sweeps: int = 1000,
        seed: int | None = None,
    ) -> None:
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.num_sweeps = num_sweeps
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
        best_solution = x.copy()
        best_energy = current_energy
        num_evaluations = 1
        accepted_moves = 0
        history = [current_energy]
        temp = self.initial_temp
        sweep = 0

        while temp > self.final_temp and sweep < self.num_sweeps:
            for i in self.rng.permutation(n):
                delta_e = flip_delta(Q, x, i)
                num_evaluations += 1
                if delta_e < 0 or self.rng.random() < np.exp(-delta_e / temp):
                    x[i] = 1 - x[i]
                    current_energy += delta_e
                    accepted_moves += 1
                if current_energy < best_energy:
                    best_solution = x.copy()
                    best_energy = current_energy
            temp *= self.cooling_rate
            sweep += 1
            history.append(best_energy)

        solve_time = perf_counter() - start
        logger.debug(
            "SA n=%d: %d sweeps, %d accepted moves, best %.4f in %.3fs",
            n,
            sweep,
            accepted_moves,
            best_energy,
            solve_time,
        )
        return QUBOResult(
            solution=best_solution,
            energy=best_energy,
            num_evaluations=num_evaluations,
            solve_time=solve_time,
            solver_name="SimulatedAnnealing",
            iterations=sweep,
            history=history,
        )
