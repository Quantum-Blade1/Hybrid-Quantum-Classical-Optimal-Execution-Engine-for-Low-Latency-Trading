"""Simulated annealing QUBO solver with single-bit-flip Metropolis moves and restarts.

Schedule (Phase 7): exactly `num_sweeps` sweeps at geometrically spaced temperatures

    T_s = T0 (T_f / T0)^(s / (num_sweeps - 1)),   s = 0, ..., num_sweeps - 1,

so `num_sweeps` is honoured (the Phase 6 schedule, T <- 0.95 T from 10 to 0.01, stopped
after 135 sweeps whatever `num_sweeps` said). A sweep visits every bit once in a random
order. `num_restarts` independent replicas run in lock-step (vectorised over replicas);
the best state seen by any replica is returned.

Temperatures default to the bounds used by D-Wave's `neal` sampler: T0 accepts the
largest possible uphill single flip with probability 1/2, T_f accepts the smallest
non-zero coefficient-sized uphill flip with probability 1/100:

    T0 = max_i (|Q_ii| + 2 sum_{j!=i} |Q_ij|) / ln 2
    T_f = min{|Q_ii|, 2|Q_ij| : non-zero} / ln 100
"""

import logging
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from qexec.optimization.solvers.result import QUBOResult

logger = logging.getLogger(__name__)

_COEFF_TOL = 1e-12


def flip_delta(Q: NDArray[np.float64], x: NDArray[np.int8], i: int) -> float:
    """Energy change of flipping bit i for symmetric Q: (1 - 2x_i)(Q_ii + 2 sum_{j!=i} Q_ij x_j)."""
    flip = 1 - 2 * x[i]
    return float(Q[i, i] * flip + 2 * (Q[i, :] @ x - Q[i, i] * x[i]) * flip)


def default_temperatures(Q: NDArray[np.float64]) -> tuple[float, float]:
    """(T0, T_final) from the coefficient bounds described in the module docstring."""
    diag = np.abs(np.diag(Q))
    off = np.abs(Q - np.diag(np.diag(Q)))
    max_delta = float(np.max(diag + 2 * off.sum(axis=1))) if Q.size else 1.0
    coeffs = np.concatenate([diag, 2 * off[np.triu_indices_from(off, k=1)]])
    nonzero = coeffs[coeffs > _COEFF_TOL]
    min_delta = float(nonzero.min()) if nonzero.size else 1.0
    max_delta = max(max_delta, min_delta)
    return max_delta / np.log(2), min_delta / np.log(100)


class SimulatedAnnealingSolver:
    """Geometric-schedule SA with `num_restarts` vectorised replicas (see module docstring)."""

    def __init__(
        self,
        initial_temp: float | None = None,
        final_temp: float | None = None,
        num_sweeps: int = 1000,
        num_restarts: int = 1,
        seed: int | None = None,
    ) -> None:
        if num_sweeps < 1 or num_restarts < 1:
            raise ValueError("num_sweeps and num_restarts must be positive")
        if initial_temp is not None and final_temp is not None and final_temp > initial_temp:
            raise ValueError("final_temp must not exceed initial_temp")
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.num_sweeps = num_sweeps
        self.num_restarts = num_restarts
        self.rng = np.random.default_rng(seed)

    def temperatures(self, Q: NDArray[np.float64]) -> NDArray[np.float64]:
        """The `num_sweeps` temperatures used on `Q`."""
        auto_t0, auto_tf = default_temperatures(Q)
        t0 = self.initial_temp if self.initial_temp is not None else auto_t0
        tf = self.final_temp if self.final_temp is not None else min(auto_tf, t0)
        if self.num_sweeps == 1:
            return np.array([tf])
        return np.asarray(np.geomspace(t0, tf, self.num_sweeps), dtype=np.float64)

    def solve(
        self, Q: NDArray[np.float64], initial_solution: NDArray[np.int8] | None = None
    ) -> QUBOResult:
        Q = np.asarray(Q, dtype=np.float64)
        n = Q.shape[0]
        R = self.num_restarts
        start = perf_counter()

        if initial_solution is not None:
            x = np.tile(np.asarray(initial_solution, dtype=np.float64), (R, 1))
        else:
            x = self.rng.integers(0, 2, size=(R, n)).astype(np.float64)
        field = x @ Q  # field[r, i] = sum_j Q_ij x_rj
        energy = np.einsum("ri,ri->r", field, x)
        diag = np.diag(Q).copy()
        best_x = x.copy()
        best_energy = energy.copy()
        history = [float(best_energy.min())]
        accepted_moves = 0
        rows = np.arange(R)
        temps = self.temperatures(Q)

        for temp in temps:
            uniforms = self.rng.random((n, R))
            for step, i in enumerate(self.rng.permutation(n)):
                xi = x[:, i]
                sign = 1.0 - 2.0 * xi
                delta = sign * (diag[i] + 2.0 * (field[:, i] - diag[i] * xi))
                accept = (delta < 0) | (uniforms[step] < np.exp(-np.maximum(delta, 0) / temp))
                if not accept.any():
                    continue
                idx = rows[accept]
                x[idx, i] = 1.0 - x[idx, i]
                field[idx] += sign[idx, None] * Q[i]
                energy[idx] += delta[idx]
                accepted_moves += idx.size
                improved = energy < best_energy
                if improved.any():
                    best_energy[improved] = energy[improved]
                    best_x[improved] = x[improved]
            history.append(float(best_energy.min()))

        winner = int(np.argmin(best_energy))
        solution = best_x[winner].astype(np.int8)
        solve_time = perf_counter() - start
        logger.debug(
            "SA n=%d: %d sweeps x %d restarts, %d accepted moves, best %.6g in %.3fs",
            n,
            temps.size,
            R,
            accepted_moves,
            best_energy[winner],
            solve_time,
        )
        return QUBOResult(
            solution=solution,
            energy=float(solution @ Q @ solution),
            num_evaluations=1 + temps.size * n * R,
            solve_time=solve_time,
            solver_name="SimulatedAnnealing",
            iterations=int(temps.size),
            history=history,
        )
