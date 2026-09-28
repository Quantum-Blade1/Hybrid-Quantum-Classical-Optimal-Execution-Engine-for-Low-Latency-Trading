"""
From an execution QUBO to an execution schedule.

Every caller that turns a QUBO solution into shares-per-period goes through
this module: the slow-path optimizer, the HFT pipeline, QUBOStrategy, the
walk-forward backtest and the analysis runners.
"""

from typing import Optional, Protocol, Tuple

import numpy as np

from qexec.optimization.qubo import QUBOConfig
from qexec.optimization.solvers.result import QUBOResult


class ScheduleQUBO(Protocol):
    """An execution QUBO whose solutions decode to per-slice quantities."""

    def build_qubo_matrix(self) -> np.ndarray: ...

    def slice_quantities(self, x: np.ndarray) -> np.ndarray: ...


class QUBOSolver(Protocol):
    def solve(self, Q: np.ndarray, verbose: bool = False) -> QUBOResult: ...


def slice_level_config(total_shares: int, num_slices: int, **overrides) -> QUBOConfig:
    """Single-venue QUBOConfig with quantity levels {0, N/(2T), N/T} and equality penalty 100."""
    params = dict(
        total_shares=total_shares,
        num_time_slices=num_slices,
        num_venues=1,
        quantity_levels=[0, total_shares // (num_slices * 2), total_shares // num_slices],
        equality_penalty=100.0,
    )
    params.update(overrides)
    return QUBOConfig(**params)


def optimize_schedule(
    qubo: ScheduleQUBO,
    solver: QUBOSolver,
    Q: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, QUBOResult]:
    """
    Solve an execution QUBO and decode the solution to quantity per time slice.

    Args:
        qubo: ExecutionQUBO or HFTExecutionQUBO.
        solver: Any QUBO solver with ``solve(Q, verbose=False) -> QUBOResult``.
        Q: Pre-built QUBO matrix; built from ``qubo`` when omitted.

    Returns:
        (quantity per slice, solver result)
    """
    if Q is None:
        Q = qubo.build_qubo_matrix()
    result = solver.solve(Q, verbose=False)
    return qubo.slice_quantities(result.solution), result


def spread_over_minutes(slice_quantities: np.ndarray, num_minutes: int) -> np.ndarray:
    """
    Spread per-slice quantities evenly over the minutes of each slice.

    Slices are ``num_minutes // num_slices`` minutes long; the integer
    remainder of each slice goes to its first minute. Minutes after the
    last full slice receive nothing.
    """
    schedule = np.zeros(num_minutes)
    minutes_per_slice = num_minutes // len(slice_quantities)
    for t, quantity in enumerate(slice_quantities):
        start = t * minutes_per_slice
        end = min((t + 1) * minutes_per_slice, num_minutes)
        if end > start and quantity > 0:
            quantity = int(quantity)
            schedule[start:end] = quantity // (end - start)
            schedule[start] += quantity % (end - start)
    return schedule
