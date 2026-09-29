from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qexec.optimization.qubo import QUBOConfig
from qexec.optimization.solvers.result import BinaryVector, QUBOResult, QUBOSolver


class ScheduleQUBO(Protocol):
    def build_qubo_matrix(self) -> NDArray[np.float64]: ...

    def slice_quantities(self, x: BinaryVector) -> NDArray[np.float64]: ...


def slice_level_config(
    total_shares: int,
    num_slices: int,
    *,
    volatility: float | None = None,
    impact_coefficient: float | None = None,
) -> QUBOConfig:
    config = QUBOConfig(
        total_shares=total_shares,
        num_time_slices=num_slices,
        num_venues=1,
        quantity_levels=[0, total_shares // (num_slices * 2), total_shares // num_slices],
        equality_penalty=100.0,
    )
    if volatility is not None:
        config.volatility = volatility
    if impact_coefficient is not None:
        config.impact_coefficient = impact_coefficient
    return config


def repair_schedule(quantities: ArrayLike, total_shares: int) -> NDArray[np.int_]:
    """Rescale `quantities` to integers summing exactly to `total_shares` (largest remainder)."""
    q = np.clip(np.asarray(quantities, dtype=np.float64), 0.0, None)
    if q.ndim != 1 or q.size == 0:
        raise ValueError("quantities must be a non-empty 1-D array")
    if total_shares < 0:
        raise ValueError("total_shares must be non-negative")
    peak = q.max()
    # Normalise by the peak first so tiny (e.g. subnormal) masses cannot overflow the scale.
    shape = q / peak if peak > 0 else np.ones(q.size)
    scaled = shape * (total_shares / shape.sum())
    rounded = np.floor(scaled).astype(np.int_)
    missing = total_shares - int(rounded.sum())
    # Largest fractional parts first; ties go to the earliest slice.
    order = np.argsort(-(scaled - rounded), kind="stable")
    rounded[order[:missing]] += 1
    return rounded


def optimize_schedule(
    qubo: ScheduleQUBO,
    solver: QUBOSolver,
    Q: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], QUBOResult]:
    """Solve `qubo` (or the pre-built `Q`); returns (quantity per slice, solver result)."""
    if Q is None:
        Q = qubo.build_qubo_matrix()
    result = solver.solve(Q)
    return qubo.slice_quantities(result.solution), result


def spread_over_minutes(
    slice_quantities: NDArray[np.float64], num_minutes: int
) -> NDArray[np.float64]:
    """Even split within each slice; the integer remainder goes to the slice's first minute."""
    schedule = np.zeros(num_minutes)
    minutes_per_slice = num_minutes // len(slice_quantities)
    for t, quantity in enumerate(slice_quantities):
        start = t * minutes_per_slice
        end = min((t + 1) * minutes_per_slice, num_minutes)
        if end > start and quantity > 0:
            shares = int(quantity)
            schedule[start:end] = shares // (end - start)
            schedule[start] += shares % (end - start)
    return schedule
