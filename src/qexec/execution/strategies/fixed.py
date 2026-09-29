"""Strategy that executes a precomputed schedule, rescaled to the order size."""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from qexec.execution.strategies.base import BaseStrategy


class FixedScheduleStrategy(BaseStrategy):
    """Execute a schedule computed elsewhere (e.g. by a QUBO solve).

    The schedule is truncated to the market-data length and rescaled to sum exactly to
    the order quantity (largest-remainder rounding); an all-zero schedule falls back to
    a uniform one.
    """

    strategy_name = "FixedSchedule"

    def __init__(self, schedule: ArrayLike) -> None:
        super().__init__()
        self._schedule = np.asarray(schedule, dtype=float)

    def calculate_schedule(self, total_shares: int, market_data: pd.DataFrame) -> NDArray[np.int_]:
        n = len(market_data)
        sched = self._schedule[:n].copy()
        sched_sum = sched.sum()
        if sched_sum > 0:
            sched = sched * (total_shares / sched_sum)
        else:
            sched = np.full(n, total_shares / n)
        # Largest-remainder rounding: integer shares that still sum to the order size.
        rounded = np.floor(sched).astype(int)
        shortfall = total_shares - int(rounded.sum())
        rounded[np.argsort(rounded - sched, kind="stable")[:shortfall]] += 1
        return rounded
