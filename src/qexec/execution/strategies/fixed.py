"""Strategy that executes a precomputed schedule, rescaled to the order size."""

import numpy as np
import pandas as pd

from qexec.execution.strategies.base import BaseStrategy


class FixedScheduleStrategy(BaseStrategy):
    """
    Execute a schedule computed elsewhere (e.g. by a QUBO solve).

    The schedule is truncated to the length of the market data and
    rescaled so that it sums to the order quantity; an all-zero schedule
    falls back to a uniform one.
    """

    strategy_name = "FixedSchedule"

    def __init__(self, schedule: np.ndarray):
        super().__init__()
        self._schedule = np.asarray(schedule, dtype=float)

    def calculate_schedule(self, total_quantity: int, market_data: pd.DataFrame) -> np.ndarray:
        n = len(market_data)
        sched = self._schedule[:n].copy()
        sched_sum = sched.sum()
        if sched_sum > 0:
            sched = sched * (total_quantity / sched_sum)
        else:
            sched = np.full(n, total_quantity / n)
        return sched.astype(int)
