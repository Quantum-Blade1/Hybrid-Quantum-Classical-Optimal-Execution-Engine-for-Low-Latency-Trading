import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from qexec.execution.strategies.base import BaseStrategy
from qexec.optimization.schedule import repair_schedule


class FixedScheduleStrategy(BaseStrategy):
    """Precomputed schedule, padded or truncated, then repaired to sum exactly to the order."""

    strategy_name = "FixedSchedule"

    def __init__(self, schedule: ArrayLike) -> None:
        super().__init__()
        self._schedule = np.asarray(schedule, dtype=float)

    def calculate_schedule(self, total_shares: int, market_data: pd.DataFrame) -> NDArray[np.int_]:
        n = len(market_data)
        sched = np.zeros(n)
        head = self._schedule[:n]
        sched[: len(head)] = head
        return repair_schedule(sched, total_shares)
