"""Adverse selection cost model for passive and aggressive fills."""

import numpy as np
from typing import Tuple
from collections import deque


class AdverseSelectionModel:
    """
    Estimates adverse selection cost from trade and quote data.

    Adverse selection = cost of trading against informed counterparties.
    Measured as the difference between the effective spread (what you paid)
    and the realized spread (what you kept after price moved).

    AS_cost = (effective_spread - realized_spread) / 2

    When AS_cost is high, the QUBO should:
    - Increase dark pool routing weight (avoid lit adverse selection)
    - Widen the timing window (patient execution)
    - Reduce per-slice quantity (smaller footprint)
    """

    def __init__(self, lookback_ticks: int = 100, realized_horizon: int = 10):
        self._lookback = lookback_ticks
        self._horizon = realized_horizon
        self._trades: deque = deque(maxlen=lookback_ticks + realized_horizon)
        self._midpoints: deque = deque(maxlen=lookback_ticks + realized_horizon)

    def update(self, trade_price: float, midpoint: float, side: str) -> None:
        self._trades.append((trade_price, midpoint, side))
        self._midpoints.append(midpoint)

    def estimate(self) -> Tuple[float, float, float]:
        if len(self._trades) < self._lookback + self._horizon:
            return 0.0, 0.0, 0.0

        effective_spreads = []
        realized_spreads = []

        for i in range(len(self._trades) - self._horizon):
            price, mid_at_trade, side = self._trades[i]
            mid_after = self._midpoints[i + self._horizon]

            if side == "buy":
                eff = 2 * (price - mid_at_trade)
                real = 2 * (price - mid_after)
            else:
                eff = 2 * (mid_at_trade - price)
                real = 2 * (mid_after - price)

            effective_spreads.append(eff)
            realized_spreads.append(real)

        avg_effective = np.mean(effective_spreads) if effective_spreads else 0
        avg_realized = np.mean(realized_spreads) if realized_spreads else 0
        adverse_selection = max(0, avg_effective - avg_realized)

        return avg_effective, avg_realized, adverse_selection
