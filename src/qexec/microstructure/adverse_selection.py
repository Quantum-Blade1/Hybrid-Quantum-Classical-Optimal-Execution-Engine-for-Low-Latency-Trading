"""Adverse selection as effective minus realised spread over a fixed horizon."""

from collections import deque

import numpy as np


class AdverseSelectionModel:
    """Effective spread 2d(p - m_t) and realised spread 2d(p - m_{t+h}), d = +1 buy, -1 sell.

    `estimate` averages both over the window and reports max(0, effective - realised),
    i.e. the adverse mid-price move 2d(m_{t+h} - m_t).
    """

    def __init__(self, lookback_ticks: int = 100, realized_horizon: int = 10) -> None:
        self._lookback = lookback_ticks
        self._horizon = realized_horizon
        self._trades: deque[tuple[float, float, str]] = deque(
            maxlen=lookback_ticks + realized_horizon
        )

    def update(self, trade_price: float, midpoint: float, side: str) -> None:
        self._trades.append((trade_price, midpoint, side))

    def estimate(self) -> tuple[float, float, float]:
        """(mean effective spread, mean realised spread, adverse selection); zeros until full."""
        if len(self._trades) < self._lookback + self._horizon:
            return 0.0, 0.0, 0.0

        effective_spreads = []
        realized_spreads = []
        for i in range(len(self._trades) - self._horizon):
            price, mid_at_trade, side = self._trades[i]
            mid_after = self._trades[i + self._horizon][1]
            if side == "buy":
                effective_spreads.append(2 * (price - mid_at_trade))
                realized_spreads.append(2 * (price - mid_after))
            else:
                effective_spreads.append(2 * (mid_at_trade - price))
                realized_spreads.append(2 * (mid_after - price))

        avg_effective = float(np.mean(effective_spreads))
        avg_realized = float(np.mean(realized_spreads))
        return avg_effective, avg_realized, max(0.0, avg_effective - avg_realized)
