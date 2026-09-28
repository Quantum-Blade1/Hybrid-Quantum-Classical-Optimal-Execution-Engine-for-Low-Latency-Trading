"""Queue position model for limit order placement."""

import numpy as np
from typing import List, Tuple


class QueuePositionModel:
    """
    Models queue position dynamics for limit order placement.

    In HFT, queue position determines fill probability. Orders at the
    front of the queue fill first; orders at the back may never fill.

    The QUBO formulation incorporates queue position as:
    - Expected fill probability per price level
    - Opportunity cost of waiting (price may move away)
    - Adverse selection risk of being picked off

    This allows the quantum optimizer to decide between:
    - Aggressive (market order): guaranteed fill, pays spread
    - Passive (limit order at best): uncertain fill, earns spread
    - Passive (limit order deeper): very uncertain fill, earns more spread
    """

    def __init__(self, tick_size: float = 0.01):
        self._tick_size = tick_size
        self._fill_rates: dict = {}
        self._cancel_rates: dict = {}

    def estimate_fill_probability(
        self,
        queue_position: int,
        total_queue: int,
        time_horizon_ms: float,
        arrival_rate: float
    ) -> float:
        if total_queue <= 0:
            return 1.0

        rate = arrival_rate * (time_horizon_ms / 1000.0)
        prob = 1.0 - np.exp(-rate * (1.0 - queue_position / total_queue))
        return np.clip(prob, 0.0, 1.0)

    def optimal_placement_level(
        self,
        book_levels: List[Tuple[float, int]],
        mid_price: float,
        urgency: float,
        time_horizon_ms: float,
        arrival_rate: float = 10.0
    ) -> Tuple[int, float]:
        best_level = 0
        best_score = -np.inf

        for level_idx, (price, queue_size) in enumerate(book_levels):
            distance_ticks = abs(price - mid_price) / self._tick_size
            fill_prob = self.estimate_fill_probability(
                queue_size, queue_size, time_horizon_ms, arrival_rate
            )
            spread_earned = abs(mid_price - price)
            adverse_risk = 0.001 * distance_ticks
            score = (
                fill_prob * (spread_earned - adverse_risk)
                - (1 - fill_prob) * urgency * abs(mid_price - price)
            )

            if score > best_score:
                best_score = score
                best_level = level_idx

        return best_level, best_score
