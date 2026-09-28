"""Kyle's lambda (price impact coefficient) estimated by rolling regression."""

import numpy as np
from collections import deque


class KyleLambdaEstimator:
    """
    Estimates Kyle's lambda (price impact coefficient) from trade data.

    Kyle (1985) model: Delta_p = lambda * (signed_volume) + noise

    lambda represents the permanent price impact per unit of signed order flow.
    Higher lambda = less liquid market = higher execution cost.

    In the QUBO context, lambda scales the market impact quadratic term,
    making the optimizer naturally route less aggressively when impact is high.
    """

    def __init__(self, window_size: int = 100):
        self._price_changes: deque = deque(maxlen=window_size)
        self._signed_volumes: deque = deque(maxlen=window_size)
        self._lambda: float = 0.0
        self._window = window_size

    def update(self, price_change: float, signed_volume: float) -> float:
        self._price_changes.append(price_change)
        self._signed_volumes.append(signed_volume)

        if len(self._price_changes) < 20:
            return self._lambda

        dp = np.array(self._price_changes)
        sv = np.array(self._signed_volumes)

        sv_var = np.var(sv)
        if sv_var < 1e-12:
            return self._lambda

        self._lambda = np.cov(dp, sv)[0, 1] / sv_var
        return self._lambda

    @property
    def lambda_value(self) -> float:
        return max(0.0, self._lambda)

    def estimate_impact(self, order_size: float) -> float:
        return self._lambda * order_size
