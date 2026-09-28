"""Kyle's lambda estimated by rolling OLS of price changes on signed volume."""

from collections import deque

import numpy as np

_MIN_OBSERVATIONS = 20
_MIN_VOLUME_VARIANCE = 1e-12


class KyleLambdaEstimator:
    """Slope of dp = lambda * signed_volume + noise (Kyle, 1985) over a rolling window.

    Equal-weight OLS; the estimate is held until 20 observations are available.
    """

    def __init__(self, window_size: int = 100) -> None:
        self._price_changes: deque[float] = deque(maxlen=window_size)
        self._signed_volumes: deque[float] = deque(maxlen=window_size)
        self._lambda = 0.0

    def update(self, price_change: float, signed_volume: float) -> float:
        """Add one observation and return the (unclamped) OLS slope."""
        self._price_changes.append(price_change)
        self._signed_volumes.append(signed_volume)
        if len(self._price_changes) < _MIN_OBSERVATIONS:
            return self._lambda

        dp = np.array(self._price_changes)
        sv = np.array(self._signed_volumes)
        sv_var = np.var(sv)
        if sv_var < _MIN_VOLUME_VARIANCE:
            return self._lambda
        # Same ddof in numerator and denominator, so this is the OLS slope.
        self._lambda = float(np.cov(dp, sv, ddof=0)[0, 1] / sv_var)
        return self._lambda

    @property
    def lambda_value(self) -> float:
        """Estimate clamped at zero (negative impact is treated as no impact)."""
        return max(0.0, self._lambda)
