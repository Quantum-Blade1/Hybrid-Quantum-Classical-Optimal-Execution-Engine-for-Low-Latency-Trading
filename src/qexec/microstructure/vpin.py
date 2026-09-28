"""VPIN (volume-synchronized probability of informed trading)."""

from collections import deque


class VPINEstimator:
    """
    Volume-Synchronized Probability of Informed Trading.

    VPIN (Easley, Lopez de Prado, O'Hara 2012) estimates the probability
    that a trade is information-driven rather than noise. High VPIN signals
    that the market is toxic for market makers and liquidity providers.

    For HFT execution, high VPIN means:
    - Wider effective spreads (adverse selection)
    - Higher market impact (information leakage)
    - The QUBO should shift to more passive, spread-crossing execution

    The quantum optimizer uses VPIN to dynamically adjust:
    - impact_coefficient in QUBOConfig
    - Venue routing weights (avoid lit venues when toxic)
    - Urgency parameter (slow down in toxic flow)
    """

    def __init__(self, bucket_size: int = 1000, num_buckets: int = 50):
        self._bucket_size = bucket_size
        self._num_buckets = num_buckets
        self._current_bucket_volume = 0
        self._current_buy_volume = 0
        self._buckets: deque = deque(maxlen=num_buckets)
        self._vpin: float = 0.0

    def update(self, price: float, volume: int, prev_price: float) -> float:
        if price > prev_price:
            buy_volume = volume
        elif price < prev_price:
            buy_volume = 0
        else:
            buy_volume = volume // 2

        self._current_buy_volume += buy_volume
        self._current_bucket_volume += volume

        if self._current_bucket_volume >= self._bucket_size:
            sell_volume = self._current_bucket_volume - self._current_buy_volume
            imbalance = abs(self._current_buy_volume - sell_volume)
            self._buckets.append((imbalance, self._current_bucket_volume))

            self._current_bucket_volume = 0
            self._current_buy_volume = 0

            if len(self._buckets) >= 10:
                total_imbalance = sum(b[0] for b in self._buckets)
                total_volume = sum(b[1] for b in self._buckets)
                self._vpin = total_imbalance / total_volume if total_volume > 0 else 0

        return self._vpin

    @property
    def vpin(self) -> float:
        return self._vpin

    @property
    def is_toxic(self) -> bool:
        return self._vpin > 0.7
