from collections import deque

TOXICITY_THRESHOLD = 0.7
_MIN_BUCKETS = 10


class VPINEstimator:
    """VPIN (Easley, Lopez de Prado & O'Hara, 2012) with tick-rule trade classification."""

    def __init__(self, bucket_size: int = 1000, num_buckets: int = 50) -> None:
        self._bucket_size = bucket_size
        self._current_bucket_volume = 0
        self._current_buy_volume = 0
        self._buckets: deque[tuple[int, int]] = deque(maxlen=num_buckets)
        self._vpin = 0.0

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

            if len(self._buckets) >= _MIN_BUCKETS:
                total_imbalance = sum(b[0] for b in self._buckets)
                total_volume = sum(b[1] for b in self._buckets)
                self._vpin = total_imbalance / total_volume if total_volume > 0 else 0.0

        return self._vpin

    @property
    def vpin(self) -> float:
        return self._vpin

    @property
    def is_toxic(self) -> bool:
        return self._vpin > TOXICITY_THRESHOLD
