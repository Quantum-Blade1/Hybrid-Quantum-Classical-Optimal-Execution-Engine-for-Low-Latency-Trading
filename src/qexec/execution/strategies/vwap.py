"""VWAP: slices proportional to a volume profile, capped by participation and slice size."""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from qexec.execution.strategies.base import BaseStrategy
from qexec.market.order_book import OrderBook


class VWAPStrategy(BaseStrategy):
    """Volume-weighted execution.

    With `historical_profile` the schedule follows that profile (resampled to the
    execution length); otherwise it uses the realised volume of `market_data`,
    i.e. a perfect-foresight VWAP.
    """

    strategy_name = "VWAP"
    benchmark_name = "VWAP"

    def __init__(
        self,
        participation_rate: float = 0.1,
        max_slice_pct: float = 0.05,
        historical_profile: NDArray[np.float64] | None = None,
        order_book: OrderBook | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(order_book=order_book, seed=seed)
        self.participation_rate = participation_rate
        self.max_slice_pct = max_slice_pct
        self.historical_profile = historical_profile

    def _volume_profile(self, market_data: pd.DataFrame) -> NDArray[np.float64]:
        if self.historical_profile is None:
            return market_data["volume"].to_numpy(dtype=np.float64)
        profile = np.asarray(self.historical_profile, dtype=np.float64)
        target_len = len(market_data)
        if len(profile) != target_len:
            positions = np.linspace(0, len(profile) - 1, target_len)
            profile = np.interp(positions, np.arange(len(profile)), profile)
        return profile

    def calculate_schedule(self, total_shares: int, market_data: pd.DataFrame) -> NDArray[np.int_]:
        """Round the capped volume-weighted schedule, then allocate leftover shares at random
        (weighted by volume) among slices still below both the participation and slice caps."""
        volume_profile = self._volume_profile(market_data)
        vol_weights = volume_profile / volume_profile.sum()
        participation_cap = volume_profile * self.participation_rate
        raw_schedule = np.minimum(vol_weights * total_shares, participation_cap)
        raw_schedule = np.minimum(raw_schedule, total_shares * self.max_slice_pct)
        schedule = np.round(raw_schedule).astype(int)

        while schedule.sum() > total_shares:
            schedule[np.argmax(schedule)] -= 1

        slice_cap = np.minimum(participation_cap, total_shares * self.max_slice_pct)
        for _ in range(total_shares - int(schedule.sum())):
            weights = vol_weights * (schedule < slice_cap)
            if weights.sum() > 0:
                idx = self.rng.choice(len(schedule), p=weights / weights.sum())
                schedule[idx] += 1

        return schedule
