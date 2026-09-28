"""Synthetic intraday market data: GBM prices, U-shaped volume, volume-dependent spreads."""

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from numpy.typing import NDArray

TRADING_MINUTES_PER_DAY = 390
TRADING_DAYS_PER_YEAR = 252
_MIN_MINUTE_VOLUME = 100

SeedLike = int | np.random.SeedSequence | None


@dataclass
class MarketParams:
    """Parameters of the synthetic market."""

    symbol: str = "AAPL"
    initial_price: float = 150.0
    annual_volatility: float = 0.25
    annual_drift: float = 0.08
    trading_start: str = "09:30"
    trading_end: str = "16:00"
    tick_size: float = 0.01
    min_spread_bps: float = 1.0
    max_spread_bps: float = 10.0


class IntraDayPriceGenerator:
    """Minute-level GBM with a U-shaped (open/close) volatility multiplier in [1, 2.5]."""

    def __init__(self, params: MarketParams, seed: SeedLike = None) -> None:
        self.params = params
        self.rng = np.random.default_rng(seed)
        minutes_per_year = TRADING_MINUTES_PER_DAY * TRADING_DAYS_PER_YEAR
        self.minute_volatility = params.annual_volatility / np.sqrt(minutes_per_year)
        self.minute_drift = params.annual_drift / minutes_per_year

    @staticmethod
    def _intraday_volatility_multiplier(
        minutes_from_open: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        t = minutes_from_open / TRADING_MINUTES_PER_DAY
        u_shape = 4 * (t - 0.5) ** 2 + 0.5
        return 1.0 + 1.5 * (u_shape - 0.5) / 0.5

    def generate(
        self,
        date: datetime,
        num_minutes: int = TRADING_MINUTES_PER_DAY,
        initial_price: float | None = None,
    ) -> pd.DataFrame:
        """Columns: timestamp, price (rounded to tick). `initial_price` overrides the params."""
        if initial_price is None:
            initial_price = self.params.initial_price
        start_time = datetime.strptime(self.params.trading_start, "%H:%M")
        timestamps = [
            datetime.combine(date.date(), start_time.time()) + timedelta(minutes=i)
            for i in range(num_minutes)
        ]
        vol_multipliers = self._intraday_volatility_multiplier(
            np.arange(num_minutes, dtype=np.float64)
        )
        shocks = self.rng.standard_normal(num_minutes)
        log_returns = self.minute_drift + self.minute_volatility * vol_multipliers * shocks
        prices = np.exp(np.log(initial_price) + np.cumsum(log_returns))
        prices = np.round(prices / self.params.tick_size) * self.params.tick_size
        return pd.DataFrame({"timestamp": timestamps, "price": prices})


class VolumeProfileGenerator:
    """U-shaped intraday volume (opening peak > closing peak) with lognormal noise."""

    def __init__(self, total_daily_volume: int = 50_000_000, seed: SeedLike = None) -> None:
        self.total_daily_volume = total_daily_volume
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _volume_profile_weights(num_minutes: int) -> NDArray[np.float64]:
        t = np.linspace(0, 1, num_minutes)
        opening_peak = 1.5 * np.exp(-(t**2) / 0.01)
        closing_peak = 1.2 * np.exp(-((t - 1.0) ** 2) / 0.02)
        baseline = 0.3 + 0.4 * (4 * (t - 0.5) ** 2)
        weights = opening_peak + closing_peak + baseline
        return np.asarray(weights / weights.sum())

    def generate(self, num_minutes: int = TRADING_MINUTES_PER_DAY) -> NDArray[np.int_]:
        """Per-minute volume, floored at 100 shares."""
        weights = self._volume_profile_weights(num_minutes)
        noisy_weights = weights * self.rng.lognormal(0, 0.3, num_minutes)
        noisy_weights /= noisy_weights.sum()
        volumes = (noisy_weights * self.total_daily_volume).astype(int)
        return np.maximum(volumes, _MIN_MINUTE_VOLUME)


class MarketDataSimulator:
    """Combines price, volume and spread generation into one minute-bar DataFrame.

    Prices, volumes and spreads draw from independent child streams of `seed`.
    """

    def __init__(
        self,
        params: MarketParams | None = None,
        total_daily_volume: int = 50_000_000,
        seed: int | None = None,
    ) -> None:
        self.params = params or MarketParams()
        price_seed, volume_seed, spread_seed = np.random.SeedSequence(seed).spawn(3)
        self.price_generator = IntraDayPriceGenerator(self.params, price_seed)
        self.volume_generator = VolumeProfileGenerator(total_daily_volume, volume_seed)
        self.rng = np.random.default_rng(spread_seed)

    def _generate_spread(
        self, prices: NDArray[np.float64], volumes: NDArray[np.int_]
    ) -> NDArray[np.float64]:
        """Spread in dollars, widening as volume falls, with +/-20% uniform noise."""
        vol_factor = volumes.max() / (volumes + 1)
        vol_factor = vol_factor / vol_factor.max()
        p = self.params
        spread_bps = p.min_spread_bps + (p.max_spread_bps - p.min_spread_bps) * vol_factor
        spread_bps = spread_bps * self.rng.uniform(0.8, 1.2, len(prices))
        spreads = prices * spread_bps / 10_000
        return np.asarray(
            np.maximum(np.round(spreads / p.tick_size) * p.tick_size, p.tick_size),
            dtype=np.float64,
        )

    def generate(
        self,
        date: datetime | None = None,
        num_minutes: int = TRADING_MINUTES_PER_DAY,
        initial_price: float | None = None,
    ) -> pd.DataFrame:
        """Columns: timestamp, symbol, price (mid), bid, ask, spread, volume.

        Successive calls continue the random streams, so days generated in sequence differ.
        """
        if date is None:
            date = datetime.now()

        price_df = self.price_generator.generate(date, num_minutes, initial_price)
        volumes = self.volume_generator.generate(num_minutes)
        prices = price_df["price"].to_numpy(dtype=np.float64)
        spreads = self._generate_spread(prices, volumes)

        tick = self.params.tick_size
        bids = np.round((prices - spreads / 2) / tick) * tick
        asks = np.round((prices + spreads / 2) / tick) * tick
        asks = np.maximum(asks, bids + tick)

        return pd.DataFrame(
            {
                "timestamp": price_df["timestamp"],
                "symbol": self.params.symbol,
                "price": prices,
                "bid": bids,
                "ask": asks,
                "spread": asks - bids,
                "volume": volumes,
            }
        )


def calculate_vwap(market_data: pd.DataFrame) -> float:
    """Volume-weighted average of the `price` column."""
    return float((market_data["price"] * market_data["volume"]).sum() / market_data["volume"].sum())
