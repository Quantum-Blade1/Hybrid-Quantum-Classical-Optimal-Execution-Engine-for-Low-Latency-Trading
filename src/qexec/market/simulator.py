"""
Market Data Simulator

Generates realistic intraday market data for equity trading simulation.
Includes price generation using Geometric Brownian Motion, volume profiles,
and bid-ask spread simulation.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timedelta


@dataclass
class MarketParams:
    """Parameters for market data simulation."""
    symbol: str = "AAPL"
    initial_price: float = 150.0
    annual_volatility: float = 0.25  # 25% annual volatility
    annual_drift: float = 0.08  # 8% annual drift
    trading_start: str = "09:30"
    trading_end: str = "16:00"
    tick_size: float = 0.01
    min_spread_bps: float = 1.0  # Minimum spread in basis points
    max_spread_bps: float = 10.0  # Maximum spread in basis points


class IntraDayPriceGenerator:
    """
    Generates realistic intraday price movements using Geometric Brownian Motion (GBM).
    
    The model accounts for:
    - Higher volatility at market open and close
    - Mean-reverting intraday patterns
    - Realistic tick-by-tick noise
    """
    
    def __init__(self, params: MarketParams, seed: Optional[int] = None):
        """
        Initialize the price generator.
        
        Args:
            params: Market parameters for simulation
            seed: Random seed for reproducibility
        """
        self.params = params
        self.rng = np.random.default_rng(seed)
        
        # Convert annual parameters to minute-level
        trading_minutes_per_day = 390  # 6.5 hours
        trading_days_per_year = 252
        minutes_per_year = trading_minutes_per_day * trading_days_per_year
        
        self.minute_volatility = params.annual_volatility / np.sqrt(minutes_per_year)
        self.minute_drift = params.annual_drift / minutes_per_year
    
    def _intraday_volatility_multiplier(self, minutes_from_open: np.ndarray) -> np.ndarray:
        """
        Calculate volatility multiplier based on time of day.
        
        Creates a U-shaped pattern with higher volatility at open and close.
        
        Args:
            minutes_from_open: Array of minutes since market open
            
        Returns:
            Array of volatility multipliers
        """
        total_minutes = 390  # Full trading day
        
        # Normalize to [0, 1]
        t = minutes_from_open / total_minutes
        
        # U-shaped curve: higher at edges (open/close), lower in middle
        # Using a quadratic function: 4*(t - 0.5)^2 + 0.5
        multiplier = 4 * (t - 0.5) ** 2 + 0.5
        
        # Scale to reasonable range [1.0, 2.5]
        multiplier = 1.0 + 1.5 * (multiplier - 0.5) / 0.5
        
        return multiplier
    
    def generate(self, date: datetime, num_minutes: int = 390) -> pd.DataFrame:
        """
        Generate intraday price data for a single trading day.
        
        Args:
            date: Trading date
            num_minutes: Number of minutes to generate (default: full day)
            
        Returns:
            DataFrame with columns: timestamp, price
        """
        # Parse trading times
        start_time = datetime.strptime(self.params.trading_start, "%H:%M")
        
        # Generate timestamps
        timestamps = [
            datetime.combine(date.date(), start_time.time()) + timedelta(minutes=i)
            for i in range(num_minutes)
        ]
        
        # Generate minute indices for volatility calculation
        minutes_from_open = np.arange(num_minutes)
        
        # Get time-varying volatility
        vol_multipliers = self._intraday_volatility_multiplier(minutes_from_open)
        
        # Generate GBM returns
        # dS/S = mu*dt + sigma*dW where dW ~ N(0, dt)
        random_shocks = self.rng.standard_normal(num_minutes)
        
        # Apply time-varying volatility
        returns = (
            self.minute_drift 
            + self.minute_volatility * vol_multipliers * random_shocks
        )
        
        # Convert returns to prices
        log_prices = np.log(self.params.initial_price) + np.cumsum(returns)
        prices = np.exp(log_prices)
        
        # Round to tick size
        prices = np.round(prices / self.params.tick_size) * self.params.tick_size
        
        return pd.DataFrame({
            "timestamp": timestamps,
            "price": prices
        })


class VolumeProfileGenerator:
    """
    Generates realistic intraday volume profiles.
    
    Uses a U-shaped pattern typical of equity markets:
    - High volume at market open
    - Low volume during midday
    - High volume at market close
    """
    
    def __init__(
        self, 
        total_daily_volume: int = 50_000_000,
        seed: Optional[int] = None
    ):
        """
        Initialize the volume generator.
        
        Args:
            total_daily_volume: Total shares traded per day
            seed: Random seed for reproducibility
        """
        self.total_daily_volume = total_daily_volume
        self.rng = np.random.default_rng(seed)
    
    def _volume_profile_weights(self, num_minutes: int) -> np.ndarray:
        """
        Generate the theoretical volume distribution weights.
        
        Creates a U-shaped profile with peaks at open and close.
        
        Args:
            num_minutes: Number of minutes in trading day
            
        Returns:
            Normalized weight array summing to 1.0
        """
        t = np.linspace(0, 1, num_minutes)
        
        # U-shaped curve using quadratic + slight asymmetry
        # Opening auction typically has higher volume than close
        opening_peak = 1.5 * np.exp(-((t - 0.0) ** 2) / 0.01)
        closing_peak = 1.2 * np.exp(-((t - 1.0) ** 2) / 0.02)
        baseline = 0.3 + 0.4 * (4 * (t - 0.5) ** 2)
        
        weights = opening_peak + closing_peak + baseline
        
        # Normalize
        return weights / weights.sum()
    
    def generate(self, num_minutes: int = 390) -> np.ndarray:
        """
        Generate volume for each minute of the trading day.
        
        Args:
            num_minutes: Number of minutes to generate
            
        Returns:
            Array of volume per minute
        """
        # Get base weights
        weights = self._volume_profile_weights(num_minutes)
        
        # Add noise (log-normal to keep positive)
        noise = self.rng.lognormal(0, 0.3, num_minutes)
        noisy_weights = weights * noise
        
        # Renormalize
        noisy_weights = noisy_weights / noisy_weights.sum()
        
        # Scale to total volume
        volumes = (noisy_weights * self.total_daily_volume).astype(int)
        
        # Ensure minimum volume per minute
        volumes = np.maximum(volumes, 100)
        
        return volumes


class MarketDataSimulator:
    """
    Complete market data simulator combining price and volume generation.
    
    Generates a unified DataFrame with all market data columns needed
    for trading simulation.
    """
    
    def __init__(
        self,
        params: Optional[MarketParams] = None,
        total_daily_volume: int = 50_000_000,
        seed: Optional[int] = None
    ):
        """
        Initialize the market data simulator.
        
        Args:
            params: Market parameters (uses defaults if None)
            total_daily_volume: Total daily trading volume
            seed: Random seed for reproducibility
        """
        self.params = params or MarketParams()
        self.price_generator = IntraDayPriceGenerator(self.params, seed)
        self.volume_generator = VolumeProfileGenerator(total_daily_volume, seed)
        self.rng = np.random.default_rng(seed)
    
    def _generate_spread(self, prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """
        Generate realistic bid-ask spreads.
        
        Spread is inversely related to volume and directly related to volatility.
        
        Args:
            prices: Array of mid prices
            volumes: Array of volumes per minute
            
        Returns:
            Array of spreads in dollars
        """
        # Normalize volume (inverse relationship with spread)
        vol_factor = volumes.max() / (volumes + 1)
        vol_factor = vol_factor / vol_factor.max()  # Scale to [0, 1]
        
        # Calculate spread in basis points
        spread_bps = (
            self.params.min_spread_bps 
            + (self.params.max_spread_bps - self.params.min_spread_bps) * vol_factor
        )
        
        # Add noise
        noise = self.rng.uniform(0.8, 1.2, len(prices))
        spread_bps = spread_bps * noise
        
        # Convert to dollar spread
        spreads = prices * spread_bps / 10000
        
        # Round to tick size
        spreads = np.maximum(
            np.round(spreads / self.params.tick_size) * self.params.tick_size,
            self.params.tick_size
        )
        
        return spreads
    
    def generate(
        self, 
        date: Optional[datetime] = None,
        num_minutes: int = 390
    ) -> pd.DataFrame:
        """
        Generate complete market data for a trading day.
        
        Args:
            date: Trading date (defaults to today)
            num_minutes: Number of minutes to generate
            
        Returns:
            DataFrame with columns:
                - timestamp: Datetime for each minute
                - symbol: Stock symbol
                - price: Mid price
                - bid: Best bid price
                - ask: Best ask price
                - spread: Bid-ask spread
                - volume: Trading volume for the minute
        """
        if date is None:
            date = datetime.now()
        
        # Generate price data
        price_df = self.price_generator.generate(date, num_minutes)
        
        # Generate volume data
        volumes = self.volume_generator.generate(num_minutes)
        
        # Generate spreads
        prices = price_df["price"].values
        spreads = self._generate_spread(prices, volumes)
        
        # Calculate bid/ask from mid and spread
        bids = prices - spreads / 2
        asks = prices + spreads / 2
        
        # Round to tick size
        bids = np.round(bids / self.params.tick_size) * self.params.tick_size
        asks = np.round(asks / self.params.tick_size) * self.params.tick_size
        
        # Ensure spread is at least one tick
        asks = np.maximum(asks, bids + self.params.tick_size)
        
        return pd.DataFrame({
            "timestamp": price_df["timestamp"],
            "symbol": self.params.symbol,
            "price": prices,
            "bid": bids,
            "ask": asks,
            "spread": asks - bids,
            "volume": volumes
        })


def calculate_vwap(market_data: pd.DataFrame) -> float:
    """
    Calculate the Volume Weighted Average Price from market data.
    
    Args:
        market_data: DataFrame with 'price' and 'volume' columns
        
    Returns:
        VWAP value
    """
    return (market_data["price"] * market_data["volume"]).sum() / market_data["volume"].sum()
