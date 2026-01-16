"""
VWAP Execution Strategy

Implements Volume Weighted Average Price execution strategy that splits
large orders proportionally to historical volume profile to minimize
market impact and track execution quality.
"""

import numpy as np
import pandas as pd
from typing import Optional
from .base_strategy import BaseStrategy, ExecutionMetrics, ExecutionSlice
from .order_book import OrderBook


class VWAPStrategy(BaseStrategy):
    """
    VWAP (Volume Weighted Average Price) execution strategy.
    
    This strategy splits a large parent order into smaller child orders
    that are executed proportionally to the historical/expected volume
    profile throughout the trading day.
    
    Key characteristics:
    - Volume-weighted slice sizes
    - Larger executions during high-volume periods
    - Minimizes market impact by following natural volume
    
    Trade-offs vs TWAP:
    - Better market impact profile
    - Requires accurate volume prediction
    - More complex to implement
    """
    
    strategy_name = "VWAP"
    benchmark_name = "VWAP"
    
    def __init__(
        self,
        participation_rate: float = 0.1,
        max_slice_pct: float = 0.05,
        historical_profile: Optional[np.ndarray] = None,
        order_book: Optional[OrderBook] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the VWAP strategy.
        
        Args:
            participation_rate: Target percentage of each minute's volume to capture
            max_slice_pct: Maximum percentage of order to execute in single slice
            historical_profile: Optional pre-computed volume probability distribution
            order_book: OrderBook instance for execution simulation
            seed: Random seed for reproducibility
        """
        super().__init__(order_book=order_book, seed=seed)
        self.participation_rate = participation_rate
        self.max_slice_pct = max_slice_pct
        self.historical_profile = historical_profile
    
    def calculate_schedule(
        self,
        total_shares: int,
        market_data: pd.DataFrame
    ) -> np.ndarray:
        """
        Calculate the execution schedule based on volume profile.
        
        Args:
            total_shares: Total order size
            market_data: Market data for the execution period
            
        Returns:
            Array of target shares per minute
        """
        if self.historical_profile is not None:
            # Use provided historical profile (Adaptive VWAP)
            # Ensure length matches or interpolate? For simplicity, assume length matches duration
            target_len = len(market_data)
            prof = self.historical_profile
            if len(prof) != target_len:
                # Simple resize/resample
                indices = np.linspace(0, len(prof)-1, target_len)
                prof = np.interp(indices, np.arange(len(prof)), prof)
            volume_profile = prof
        else:
            # Use current day's actual volume (Perfect Foresight / Baseline)
            volume_profile = market_data["volume"].values
        
        # Normalize volume profile
        vol_weights = volume_profile / volume_profile.sum()
        
        # Calculate raw schedule
        raw_schedule = vol_weights * total_shares
        
        # Apply participation rate cap
        participation_cap = volume_profile * self.participation_rate
        raw_schedule = np.minimum(raw_schedule, participation_cap)
        
        # Apply max slice cap
        max_slice_shares = total_shares * self.max_slice_pct
        raw_schedule = np.minimum(raw_schedule, max_slice_shares)
        
        # Round to integers
        schedule = np.round(raw_schedule).astype(int)
        
        # Ensure we're not over-allocated
        while schedule.sum() > total_shares:
            # Reduce the largest slice
            max_idx = np.argmax(schedule)
            schedule[max_idx] -= 1
        
        # Distribute any remaining shares
        remaining = total_shares - schedule.sum()
        if remaining > 0:
            # Add to slices proportionally to volume
            for _ in range(remaining):
                # Weight by volume but avoid already-capped slices
                weights = vol_weights * (schedule < participation_cap)
                if weights.sum() > 0:
                    weights = weights / weights.sum()
                    idx = self.rng.choice(len(schedule), p=weights)
                    schedule[idx] += 1
        
        return schedule
    
    def calculate_benchmark(self, market_data: pd.DataFrame) -> float:
        """
        Calculate VWAP benchmark (volume-weighted average price).
        
        Args:
            market_data: Market data for the execution period
            
        Returns:
            VWAP price
        """
        return (market_data["price"] * market_data["volume"]).sum() / market_data["volume"].sum()


# Re-export for backward compatibility
__all__ = ["VWAPStrategy", "ExecutionMetrics", "ExecutionSlice"]
