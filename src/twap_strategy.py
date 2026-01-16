"""
TWAP Execution Strategy

Implements Time Weighted Average Price execution strategy that splits
orders into equal-sized slices executed at regular intervals.
"""

import numpy as np
import pandas as pd
from typing import Optional
from .base_strategy import BaseStrategy
from .order_book import OrderBook


class TWAPStrategy(BaseStrategy):
    """
    TWAP (Time Weighted Average Price) execution strategy.
    
    This strategy splits a large parent order into equal-sized child orders
    that are executed at regular time intervals throughout the execution period.
    
    Key characteristics:
    - Equal slice sizes (time-weighted, not volume-weighted)
    - Regular execution intervals
    - Simple and predictable execution pattern
    - Lower implementation complexity than VWAP
    
    Trade-offs vs VWAP:
    - May have higher market impact during low-volume periods
    - More predictable and harder to game by market makers
    - Better for less liquid securities where volume prediction is unreliable
    """
    
    strategy_name = "TWAP"
    benchmark_name = "TWAP"
    
    def __init__(
        self,
        interval_minutes: int = 1,
        max_slice_pct: float = 0.05,
        order_book: Optional[OrderBook] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the TWAP strategy.
        
        Args:
            interval_minutes: Minutes between each execution slice
            max_slice_pct: Maximum percentage of order to execute in single slice
            order_book: OrderBook instance for execution simulation
            seed: Random seed for reproducibility
        """
        super().__init__(order_book=order_book, seed=seed)
        self.interval_minutes = interval_minutes
        self.max_slice_pct = max_slice_pct
    
    def calculate_schedule(
        self,
        total_shares: int,
        market_data: pd.DataFrame
    ) -> np.ndarray:
        """
        Calculate equal-sized execution schedule at regular intervals.
        
        Args:
            total_shares: Total order size
            market_data: Market data for the execution period
            
        Returns:
            Array of target shares per minute
        """
        num_minutes = len(market_data)
        schedule = np.zeros(num_minutes, dtype=int)
        
        # Calculate execution points (every interval_minutes)
        execution_points = list(range(0, num_minutes, self.interval_minutes))
        num_slices = len(execution_points)
        
        if num_slices == 0:
            return schedule
        
        # Calculate base slice size (equal distribution)
        base_slice_size = total_shares // num_slices
        remainder = total_shares % num_slices
        
        # Apply max slice cap
        max_slice_shares = int(total_shares * self.max_slice_pct)
        base_slice_size = min(base_slice_size, max_slice_shares)
        
        # Distribute shares across execution points
        shares_allocated = 0
        for i, point in enumerate(execution_points):
            if shares_allocated >= total_shares:
                break
            
            # Add one extra share to first 'remainder' slices for even distribution
            slice_size = base_slice_size + (1 if i < remainder else 0)
            slice_size = min(slice_size, total_shares - shares_allocated)
            
            schedule[point] = slice_size
            shares_allocated += slice_size
        
        # If we still have remaining shares (due to max_slice_pct cap), distribute them
        if shares_allocated < total_shares:
            remaining = total_shares - shares_allocated
            # Distribute to points that have capacity
            for point in execution_points:
                if remaining <= 0:
                    break
                additional = min(remaining, max_slice_shares - schedule[point])
                if additional > 0:
                    schedule[point] += additional
                    remaining -= additional
        
        return schedule
    
    def calculate_benchmark(self, market_data: pd.DataFrame) -> float:
        """
        Calculate TWAP benchmark (simple time-weighted average price).
        
        Args:
            market_data: Market data for the execution period
            
        Returns:
            TWAP price
        """
        # Simple average of prices at execution intervals
        execution_points = range(0, len(market_data), self.interval_minutes)
        prices_at_points = market_data.iloc[list(execution_points)]["price"]
        return prices_at_points.mean()


def compare_strategies(
    vwap_strategy,
    twap_strategy,
    total_shares: int,
    side: str,
    market_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Compare VWAP and TWAP strategy performance.
    
    Args:
        vwap_strategy: VWAPStrategy instance
        twap_strategy: TWAPStrategy instance
        total_shares: Order size
        side: 'buy' or 'sell'
        market_data: Market data
        
    Returns:
        DataFrame comparing key metrics
    """
    vwap_metrics = vwap_strategy.execute(total_shares, side, market_data)
    twap_metrics = twap_strategy.execute(total_shares, side, market_data)
    
    # Calculate VWAP for both to compare against same benchmark
    vwap_price = (market_data["price"] * market_data["volume"]).sum() / market_data["volume"].sum()
    
    return pd.DataFrame({
        "Metric": [
            "Average Execution Price",
            "Benchmark Price",
            "Slippage (bps)",
            "Spread Cost ($)",
            "Impact Cost ($)",
            "Total Cost ($)",
            "Timing Risk ($)",
            "Fill Rate (%)",
            "Num Slices",
            "Execution Time (min)"
        ],
        "VWAP": [
            f"${vwap_metrics.average_execution_price:.4f}",
            f"${vwap_price:.4f} (VWAP)",
            f"{vwap_metrics.slippage_bps:+.2f}",
            f"${vwap_metrics.spread_cost:.2f}",
            f"${vwap_metrics.impact_cost:.2f}",
            f"${vwap_metrics.total_cost:.2f}",
            f"${vwap_metrics.timing_risk:.4f}",
            f"{vwap_metrics.fill_rate * 100:.1f}%",
            str(vwap_metrics.num_slices),
            str(vwap_metrics.execution_time_minutes)
        ],
        "TWAP": [
            f"${twap_metrics.average_execution_price:.4f}",
            f"${twap_metrics.benchmark_price:.4f} (TWAP)",
            f"{twap_metrics.slippage_bps:+.2f}",
            f"${twap_metrics.spread_cost:.2f}",
            f"${twap_metrics.impact_cost:.2f}",
            f"${twap_metrics.total_cost:.2f}",
            f"${twap_metrics.timing_risk:.4f}",
            f"{twap_metrics.fill_rate * 100:.1f}%",
            str(twap_metrics.num_slices),
            str(twap_metrics.execution_time_minutes)
        ]
    })
