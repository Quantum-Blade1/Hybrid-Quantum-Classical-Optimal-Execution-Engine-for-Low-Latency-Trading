"""
Base Execution Strategy

Abstract base class for execution strategies providing a common interface
and shared functionality for VWAP, TWAP, and other execution algorithms.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional
from .order_book import OrderBook


@dataclass
class ExecutionMetrics:
    """
    Comprehensive metrics for evaluating execution quality.
    
    Attributes:
        strategy_name: Name of the execution strategy used
        average_execution_price: Volume-weighted average price of all fills
        benchmark_price: Benchmark price (VWAP, TWAP, arrival, etc.)
        benchmark_name: Name of the benchmark used
        slippage_bps: Slippage vs benchmark in basis points (positive = worse for buyer)
        total_cost: Total execution cost including spread and impact
        spread_cost: Cost from crossing the spread
        impact_cost: Cost from market impact
        timing_risk: Standard deviation of execution prices (higher = more timing risk)
        fill_rate: Percentage of order filled (0 to 1)
        num_slices: Number of execution slices
        total_shares: Total shares in order
        filled_shares: Total shares actually filled
        execution_time_minutes: Time from first to last fill
    """
    strategy_name: str
    average_execution_price: float
    benchmark_price: float
    benchmark_name: str
    slippage_bps: float
    total_cost: float
    spread_cost: float
    impact_cost: float
    timing_risk: float
    fill_rate: float
    num_slices: int
    total_shares: int
    filled_shares: int
    execution_time_minutes: int

    def __repr__(self) -> str:
        return (
            f"ExecutionMetrics({self.strategy_name})\n"
            f"  avg_price={self.average_execution_price:.4f}\n"
            f"  {self.benchmark_name}={self.benchmark_price:.4f}\n"
            f"  slippage={self.slippage_bps:+.2f} bps\n"
            f"  total_cost=${self.total_cost:.2f}\n"
            f"  timing_risk=${self.timing_risk:.4f}\n"
            f"  fill_rate={self.fill_rate*100:.1f}%\n"
            f"  filled={self.filled_shares:,}/{self.total_shares:,} shares"
        )


@dataclass
class ExecutionSlice:
    """Record of a single execution slice."""
    minute: int
    timestamp: pd.Timestamp
    target_quantity: int
    filled_quantity: int
    execution_price: float
    market_mid_price: float
    spread: float
    market_impact: float


class BaseStrategy(ABC):
    """
    Abstract base class for execution strategies.
    
    All execution strategies should inherit from this class and implement
    the required abstract methods.
    """
    
    strategy_name: str = "Base"
    benchmark_name: str = "VWAP"
    
    def __init__(
        self,
        order_book: Optional[OrderBook] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the base strategy.
        
        Args:
            order_book: OrderBook instance for execution simulation
            seed: Random seed for reproducibility
        """
        self.order_book = order_book or OrderBook(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.slices: List[ExecutionSlice] = []
    
    @abstractmethod
    def calculate_schedule(
        self,
        total_shares: int,
        market_data: pd.DataFrame
    ) -> np.ndarray:
        """
        Calculate the execution schedule.
        
        Args:
            total_shares: Total order size
            market_data: Market data for the execution period
            
        Returns:
            Array of target shares per minute
        """
        pass
    
    def calculate_benchmark(self, market_data: pd.DataFrame) -> float:
        """
        Calculate the benchmark price for comparison.
        
        Default implementation returns VWAP. Override for different benchmarks.
        
        Args:
            market_data: Market data for the execution period
            
        Returns:
            Benchmark price
        """
        return (market_data["price"] * market_data["volume"]).sum() / market_data["volume"].sum()
    
    def execute(
        self,
        total_shares: int,
        side: str,
        market_data: pd.DataFrame,
        start_minute: int = 0,
        end_minute: Optional[int] = None
    ) -> ExecutionMetrics:
        """
        Execute an order against simulated market data.
        
        Args:
            total_shares: Total number of shares to execute
            side: 'buy' or 'sell'
            market_data: DataFrame with market data (from MarketDataSimulator)
            start_minute: Minute index to start execution
            end_minute: Minute index to end execution (None = end of data)
            
        Returns:
            ExecutionMetrics with comprehensive quality metrics
        """
        self.slices = []  # Reset execution record
        
        if end_minute is None:
            end_minute = len(market_data)
        
        # Extract relevant portion of market data
        execution_data = market_data.iloc[start_minute:end_minute].copy()
        execution_data = execution_data.reset_index(drop=True)
        
        # Calculate execution schedule
        schedule = self.calculate_schedule(total_shares, execution_data)
        
        # Execute each slice
        total_filled = 0
        total_value = 0.0
        total_spread_cost = 0.0
        total_impact_cost = 0.0
        execution_prices = []
        
        first_fill_minute = None
        last_fill_minute = None
        
        for minute_idx, row in execution_data.iterrows():
            target_qty = schedule[minute_idx]
            
            if target_qty == 0:
                continue
            
            # Generate order book snapshot
            snapshot = self.order_book.generate_snapshot(
                mid_price=row["price"],
                spread=row["spread"],
                minute_volume=row["volume"]
            )
            
            # Simulate execution
            avg_price, filled, impact = self.order_book.simulate_execution(
                snapshot=snapshot,
                order_size=target_qty,
                side=side
            )
            
            if filled > 0:
                if first_fill_minute is None:
                    first_fill_minute = minute_idx
                last_fill_minute = minute_idx
                
                total_filled += filled
                total_value += avg_price * filled
                execution_prices.extend([avg_price] * filled)
                
                # Calculate costs
                if side.lower() == "buy":
                    spread_cost = (snapshot.best_ask - row["price"]) * filled
                else:
                    spread_cost = (row["price"] - snapshot.best_bid) * filled
                
                total_spread_cost += spread_cost
                total_impact_cost += abs(impact) * filled
                
                # Record the slice
                self.slices.append(ExecutionSlice(
                    minute=minute_idx,
                    timestamp=row["timestamp"],
                    target_quantity=target_qty,
                    filled_quantity=filled,
                    execution_price=avg_price,
                    market_mid_price=row["price"],
                    spread=row["spread"],
                    market_impact=impact
                ))
        
        # Calculate metrics
        if total_filled > 0:
            avg_execution_price = total_value / total_filled
            timing_risk = np.std(execution_prices) if len(execution_prices) > 1 else 0.0
        else:
            avg_execution_price = 0.0
            timing_risk = 0.0
        
        benchmark_price = self.calculate_benchmark(execution_data)
        
        # Calculate slippage (positive = worse for buyer, negative = worse for seller)
        if benchmark_price > 0:
            if side.lower() == "buy":
                slippage_bps = (avg_execution_price - benchmark_price) / benchmark_price * 10000
            else:
                slippage_bps = (benchmark_price - avg_execution_price) / benchmark_price * 10000
        else:
            slippage_bps = 0.0
        
        # Execution time
        if first_fill_minute is not None and last_fill_minute is not None:
            execution_time = last_fill_minute - first_fill_minute + 1
        else:
            execution_time = 0
        
        return ExecutionMetrics(
            strategy_name=self.strategy_name,
            average_execution_price=avg_execution_price,
            benchmark_price=benchmark_price,
            benchmark_name=self.benchmark_name,
            slippage_bps=slippage_bps,
            total_cost=total_spread_cost + total_impact_cost,
            spread_cost=total_spread_cost,
            impact_cost=total_impact_cost,
            timing_risk=timing_risk,
            fill_rate=total_filled / total_shares if total_shares > 0 else 0.0,
            num_slices=len(self.slices),
            total_shares=total_shares,
            filled_shares=total_filled,
            execution_time_minutes=execution_time
        )
    
    def get_execution_summary(self) -> pd.DataFrame:
        """
        Get a DataFrame summary of all execution slices.
        
        Returns:
            DataFrame with slice-by-slice execution details
        """
        if not self.slices:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {
                "minute": s.minute,
                "timestamp": s.timestamp,
                "target_qty": s.target_quantity,
                "filled_qty": s.filled_quantity,
                "fill_rate": s.filled_quantity / s.target_quantity if s.target_quantity > 0 else 0,
                "exec_price": s.execution_price,
                "mid_price": s.market_mid_price,
                "slippage": s.execution_price - s.market_mid_price,
                "spread": s.spread,
                "impact": s.market_impact
            }
            for s in self.slices
        ])
    
    def get_schedule_df(self, market_data: pd.DataFrame, total_shares: int) -> pd.DataFrame:
        """
        Get the execution schedule as a DataFrame for visualization.
        
        Args:
            market_data: Market data
            total_shares: Order size
            
        Returns:
            DataFrame with schedule and market data
        """
        schedule = self.calculate_schedule(total_shares, market_data)
        
        return pd.DataFrame({
            "minute": range(len(market_data)),
            "timestamp": market_data["timestamp"].values,
            "price": market_data["price"].values,
            "volume": market_data["volume"].values,
            "scheduled_shares": schedule
        })
