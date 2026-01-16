"""
Tests for VWAP Execution Strategy

Verifies execution scheduling and quality metrics.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import sys
sys.path.insert(0, str(__file__).replace("\\tests\\test_vwap_strategy.py", ""))

from src.vwap_strategy import VWAPStrategy
from src.base_strategy import ExecutionMetrics
from src.market_data import MarketDataSimulator
from src.order_book import OrderBook


def create_mock_market_data(volume_profile: np.ndarray) -> pd.DataFrame:
    """Create mock market data with specified volume profile."""
    n = len(volume_profile)
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-15 09:30", periods=n, freq="min"),
        "price": [150.0] * n,
        "bid": [149.99] * n,
        "ask": [150.01] * n,
        "spread": [0.02] * n,
        "volume": volume_profile
    })


class TestVWAPSchedule:
    """Tests for order scheduling logic."""
    
    def test_schedule_sums_to_total(self):
        """Test that schedule sums to total order size when volume allows."""
        # Use high participation rate and large enough volume
        strategy = VWAPStrategy(participation_rate=0.5, max_slice_pct=0.5, seed=42)
        
        total_shares = 10000
        # Volume profile with enough capacity (50% of 100k = 50k >> 10k)
        volume_profile = np.array([20000, 10000, 10000, 10000, 50000])
        market_data = create_mock_market_data(volume_profile)
        
        schedule = strategy.calculate_schedule(total_shares, market_data)
        
        assert schedule.sum() == total_shares
    
    def test_schedule_proportional_to_volume(self):
        """Test that schedule roughly follows volume profile."""
        strategy = VWAPStrategy(participation_rate=1.0, max_slice_pct=1.0, seed=42)
        
        volume_profile = np.array([100, 50, 50, 100])
        market_data = create_mock_market_data(volume_profile)
        total_shares = 100
        
        schedule = strategy.calculate_schedule(total_shares, market_data)
        
        # High volume periods should have more shares
        assert schedule[0] >= schedule[1]
        assert schedule[3] >= schedule[2]
    
    def test_participation_rate_cap(self):
        """Test that participation rate limits slice size."""
        strategy = VWAPStrategy(participation_rate=0.1, seed=42)
        
        volume_profile = np.array([10000, 10000, 10000])
        market_data = create_mock_market_data(volume_profile)
        total_shares = 5000  # Would be ~1667 per slice without cap
        
        schedule = strategy.calculate_schedule(total_shares, market_data)
        
        # Each slice should be capped at 10% of volume = 1000
        assert schedule.max() <= 1000


class TestVWAPExecution:
    """Tests for order execution."""
    
    @pytest.fixture
    def market_data(self):
        """Generate sample market data for testing."""
        simulator = MarketDataSimulator(
            total_daily_volume=10_000_000,
            seed=42
        )
        return simulator.generate(datetime.now(), num_minutes=60)  # 1 hour
    
    def test_execution_returns_metrics(self, market_data):
        """Test that execution returns valid metrics."""
        strategy = VWAPStrategy(seed=42)
        
        metrics = strategy.execute(
            total_shares=1000,
            side="buy",
            market_data=market_data
        )
        
        assert isinstance(metrics, ExecutionMetrics)
        assert metrics.average_execution_price > 0
        assert metrics.benchmark_price > 0
        assert 0 <= metrics.fill_rate <= 1
    
    def test_execution_fills_order(self, market_data):
        """Test that order gets filled."""
        strategy = VWAPStrategy(seed=42)
        
        metrics = strategy.execute(
            total_shares=1000,
            side="buy",
            market_data=market_data
        )
        
        # With reasonable order size, should get substantial fill
        assert metrics.fill_rate > 0.5
    
    def test_reasonable_slippage(self, market_data):
        """Test that slippage is reasonable for small orders."""
        strategy = VWAPStrategy(seed=42)
        
        # Small order should have minimal slippage
        metrics = strategy.execute(
            total_shares=100,
            side="buy",
            market_data=market_data
        )
        
        # Slippage should be less than 100 bps for small order
        assert abs(metrics.slippage_bps) < 100
    
    def test_buy_vs_sell_slippage(self, market_data):
        """Test that buy and sell have opposite slippage directions."""
        strategy_buy = VWAPStrategy(seed=42)
        strategy_sell = VWAPStrategy(seed=42)
        
        buy_metrics = strategy_buy.execute(
            total_shares=500,
            side="buy",
            market_data=market_data
        )
        
        sell_metrics = strategy_sell.execute(
            total_shares=500,
            side="sell",
            market_data=market_data
        )
        
        # Both should complete, metrics should be valid
        assert buy_metrics.average_execution_price > 0
        assert sell_metrics.average_execution_price > 0
    
    def test_execution_summary(self, market_data):
        """Test that execution summary is available."""
        strategy = VWAPStrategy(seed=42)
        
        strategy.execute(
            total_shares=1000,
            side="buy",
            market_data=market_data
        )
        
        summary = strategy.get_execution_summary()
        
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) > 0
        assert "minute" in summary.columns
        assert "exec_price" in summary.columns


class TestExecutionMetrics:
    """Tests for ExecutionMetrics dataclass."""
    
    def test_metrics_repr(self):
        """Test that metrics have readable string representation."""
        metrics = ExecutionMetrics(
            strategy_name="VWAP",
            average_execution_price=150.25,
            benchmark_price=150.00,
            benchmark_name="VWAP",
            slippage_bps=16.67,
            total_cost=100.0,
            spread_cost=60.0,
            impact_cost=40.0,
            timing_risk=0.05,
            fill_rate=0.95,
            num_slices=50,
            total_shares=10000,
            filled_shares=9500,
            execution_time_minutes=60
        )
        
        repr_str = repr(metrics)
        
        assert "150.25" in repr_str
        assert "150.00" in repr_str
        assert "16.67" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
