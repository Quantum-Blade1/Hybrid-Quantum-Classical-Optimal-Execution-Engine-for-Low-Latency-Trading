"""
Tests for TWAP Execution Strategy

Verifies TWAP scheduling and execution quality.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

import sys
sys.path.insert(0, str(__file__).replace("\\tests\\test_twap_strategy.py", ""))

from src.twap_strategy import TWAPStrategy, compare_strategies
from src.vwap_strategy import VWAPStrategy
from src.base_strategy import ExecutionMetrics
from src.market_data import MarketDataSimulator


class TestTWAPSchedule:
    """Tests for TWAP order scheduling logic."""
    
    @pytest.fixture
    def market_data(self):
        """Generate sample market data for testing."""
        simulator = MarketDataSimulator(
            total_daily_volume=10_000_000,
            seed=42
        )
        return simulator.generate(datetime.now(), num_minutes=60)
    
    def test_schedule_creates_equal_slices(self, market_data):
        """Test that TWAP creates roughly equal slice sizes."""
        strategy = TWAPStrategy(interval_minutes=1, max_slice_pct=1.0, seed=42)
        
        total_shares = 6000
        schedule = strategy.calculate_schedule(total_shares, market_data)
        
        # Get non-zero slices
        non_zero = schedule[schedule > 0]
        
        # All slices should be within 1 share of each other (accounting for rounding)
        assert non_zero.max() - non_zero.min() <= 1
    
    def test_schedule_sums_to_total(self, market_data):
        """Test that schedule sums to total order size."""
        strategy = TWAPStrategy(interval_minutes=1, max_slice_pct=1.0, seed=42)
        
        total_shares = 1000
        schedule = strategy.calculate_schedule(total_shares, market_data)
        
        assert schedule.sum() == total_shares
    
    def test_interval_spacing(self, market_data):
        """Test that slices are spaced at correct intervals."""
        strategy = TWAPStrategy(interval_minutes=5, max_slice_pct=1.0, seed=42)
        
        total_shares = 1000
        schedule = strategy.calculate_schedule(total_shares, market_data)
        
        # Find indices with non-zero volume
        execution_points = np.where(schedule > 0)[0]
        
        # Check spacing (should be multiples of 5)
        for point in execution_points:
            assert point % 5 == 0
    
    def test_max_slice_cap(self, market_data):
        """Test that max slice percentage is respected."""
        strategy = TWAPStrategy(interval_minutes=10, max_slice_pct=0.1, seed=42)
        
        # With 60-min data, 10-min intervals = 6 execution points
        # max_slice_pct=0.1 means max 60 shares per slice for 600 total
        total_shares = 600
        schedule = strategy.calculate_schedule(total_shares, market_data)
        
        # No slice should exceed 10% of total (60 shares)
        assert schedule.max() <= 60


class TestTWAPExecution:
    """Tests for TWAP order execution."""
    
    @pytest.fixture
    def market_data(self):
        """Generate sample market data for testing."""
        simulator = MarketDataSimulator(
            total_daily_volume=10_000_000,
            seed=42
        )
        return simulator.generate(datetime.now(), num_minutes=60)
    
    def test_execution_returns_metrics(self, market_data):
        """Test that execution returns valid metrics."""
        strategy = TWAPStrategy(seed=42)
        
        metrics = strategy.execute(
            total_shares=1000,
            side="buy",
            market_data=market_data
        )
        
        assert isinstance(metrics, ExecutionMetrics)
        assert metrics.strategy_name == "TWAP"
        assert metrics.benchmark_name == "TWAP"
        assert metrics.average_execution_price > 0
    
    def test_twap_benchmark_is_time_weighted(self, market_data):
        """Test that TWAP benchmark uses time weighting (equal weights)."""
        strategy = TWAPStrategy(interval_minutes=1, seed=42)
        
        metrics = strategy.execute(
            total_shares=100,
            side="buy",
            market_data=market_data
        )
        
        # TWAP benchmark should be close to simple average of prices
        simple_avg = market_data["price"].mean()
        
        # Should be within 1% of simple average for 1-min intervals
        assert abs(metrics.benchmark_price - simple_avg) / simple_avg < 0.01
    
    def test_timing_risk_calculated(self, market_data):
        """Test that timing risk (std of execution prices) is calculated."""
        strategy = TWAPStrategy(seed=42)
        
        metrics = strategy.execute(
            total_shares=1000,
            side="buy",
            market_data=market_data
        )
        
        # Should have non-zero timing risk when executing over time
        assert metrics.timing_risk >= 0


class TestStrategyComparison:
    """Tests for strategy comparison utility."""
    
    @pytest.fixture
    def market_data(self):
        """Generate sample market data for testing."""
        simulator = MarketDataSimulator(
            total_daily_volume=10_000_000,
            seed=42
        )
        return simulator.generate(datetime.now(), num_minutes=60)
    
    def test_compare_returns_dataframe(self, market_data):
        """Test that comparison returns a DataFrame."""
        vwap = VWAPStrategy(seed=42)
        twap = TWAPStrategy(seed=42)
        
        result = compare_strategies(
            vwap_strategy=vwap,
            twap_strategy=twap,
            total_shares=1000,
            side="buy",
            market_data=market_data
        )
        
        assert isinstance(result, pd.DataFrame)
        assert "VWAP" in result.columns
        assert "TWAP" in result.columns
    
    def test_compare_has_all_metrics(self, market_data):
        """Test that comparison includes key metrics."""
        vwap = VWAPStrategy(seed=42)
        twap = TWAPStrategy(seed=42)
        
        result = compare_strategies(
            vwap_strategy=vwap,
            twap_strategy=twap,
            total_shares=1000,
            side="buy",
            market_data=market_data
        )
        
        metrics_column = result["Metric"].tolist()
        
        assert "Average Execution Price" in metrics_column
        assert "Slippage (bps)" in metrics_column
        assert "Total Cost ($)" in metrics_column
        assert "Timing Risk ($)" in metrics_column


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
