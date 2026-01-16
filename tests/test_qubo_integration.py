"""
Tests for QUBO Integration

Verifies the integrated QUBO->Execution workflow.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

import sys
sys.path.insert(0, str(__file__).replace("\\tests\\test_qubo_integration.py", ""))

from src.qubo_integration import (
    QUBOStrategy,
    StrategyComparison,
    run_integrated_comparison
)
from src.execution_engine import ParentOrder, OrderSide, ExecutionEngine
from src.market_data import MarketDataSimulator


class TestQUBOStrategy:
    """Tests for QUBOStrategy."""
    
    @pytest.fixture
    def market_data(self):
        """Generate sample market data."""
        simulator = MarketDataSimulator(
            total_daily_volume=10_000_000,
            seed=42
        )
        return simulator.generate(datetime.now(), num_minutes=60)
    
    def test_calculate_schedule(self, market_data):
        """Test schedule calculation."""
        strategy = QUBOStrategy(
            num_time_slices=10,
            sa_sweeps=100,
            seed=42
        )
        
        schedule = strategy.calculate_schedule(1000, market_data)
        
        assert len(schedule) == len(market_data)
        assert schedule.sum() > 0
    
    def test_optimization_stats(self, market_data):
        """Test that optimization stats are recorded."""
        strategy = QUBOStrategy(
            num_time_slices=5,
            sa_sweeps=50,
            seed=42
        )
        
        strategy.calculate_schedule(500, market_data)
        stats = strategy.get_optimization_stats()
        
        assert "qubo_energy" in stats
        assert "optimization_time_s" in stats
        assert stats["optimization_time_s"] > 0
    
    def test_works_with_execution_engine(self, market_data):
        """Test integration with execution engine."""
        strategy = QUBOStrategy(
            num_time_slices=5,
            sa_sweeps=50,
            seed=42
        )
        
        engine = ExecutionEngine(seed=42)
        order = ParentOrder(
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=500,
            time_horizon_minutes=60
        )
        
        report = engine.process_order(order, market_data, strategy)
        
        assert report.filled_quantity > 0
        assert report.average_execution_price > 0


class TestRunIntegratedComparison:
    """Tests for integrated comparison workflow."""
    
    @pytest.fixture
    def market_data(self):
        """Generate sample market data."""
        simulator = MarketDataSimulator(
            total_daily_volume=10_000_000,
            seed=42
        )
        return simulator.generate(datetime.now(), num_minutes=30)
    
    def test_returns_comparison(self, market_data):
        """Test that comparison returns valid results."""
        order = ParentOrder(
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=300,
            time_horizon_minutes=30
        )
        
        comparison = run_integrated_comparison(
            parent_order=order,
            market_data=market_data,
            qubo_time_slices=5,
            qubo_sa_sweeps=50,
            seed=42,
            verbose=False
        )
        
        assert isinstance(comparison, StrategyComparison)
        assert comparison.vwap_report is not None
        assert comparison.twap_report is not None
        assert comparison.qubo_report is not None
    
    def test_all_strategies_fill(self, market_data):
        """Test that all strategies achieve some fill."""
        order = ParentOrder(
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=200,
            time_horizon_minutes=30
        )
        
        comparison = run_integrated_comparison(
            parent_order=order,
            market_data=market_data,
            qubo_time_slices=5,
            qubo_sa_sweeps=50,
            seed=42,
            verbose=False
        )
        
        assert comparison.vwap_report.fill_rate > 0
        assert comparison.twap_report.fill_rate > 0
        assert comparison.qubo_report.fill_rate > 0
    
    def test_best_strategy_identified(self, market_data):
        """Test that best strategy is identified."""
        order = ParentOrder(
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=200,
            time_horizon_minutes=30
        )
        
        comparison = run_integrated_comparison(
            parent_order=order,
            market_data=market_data,
            qubo_time_slices=5,
            qubo_sa_sweeps=50,
            seed=42,
            verbose=False
        )
        
        assert comparison.best_strategy in ["VWAP", "TWAP", "QUBO"]


class TestStrategyComparison:
    """Tests for StrategyComparison dataclass."""
    
    @pytest.fixture
    def comparison(self):
        """Create a sample comparison."""
        simulator = MarketDataSimulator(seed=42)
        market_data = simulator.generate(datetime.now(), num_minutes=30)
        
        order = ParentOrder(
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=200,
            time_horizon_minutes=30
        )
        
        return run_integrated_comparison(
            parent_order=order,
            market_data=market_data,
            qubo_time_slices=5,
            qubo_sa_sweeps=50,
            seed=42,
            verbose=False
        )
    
    def test_has_all_reports(self, comparison):
        """Test that all reports are present."""
        assert comparison.vwap_report is not None
        assert comparison.twap_report is not None
        assert comparison.qubo_report is not None
    
    def test_cost_savings_computed(self, comparison):
        """Test that cost savings is computed."""
        assert isinstance(comparison.cost_savings, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
