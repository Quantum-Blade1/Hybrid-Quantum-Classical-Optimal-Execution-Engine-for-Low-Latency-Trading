"""
Tests for Execution Engine Framework

Verifies order processing, execution, and reporting.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

import sys
sys.path.insert(0, str(__file__).replace("\\tests\\test_execution_engine.py", ""))

from src.execution_engine import (
    ExecutionEngine,
    ParentOrder,
    ChildOrder,
    OrderSide,
    OrderStatus,
    ExecutionState,
    ExecutionReport
)
from src.vwap_strategy import VWAPStrategy
from src.twap_strategy import TWAPStrategy
from src.market_data import MarketDataSimulator


class TestParentOrder:
    """Tests for ParentOrder class."""
    
    def test_parent_order_creation(self):
        """Test creating a parent order."""
        order = ParentOrder(
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=10000,
            time_horizon_minutes=60
        )
        
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.total_quantity == 10000
        assert order.status == OrderStatus.PENDING
        assert order.filled_quantity == 0
    
    def test_remaining_quantity(self):
        """Test remaining quantity calculation."""
        order = ParentOrder(
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=10000,
            time_horizon_minutes=60
        )
        
        order.filled_quantity = 3000
        
        assert order.remaining_quantity == 7000
        assert order.fill_rate == 0.3
    
    def test_is_complete(self):
        """Test order completion check."""
        order = ParentOrder(
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=1000,
            time_horizon_minutes=60
        )
        
        assert not order.is_complete
        
        order.filled_quantity = 1000
        assert order.is_complete
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        order = ParentOrder(
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=1000,
            time_horizon_minutes=60
        )
        
        d = order.to_dict()
        
        assert "order_id" in d
        assert d["symbol"] == "AAPL"
        assert d["side"] == "buy"


class TestChildOrder:
    """Tests for ChildOrder class."""
    
    def test_child_order_creation(self):
        """Test creating a child order."""
        child = ChildOrder(
            parent_id="test123",
            target_quantity=100,
            target_time=datetime.now(),
            sequence=1
        )
        
        assert child.parent_id == "test123"
        assert child.target_quantity == 100
        assert child.status == OrderStatus.PENDING
    
    def test_slippage_calculation(self):
        """Test slippage calculation."""
        child = ChildOrder(
            parent_id="test123",
            target_quantity=100,
            target_time=datetime.now()
        )
        
        child.execution_price = 150.10
        child.market_price_at_execution = 150.00
        
        assert child.slippage == pytest.approx(0.10, rel=0.01)
        assert child.slippage_bps == pytest.approx(6.67, rel=0.1)


class TestExecutionEngine:
    """Tests for ExecutionEngine."""
    
    @pytest.fixture
    def market_data(self):
        """Generate sample market data for testing."""
        simulator = MarketDataSimulator(
            total_daily_volume=10_000_000,
            seed=42
        )
        return simulator.generate(datetime.now(), num_minutes=60)
    
    @pytest.fixture
    def engine(self):
        """Create execution engine."""
        return ExecutionEngine(seed=42)
    
    def test_process_order_returns_report(self, market_data, engine):
        """Test that process_order returns an ExecutionReport."""
        order = ParentOrder(
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=1000,
            time_horizon_minutes=60
        )
        
        strategy = VWAPStrategy(seed=42)
        report = engine.process_order(order, market_data, strategy)
        
        assert isinstance(report, ExecutionReport)
        assert report.order_id == order.order_id
        assert report.symbol == "AAPL"
    
    def test_order_gets_filled(self, market_data, engine):
        """Test that order gets filled."""
        order = ParentOrder(
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=1000,
            time_horizon_minutes=60
        )
        
        strategy = VWAPStrategy(seed=42)
        report = engine.process_order(order, market_data, strategy)
        
        assert report.fill_rate > 0.5
        assert report.filled_quantity > 0
    
    def test_metrics_calculated(self, market_data, engine):
        """Test that metrics are calculated."""
        order = ParentOrder(
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=1000,
            time_horizon_minutes=60
        )
        
        strategy = VWAPStrategy(seed=42)
        report = engine.process_order(order, market_data, strategy)
        
        assert report.average_execution_price > 0
        assert report.benchmark_vwap > 0
        assert report.benchmark_twap > 0
        assert isinstance(report.slippage_vs_vwap_bps, float)
    
    def test_child_orders_generated(self, market_data, engine):
        """Test that child orders are generated."""
        order = ParentOrder(
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=1000,
            time_horizon_minutes=60
        )
        
        strategy = VWAPStrategy(seed=42)
        engine.process_order(order, market_data, strategy)
        
        children_df = engine.get_child_orders_df()
        
        assert len(children_df) > 0
        assert "child_id" in children_df.columns
    
    def test_sell_order(self, market_data, engine):
        """Test sell order execution."""
        order = ParentOrder(
            symbol="AAPL",
            side=OrderSide.SELL,
            total_quantity=500,
            time_horizon_minutes=60
        )
        
        strategy = TWAPStrategy(seed=42)
        report = engine.process_order(order, market_data, strategy)
        
        assert report.side == "sell"
        assert report.filled_quantity > 0
    
    def test_execution_history(self, market_data, engine):
        """Test execution history tracking."""
        # Execute first order
        order1 = ParentOrder(
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=500,
            time_horizon_minutes=60
        )
        engine.process_order(order1, market_data, VWAPStrategy(seed=42))
        
        # Execute second order
        order2 = ParentOrder(
            symbol="AAPL",
            side=OrderSide.SELL,
            total_quantity=300,
            time_horizon_minutes=60
        )
        engine.process_order(order2, market_data, TWAPStrategy(seed=42))
        
        history = engine.get_execution_history()
        
        assert len(history) == 2
    
    def test_get_execution_report(self, market_data, engine):
        """Test getting most recent report."""
        order = ParentOrder(
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=500,
            time_horizon_minutes=60
        )
        
        engine.process_order(order, market_data, VWAPStrategy(seed=42))
        
        report = engine.get_execution_report()
        
        assert report is not None
        assert report.order_id == order.order_id


class TestExecutionReport:
    """Tests for ExecutionReport."""
    
    def test_report_to_dataframe(self):
        """Test conversion to DataFrame."""
        report = ExecutionReport(
            order_id="test123",
            symbol="AAPL",
            side="buy",
            total_quantity=1000,
            filled_quantity=950,
            average_execution_price=150.25,
            benchmark_vwap=150.00,
            benchmark_twap=149.95,
            arrival_price=149.80,
            slippage_vs_vwap_bps=16.67,
            slippage_vs_twap_bps=20.00,
            slippage_vs_arrival_bps=30.00,
            total_cost=100.0,
            spread_cost=60.0,
            impact_cost=40.0,
            cost_per_share=0.105,
            timing_risk=0.05,
            fill_rate=0.95,
            num_child_orders=50,
            execution_time_minutes=60,
            strategy_used="VWAP",
            started_at=datetime.now(),
            completed_at=datetime.now()
        )
        
        df = report.to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "Metric" in df.columns
        assert "Value" in df.columns
    
    def test_report_repr(self):
        """Test string representation."""
        report = ExecutionReport(
            order_id="test123",
            symbol="AAPL",
            side="buy",
            total_quantity=1000,
            filled_quantity=950,
            average_execution_price=150.25,
            benchmark_vwap=150.00,
            benchmark_twap=149.95,
            arrival_price=149.80,
            slippage_vs_vwap_bps=16.67,
            slippage_vs_twap_bps=20.00,
            slippage_vs_arrival_bps=30.00,
            total_cost=100.0,
            spread_cost=60.0,
            impact_cost=40.0,
            cost_per_share=0.105,
            timing_risk=0.05,
            fill_rate=0.95,
            num_child_orders=50,
            execution_time_minutes=60,
            strategy_used="VWAP",
            started_at=datetime.now(),
            completed_at=datetime.now()
        )
        
        repr_str = repr(report)
        
        assert "test123" in repr_str
        assert "AAPL" in repr_str
        assert "VWAP" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
