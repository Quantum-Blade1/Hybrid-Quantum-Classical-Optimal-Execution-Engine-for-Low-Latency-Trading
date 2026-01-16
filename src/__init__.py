"""
Quantum-Classical Trading Execution System
Phase 1: Classical Baseline + Phase 2: Quantum Optimization

Core modules for market data simulation, execution strategies, 
execution engine, and quantum optimization formulation.
"""

from .market_data import MarketDataSimulator, IntraDayPriceGenerator, VolumeProfileGenerator
from .order_book import OrderBook
from .base_strategy import BaseStrategy, ExecutionMetrics, ExecutionSlice
from .vwap_strategy import VWAPStrategy
from .twap_strategy import TWAPStrategy, compare_strategies
from .execution_engine import (
    ExecutionEngine,
    ParentOrder,
    ChildOrder,
    OrderSide,
    OrderStatus,
    ExecutionState,
    ExecutionReport
)
from .qubo_execution import (
    QUBOConfig,
    ExecutionQUBO,
    create_random_binary_solution,
    create_uniform_solution,
    print_qubo_summary
)
from .qubo_solvers import (
    QUBOResult,
    BruteForceSolver,
    SimulatedAnnealingSolver,
    GreedySolver,
    compare_solvers
)
from .qubo_integration import (
    QUBOStrategy,
    StrategyComparison,
    run_integrated_comparison,
    plot_strategy_comparison
)

__version__ = "0.5.0"
__all__ = [
    # Market Data
    "MarketDataSimulator",
    "IntraDayPriceGenerator", 
    "VolumeProfileGenerator",
    "OrderBook",
    # Strategies
    "BaseStrategy",
    "VWAPStrategy",
    "TWAPStrategy",
    "QUBOStrategy",
    # Execution Engine
    "ExecutionEngine",
    "ParentOrder",
    "ChildOrder",
    "OrderSide",
    "OrderStatus",
    "ExecutionState",
    "ExecutionReport",
    # QUBO Optimization
    "QUBOConfig",
    "ExecutionQUBO",
    "create_random_binary_solution",
    "create_uniform_solution",
    "print_qubo_summary",
    # QUBO Solvers
    "QUBOResult",
    "BruteForceSolver",
    "SimulatedAnnealingSolver",
    "GreedySolver",
    "compare_solvers",
    # Integration
    "StrategyComparison",
    "run_integrated_comparison",
    "plot_strategy_comparison",
    # Utilities
    "ExecutionMetrics",
    "ExecutionSlice",
    "compare_strategies",
]
