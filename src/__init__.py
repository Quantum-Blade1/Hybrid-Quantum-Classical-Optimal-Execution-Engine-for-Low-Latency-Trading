"""
Hybrid Quantum-Classical Optimal Execution Engine for Low-Latency Trading

Core modules for market data simulation, execution strategies,
execution engine, quantum optimization, HFT microstructure analysis,
adaptive risk management, and latency-decoupled architecture.
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

from .hft_microstructure import (
    MicrostructureAnalyzer,
    MicrostructureState,
    KyleLambdaEstimator,
    VPINEstimator,
    AdverseSelectionModel,
    QueuePositionModel,
)
from .adaptive_risk import (
    AdaptiveRiskManager,
    VolatilityEstimator,
    RegimeState,
    VolatilityRegime,
    SpreadRegime,
)
from .hft_qubo import (
    HFTQUBOConfig,
    HFTExecutionQUBO,
)
from .latency_monitor import (
    LatencyMonitor,
    LatencySpan,
    LatencyStats,
    get_latency_monitor,
)
from .hft_pipeline import (
    HFTQuantumPipeline,
    HFTPipelineConfig,
    HFTExecutionResult,
)

__version__ = "1.0.0"
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
    # HFT Microstructure
    "MicrostructureAnalyzer",
    "MicrostructureState",
    "KyleLambdaEstimator",
    "VPINEstimator",
    "AdverseSelectionModel",
    "QueuePositionModel",
    # Adaptive Risk
    "AdaptiveRiskManager",
    "VolatilityEstimator",
    "RegimeState",
    "VolatilityRegime",
    "SpreadRegime",
    # HFT QUBO
    "HFTQUBOConfig",
    "HFTExecutionQUBO",
    # Latency Monitor
    "LatencyMonitor",
    "LatencySpan",
    "LatencyStats",
    "get_latency_monitor",
    # HFT Pipeline
    "HFTQuantumPipeline",
    "HFTPipelineConfig",
    "HFTExecutionResult",
    # Utilities
    "ExecutionMetrics",
    "ExecutionSlice",
    "compare_strategies",
]
