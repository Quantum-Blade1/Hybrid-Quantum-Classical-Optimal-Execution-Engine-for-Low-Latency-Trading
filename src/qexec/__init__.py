"""
qexec: hybrid quantum-classical optimal trade execution.

Subpackages:
    market          synthetic market data, order book, tick-data loading
    execution       execution engine and strategies (TWAP, VWAP, Almgren-Chriss, QUBO)
    optimization    execution QUBOs, Ising mapping, classical and QAOA solvers
    microstructure  Kyle's lambda, VPIN, adverse selection, queue and regime models
    runtime         latency-decoupled fast/slow path runtime and HFT pipeline
    hardware        IBM Quantum hardware access and error mitigation
    analysis        implementation shortfall, walk-forward and stress tests
"""

from qexec.execution.engine import ExecutionEngine, OrderSide, ParentOrder
from qexec.market.simulator import MarketDataSimulator
from qexec.optimization.qubo import ExecutionQUBO, QUBOConfig
from qexec.optimization.solvers.annealing import SimulatedAnnealingSolver

__version__ = "0.1.0"

__all__ = [
    "ExecutionEngine",
    "ExecutionQUBO",
    "MarketDataSimulator",
    "OrderSide",
    "ParentOrder",
    "QUBOConfig",
    "SimulatedAnnealingSolver",
    "__version__",
]
