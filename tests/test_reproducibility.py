"""Seeds fully determine results: same seed, same output; different seed, different output."""

from datetime import datetime

import numpy as np
import pandas as pd

from qexec.analysis.walk_forward import WalkForwardAnalyzer
from qexec.execution.engine import ExecutionEngine, OrderSide, ParentOrder
from qexec.execution.strategies.vwap import VWAPStrategy
from qexec.market.simulator import MarketDataSimulator
from qexec.optimization.solvers.annealing import SimulatedAnnealingSolver


def market(seed: int) -> pd.DataFrame:
    return MarketDataSimulator(seed=seed).generate(datetime(2024, 1, 2), num_minutes=60)


def walk_forward(seed: int):
    return WalkForwardAnalyzer(
        total_days=4, train_days=2, test_days=1, daily_shares=5000, seed=seed
    ).run()


def annealing(seed: int, Q: np.ndarray):
    result = SimulatedAnnealingSolver(num_sweeps=50, seed=seed).solve(Q)
    return result.solution.tolist(), result.history


def execution(seed: int, data: pd.DataFrame):
    engine = ExecutionEngine(seed=seed)
    order = ParentOrder("AAPL", OrderSide.BUY, total_quantity=20_000, time_horizon_minutes=60)
    report = engine.process_order(order, data, VWAPStrategy(seed=seed))
    return report.filled_quantity, report.average_execution_price, report.total_cost


def test_market_data_is_determined_by_seed():
    pd.testing.assert_frame_equal(market(11), market(11))
    assert not market(11)["price"].equals(market(12)["price"])


def test_walk_forward_is_determined_by_seed():
    first = walk_forward(5)
    assert first == walk_forward(5)
    assert first != walk_forward(6)


def test_simulated_annealing_is_determined_by_seed(rng):
    Q = rng.standard_normal((15, 15))
    Q = (Q + Q.T) / 2
    assert annealing(3, Q) == annealing(3, Q)
    assert annealing(3, Q)[1] != annealing(4, Q)[1]  # different random walk


def test_engine_fills_are_determined_by_seed():
    data = market(0)
    assert execution(1, data) == execution(1, data)
    assert execution(1, data) != execution(2, data)
