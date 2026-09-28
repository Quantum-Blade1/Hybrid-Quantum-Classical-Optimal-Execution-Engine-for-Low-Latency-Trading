"""Regression tests for bugs fixed during the Phase 4 clean-up."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from qexec.analysis.runners import run_hybrid_execution, run_vwap_execution
from qexec.analysis.walk_forward import WalkForwardAnalyzer
from qexec.execution.engine import ExecutionEngine, OrderSide, ParentOrder
from qexec.execution.strategies.almgren_chriss import ACConfig, AlmgrenChrissSolver
from qexec.execution.strategies.fixed import FixedScheduleStrategy
from qexec.execution.strategies.qubo import QUBOStrategy
from qexec.hardware.mitigation import ZeroNoiseExtrapolation
from qexec.market.simulator import MarketDataSimulator
from qexec.microstructure.kyle import KyleLambdaEstimator


@pytest.fixture
def market_data() -> pd.DataFrame:
    return MarketDataSimulator(total_daily_volume=10_000_000, seed=7).generate(
        datetime(2024, 1, 2), num_minutes=30
    )


def test_engine_fills_child_at_scheduled_minute(market_data):
    # Before the fix, the k-th non-zero slice was filled at minute k-1, not at its own minute.
    schedule = np.zeros(len(market_data))
    schedule[[5, 20]] = 100
    engine = ExecutionEngine(seed=1)
    order = ParentOrder("AAPL", OrderSide.BUY, total_quantity=200, time_horizon_minutes=30)
    engine.process_order(order, market_data, FixedScheduleStrategy(schedule))
    assert engine.state is not None
    children = engine.state.child_orders
    assert [c.minute_index for c in children] == [5, 20]
    for child in children:
        assert child.market_price_at_execution == market_data.iloc[child.minute_index]["price"]
        assert child.executed_at == market_data.iloc[child.minute_index]["timestamp"]


def test_simulator_streams_are_independent():
    # Prices, volumes and spreads used to share one seed, so volume noise reused the price shocks.
    sim = MarketDataSimulator(seed=3)
    price_draw = sim.price_generator.rng.standard_normal(100)
    volume_draw = sim.volume_generator.rng.standard_normal(100)
    assert abs(np.corrcoef(price_draw, volume_draw)[0, 1]) < 0.5


def test_simulator_is_reproducible():
    a = MarketDataSimulator(seed=11).generate(datetime(2024, 1, 2), num_minutes=50)
    b = MarketDataSimulator(seed=11).generate(datetime(2024, 1, 2), num_minutes=50)
    pd.testing.assert_frame_equal(a, b)


def test_almgren_chriss_variance_uses_end_of_interval_holdings():
    # V = sigma^2 tau sum_{k=1}^N x_k^2 with x_N = 0; linear 2-step path has x_1 = X/2.
    config = ACConfig(total_shares=1000, n_steps=2, risk_aversion=0.0)
    solver = AlmgrenChrissSolver(config)
    variance = solver.calculate_variance(solver.compute_trajectory())
    tau = config.n_days / config.n_steps
    expected = (config.sigma * config.price) ** 2 * tau * 500.0**2
    assert variance == pytest.approx(expected)


def test_kyle_lambda_is_ols_slope():
    rng = np.random.default_rng(0)
    volume = rng.normal(0, 1000, 60)
    dp = 2e-4 * volume + rng.normal(0, 0.01, 60)
    estimator = KyleLambdaEstimator(window_size=60)
    for p, v in zip(dp, volume, strict=True):
        estimator.update(p, v)
    assert estimator.lambda_value == pytest.approx(np.polyfit(volume, dp, 1)[0])


def test_qubo_strategy_initialises_base_state(market_data):
    strategy = QUBOStrategy(num_time_slices=5, sa_sweeps=50, seed=1)
    metrics = strategy.execute(500, "buy", market_data)
    summary = strategy.get_execution_summary()
    assert metrics.filled_shares > 0
    assert len(summary) == metrics.num_slices > 0


def test_qubo_strategy_levels_follow_order_size(market_data):
    # Default quantity levels were cached from the first call and reused for other sizes.
    strategy = QUBOStrategy(num_time_slices=5, sa_sweeps=50, seed=1)
    strategy.calculate_schedule(500, market_data)
    strategy.calculate_schedule(5000, market_data)
    assert strategy.qubo is not None
    assert strategy.qubo.config.quantity_levels == [0, 500, 1000, 1500, 2000]


def test_runner_cost_uses_executed_shares(market_data):
    result = run_hybrid_execution(market_data, total_shares=5000, seed=1)
    benchmark = (market_data["price"] * market_data["volume"]).sum() / market_data["volume"].sum()
    value = sum(e["shares"] * e["price"] for e in result.execution_log)
    assert result.total_cost == pytest.approx(value - result.executed_shares * benchmark)


def test_vwap_runner_log_is_engine_fills(market_data):
    result = run_vwap_execution(market_data, total_shares=2000, seed=1)
    assert sum(e["shares"] for e in result.execution_log) == result.executed_shares
    value = sum(e["shares"] * e["price"] for e in result.execution_log)
    assert value / result.executed_shares == pytest.approx(result.avg_price)


def test_walk_forward_is_reproducible():
    kwargs = {"total_days": 4, "train_days": 2, "test_days": 1, "daily_shares": 5000, "seed": 5}
    first = WalkForwardAnalyzer(**kwargs).run()
    second = WalkForwardAnalyzer(**kwargs).run()
    assert len(first) == 2
    assert first == second


def test_zne_exponential_keeps_sign():
    zne = ZeroNoiseExtrapolation(noise_factors=[1.0, 2.0, 3.0], extrapolation="exponential")
    values = [-np.exp(-0.5 * c) * 10 for c in (1.0, 2.0, 3.0)]
    assert zne._extrapolate([1.0, 2.0, 3.0], values) == pytest.approx(-10.0)
