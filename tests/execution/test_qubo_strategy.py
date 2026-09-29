"""QUBO-scheduled strategy: regressions and the three-way comparison through the engine."""

import numpy as np
import pytest

from qexec.execution.engine import OrderSide, ParentOrder
from qexec.execution.strategies.qubo import QUBOStrategy, run_integrated_comparison


def test_qubo_strategy_levels_follow_each_order_size(small_market):
    # Regression: default quantity levels were cached from the first call and reused.
    strategy = QUBOStrategy(num_time_slices=5, sa_sweeps=50, seed=1)
    strategy.calculate_schedule(500, small_market)
    strategy.calculate_schedule(5000, small_market)
    assert strategy.qubo is not None
    assert strategy.qubo.config.quantity_levels == [0, 500, 1000, 1500, 2000]


def test_qubo_strategy_schedule_matches_its_solved_qubo(small_market):
    strategy = QUBOStrategy(num_time_slices=5, sa_sweeps=200, seed=1)
    schedule = strategy.calculate_schedule(1000, small_market)
    stats = strategy.get_optimization_stats()
    assert np.all(schedule >= 0)
    assert schedule.sum() == stats["total_shares"]
    assert stats["qubo_energy"] == pytest.approx(stats["total_cost"])


def test_qubo_strategy_execute_records_every_filled_slice(small_market):
    # Regression: execute() failed because BaseStrategy state was not initialised.
    strategy = QUBOStrategy(num_time_slices=5, sa_sweeps=50, seed=1)
    metrics = strategy.execute(500, "buy", small_market)
    summary = strategy.get_execution_summary()
    assert len(summary) == metrics.num_slices > 0
    assert summary["filled_qty"].sum() == metrics.filled_shares <= 500


def test_integrated_comparison_runs_the_same_order_and_ranks_by_cost(small_market):
    order = ParentOrder("AAPL", OrderSide.BUY, total_quantity=300, time_horizon_minutes=30)
    comparison = run_integrated_comparison(
        order, small_market, qubo_time_slices=5, qubo_sa_sweeps=50, seed=0
    )
    reports = comparison.reports
    assert {r.total_quantity for r in reports.values()} == {300}
    assert all(0 < r.filled_quantity <= 300 for r in reports.values())
    costs = {name: r.total_cost for name, r in reports.items()}
    assert comparison.best_strategy == min(costs, key=costs.__getitem__)
