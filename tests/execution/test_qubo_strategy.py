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
    schedule = strategy.calculate_schedule(1003, small_market)
    stats = strategy.get_optimization_stats()
    assert np.all(schedule >= 0)
    # The QUBO's discrete levels cannot select 1003 shares; the schedule is repaired to it.
    assert stats["total_shares"] != 1003
    assert schedule.sum() == 1003
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
    costs = {name: r.implementation_shortfall for name, r in reports.items()}
    assert comparison.best_strategy == min(costs, key=costs.__getitem__)


def test_integrated_comparison_does_not_reward_underfilling(small_market):
    # Regression (claims audit): ranking by spread + impact on filled shares favoured a
    # strategy that filled fewer shares. With liquidity only in the first three minutes,
    # every strategy underfills and its remainder is charged at the final far touch.
    data = small_market.copy()
    data.loc[3:, "volume"] = 0
    order = ParentOrder("AAPL", OrderSide.BUY, total_quantity=3000, time_horizon_minutes=30)
    comparison = run_integrated_comparison(
        order, data, qubo_time_slices=5, qubo_sa_sweeps=50, seed=0
    )
    last = data.iloc[-1]
    for report in comparison.reports.values():
        assert report.unfilled_quantity > 0
        assert report.completion_price == pytest.approx(last["price"] + last["spread"] / 2)
    shortfalls = {n: r.implementation_shortfall for n, r in comparison.reports.items()}
    assert comparison.best_strategy == min(shortfalls, key=shortfalls.__getitem__)
