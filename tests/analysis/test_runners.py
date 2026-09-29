"""Execution runners used by the experiments: cost accounting and fill bounds."""

import pytest

from qexec.analysis.runners import run_hybrid_execution, run_sa_execution, run_vwap_execution
from qexec.analysis.walk_forward import WalkForwardAnalyzer
from qexec.market.simulator import calculate_vwap


@pytest.mark.parametrize("runner", [run_vwap_execution, run_sa_execution, run_hybrid_execution])
def test_runner_logs_account_for_every_executed_share(small_market, runner):
    result = runner(small_market, total_shares=3000, seed=1)
    log = result.execution_log
    assert result.executed_shares == sum(e["shares"] for e in log) <= 3000
    assert all(e["shares"] > 0 for e in log)
    value = sum(e["shares"] * e["price"] for e in log)
    assert result.avg_price == pytest.approx(value / result.executed_shares)


@pytest.mark.parametrize("runner", [run_sa_execution, run_hybrid_execution])
def test_runner_cost_is_value_over_vwap_of_executed_shares(small_market, runner):
    # Regression: the cost used the order size rather than the executed shares.
    result = runner(small_market, total_shares=5000, seed=1)
    value = sum(e["shares"] * e["price"] for e in result.execution_log)
    benchmark = calculate_vwap(small_market)
    assert result.total_cost == pytest.approx(value - result.executed_shares * benchmark)


@pytest.mark.parametrize(("total", "train", "test", "windows"), [(4, 2, 1, 2), (7, 3, 2, 2)])
def test_walk_forward_window_count(total, train, test, windows):
    analyzer = WalkForwardAnalyzer(
        total_days=total, train_days=train, test_days=test, daily_shares=2000, seed=0
    )
    results = analyzer.run()
    assert [r.window_id for r in results] == list(range(windows))
    assert all(r.train_volatility > 0 for r in results)
