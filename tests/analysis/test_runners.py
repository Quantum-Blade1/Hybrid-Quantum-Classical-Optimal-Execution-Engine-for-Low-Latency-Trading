"""Execution runners used by the experiments: one fill model for every strategy."""

import numpy as np
import pytest

from qexec.analysis.runners import STRATEGIES, HybridStrategy, run_strategy
from qexec.analysis.stress import StressGenerator, StressRunner
from qexec.analysis.walk_forward import WALK_FORWARD_STRATEGIES, WalkForwardAnalyzer


@pytest.mark.parametrize("name", STRATEGIES)
def test_runner_logs_account_for_every_executed_share(small_market, name):
    result = run_strategy(name, small_market, total_shares=3000, seed=1)
    log = result.execution_log
    assert result.executed_shares == sum(e["shares"] for e in log) <= 3000
    assert all(e["shares"] > 0 for e in log)
    value = sum(e["shares"] * e["price"] for e in log)
    assert result.avg_price == pytest.approx(value / result.executed_shares)


@pytest.mark.parametrize("name", STRATEGIES)
def test_shortfall_is_execution_plus_opportunity_cost(small_market, name):
    result = run_strategy(name, small_market, total_shares=5000, seed=1)
    arrival = small_market["price"].iloc[0]
    value = sum(e["shares"] * e["price"] for e in result.execution_log)
    assert result.execution_cost == pytest.approx(value - result.executed_shares * arrival)
    assert result.shortfall == pytest.approx(result.execution_cost + result.opportunity_cost)
    assert result.shortfall_bps == pytest.approx(result.shortfall / (5000 * arrival) * 1e4)
    m = result.metrics()
    decomposed = m["spread_cost_bps"] + m["impact_cost_bps"] + m["timing_cost_bps"]
    assert decomposed + m["opportunity_cost_bps"] == pytest.approx(result.shortfall_bps)


@pytest.mark.parametrize("name", STRATEGIES)
def test_no_strategy_fills_in_a_zero_volume_bar(name):
    # Regression (claims audit F6): the hybrid runner filled at mid + spread/2 during the
    # outage, which with the outage's $1000 spread gave absurd slippage.
    data = StressGenerator.market_outage(seed=3)
    outage = set(np.flatnonzero(data["volume"].to_numpy() == 0))
    assert outage
    result = run_strategy(name, data, total_shares=20_000, seed=3)
    assert not outage & {e["minute"] for e in result.execution_log}
    # Shares planned for the outage are carried forward and filled after it.
    assert result.executed_shares == 20_000
    m = result.metrics()
    assert m["spread_cost_bps"] + m["impact_cost_bps"] < 20


def test_unfilled_shares_are_charged_not_dropped(small_market):
    # No liquidity after minute 4: most of the order cannot fill, and the remainder is
    # priced at the last bar's ask (the final bar has no book), not dropped.
    data = small_market.copy()
    data.loc[5:, "volume"] = 0
    result = run_strategy("TWAP", data, total_shares=3000, seed=0)
    unfilled = 3000 - result.executed_shares
    assert unfilled > 0
    last = data.iloc[-1]
    arrival = data["price"].iloc[0]
    expected = unfilled * (last["price"] + last["spread"] / 2 - arrival)
    assert result.opportunity_cost == pytest.approx(expected)


def test_strategies_with_the_same_seed_see_the_same_books(small_market):
    a = run_strategy("TWAP", small_market, total_shares=300_000, seed=9)
    b = run_strategy("TWAP", small_market, total_shares=300_000, seed=9)
    c = run_strategy("TWAP", small_market, total_shares=300_000, seed=10)
    assert a.execution_log == b.execution_log
    assert a.shortfall == b.shortfall != c.shortfall


def test_hybrid_replans_only_at_checkpoints_without_look_ahead(small_market):
    seen = []

    class Spy(HybridStrategy):
        def replan(self, minute, remaining_shares, observed, num_minutes):
            seen.append((minute, len(observed)))
            return super().replan(minute, remaining_shares, observed, num_minutes)

    strategy = Spy(seed=0)
    strategy.calculate_schedule(5000, small_market)
    for minute in range(len(small_market)):
        strategy.replan(minute, 5000, small_market.iloc[: minute + 1], len(small_market))
    assert all(length == minute + 1 for minute, length in seen)
    assert strategy.invocations <= 5


def test_stress_runner_runs_every_strategy_on_every_scenario():
    results = StressRunner(total_shares=5000, seed=0).run_suite()
    assert len(results) == 4 * len(STRATEGIES)
    assert not any(r.crashed for r in results)
    assert all(0 <= r.fill_rate <= 1 for r in results)


@pytest.mark.parametrize(("total", "train", "test", "windows"), [(4, 2, 1, 2), (7, 3, 2, 2)])
def test_walk_forward_window_count(total, train, test, windows):
    analyzer = WalkForwardAnalyzer(
        total_days=total, train_days=train, test_days=test, daily_shares=2000, seed=0
    )
    results = analyzer.run()
    assert [r.window_id for r in results] == list(range(windows))
    assert all(r.train_volatility > 0 for r in results)
    assert all(set(r.shortfall_bps) == set(WALK_FORWARD_STRATEGIES) for r in results)
    assert all(r.fill_rate["TWAP"] == 1.0 for r in results)
