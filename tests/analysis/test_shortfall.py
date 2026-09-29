"""Implementation shortfall decomposition on a hand-computed example."""

import pandas as pd
import pytest

from qexec.analysis.shortfall import ISAnalyzer

TIMES = pd.date_range("2024-01-02 09:30", periods=4, freq="min")
MARKET = pd.DataFrame({"timestamp": TIMES, "price": [101.0, 102.0, 103.0, 104.0]})


def test_shortfall_components_sum_to_perold_shortfall():
    # Buy 1,000 decided at 100; mids 101 (arrival) .. 104 (close); 700 filled, 300 unfilled.
    fills = pd.DataFrame(
        {"timestamp": TIMES[[1, 2]], "shares": [300, 400], "price": [102.05, 103.10]}
    )
    result = ISAnalyzer(decision_price=100.0, total_orders=1000).analyze(fills, MARKET)

    assert result.delay_cost == pytest.approx((101 - 100) * 1000)
    assert result.market_impact == pytest.approx(300 * 0.05 + 400 * 0.10)
    assert result.timing_risk == pytest.approx(300 * (102 - 101) + 400 * (103 - 101))
    # Delay already charges the unfilled shares from decision to arrival.
    assert result.opportunity_cost == pytest.approx((104 - 101) * 300)

    # Perold: paper-portfolio return minus real return,
    # sum n_i (p_i - P_d) + U (P_T - P_d) = 615 + 1240 + 1200.
    perold = 300 * (102.05 - 100) + 400 * (103.10 - 100) + 300 * (104 - 100)
    parts = (result.delay_cost, result.market_impact, result.timing_risk, result.opportunity_cost)
    assert sum(parts) == pytest.approx(result.total_shortfall)
    assert result.total_shortfall == pytest.approx(perold) == pytest.approx(3055.0)
    assert result.executed_shares == 700
    assert result.avg_exec_price == pytest.approx((300 * 102.05 + 400 * 103.10) / 700)


def test_fully_filled_order_has_no_opportunity_cost():
    fills = pd.DataFrame(
        {"timestamp": TIMES[[0, 3]], "shares": [500, 500], "price": [101.0, 104.0]}
    )
    result = ISAnalyzer(decision_price=101.0, total_orders=1000).analyze(fills, MARKET)
    assert result.opportunity_cost == 0.0
    assert result.delay_cost == 0.0
    assert result.total_shortfall == pytest.approx(500 * 3.0)


def test_fills_are_matched_to_the_nearest_bar():
    fills = pd.DataFrame(
        {"timestamp": [TIMES[1] + pd.Timedelta(seconds=20)], "shares": [100], "price": [102.5]}
    )
    result = ISAnalyzer(decision_price=101.0, total_orders=100).analyze(fills, MARKET)
    assert result.market_impact == pytest.approx(100 * 0.5)  # matched to the 102.0 bar
