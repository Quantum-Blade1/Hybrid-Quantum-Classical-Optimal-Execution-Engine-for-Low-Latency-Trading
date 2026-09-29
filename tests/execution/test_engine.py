import numpy as np
import pandas as pd
import pytest

from qexec.execution.engine import ExecutionEngine, OrderSide, OrderStatus, ParentOrder
from qexec.execution.strategies.fixed import FixedScheduleStrategy
from qexec.execution.strategies.qubo import QUBOStrategy
from qexec.execution.strategies.twap import TWAPStrategy
from qexec.execution.strategies.vwap import VWAPStrategy
from qexec.market.order_book import OrderBook, OrderBookSnapshot, PriceLevel


class TwoLevelBook(OrderBook):
    def generate_snapshot(self, mid_price, spread, minute_volume, key=None):
        return OrderBookSnapshot(
            bids=[PriceLevel(mid_price - 0.01, 100), PriceLevel(mid_price - 0.02, 1000)],
            asks=[PriceLevel(mid_price + 0.01, 100), PriceLevel(mid_price + 0.02, 1000)],
        )


def hand_market(prices) -> pd.DataFrame:
    n = len(prices)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-02 09:30", periods=n, freq="min"),
            "price": np.asarray(prices, dtype=float),
            "spread": np.full(n, 0.02),
            "volume": np.full(n, 1000),
        }
    )


def order(side: OrderSide, quantity: int, horizon: int = 4) -> ParentOrder:
    return ParentOrder("TEST", side, total_quantity=quantity, time_horizon_minutes=horizon)


@pytest.mark.parametrize(
    ("side", "first_price", "second_price", "arrival_slippage_sign"),
    [(OrderSide.BUY, 100.01, 102.015, 1), (OrderSide.SELL, 99.99, 101.985, -1)],
)
def test_fill_prices_and_average_on_hand_built_book(
    side, first_price, second_price, arrival_slippage_sign
):
    # Minute 2 (mid 102), 200 shares: buy 100 @ 102.01 + 100 @ 102.02 = 102.015, sell 101.985.
    market = hand_market([100.0, 101.0, 102.0, 103.0])
    engine = ExecutionEngine(order_book=TwoLevelBook())
    report = engine.process_order(order(side, 300), market, FixedScheduleStrategy([1, 0, 2, 0]))

    assert engine.state is not None
    children = engine.state.child_orders
    assert [c.minute_index for c in children] == [0, 2]
    assert [c.filled_quantity for c in children] == [100, 200]
    assert [c.execution_price for c in children] == pytest.approx([first_price, second_price])

    prices = np.array([first_price, second_price])
    shares = np.array([100, 200])
    average = float(prices @ shares / 300)
    assert report.filled_quantity == 300
    assert report.average_execution_price == pytest.approx(average)
    assert report.benchmark_vwap == pytest.approx(101.5)  # equal volumes: mean price
    assert report.arrival_price == 100.0
    assert np.sign(report.slippage_vs_arrival_bps) == arrival_slippage_sign
    # Half-spread (1 tick) on all 300 shares; impact adds one tick on the second child's 200.
    assert report.spread_cost == pytest.approx(300 * 0.01)
    assert report.impact_cost == pytest.approx(200 * 0.01)
    assert report.timing_risk == pytest.approx(
        np.sqrt(np.average((prices - average) ** 2, weights=shares))
    )
    assert engine.state.parent_order.status is OrderStatus.FILLED


def test_child_fills_happen_at_their_scheduled_minute(small_market):
    schedule = np.zeros(len(small_market))
    schedule[[5, 20]] = 100
    engine = ExecutionEngine(seed=1)
    engine.process_order(
        order(OrderSide.BUY, 200, 30), small_market, FixedScheduleStrategy(schedule)
    )
    assert engine.state is not None
    children = engine.state.child_orders
    assert [c.minute_index for c in children] == [5, 20]
    for child in children:
        row = small_market.iloc[child.minute_index]
        assert child.market_price_at_execution == row["price"]
        assert child.executed_at == row["timestamp"]


def test_thin_book_partially_fills_and_reports_it():
    market = hand_market([100.0, 100.0])
    engine = ExecutionEngine(order_book=TwoLevelBook(), carry_forward=False)
    report = engine.process_order(
        order(OrderSide.BUY, 2000, 2), market, FixedScheduleStrategy([1, 0])
    )
    assert report.filled_quantity == 1100  # both levels exhausted
    assert report.fill_rate == pytest.approx(0.55)
    assert report.unfilled_quantity == 900
    assert engine.state is not None
    assert engine.state.parent_order.status is OrderStatus.PARTIALLY_FILLED


def test_unfilled_shares_carry_forward_to_the_next_bar():
    market = hand_market([100.0, 100.0, 100.0])
    engine = ExecutionEngine(order_book=TwoLevelBook())
    report = engine.process_order(
        order(OrderSide.BUY, 2000, 3), market, FixedScheduleStrategy([1, 0, 0])
    )
    assert engine.state is not None
    assert [c.target_quantity for c in engine.state.child_orders] == [2000, 900]
    assert [c.filled_quantity for c in engine.state.child_orders] == [1100, 900]
    assert report.filled_quantity == 2000


def test_zero_volume_bar_fills_nothing_and_delays_the_shares():
    market = hand_market([100.0, 101.0, 102.0])
    market.loc[0, "volume"] = 0
    engine = ExecutionEngine(seed=0)
    report = engine.process_order(
        order(OrderSide.BUY, 300, 3), market, FixedScheduleStrategy([1, 0, 0])
    )
    assert engine.state is not None
    fills = [(c.minute_index, c.filled_quantity) for c in engine.state.child_orders]
    assert fills == [(0, 0), (1, 300)]
    assert report.filled_quantity == 300


def test_shortfall_charges_unfilled_shares_at_the_final_far_touch():
    market = hand_market([100.0, 101.0, 103.0])
    market["volume"] = 0
    report = ExecutionEngine(seed=0).process_order(
        order(OrderSide.BUY, 500, 3), market, FixedScheduleStrategy([1, 1, 1])
    )
    assert report.filled_quantity == 0
    assert report.execution_shortfall == 0
    assert report.opportunity_cost == pytest.approx(500 * (103.01 - 100.0))
    assert report.implementation_shortfall_bps == pytest.approx(3.01 / 100 * 1e4)


def test_unfilled_remainder_pays_the_impact_of_a_clean_up_order():
    # 1,100 fill in the only minute; the other 900 walk the final book: 100 @ 100.01 + 800 @ 100.02.
    market = hand_market([100.0])
    report = ExecutionEngine(order_book=TwoLevelBook(), carry_forward=False).process_order(
        order(OrderSide.BUY, 2000, 1), market, FixedScheduleStrategy([1])
    )
    assert report.unfilled_quantity == 900
    assert report.completion_price == pytest.approx((100 * 100.01 + 800 * 100.02) / 900)
    assert report.opportunity_cost > 900 * 0.01  # more than the half spread alone


def test_replan_hook_replaces_the_rest_of_the_plan():
    class SwitchAtTwo(FixedScheduleStrategy):
        def replan(self, minute, remaining_shares, observed, num_minutes):
            assert len(observed) == minute + 1
            return np.array([0, 1]) if minute == 2 else None

    market = hand_market([100.0] * 4)
    engine = ExecutionEngine(order_book=TwoLevelBook())
    report = engine.process_order(order(OrderSide.BUY, 40, 4), market, SwitchAtTwo([1, 1, 1, 1]))
    assert engine.state is not None
    assert [c.filled_quantity for c in engine.state.child_orders] == [10, 10, 20]
    assert [c.minute_index for c in engine.state.child_orders] == [0, 1, 3]
    assert report.filled_quantity == 40


@pytest.mark.parametrize(
    "strategy",
    [
        VWAPStrategy(seed=0),
        TWAPStrategy(seed=0),
        FixedScheduleStrategy([3, 1, 4, 1, 5, 9, 2, 6]),
        QUBOStrategy(num_time_slices=5, sa_sweeps=50, seed=0),
    ],
    ids=["vwap", "twap", "fixed", "qubo"],
)
@pytest.mark.parametrize("side", [OrderSide.BUY, OrderSide.SELL])
@pytest.mark.parametrize("quantity", [1, 777, 50_000])
def test_fills_never_exceed_order_size(small_market, strategy, side, quantity):
    engine = ExecutionEngine(seed=3)
    report = engine.process_order(order(side, quantity, 30), small_market, strategy)
    assert engine.state is not None
    children = engine.state.child_orders
    assert all(0 <= c.filled_quantity <= c.target_quantity <= quantity for c in children)
    assert report.filled_quantity == sum(c.filled_quantity for c in children) <= quantity
    assert 0.0 <= report.fill_rate <= 1.0


def test_engine_rejects_non_positive_quantity(small_market):
    with pytest.raises(ValueError, match="positive"):
        ExecutionEngine(seed=0).process_order(
            order(OrderSide.BUY, 0), small_market, TWAPStrategy(seed=0)
        )
