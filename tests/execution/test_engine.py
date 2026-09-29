"""Execution engine: fill accounting on a hand-built book, fill timing, and fill bounds."""

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
    """Deterministic book: 100 shares one tick from mid, then 1,000 shares two ticks away."""

    def generate_snapshot(self, mid_price: float, spread: float, minute_volume: int):
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
    # Minute 0 (mid 100): 100 shares, all at the touch.
    # Minute 2 (mid 102): 200 shares; a buy takes 100 @ 102.01 + 100 @ 102.02 = 102.015 average,
    # a sell 100 @ 101.99 + 100 @ 101.98 = 101.985.
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
    # Half-spread (1 tick) on every share; impact = last fill minus touch (1 tick) on the
    # second child's 200 shares.
    assert report.spread_cost == pytest.approx(300 * 0.01)
    assert report.impact_cost == pytest.approx(200 * 0.01)
    assert report.timing_risk == pytest.approx(
        np.sqrt(np.average((prices - average) ** 2, weights=shares))
    )
    assert engine.state.parent_order.status is OrderStatus.FILLED


def test_child_fills_happen_at_their_scheduled_minute(small_market):
    # Regression: the k-th non-zero slice used to be filled at minute k-1, not its own minute.
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
    engine = ExecutionEngine(order_book=TwoLevelBook())
    report = engine.process_order(
        order(OrderSide.BUY, 2000, 2), market, FixedScheduleStrategy([1, 0])
    )
    assert report.filled_quantity == 1100  # both levels exhausted
    assert report.fill_rate == pytest.approx(0.55)
    assert engine.state is not None
    assert engine.state.parent_order.status is OrderStatus.PARTIALLY_FILLED


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
    assert all(0 <= c.filled_quantity <= c.target_quantity for c in children)
    assert sum(c.target_quantity for c in children) <= quantity
    assert report.filled_quantity == sum(c.filled_quantity for c in children) <= quantity
    assert 0.0 <= report.fill_rate <= 1.0


def test_engine_rejects_non_positive_quantity(small_market):
    with pytest.raises(ValueError, match="positive"):
        ExecutionEngine(seed=0).process_order(
            order(OrderSide.BUY, 0), small_market, TWAPStrategy(seed=0)
        )
