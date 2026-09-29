"""Synthetic order book: snapshot shape and book-walking fills."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from qexec.market.order_book import OrderBook, OrderBookSnapshot, PriceLevel


@given(
    mid=st.floats(1.0, 1000.0),
    spread_ticks=st.integers(1, 20),
    volume=st.integers(0, 1_000_000),
    seed=st.integers(0, 1000),
)
def test_snapshot_levels_are_ordered_outside_the_mid(mid, spread_ticks, volume, seed):
    snap = OrderBook(seed=seed).generate_snapshot(mid, spread_ticks * 0.01, volume)
    bids = [level.price for level in snap.bids]
    asks = [level.price for level in snap.asks]
    assert bids[0] <= mid <= asks[0]
    assert np.all(np.diff(bids) < 0) and np.all(np.diff(asks) > 0)
    assert all(level.quantity >= 10 for level in snap.bids + snap.asks)


BOOK = OrderBookSnapshot(
    bids=[PriceLevel(99.99, 100), PriceLevel(99.98, 200), PriceLevel(99.97, 300)],
    asks=[PriceLevel(100.01, 100), PriceLevel(100.02, 200), PriceLevel(100.03, 300)],
)


@pytest.mark.parametrize(
    ("side", "size", "filled", "avg_price", "impact"),
    [
        ("buy", 50, 50, 100.01, 0.0),
        ("buy", 250, 250, (100 * 100.01 + 150 * 100.02) / 250, 0.01),
        ("buy", 1000, 600, (100 * 100.01 + 200 * 100.02 + 300 * 100.03) / 600, 0.02),
        ("sell", 250, 250, (100 * 99.99 + 150 * 99.98) / 250, 0.01),
    ],
)
def test_marketable_order_walks_the_book(side, size, filled, avg_price, impact):
    got_price, got_filled, got_impact = OrderBook().simulate_execution(BOOK, size, side)
    assert got_filled == filled
    assert got_price == pytest.approx(avg_price)
    assert got_impact == pytest.approx(impact)
