"""TWAP, VWAP and fixed schedules: conservation of shares and the caps they promise."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from qexec.execution.strategies.fixed import FixedScheduleStrategy
from qexec.execution.strategies.twap import TWAPStrategy
from qexec.execution.strategies.vwap import VWAPStrategy


def flat_market(volumes, price: float = 100.0) -> pd.DataFrame:
    n = len(volumes)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-02 09:30", periods=n, freq="min"),
            "price": np.full(n, price),
            "bid": np.full(n, price - 0.01),
            "ask": np.full(n, price + 0.01),
            "spread": np.full(n, 0.02),
            "volume": np.asarray(volumes),
        }
    )


@given(
    total=st.integers(1, 20_000),
    minutes=st.integers(1, 120),
    interval=st.integers(1, 15),
    max_slice_pct=st.floats(0.01, 1.0),
)
def test_twap_schedule_conserves_shares_and_respects_caps(total, minutes, interval, max_slice_pct):
    schedule = TWAPStrategy(interval, max_slice_pct).calculate_schedule(
        total, flat_market([1] * minutes)
    )
    points = np.arange(0, minutes, interval)
    cap = int(total * max_slice_pct)
    assert np.all(schedule >= 0)
    assert np.all(schedule[np.setdiff1d(np.arange(minutes), points)] == 0)
    assert schedule.max() <= cap
    if len(points) * cap >= total:
        assert schedule.sum() == total
    else:
        assert schedule.sum() == len(points) * cap  # every slice filled to its cap


@given(total=st.integers(1, 20_000), minutes=st.integers(1, 120))
def test_uncapped_twap_slices_differ_by_at_most_one_share(total, minutes):
    schedule = TWAPStrategy(1, max_slice_pct=1.0).calculate_schedule(
        total, flat_market([1] * minutes)
    )
    assert schedule.sum() == total
    assert schedule.max() - schedule.min() <= 1


def test_twap_benchmark_is_mean_price_at_execution_points():
    market = flat_market([1000] * 10)
    market["price"] = np.arange(100.0, 110.0)
    assert TWAPStrategy(interval_minutes=3).calculate_benchmark(market) == pytest.approx(
        np.mean([100.0, 103.0, 106.0, 109.0])
    )


def test_vwap_schedule_is_proportional_to_volume():
    volumes = np.array([100, 200, 300, 400, 1000])
    schedule = VWAPStrategy(participation_rate=1.0, max_slice_pct=1.0, seed=0).calculate_schedule(
        2000, flat_market(volumes)
    )
    np.testing.assert_array_equal(schedule, 2000 * volumes / volumes.sum())


@given(
    volumes=st.lists(st.integers(100, 50_000), min_size=1, max_size=60),
    total=st.integers(1, 20_000),
    participation=st.floats(0.05, 1.0),
    max_slice_pct=st.floats(0.05, 1.0),
    seed=st.integers(0, 1000),
)
def test_vwap_schedule_never_overfills_and_respects_caps(
    volumes, total, participation, max_slice_pct, seed
):
    strategy = VWAPStrategy(participation, max_slice_pct, seed=seed)
    schedule = strategy.calculate_schedule(total, flat_market(volumes))
    cap = np.minimum(np.asarray(volumes) * participation, total * max_slice_pct)
    assert np.all(schedule >= 0)
    assert schedule.sum() <= total
    assert np.all(schedule <= np.ceil(cap))
    if np.floor(cap).sum() >= total:
        assert schedule.sum() == total


@given(
    volumes=st.lists(st.integers(100, 50_000), min_size=1, max_size=60),
    total=st.integers(1, 20_000),
    seed=st.integers(0, 1000),
)
def test_vwap_schedule_tracks_volume_share_within_rounding(volumes, total, seed):
    assume(total <= sum(volumes))
    schedule = VWAPStrategy(1.0, 1.0, seed=seed).calculate_schedule(total, flat_market(volumes))
    ideal = total * np.asarray(volumes) / sum(volumes)
    assert schedule.sum() == total
    # Rounding moves each slice by < 1 share; the leftover (< n shares) is spread by volume.
    assert np.abs(schedule - ideal).sum() <= len(volumes)


@given(
    weights=st.lists(st.integers(0, 1000), min_size=1, max_size=50),
    total=st.integers(1, 100_000),
)
def test_fixed_schedule_is_rescaled_to_exactly_the_order_size(weights, total):
    schedule = FixedScheduleStrategy(weights).calculate_schedule(
        total, flat_market([1] * len(weights))
    )
    assert np.all(schedule >= 0)
    assert schedule.sum() == total
    if sum(weights) > 0:
        # Zero-weight minutes stay empty; others get their share up to one rounding unit.
        w = np.asarray(weights, dtype=float)
        assert np.all(schedule[w == 0] == 0)
        assert np.all(np.abs(schedule - total * w / w.sum()) < 1)
