"""Synthetic market data: quote invariants, volume profile shape, stream independence."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from qexec.market.simulator import MarketDataSimulator, VolumeProfileGenerator, calculate_vwap


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_quotes_are_consistent_on_the_tick_grid(seed):
    data = MarketDataSimulator(seed=seed).generate(datetime(2024, 1, 2))
    assert len(data) == 390
    assert (data["price"] > 0).all()
    assert (data["bid"] < data["ask"]).all()
    assert (data["bid"] <= data["price"]).all() and (data["price"] <= data["ask"]).all()
    np.testing.assert_allclose(data["spread"], data["ask"] - data["bid"])
    assert (data["spread"] / data["price"] < 0.01).all()  # below 100 bps
    for column in ("price", "bid", "ask"):
        ticks = data[column] / 0.01
        np.testing.assert_allclose(ticks, ticks.round(), atol=1e-6)
    assert (data["volume"] >= 100).all()
    assert data["timestamp"].diff().dropna().eq(pd.Timedelta(minutes=1)).all()


def test_volume_profile_is_u_shaped_and_sums_near_target():
    volumes = VolumeProfileGenerator(total_daily_volume=50_000_000, seed=0).generate(390)
    midday = volumes[150:240].mean()
    assert volumes[:30].mean() > 2 * midday
    assert volumes[-30:].mean() > 2 * midday
    # Integer truncation and the 100-share floor move the total by well under 1%.
    assert volumes.sum() == pytest.approx(50_000_000, rel=0.01)


def test_daily_price_volatility_matches_parameters():
    # 1-minute GBM with a U-shaped vol multiplier in [1, 2.5]; annual vol 25%.
    sim = MarketDataSimulator(seed=4)
    closes = np.array(
        [
            sim.generate(datetime(2024, 1, 2), initial_price=150.0)["price"].iloc[-1]
            for _ in range(300)
        ]
    )
    daily_vol = np.log(closes / 150.0).std()
    base = 0.25 / np.sqrt(252)
    assert base < daily_vol < 2.5 * base


def test_price_volume_and_spread_streams_are_independent():
    # Regression: the three generators shared one seed, so volume noise reused price shocks.
    sim = MarketDataSimulator(seed=3)
    price_draw = sim.price_generator.rng.standard_normal(1000)
    volume_draw = sim.volume_generator.rng.standard_normal(1000)
    spread_draw = sim.rng.standard_normal(1000)
    corr = np.corrcoef([price_draw, volume_draw, spread_draw])
    assert np.all(np.abs(corr[np.triu_indices(3, 1)]) < 0.1)


def test_calculate_vwap_hand_example():
    df = pd.DataFrame({"price": [100.0, 101.0, 102.0], "volume": [1000, 2000, 1000]})
    assert calculate_vwap(df) == pytest.approx((100_000 + 202_000 + 102_000) / 4000)
