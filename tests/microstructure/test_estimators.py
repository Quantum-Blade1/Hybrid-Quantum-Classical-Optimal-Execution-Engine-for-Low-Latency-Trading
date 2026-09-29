"""Kyle's lambda, VPIN and adverse selection on synthetic data with known answers."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from qexec.microstructure.adverse_selection import AdverseSelectionModel
from qexec.microstructure.analyzer import MicrostructureAnalyzer
from qexec.microstructure.kyle import KyleLambdaEstimator
from qexec.microstructure.vpin import VPINEstimator


@pytest.mark.parametrize("true_lambda", [5e-5, 2e-4, 1e-3])
def test_kyle_lambda_recovers_the_generating_slope(true_lambda, rng):
    signed_volume = rng.normal(0, 1000, 500)
    price_change = true_lambda * signed_volume + rng.normal(0, 0.02, 500)
    estimator = KyleLambdaEstimator(window_size=500)
    for dp, sv in zip(price_change, signed_volume, strict=True):
        estimator.update(dp, sv)
    assert estimator.lambda_value == pytest.approx(true_lambda, rel=0.1)


def test_kyle_lambda_is_the_ols_slope_of_the_window(rng):
    # Regression: numerator and denominator used different ddof, biasing the slope.
    volume = rng.normal(0, 1000, 60)
    dp = 2e-4 * volume + rng.normal(0, 0.01, 60)
    estimator = KyleLambdaEstimator(window_size=60)
    for p, v in zip(dp, volume, strict=True):
        estimator.update(p, v)
    assert estimator.lambda_value == pytest.approx(np.polyfit(volume, dp, 1)[0])


def test_kyle_lambda_waits_for_twenty_observations_and_clamps_negative_slopes():
    estimator = KyleLambdaEstimator()
    for k in range(19):
        assert estimator.update(-1e-4 * (k - 9), float(k - 9)) == 0.0
    assert estimator.update(1e-4, -10.0) < 0  # raw slope is negative ...
    assert estimator.lambda_value == 0.0  # ... but the reported impact is clamped


@given(
    prices=st.lists(st.floats(90, 110), min_size=2, max_size=300),
    volumes=st.lists(st.integers(1, 2000), min_size=300, max_size=300),
    bucket=st.integers(100, 3000),
)
def test_vpin_is_a_fraction(prices, volumes, bucket):
    estimator = VPINEstimator(bucket_size=bucket, num_buckets=20)
    for prev, price, volume in zip(prices, prices[1:], volumes, strict=False):
        assert 0.0 <= estimator.update(price, volume, prev) <= 1.0


@pytest.mark.parametrize(
    ("step", "expected"),
    [(0.01, 1.0), (-0.01, 1.0), (0.0, 0.0)],
    ids=["all-buys", "all-sells", "unchanged"],
)
def test_vpin_extremes(step, expected):
    estimator = VPINEstimator(bucket_size=1000, num_buckets=20)
    price = 100.0
    for _ in range(200):
        estimator.update(price + step, 500, price)
        price += step
    assert estimator.vpin == pytest.approx(expected)
    assert estimator.is_toxic == (expected > 0.7)


def test_adverse_selection_equals_twice_the_post_trade_mid_move():
    # Buys at mid + half-spread while the mid drifts up by `drift` per trade:
    # effective = spread, realised = spread - 2 * horizon * drift.
    spread, drift, horizon = 0.02, 0.001, 10
    model = AdverseSelectionModel(lookback_ticks=50, realized_horizon=horizon)
    for t in range(60):
        mid = 100.0 + drift * t
        model.update(mid + spread / 2, mid, "buy")
    effective, realised, adverse = model.estimate()
    assert effective == pytest.approx(spread)
    assert realised == pytest.approx(spread - 2 * horizon * drift)
    assert adverse == pytest.approx(2 * horizon * drift)


def test_analyzer_flags_one_sided_flow_as_toxic():
    analyzer = MicrostructureAnalyzer(vpin_bucket_size=1000, vpin_buckets=20)
    price = 100.0
    for _ in range(100):
        price += 0.01
        state = analyzer.process_tick(price, 500, price - 0.01, price + 0.01)
    assert state.vpin == pytest.approx(1.0)
    assert state.toxicity_flag
    assert state.spread_regime == "toxic"
