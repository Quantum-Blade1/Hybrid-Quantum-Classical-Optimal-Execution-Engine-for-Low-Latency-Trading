"""Dev-day calibration: profiles, Kyle-style impact regression and lot sizes."""

import numpy as np
import pandas as pd
import pytest

from qexec.market.calibration import (
    MINUTES_PER_DAY,
    calibrate,
    fit_impact,
    intraday_profile,
    log_returns_bps,
    lot_size_for,
)


def synthetic_bars(days: int, beta: float, seed: int = 0) -> pd.DataFrame:
    """Bars whose returns are exactly beta * SV / Vbar plus noise (Vbar = 100 everywhere)."""
    rng = np.random.default_rng(seed)
    n = days * MINUTES_PER_DAY
    ts = pd.date_range("2026-07-22", periods=n, freq="min")
    volume = rng.uniform(50, 150, n)
    volume *= 100 / np.mean(volume)
    signed = rng.uniform(-1, 1, n) * volume
    r = beta * signed / 100 + rng.normal(0, 0.5, n)
    close = 100 * np.exp(np.cumsum(r) / 1e4)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "close": close,
            "volume": volume,
            "signed_volume": signed,
            "half_spread_bps": np.where(rng.random(n) < 0.5, 0.6, np.nan),
        }
    )


def test_log_returns_reset_at_each_day():
    bars = synthetic_bars(2, 1.0)
    r = log_returns_bps(bars)
    assert np.isnan(r[0]) and np.isnan(r[MINUTES_PER_DAY])
    assert r[1] == pytest.approx(np.log(bars.close[1] / bars.close[0]) * 1e4)


def test_impact_regression_recovers_beta():
    bars = synthetic_bars(4, beta=2.0)
    profile = intraday_profile(bars, bucket_minutes=MINUTES_PER_DAY)
    fit = fit_impact(bars, profile, n_boot=50)
    assert fit.beta_bps == pytest.approx(2.0, rel=0.1)
    assert 0 < fit.std_error < 0.5
    assert fit.num_bars == 4 * (MINUTES_PER_DAY - 1)


def test_profile_shapes_and_values():
    bars = synthetic_bars(2, 1.0)
    profile = intraday_profile(bars, bucket_minutes=60)
    assert profile.volume.shape == (MINUTES_PER_DAY,)
    assert np.allclose(profile.half_spread_bps, 0.6)
    assert np.all(profile.sigma_bps > 0)
    volume, spread, sigma = profile.window(1430, 20)  # wraps past midnight
    assert volume.shape == spread.shape == sigma.shape == (20,)
    assert len(profile.as_frame()) == MINUTES_PER_DAY
    with pytest.raises(ValueError):
        intraday_profile(bars, bucket_minutes=7)


def test_lot_size_rule():
    assert lot_size_for(13.6, 10_000) == pytest.approx(0.001)
    assert lot_size_for(1156.6, 10_000) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        lot_size_for(0, 10)


def test_calibrate_and_cost_model():
    bars = synthetic_bars(2, 1.5)
    cal = calibrate("X", bars, smallest_order_pct_adv=0.1, min_lots=100)
    assert cal.days == ("2026-07-22", "2026-07-23")
    assert cal.adv == pytest.approx(bars.volume.sum() / 2)
    model = cal.cost_model(60, 30, impact_multiple=2.0, risk_aversion=0.01)
    assert model.num_minutes == 30
    assert model.impact_bps == pytest.approx(2 * cal.impact.beta_bps)
    assert np.allclose(model.expected_volume, cal.profile.volume[60:90] / cal.lot_size)
    summary = cal.summary()
    assert summary["num_days"] == 2 and summary["lot_size"] == cal.lot_size
