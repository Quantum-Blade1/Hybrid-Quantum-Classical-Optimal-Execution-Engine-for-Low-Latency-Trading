"""Volatility/spread regime detection and the regime-dependent risk aversion."""

import numpy as np
import pytest

from qexec.microstructure.regime import (
    AdaptiveRiskManager,
    SpreadEstimator,
    SpreadRegime,
    VolatilityEstimator,
    VolatilityRegime,
)


def gbm_prices(rng: np.random.Generator, vols: np.ndarray, start: float = 100.0) -> np.ndarray:
    return start * np.exp(np.cumsum(vols * rng.standard_normal(len(vols))))


def test_volatility_regime_switches_on_a_volatility_step(rng):
    # The regime is the ratio of a fast to a slow EWMA volatility, i.e. a change detector:
    # a 5x volatility step must register as stressed within a few ticks.
    vols = np.r_[np.full(400, 1e-4), np.full(100, 5e-4)]
    estimator = VolatilityEstimator()
    regimes = []
    for price in gbm_prices(rng, vols):
        estimator.update(price)
        regimes.append(estimator.detect_regime())
    assert set(regimes[100:400]) == {VolatilityRegime.NORMAL}
    stressed = {VolatilityRegime.HIGH, VolatilityRegime.EXTREME}
    first_stressed = next(i for i, r in enumerate(regimes) if r in stressed)
    assert 400 <= first_stressed < 415
    assert VolatilityRegime.EXTREME in regimes[400:440]


def test_volatility_estimates_converge_to_the_true_volatility(rng):
    estimator = VolatilityEstimator()
    for price in gbm_prices(rng, np.full(3000, 2e-4)):
        estimator.update(price)
    assert estimator.slow_vol == pytest.approx(2e-4, rel=0.15)
    assert estimator.detect_regime() is VolatilityRegime.NORMAL


def test_spread_regime_follows_spread_relative_to_baseline():
    estimator = SpreadEstimator()
    for _ in range(40):
        regime = estimator.update(99.99, 100.01)  # 2 bps
    assert regime is SpreadRegime.NORMAL
    for _ in range(10):
        regime = estimator.update(99.95, 100.05)  # 10 bps = 5x baseline
    assert regime is SpreadRegime.GAPPED


def test_risk_aversion_rises_after_a_volatility_step_and_stays_within_bounds(rng):
    manager = AdaptiveRiskManager(base_lambda=1.0, lambda_min=0.1, lambda_max=10.0)
    vols = np.r_[np.full(300, 1e-4), np.full(30, 8e-4)]
    lambdas, params = [], []
    for price in gbm_prices(rng, vols):
        state = manager.update(price, price - 0.01, price + 0.01, volume=1.0, avg_volume=1.0)
        lambdas.append(state.lambda_value)
        params.append(manager.get_qubo_params())
    assert all(0.1 <= lam <= 10.0 for lam in lambdas)
    assert lambdas[299] == pytest.approx(1.0)  # calm market, normal spread and volume
    assert max(lambdas[300:]) >= 2.0  # HIGH multiplier 2, EXTREME 4, EWMA-smoothed
    stressed = [p for p in params[300:] if p["vol_regime"] == "extreme"]
    assert stressed and all(
        p["impact_scale"] == 2.5 and p["timing_weight"] == 0.5 for p in stressed
    )
