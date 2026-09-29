import numpy as np
import pytest

from qexec.execution.strategies.almgren_chriss import ACConfig, AlmgrenChrissSolver

RISK_AVERSIONS = [0.0, 1e-8, 1e-6, 1e-4]


def simulate_costs(config: ACConfig, trades: np.ndarray, shocks: np.ndarray) -> np.ndarray:
    # S_k = S_{k-1} + sigma S_0 sqrt(tau) xi_k - rho n_k; fills at S_{k-1} - eta n_k / tau.
    tau = config.n_days / config.n_steps
    sigma_dollars = config.sigma * config.price
    steps = sigma_dollars * np.sqrt(tau) * shocks - config.rho * trades
    prior_prices = config.price + np.cumsum(steps, axis=1) - steps  # S_{k-1}
    fill_prices = prior_prices - config.eta * trades / tau
    return config.total_shares * config.price - (fill_prices * trades).sum(axis=1)


@pytest.mark.parametrize("risk_aversion", RISK_AVERSIONS)
def test_sell_trajectory_starts_at_x_ends_at_zero_and_is_monotone(risk_aversion):
    config = ACConfig(total_shares=10_000, n_steps=20, risk_aversion=risk_aversion)
    traj = AlmgrenChrissSolver(config).compute_trajectory()
    assert traj["shares_held_start"].iloc[0] == pytest.approx(10_000)
    assert traj["shares_held_end"].iloc[-1] == pytest.approx(0, abs=1e-6)
    assert traj["shares_to_trade"].sum() == pytest.approx(10_000)
    assert np.all(traj["shares_to_trade"] > 0)
    assert np.all(np.diff(traj["shares_to_trade"]) <= 1e-9)


def test_zero_risk_aversion_limit_is_twap():
    twap = np.full(10, 1000.0)
    for risk_aversion in (0.0, 1e-12):
        config = ACConfig(total_shares=10_000, n_steps=10, risk_aversion=risk_aversion)
        trades = AlmgrenChrissSolver(config).compute_trajectory()["shares_to_trade"]
        np.testing.assert_allclose(trades, twap, rtol=1e-4)


def test_higher_risk_aversion_front_loads_execution():
    first_trade = [
        AlmgrenChrissSolver(ACConfig(total_shares=10_000, risk_aversion=lam))
        .compute_trajectory()["shares_to_trade"]
        .iloc[0]
        for lam in RISK_AVERSIONS
    ]
    assert np.all(np.diff(first_trade) > 0)


@pytest.mark.parametrize("risk_aversion", [0.0, 1e-6, 1e-4])
def test_analytic_cost_moments_match_simulated_shortfall(risk_aversion):
    config = ACConfig(total_shares=50_000, n_steps=8, risk_aversion=risk_aversion)
    solver = AlmgrenChrissSolver(config)
    traj = solver.compute_trajectory()
    trades = traj["shares_to_trade"].to_numpy()[None, :]

    # The shortfall is linear in the shocks, so the zero-shock path gives E[C] exactly.
    noiseless = simulate_costs(config, trades, np.zeros_like(trades))[0]
    assert solver.calculate_expected_cost(traj) == pytest.approx(noiseless, rel=1e-10)

    shocks = np.random.default_rng(0).standard_normal((40_000, config.n_steps))
    costs = simulate_costs(config, np.repeat(trades, len(shocks), axis=0), shocks)
    variance = solver.calculate_variance(traj)
    assert costs.var() == pytest.approx(variance, rel=0.03)
    assert costs.mean() == pytest.approx(noiseless, abs=4 * np.sqrt(variance / len(shocks)))


def test_expected_cost_is_minimised_by_twap_when_risk_neutral():
    # With lambda = 0 the objective is E[C], which the linear schedule minimises.
    config = ACConfig(total_shares=10_000, n_steps=10, risk_aversion=0.0)
    solver = AlmgrenChrissSolver(config)
    twap = solver.compute_trajectory()
    front_loaded = AlmgrenChrissSolver(
        ACConfig(total_shares=10_000, n_steps=10, risk_aversion=1e-4)
    ).compute_trajectory()
    assert solver.calculate_expected_cost(twap) < solver.calculate_expected_cost(front_loaded)
    assert solver.calculate_variance(twap) > solver.calculate_variance(front_loaded)
