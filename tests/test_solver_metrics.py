"""Approximation ratio and optimality gap for signed QUBO energies."""

import numpy as np
import pytest

from qexec.optimization.ising import binary_to_spins, qubo_to_ising
from qexec.optimization.solvers.exact import BruteForceSolver
from qexec.optimization.solvers.metrics import (
    EnergyBounds,
    approximation_ratio,
    energy_bounds,
    enumerate_energies,
    optimality_gap,
)


def _negative_qubo() -> np.ndarray:
    # Every non-empty assignment has negative energy, like the penalty-dominated execution QUBOs.
    return np.array([[-10.0, 1.0, 0.5], [1.0, -8.0, 0.5], [0.5, 0.5, -6.0]])


def test_bounds_match_exhaustive_search():
    rng = np.random.default_rng(0)
    Q = rng.standard_normal((8, 8))
    Q = (Q + Q.T) / 2
    bounds = energy_bounds(Q)
    energies = [
        float(x @ Q @ x) for x in (np.array([(i >> b) & 1 for b in range(8)]) for i in range(256))
    ]
    assert bounds.min_energy == pytest.approx(min(energies))
    assert bounds.max_energy == pytest.approx(max(energies))
    assert bounds.min_energy == pytest.approx(BruteForceSolver().solve(Q).energy)


def test_enumeration_order_matches_bit_index():
    Q = _negative_qubo()
    energies = enumerate_energies(Q)
    x = np.array([1, 0, 1])  # integer 5
    assert energies[5] == pytest.approx(x @ Q @ x)


def test_suboptimal_negative_energy_has_ratio_below_one():
    Q = _negative_qubo()
    bounds = energy_bounds(Q)
    suboptimal = sorted(set(np.round(enumerate_energies(Q), 9)))[1]
    assert bounds.min_energy < suboptimal < 0
    # The old metric min(E_opt / E, 1) reports a perfect score here.
    assert min(bounds.min_energy / suboptimal, 1.0) == 1.0
    assert approximation_ratio(suboptimal, bounds) < 1.0
    assert optimality_gap(suboptimal, bounds) > 0.0


def test_ratio_endpoints_and_monotonicity():
    bounds = EnergyBounds(min_energy=-20.0, max_energy=5.0)
    assert approximation_ratio(-20.0, bounds) == 1.0
    assert approximation_ratio(5.0, bounds) == 0.0
    assert approximation_ratio(-15.0, bounds) > approximation_ratio(-10.0, bounds)
    assert optimality_gap(-20.0, bounds) == 0.0
    assert optimality_gap(-15.0, bounds) == pytest.approx(0.25)


def test_degenerate_bounds():
    bounds = EnergyBounds(min_energy=0.0, max_energy=0.0)
    assert approximation_ratio(0.0, bounds) == 1.0
    assert optimality_gap(0.5, bounds) == 0.5


def test_qubo_to_ising_preserves_energy():
    rng = np.random.default_rng(1)
    Q = rng.standard_normal((6, 6))
    ising = qubo_to_ising(Q)
    for _ in range(50):
        x = rng.integers(0, 2, 6)
        assert ising.evaluate(binary_to_spins(x)) == pytest.approx(x @ Q @ x)
