"""Approximation ratio and optimality gap for signed QUBO energies."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from qexec.optimization.solvers.metrics import (
    EnergyBounds,
    approximation_ratio,
    energy_bounds,
    enumerate_energies,
    optimality_gap,
)


def random_symmetric(n: int, seed: int, shift: float = 0.0) -> np.ndarray:
    Q = np.random.default_rng(seed).standard_normal((n, n))
    return (Q + Q.T) / 2 + shift * np.eye(n)


def test_enumeration_index_bit_k_is_variable_k(small_qubo):
    x = np.array([1, 0, 1, 1, 0, 0])  # integer 1 + 4 + 8 = 13
    assert enumerate_energies(small_qubo)[13] == pytest.approx(x @ small_qubo @ x)


# shift < 0 makes most energies negative (like the penalty-dominated execution QUBOs).
@given(n=st.integers(2, 8), seed=st.integers(0, 2**32 - 1), shift=st.sampled_from([-5.0, 0, 5]))
def test_ratio_and_gap_properties_over_all_assignments(n, seed, shift):
    Q = random_symmetric(n, seed, shift)
    bounds = energy_bounds(Q)
    energies = enumerate_energies(Q)
    assert bounds.min_energy == pytest.approx(energies.min())
    assert bounds.max_energy == pytest.approx(energies.max())

    ratios = np.array([approximation_ratio(e, bounds) for e in energies])
    gaps = np.array([optimality_gap(e, bounds) for e in energies])
    assert np.all((ratios >= -1e-12) & (ratios <= 1 + 1e-12))
    assert np.all(gaps >= -1e-12)
    assert approximation_ratio(bounds.min_energy, bounds) == 1.0
    assert approximation_ratio(bounds.max_energy, bounds) == 0.0
    assert optimality_gap(bounds.min_energy, bounds) == 0.0

    suboptimal = energies > bounds.min_energy + 1e-9
    assert np.all(ratios[suboptimal] < 1.0)
    assert np.all(gaps[suboptimal] > 0.0)
    # Monotone: lower energy never has a lower ratio.
    order = np.argsort(energies)
    assert np.all(np.diff(ratios[order]) <= 1e-12)


def test_ratio_is_below_one_for_suboptimal_negative_energy():
    # Regression: the old metric min(E_opt / E, 1) scored this 1.0 (claims audit F1).
    bounds = EnergyBounds(min_energy=-248.9, max_energy=10.0)
    assert min(bounds.min_energy / -248.7, 1.0) == 1.0
    assert approximation_ratio(-248.7, bounds) < 1.0
    assert optimality_gap(-248.7, bounds) == pytest.approx(0.2 / 248.9)


def test_ratio_and_gap_hand_values():
    bounds = EnergyBounds(min_energy=-20.0, max_energy=5.0)
    assert approximation_ratio(-15.0, bounds) == pytest.approx(0.8)
    assert optimality_gap(-15.0, bounds) == pytest.approx(0.25)


def test_degenerate_bounds_score_every_energy_as_optimal():
    bounds = EnergyBounds(min_energy=0.0, max_energy=0.0)
    assert approximation_ratio(0.0, bounds) == 1.0
    # E_min = 0: the gap is absolute rather than relative.
    assert optimality_gap(0.5, bounds) == 0.5


def test_counts_quality_counts_every_optimal_bitstring():
    # Two degenerate optima (x = 01 and 10 with Q = [[-1, 1], [1, -1]]: E = -1, E(11) = 0).
    from qexec.optimization.solvers.metrics import counts_quality, energy_bounds

    Q = np.array([[-1.0, 1.0], [1.0, -1.0]])
    bounds = energy_bounds(Q)
    q = counts_quality({"01": 30, "10": 20, "11": 40, "00": 10}, Q, bounds)
    assert q.success_probability == pytest.approx(0.5)
    assert q.mean_energy == pytest.approx((30 * -1 + 20 * -1 + 0 + 0) / 100)
    assert q.best_energy == -1.0 and q.shots == 100


def test_random_baseline_is_exact_for_uniform_sampling():
    from qexec.optimization.solvers.metrics import random_sampling_baseline

    Q = np.array([[-1.0, 1.0], [1.0, -1.0]])
    base = random_sampling_baseline(Q, shots=1000, rng=np.random.default_rng(0))
    assert base.num_optimal == 2
    assert base.success_probability == 0.5
    assert base.mean_energy == pytest.approx((0 - 1 - 1 + 0) / 4)
    assert base.best_energy == -1.0
