import numpy as np
import pytest
from scipy import stats

from qexec.hardware.stats import (
    binomial_test_greater,
    binomial_test_less,
    bootstrap_ratio_advantage,
    count_energies,
    sign_test_greater,
    wilson_interval,
)
from qexec.optimization.solvers.metrics import EnergyBounds, energy_bounds
from qexec.optimization.toy import toy_execution_qubo


def test_wilson_interval_known_values():
    # Reference: 1872 / 10000 (ibm_fez n=4, run 1) and the textbook 0 / 10 case.
    low, high = wilson_interval(1872, 10_000)
    assert low == pytest.approx(0.17968, abs=1e-5)
    assert high == pytest.approx(0.19496, abs=1e-5)
    low, high = wilson_interval(0, 10)
    assert low == 0.0 and high == pytest.approx(0.2775, abs=1e-4)
    with pytest.raises(ValueError):
        wilson_interval(1, 0)
    with pytest.raises(ValueError):
        wilson_interval(5, 4)


def test_binomial_tests_are_one_sided_exact():
    assert binomial_test_greater(15, 10_000, 2**-10) == pytest.approx(
        stats.binom.sf(14, 10_000, 2**-10)
    )
    assert binomial_test_less(232, 10_000, 1 / 16) < 1e-50
    assert binomial_test_greater(232, 10_000, 1 / 16) == pytest.approx(1.0)
    # With 3 runs a sign test cannot go below 1/8.
    assert sign_test_greater(3, 3) == pytest.approx(0.125)
    assert sign_test_greater(0, 3) == pytest.approx(1.0)


def test_count_energies_use_qiskit_bit_order():
    Q = toy_execution_qubo(4)
    energies, weights = count_energies({"0101": 3, "1010": 1}, Q)
    x = np.array([1, 0, 1, 0])  # "0101" read right to left
    assert energies[0] == pytest.approx(x @ Q @ x)
    assert energies[0] == pytest.approx(energy_bounds(Q).min_energy)
    assert list(weights) == [3, 1]


def test_bootstrap_ratio_advantage():
    bounds = EnergyBounds(min_energy=0.0, max_energy=10.0)
    energies = np.array([0.0, 10.0])
    rng = np.random.default_rng(0)
    adv = bootstrap_ratio_advantage(
        energies, np.array([7_000, 3_000]), bounds, 0.5, resamples=2_000, rng=rng
    )
    assert adv.ratio == pytest.approx(0.7)
    assert adv.difference == pytest.approx(0.2)
    # Binomial standard error of the ratio: sqrt(0.7 * 0.3 / 10000) ~ 0.0046.
    assert adv.ci_low == pytest.approx(0.2 - 1.96 * 0.00458, abs=0.002)
    assert adv.ci_high == pytest.approx(0.2 + 1.96 * 0.00458, abs=0.002)
    # A point mass has no sampling spread; a degenerate energy range gives ratio 1.
    flat = bootstrap_ratio_advantage(
        np.array([3.0]), np.array([5]), EnergyBounds(3.0, 3.0), 1.0, resamples=10, rng=rng
    )
    assert flat.ci_low == flat.ci_high == 0.0
    with pytest.raises(ValueError):
        bootstrap_ratio_advantage(energies, np.array([0, 0]), bounds, 0.5, rng=rng)
