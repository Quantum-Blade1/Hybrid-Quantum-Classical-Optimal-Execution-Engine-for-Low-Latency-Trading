"""Exact, simulated-annealing and greedy QUBO solvers against exhaustive enumeration."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from qexec.optimization.solvers.annealing import SimulatedAnnealingSolver, flip_delta
from qexec.optimization.solvers.compare import compare_solvers
from qexec.optimization.solvers.exact import BruteForceSolver
from qexec.optimization.solvers.greedy import GreedySolver
from qexec.optimization.solvers.metrics import enumerate_energies


def random_symmetric(n: int, seed: int) -> np.ndarray:
    Q = np.random.default_rng(seed).standard_normal((n, n))
    return (Q + Q.T) / 2


@given(n=st.integers(1, 8), seed=st.integers(0, 2**32 - 1))
def test_exact_solver_returns_enumerated_minimum(n, seed):
    Q = random_symmetric(n, seed)
    result = BruteForceSolver().solve(Q)
    x = result.solution
    assert result.energy == pytest.approx(enumerate_energies(Q).min())
    assert result.energy == pytest.approx(float(x @ Q @ x))


def test_exact_solver_refuses_problems_above_its_limit():
    with pytest.raises(ValueError, match="exceeds max_variables"):
        BruteForceSolver(max_variables=4).solve(np.zeros((5, 5)))


@pytest.mark.parametrize("n", [4, 6, 8, 10])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_simulated_annealing_finds_optimum_on_small_problems(n, seed):
    Q = random_symmetric(n, seed)
    result = SimulatedAnnealingSolver(num_sweeps=300, seed=seed).solve(Q)
    x = result.solution
    assert result.energy == pytest.approx(float(x @ Q @ x))
    assert result.energy == pytest.approx(enumerate_energies(Q).min())


def test_simulated_annealing_best_energy_history_is_non_increasing(small_qubo):
    history = SimulatedAnnealingSolver(num_sweeps=100, seed=0).solve(small_qubo).history
    assert history is not None
    assert np.all(np.diff(history) <= 0)


def test_simulated_annealing_does_not_mutate_initial_solution(small_qubo):
    x0 = np.ones(6, dtype=np.int8)
    SimulatedAnnealingSolver(num_sweeps=50, seed=0).solve(small_qubo, initial_solution=x0)
    assert x0.tolist() == [1] * 6


@given(n=st.integers(1, 10), seed=st.integers(0, 2**32 - 1), bit=st.integers(0, 9))
def test_flip_delta_equals_energy_difference(n, seed, bit):
    rng = np.random.default_rng(seed)
    Q = random_symmetric(n, seed)
    x = rng.integers(0, 2, n).astype(np.int8)
    i = bit % n
    flipped = x.copy()
    flipped[i] = 1 - flipped[i]
    assert flip_delta(Q, x, i) == pytest.approx(
        float(flipped @ Q @ flipped) - float(x @ Q @ x), abs=1e-9
    )


@pytest.mark.parametrize("seed", range(5))
def test_greedy_result_is_a_single_flip_local_minimum(seed):
    Q = random_symmetric(10, seed)
    result = GreedySolver(seed=seed).solve(Q)
    x = result.solution
    assert result.energy == pytest.approx(float(x @ Q @ x))
    for i in range(10):
        assert flip_delta(Q, x, i) >= -1e-9


def test_no_default_solver_beats_the_exact_optimum():
    Q = random_symmetric(10, 7)
    results = compare_solvers(Q)
    exact = next(r for r in results if r.solver_name == "BruteForce")
    assert len(results) == 4
    assert all(r.energy >= exact.energy - 1e-9 for r in results)
