"""Exact, simulated-annealing and greedy QUBO solvers against exhaustive enumeration."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from qexec.execution.cost_model import CostModel
from qexec.optimization.slice_program import SliceProgram
from qexec.optimization.solvers.annealing import (
    SimulatedAnnealingSolver,
    default_temperatures,
    flip_delta,
)
from qexec.optimization.solvers.compare import compare_solvers
from qexec.optimization.solvers.exact import BruteForceSolver
from qexec.optimization.solvers.greedy import GreedySolver
from qexec.optimization.solvers.metrics import enumerate_energies
from qexec.optimization.toy import toy_execution_qubo


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


def test_simulated_annealing_runs_exactly_num_sweeps_on_a_geometric_schedule():
    Q = random_symmetric(6, 0)
    solver = SimulatedAnnealingSolver(initial_temp=5.0, final_temp=0.05, num_sweeps=250, seed=0)
    temps = solver.temperatures(Q)
    assert temps.size == 250
    assert temps[0] == pytest.approx(5.0) and temps[-1] == pytest.approx(0.05)
    assert np.allclose(temps[1:] / temps[:-1], (0.05 / 5.0) ** (1 / 249))
    result = solver.solve(Q)
    assert result.iterations == 250
    assert len(result.history) == 251


def test_default_temperatures_bracket_the_single_flip_deltas():
    Q = random_symmetric(8, 3)
    t0, tf = default_temperatures(Q)
    rng = np.random.default_rng(0)
    deltas = [
        abs(flip_delta(Q, rng.integers(0, 2, 8).astype(np.int8), i))
        for i in range(8)
        for _ in range(20)
    ]
    assert tf < t0
    assert max(deltas) <= t0 * np.log(2) + 1e-12


def test_simulated_annealing_validates_arguments():
    with pytest.raises(ValueError):
        SimulatedAnnealingSolver(num_sweeps=0)
    with pytest.raises(ValueError):
        SimulatedAnnealingSolver(num_restarts=0)
    with pytest.raises(ValueError):
        SimulatedAnnealingSolver(initial_temp=1.0, final_temp=2.0)


@pytest.mark.parametrize("n", [8, 12, 16])
def test_simulated_annealing_with_restarts_reaches_toy_optimum(n):
    Q = toy_execution_qubo(n)
    optimum = enumerate_energies(Q).min()
    for seed in range(3):
        result = SimulatedAnnealingSolver(num_sweeps=1000, num_restarts=16, seed=seed).solve(Q)
        assert result.energy == pytest.approx(optimum, abs=1e-9)


@pytest.mark.parametrize(
    ("slices", "bits", "units", "lam"),
    [(4, 3, 12, 0.0), (5, 3, 20, 0.0), (5, 3, 20, 5e-3), (4, 4, 24, 0.0), (5, 4, 30, 1e-3)],
)
def test_simulated_annealing_reaches_the_slice_program_optimum(slices, bits, units, lam):
    rng = np.random.default_rng(slices * 100 + units)
    minutes = 30
    model = CostModel(
        expected_volume=rng.uniform(500, 3000, minutes),
        half_spread_bps=rng.uniform(0, 1.5, minutes),
        sigma_bps=rng.uniform(2, 6, minutes),
        impact_bps=1.5,
        risk_aversion=lam,
    )
    program = SliceProgram(model, total=20_000, num_slices=slices, units=units, bits=bits)
    _, optimum = program.solve_dp()
    problem = program.qubo()
    for seed in range(3):
        result = SimulatedAnnealingSolver(num_sweeps=1000, num_restarts=64, seed=seed).solve(
            problem.Q
        )
        counts = program.decode(result.solution)
        assert program.is_feasible(counts)
        assert program.objective(counts) == pytest.approx(optimum, rel=1e-9, abs=1e-12)
