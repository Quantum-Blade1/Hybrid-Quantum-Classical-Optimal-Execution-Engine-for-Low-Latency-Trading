"""QAOA driver: sample post-processing with a fake sampler, and seeded runs on Aer."""

import numpy as np
import pytest

from qexec.optimization.solvers.metrics import energy_bounds, enumerate_energies
from qexec.optimization.solvers.qaoa import (
    QAOASolver,
    best_bitstring,
    expected_energy,
    run_qaoa,
)
from qexec.optimization.toy import toy_execution_qubo

# Separable QUBO with the unique optimum x = (1, 0, 1, 0), energy -2.5.
TRIVIAL_QUBO = np.diag([-1.0, 2.0, -1.5, 2.0])


def uniform_counts(n: int, shots_per_state: int = 3) -> dict[str, int]:
    return {format(i, f"0{n}b"): shots_per_state for i in range(2**n)}


def test_sample_statistics_use_qiskit_bit_order(small_qubo):
    Q = small_qubo[:4, :4]
    energies = enumerate_energies(Q)  # index bit k = variable k = Qiskit qubit k
    counts = {format(i, "04b"): int(c) for i, c in enumerate(np.arange(1, 17))}
    weights = np.arange(1, 17) / np.arange(1, 17).sum()
    assert expected_energy(counts, Q) == pytest.approx(float(weights @ energies))
    best, energy = best_bitstring(counts, Q)
    assert energy == pytest.approx(energies.min())
    assert best == format(int(np.argmin(energies)), "04b")


def test_run_qaoa_reports_best_sample_and_its_frequency():
    # A sampler that ignores the circuit and returns every bitstring equally often.
    calls: list[tuple[str, int]] = []

    def sampler(circuit, shots):
        calls.append((circuit.metadata["stage"], shots))
        return uniform_counts(4)

    result = run_qaoa(
        TRIVIAL_QUBO,
        p=1,
        sample=sampler,
        shots=100,
        maxiter=5,
        final_shots=1000,
        rng=np.random.default_rng(0),
    )
    assert result.energy == -2.5
    assert result.solution.tolist() == [1, 0, 1, 0]
    assert result.success_probability == pytest.approx(1 / 16)
    uniform_mean = float(np.mean(enumerate_energies(TRIVIAL_QUBO)))
    assert result.history == pytest.approx([uniform_mean] * len(result.history))
    assert result.num_iterations == len(result.history) > 0
    assert calls[-1] == ("final", 1000)
    assert {stage for stage, _ in calls[:-1]} == {"optimize"}


def test_qaoa_on_aer_finds_optimum_of_trivial_qubo():
    result = QAOASolver(p=1, shots=256, maxiter=40, seed=0).solve(TRIVIAL_QUBO)
    assert result.energy == energy_bounds(TRIVIAL_QUBO).min_energy
    assert result.solution.tolist() == [1, 0, 1, 0]


@pytest.mark.slow
def test_seeded_qaoa_on_aer_is_reproducible():
    first = QAOASolver(p=1, shots=256, maxiter=30, seed=5).solve(TRIVIAL_QUBO)
    second = QAOASolver(p=1, shots=256, maxiter=30, seed=5).solve(TRIVIAL_QUBO)
    assert first.counts == second.counts
    np.testing.assert_array_equal(first.optimal_params, second.optimal_params)


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("Q", [TRIVIAL_QUBO, toy_execution_qubo(4)], ids=["trivial", "toy4"])
def test_qaoa_optimised_expectation_beats_uniform_sampling(Q, seed):
    # Finding the optimum among 10x shots samples of 16 states says little (claims audit F2);
    # the optimised <H> falling below the uniform-superposition mean shows the angles matter.
    result = QAOASolver(p=2, shots=512, maxiter=60, seed=seed).solve(Q)
    assert result.energy == energy_bounds(Q).min_energy
    assert min(result.history) < float(np.mean(enumerate_energies(Q)))
