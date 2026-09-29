import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from qiskit.quantum_info import Statevector

from qexec.optimization.ising import binary_to_spins, build_qaoa_circuit_from_ising, qubo_to_ising
from qexec.optimization.solvers.qaoa import bitstring_to_binary

square_matrices = st.integers(1, 8).flatmap(
    lambda n: arrays(np.float64, (n, n), elements=st.floats(-10, 10, allow_subnormal=False))
)


@given(Q=square_matrices)
def test_ising_energy_matches_qubo_energy_for_every_bitstring(Q, bitstrings):
    # Q need not be symmetric: x^T Q x only sees (Q + Q^T)/2, and neither may the mapping.
    ising = qubo_to_ising(Q)
    for x in bitstrings(Q.shape[0]):
        assert ising.evaluate(binary_to_spins(x)) == pytest.approx(
            float(x @ Q @ x), rel=1e-9, abs=1e-9
        )


@pytest.mark.parametrize("gamma", [0.3, 1.1])
def test_qaoa_cost_layer_applies_phase_exp_minus_i_gamma_energy(small_qubo, bitstrings, gamma):
    # beta = 0: amplitude of x is 2^{-n/2} exp(-i gamma (E(x) - offset)) up to a global phase.
    Q = small_qubo[:4, :4]
    ising = qubo_to_ising(Q)
    qc = build_qaoa_circuit_from_ising(ising, gamma=gamma, beta=0.0, p=1)
    qc.remove_final_measurements()
    amplitudes = Statevector(qc).data

    n = Q.shape[0]
    xs = bitstrings(n)  # row i is the basis state with qubit k = bit k of i (Qiskit order)
    energies = np.array([float(x @ Q @ x) for x in xs]) - ising.offset
    expected = np.exp(-1j * gamma * energies) / np.sqrt(2**n)
    global_phase = amplitudes[0] / expected[0]
    np.testing.assert_allclose(amplitudes, global_phase * expected, atol=1e-9)


def test_bitstring_to_binary_reads_qubit_zero_from_the_right():
    # Qiskit prints qubit 0 as the rightmost character.
    assert bitstring_to_binary("0011", 4).tolist() == [1, 1, 0, 0]
    assert bitstring_to_binary("1000", 4).tolist() == [0, 0, 0, 1]
