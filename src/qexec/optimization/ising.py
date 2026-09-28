"""QUBO <-> Ising mapping and the QAOA circuit for an Ising cost Hamiltonian.

With x_i = (1 - z_i)/2 and symmetric Q, x^T Q x = sum_i h_i z_i + sum_{i<j} J_ij z_i z_j + c,
where h_i = -(sum_j Q_ij)/2, J_ij = Q_ij/2 and c = (tr Q + sum_{i<j} Q_ij)/2.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit

_COEFF_TOL = 1e-10


@dataclass
class IsingHamiltonian:
    """H = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j + offset, with J upper triangular."""

    h: NDArray[np.float64]
    J: NDArray[np.float64]
    offset: float
    num_qubits: int

    def evaluate(self, spins: NDArray[np.int_]) -> float:
        """Energy of a +/-1 spin configuration."""
        s = np.asarray(spins, dtype=np.float64)
        return float(self.h @ s + s @ np.triu(self.J, k=1) @ s + self.offset)

    def __repr__(self) -> str:
        num_linear = int(np.sum(np.abs(self.h) > _COEFF_TOL))
        num_quadratic = int(np.sum(np.abs(np.triu(self.J, k=1)) > _COEFF_TOL))
        return (
            f"IsingHamiltonian(n={self.num_qubits}, "
            f"linear_terms={num_linear}, quadratic_terms={num_quadratic}, "
            f"offset={self.offset:.4f})"
        )


def qubo_to_ising(Q: NDArray[np.float64]) -> IsingHamiltonian:
    """Ising form of min x^T Q x (Q is symmetrised first)."""
    n = Q.shape[0]
    Q_sym = (Q + Q.T) / 2
    offset = float(np.trace(Q_sym) / 2 + np.sum(np.triu(Q_sym, k=1)) / 2)
    h = -Q_sym.sum(axis=1) / 2
    J = np.triu(Q_sym, k=1) / 2
    return IsingHamiltonian(h=h, J=J, offset=offset, num_qubits=n)


def build_qaoa_circuit_from_ising(
    ising: IsingHamiltonian,
    gamma: float | Sequence[float] | NDArray[np.float64],
    beta: float | Sequence[float] | NDArray[np.float64],
    p: int = 1,
) -> QuantumCircuit:
    """p-layer QAOA circuit on |+>^n with measurement of every qubit.

    Layer k applies exp(-i gamma_k H_C) as RZ(2 gamma_k h_i) and RZZ(2 gamma_k J_ij),
    then the mixer exp(-i beta_k sum X_i) as RX(2 beta_k). Scalar angles are reused per layer.
    """
    gammas = np.broadcast_to(np.asarray(gamma, dtype=np.float64), (p,))
    betas = np.broadcast_to(np.asarray(beta, dtype=np.float64), (p,))
    n = ising.num_qubits
    qc = QuantumCircuit(n, n)
    for i in range(n):
        qc.h(i)
    qc.barrier()

    for g, b in zip(gammas, betas, strict=True):
        for i in range(n):
            if abs(ising.h[i]) > _COEFF_TOL:
                qc.rz(2 * g * ising.h[i], i)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(ising.J[i, j]) > _COEFF_TOL:
                    qc.rzz(2 * g * ising.J[i, j], i, j)
        qc.barrier()
        for i in range(n):
            qc.rx(2 * b, i)
        qc.barrier()

    qc.measure(range(n), range(n))
    return qc


def binary_to_spins(binary: NDArray[np.int_]) -> NDArray[np.int_]:
    """z_i = 1 - 2 x_i."""
    return (1 - 2 * np.asarray(binary)).astype(int)
