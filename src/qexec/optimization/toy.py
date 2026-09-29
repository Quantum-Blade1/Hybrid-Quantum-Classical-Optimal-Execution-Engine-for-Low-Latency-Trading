import numpy as np
from numpy.typing import NDArray


def toy_execution_qubo(n_qubits: int) -> NDArray[np.float64]:
    """Fixed toy QUBO behind results/bench_*.json; changing it invalidates those results."""
    Q = np.zeros((n_qubits, n_qubits))
    num_levels = 2
    num_slices = n_qubits // num_levels
    impact_coeff = 0.1
    timing_coeff = 0.05
    penalty = 10.0

    for t in range(num_slices):
        for k in range(num_levels):
            i = t * num_levels + k
            q = k + 1
            Q[i, i] += impact_coeff * q * q
            Q[i, i] += timing_coeff * (t + 1) * q

    for i in range(n_qubits):
        q_i = i % num_levels + 1
        Q[i, i] += penalty * q_i * q_i - 2 * penalty * num_slices * q_i
        for j in range(i + 1, n_qubits):
            q_j = j % num_levels + 1
            Q[i, j] += 2 * penalty * q_i * q_j

    return (Q + Q.T) / 2
