"""Simulated device noise for QAOA benchmarks (not calibrated to any IBM backend)."""

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error


def noisy_aer_backend(noise_level: float = 0.02) -> AerSimulator:
    """Aer simulator with depolarizing gate noise and asymmetric readout error.

    Depolarizing probability p on the 1-qubit gates (rx, rz, h) and 5p on the 2-qubit
    gates (rzz, cx); readout p(0|1) = p, p(1|0) = 0.8p.
    """
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(noise_level, 1), ["rx", "rz", "h"])
    noise_model.add_all_qubit_quantum_error(depolarizing_error(noise_level * 5, 2), ["rzz", "cx"])
    p_0_given_1 = noise_level
    p_1_given_0 = noise_level * 0.8
    noise_model.add_all_qubit_readout_error(
        ReadoutError([[1 - p_1_given_0, p_1_given_0], [p_0_given_1, 1 - p_0_given_1]])
    )
    return AerSimulator(noise_model=noise_model)
