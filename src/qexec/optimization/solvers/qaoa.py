"""QAOA for QUBO problems with a pluggable sampler (ideal Aer, noisy Aer or IBM hardware).

QUBO -> Ising -> p-layer QAOA circuit; the 2p angles are optimised classically
(COBYLA by default) on the sampled mean energy, then the optimised circuit is
sampled once more and the lowest-energy measured bitstring is returned.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit, transpile
from qiskit.providers import BackendV2
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

from qexec.optimization.ising import build_qaoa_circuit_from_ising, qubo_to_ising

logger = logging.getLogger(__name__)

Counts = dict[str, int]
Sampler = Callable[[QuantumCircuit, int], Counts]


@dataclass
class QAOAResult:
    """Lowest-energy sampled solution and the variational run that produced it."""

    solution: NDArray[np.int_]
    energy: float
    optimal_params: NDArray[np.float64]
    num_iterations: int
    solve_time: float
    counts: Counts
    history: list[float]
    success_probability: float


def bitstring_to_binary(bitstring: str, n: int) -> NDArray[np.int_]:
    """Qiskit bitstring (qubit 0 rightmost) to the binary vector of the first n qubits."""
    return np.array([int(b) for b in bitstring[::-1]])[:n]


def expected_energy(counts: Counts, Q: NDArray[np.float64]) -> float:
    """Sample mean of x^T Q x over the measured bitstrings."""
    n = Q.shape[0]
    total = sum(counts.values())
    energy = 0.0
    for bitstring, count in counts.items():
        binary = bitstring_to_binary(bitstring, n)
        energy += float(binary @ Q @ binary) * count / total
    return energy


def best_bitstring(counts: Counts, Q: NDArray[np.float64]) -> tuple[str | None, float]:
    """Lowest-energy measured bitstring and its energy; (None, inf) for empty counts."""
    n = Q.shape[0]
    best_bs = None
    best_energy = float("inf")
    for bitstring in counts:
        binary = bitstring_to_binary(bitstring, n)
        energy = float(binary @ Q @ binary)
        if energy < best_energy:
            best_energy = energy
            best_bs = bitstring
    return best_bs, best_energy


def aer_sampler(backend: BackendV2, seed: int | None = None) -> Sampler:
    """Sampler that transpiles for and runs on an Aer backend.

    With `seed`, each call passes the next seed of a generator seeded with it as
    `seed_simulator`, so shot noise is reproducible.
    """
    rng = np.random.default_rng(seed)

    def sample(circuit: QuantumCircuit, shots: int) -> Counts:
        compiled = transpile(circuit, backend)
        options = {} if seed is None else {"seed_simulator": int(rng.integers(2**31))}
        counts: Counts = backend.run(compiled, shots=shots, **options).result().get_counts()
        return counts

    return sample


def run_qaoa(
    Q: NDArray[np.float64],
    p: int,
    sample: Sampler,
    *,
    shots: int,
    maxiter: int,
    final_shots: int,
    rng: np.random.Generator,
    method: str = "COBYLA",
    on_iteration: Callable[[int, float], None] | None = None,
) -> QAOAResult:
    """Optimise the QAOA angles for `Q` and sample the final circuit with `final_shots` shots.

    Initial angles: gammas ~ U[0, 2pi), betas ~ U[0, pi) drawn from `rng`.
    `on_iteration(k, energy)` is called after each expectation-value evaluation.
    """
    start = perf_counter()
    ising = qubo_to_ising(Q)
    history: list[float] = []

    def circuit(params: NDArray[np.float64], stage: str) -> QuantumCircuit:
        qc = build_qaoa_circuit_from_ising(ising, params[:p], params[p:], p)
        qc.metadata = {
            "stage": stage,
            "gammas": [float(g) for g in params[:p]],
            "betas": [float(b) for b in params[p:]],
        }
        return qc

    def cost(params: NDArray[np.float64]) -> float:
        energy = expected_energy(sample(circuit(params, "optimize"), shots), Q)
        history.append(energy)
        if on_iteration is not None:
            on_iteration(len(history), energy)
        return energy

    x0 = np.concatenate([rng.uniform(0, 2 * np.pi, p), rng.uniform(0, np.pi, p)])
    opt = minimize(cost, x0, method=method, options={"maxiter": maxiter})

    counts = sample(circuit(opt.x, "final"), final_shots)
    best_bs, best_energy = best_bitstring(counts, Q)
    n = Q.shape[0]
    if best_bs is None:
        solution = np.zeros(n, dtype=int)
        success_prob = 0.0
    else:
        solution = bitstring_to_binary(best_bs, n)
        success_prob = counts[best_bs] / sum(counts.values())

    return QAOAResult(
        solution=solution,
        energy=best_energy,
        optimal_params=opt.x,
        num_iterations=len(history),
        solve_time=perf_counter() - start,
        counts=counts,
        history=history,
        success_probability=success_prob,
    )


class QAOASolver:
    """QAOA on the ideal Aer simulator; final sampling uses 10x `shots`."""

    def __init__(
        self,
        p: int = 2,
        shots: int = 1000,
        maxiter: int = 100,
        optimizer: str = "COBYLA",
        seed: int | None = None,
    ) -> None:
        self.p = p
        self.shots = shots
        self.maxiter = maxiter
        self.optimizer = optimizer
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def solve(self, Q: NDArray[np.float64]) -> QAOAResult:
        def report(iteration: int, energy: float) -> None:
            if iteration % 10 == 0:
                logger.debug("QAOA iteration %d: E=%.4f", iteration, energy)

        result = run_qaoa(
            Q,
            p=self.p,
            sample=aer_sampler(AerSimulator(), seed=self.seed),
            shots=self.shots,
            maxiter=self.maxiter,
            final_shots=self.shots * 10,
            rng=self.rng,
            method=self.optimizer,
            on_iteration=report,
        )
        logger.info(
            "QAOA p=%d n=%d: best %.4f, success probability %.3f, %.2fs",
            self.p,
            Q.shape[0],
            result.energy,
            result.success_probability,
            result.solve_time,
        )
        return result
