"""
QAOA for QUBO problems.

QUBO -> Ising -> parameterised QAOA circuit; parameters are optimised
classically (COBYLA by default) against the sampled expectation value, then
the circuit is sampled once more with more shots and the lowest-energy
bitstring is returned.

`run_qaoa` is the single variational loop. It takes a `sample` callable
(circuit, shots) -> counts, so the same loop runs on the ideal Aer
simulator (QAOASolver), on a noisy Aer backend and on IBM hardware
(qexec.hardware.ibm).
"""

import numpy as np
from typing import Callable, Dict, List, Optional
from dataclasses import dataclass
from time import time

from qexec.optimization.ising import qubo_to_ising, build_qaoa_circuit_from_ising


Counts = Dict[str, int]
Sampler = Callable[[object, int], Counts]


# =============================================================================
# QAOA Result
# =============================================================================

@dataclass
class QAOAResult:
    """Result from QAOA optimization."""
    solution: np.ndarray
    energy: float
    optimal_params: np.ndarray
    num_iterations: int
    solve_time: float
    counts: Dict[str, int]
    history: List[float]
    success_probability: float


# =============================================================================
# Counts post-processing
# =============================================================================

def bitstring_to_binary(bitstring: str, n: int) -> np.ndarray:
    """Qiskit bitstring (qubit 0 rightmost) -> binary vector of the first n qubits."""
    return np.array([int(b) for b in bitstring[::-1]])[:n]


def expected_energy(counts: Counts, Q: np.ndarray) -> float:
    """Sample mean of x^T Q x over measured bitstrings."""
    n = Q.shape[0]
    total = sum(counts.values())
    energy = 0.0
    for bitstring, count in counts.items():
        binary = bitstring_to_binary(bitstring, n)
        energy += float(binary @ Q @ binary) * count / total
    return energy


def best_bitstring(counts: Counts, Q: np.ndarray) -> tuple:
    """Lowest-energy measured bitstring: (bitstring, energy)."""
    n = Q.shape[0]
    best_bs = None
    best_energy = float('inf')
    for bitstring in counts:
        binary = bitstring_to_binary(bitstring, n)
        energy = float(binary @ Q @ binary)
        if energy < best_energy:
            best_energy = energy
            best_bs = bitstring
    return best_bs, best_energy


def aer_sampler(backend) -> Sampler:
    """Sampler that transpiles for and runs on an Aer backend."""
    from qiskit import transpile

    def sample(circuit, shots: int) -> Counts:
        compiled = transpile(circuit, backend)
        return backend.run(compiled, shots=shots).result().get_counts()

    return sample


# =============================================================================
# Variational loop
# =============================================================================

def run_qaoa(
    Q: np.ndarray,
    p: int,
    sample: Sampler,
    shots: int,
    maxiter: int,
    final_shots: int,
    rng: np.random.Generator,
    method: str = 'COBYLA',
    on_iteration: Optional[Callable[[int, float], None]] = None,
) -> QAOAResult:
    """
    Optimise QAOA parameters for QUBO matrix Q and sample the final circuit.

    Args:
        Q: QUBO matrix
        p: Number of QAOA layers
        sample: (circuit, shots) -> counts
        shots: Shots per expectation-value evaluation
        maxiter: Maximum classical optimizer iterations
        final_shots: Shots for the final sampling of the optimised circuit
        rng: Generator for the initial parameters (gammas in [0, 2pi), betas in [0, pi))
        method: scipy.optimize.minimize method
        on_iteration: Called with (iteration, expectation) after each evaluation
    """
    from scipy.optimize import minimize

    start_time = time()
    ising = qubo_to_ising(Q)
    history: List[float] = []

    def circuit(params, stage: str):
        qc = build_qaoa_circuit_from_ising(ising, params[:p], params[p:], p)
        qc.metadata = {"stage": stage, "gammas": [float(g) for g in params[:p]],
                       "betas": [float(b) for b in params[p:]]}
        return qc

    def cost(params):
        qc = circuit(params, "optimize")
        energy = expected_energy(sample(qc, shots), Q)
        history.append(energy)
        if on_iteration is not None:
            on_iteration(len(history), energy)
        return energy

    x0 = np.concatenate([
        rng.uniform(0, 2 * np.pi, p),  # gammas
        rng.uniform(0, np.pi, p)       # betas
    ])
    opt = minimize(cost, x0, method=method, options={'maxiter': maxiter})

    counts = sample(circuit(opt.x, "final"), final_shots)
    best_bs, best_energy = best_bitstring(counts, Q)
    success_prob = counts.get(best_bs, 0) / sum(counts.values()) if best_bs else 0.0

    return QAOAResult(
        solution=bitstring_to_binary(best_bs, Q.shape[0]) if best_bs else np.zeros(Q.shape[0], dtype=int),
        energy=best_energy,
        optimal_params=opt.x,
        num_iterations=len(history),
        solve_time=time() - start_time,
        counts=counts,
        history=history,
        success_probability=success_prob,
    )


# =============================================================================
# QAOA Solver (ideal Aer simulator)
# =============================================================================

class QAOASolver:
    """
    QAOA solver for QUBO problems on the ideal Aer simulator.

    Implements the full QAOA workflow:
    1. Convert QUBO to Ising Hamiltonian
    2. Build parameterized QAOA circuit
    3. Optimize parameters using classical optimizer
    4. Extract and decode solution
    """

    def __init__(
        self,
        p: int = 2,
        shots: int = 1000,
        maxiter: int = 100,
        optimizer: str = 'COBYLA',
        seed: Optional[int] = None
    ):
        """
        Initialize QAOA solver.

        Args:
            p: Number of QAOA layers (circuit depth)
            shots: Measurement shots per circuit
            maxiter: Maximum optimizer iterations
            optimizer: Classical optimizer ('COBYLA', 'SPSA', etc.)
            seed: Random seed
        """
        self.p = p
        self.shots = shots
        self.maxiter = maxiter
        self.optimizer = optimizer
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def solve(
        self,
        Q: np.ndarray,
        verbose: bool = True
    ) -> QAOAResult:
        """
        Solve QUBO using QAOA.

        Args:
            Q: QUBO matrix
            verbose: Print progress

        Returns:
            QAOAResult with solution and statistics
        """
        from qiskit_aer import AerSimulator

        n = Q.shape[0]

        if verbose:
            print(f"\n{'='*60}")
            print(f" QAOA Solver (p={self.p}, n={n})")
            print(f"{'='*60}")
            print(f" Converted to Ising: {qubo_to_ising(Q)}")
            print(f" Optimizing {2*self.p} parameters...")

        def report(iteration: int, energy: float) -> None:
            if verbose and iteration % 10 == 0:
                print(f"   Iter {iteration}: E={energy:.4f}")

        result = run_qaoa(
            Q,
            p=self.p,
            sample=aer_sampler(AerSimulator()),
            shots=self.shots,
            maxiter=self.maxiter,
            final_shots=self.shots * 10,
            rng=self.rng,
            method=self.optimizer,
            on_iteration=report,
        )

        if verbose:
            print(f"\n Optimization complete!")
            print(f" Best energy: {result.energy:.4f}")
            print(f" Success probability: {result.success_probability:.1%}")
            print(f" Time: {result.solve_time:.2f}s")

        return result
