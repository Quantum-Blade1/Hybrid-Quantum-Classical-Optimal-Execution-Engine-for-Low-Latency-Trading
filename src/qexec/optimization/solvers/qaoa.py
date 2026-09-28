"""
QAOA solver for QUBO problems on the Qiskit Aer simulator.

QUBO -> Ising -> parameterised QAOA circuit; parameters are optimised
classically (COBYLA by default) against the sampled expectation value.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from time import time

from qexec.optimization.ising import (
    qubo_to_ising, 
    IsingHamiltonian,
    build_qaoa_circuit_from_ising,
    binary_to_spins,
    spins_to_binary
)


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
# QAOA Solver
# =============================================================================

class QAOASolver:
    """
    QAOA solver for QUBO problems using Qiskit.
    
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
        from qiskit import transpile
        from scipy.optimize import minimize
        
        start_time = time()
        n = Q.shape[0]
        
        if verbose:
            print(f"\n{'='*60}")
            print(f" QAOA Solver (p={self.p}, n={n})")
            print(f"{'='*60}")
        
        # Convert to Ising
        ising = qubo_to_ising(Q)
        
        if verbose:
            print(f" Converted to Ising: {ising}")
        
        # Cost function for optimizer
        history = []
        iteration = [0]
        simulator = AerSimulator()
        
        def qaoa_cost(params):
            gammas = params[:self.p]
            betas = params[self.p:]
            
            # Build and run circuit
            qc = build_qaoa_circuit_from_ising(ising, gammas, betas, self.p)
            compiled = transpile(qc, simulator)
            result = simulator.run(compiled, shots=self.shots).result()
            counts = result.get_counts()
            
            # Calculate expected energy
            exp_energy = 0.0
            for bitstring, count in counts.items():
                # Convert bitstring to binary (reverse for Qiskit)
                binary = np.array([int(b) for b in bitstring[::-1]])
                qubo_val = binary @ Q @ binary
                exp_energy += qubo_val * count / self.shots
            
            history.append(exp_energy)
            iteration[0] += 1
            
            if verbose and iteration[0] % 10 == 0:
                print(f"   Iter {iteration[0]}: E={exp_energy:.4f}")
            
            return exp_energy
        
        # Initial parameters
        x0 = np.concatenate([
            self.rng.uniform(0, 2*np.pi, self.p),  # gammas
            self.rng.uniform(0, np.pi, self.p)      # betas
        ])
        
        if verbose:
            print(f" Optimizing {2*self.p} parameters...")
        
        # Optimize
        result = minimize(
            qaoa_cost,
            x0,
            method=self.optimizer,
            options={'maxiter': self.maxiter}
        )
        
        optimal_params = result.x
        
        # Get final measurements with more shots
        gammas = optimal_params[:self.p]
        betas = optimal_params[self.p:]
        
        qc = build_qaoa_circuit_from_ising(ising, gammas, betas, self.p)
        compiled = transpile(qc, simulator)
        final_result = simulator.run(compiled, shots=self.shots * 10).result()
        counts = final_result.get_counts()
        
        # Find best solution
        best_bitstring = None
        best_energy = float('inf')
        
        for bitstring, count in counts.items():
            binary = np.array([int(b) for b in bitstring[::-1]])
            energy = binary @ Q @ binary
            if energy < best_energy:
                best_energy = energy
                best_bitstring = bitstring
        
        best_solution = np.array([int(b) for b in best_bitstring[::-1]])
        
        # Calculate success probability
        total_counts = sum(counts.values())
        success_count = counts.get(best_bitstring, 0)
        success_prob = success_count / total_counts
        
        solve_time = time() - start_time
        
        if verbose:
            print(f"\n Optimization complete!")
            print(f" Best energy: {best_energy:.4f}")
            print(f" Success probability: {success_prob:.1%}")
            print(f" Time: {solve_time:.2f}s")
        
        return QAOAResult(
            solution=best_solution,
            energy=best_energy,
            optimal_params=optimal_params,
            num_iterations=iteration[0],
            solve_time=solve_time,
            counts=counts,
            history=history,
            success_probability=success_prob
        )
