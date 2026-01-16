"""
QAOA Solver for Execution QUBO

This module solves the execution optimization QUBO using QAOA
and compares results against classical simulated annealing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from time import time

from .qubo_execution import QUBOConfig, ExecutionQUBO
from .qubo_to_ising import (
    qubo_to_ising, 
    IsingHamiltonian,
    build_qaoa_circuit_from_ising,
    binary_to_spins,
    spins_to_binary
)
from .qubo_solvers import SimulatedAnnealingSolver, QUBOResult


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


# =============================================================================
# Comparison Framework
# =============================================================================

@dataclass
class SolverComparison:
    """Results comparing QAOA vs SA."""
    qaoa_energies: List[float]
    sa_energies: List[float]
    qaoa_times: List[float]
    sa_times: List[float]
    qaoa_best: float
    sa_best: float
    optimal_energy: Optional[float]
    
    @property
    def qaoa_success_rate(self) -> float:
        """Rate at which QAOA matches best found."""
        best = min(self.qaoa_best, self.sa_best)
        return sum(1 for e in self.qaoa_energies if abs(e - best) < 1e-6) / len(self.qaoa_energies)
    
    @property
    def sa_success_rate(self) -> float:
        """Rate at which SA matches best found."""
        best = min(self.qaoa_best, self.sa_best)
        return sum(1 for e in self.sa_energies if abs(e - best) < 1e-6) / len(self.sa_energies)


def compare_qaoa_vs_sa(
    Q: np.ndarray,
    num_runs: int = 10,
    qaoa_p: int = 2,
    qaoa_maxiter: int = 50,
    sa_sweeps: int = 500,
    verbose: bool = True
) -> SolverComparison:
    """
    Compare QAOA against Simulated Annealing.
    
    Args:
        Q: QUBO matrix
        num_runs: Number of runs for each solver
        qaoa_p: QAOA layer depth
        qaoa_maxiter: QAOA optimizer iterations
        sa_sweeps: SA sweeps per run
        verbose: Print progress
        
    Returns:
        SolverComparison with all results
    """
    if verbose:
        print("\n" + "="*70)
        print(" QAOA vs Simulated Annealing Comparison")
        print("="*70)
        print(f" Problem size: {Q.shape[0]} qubits")
        print(f" Runs: {num_runs}")
    
    qaoa_energies = []
    qaoa_times = []
    sa_energies = []
    sa_times = []
    
    # Run QAOA
    if verbose:
        print(f"\n Running QAOA (p={qaoa_p})...")
    
    for i in range(num_runs):
        qaoa_solver = QAOASolver(
            p=qaoa_p,
            shots=500,
            maxiter=qaoa_maxiter,
            seed=42 + i
        )
        result = qaoa_solver.solve(Q, verbose=False)
        qaoa_energies.append(result.energy)
        qaoa_times.append(result.solve_time)
        
        if verbose:
            print(f"   Run {i+1}: E={result.energy:.4f}, t={result.solve_time:.2f}s")
    
    # Run SA
    if verbose:
        print(f"\n Running SA (sweeps={sa_sweeps})...")
    
    for i in range(num_runs):
        sa_solver = SimulatedAnnealingSolver(
            num_sweeps=sa_sweeps,
            seed=42 + i
        )
        start = time()
        result = sa_solver.solve(Q, verbose=False)
        sa_time = time() - start
        
        sa_energies.append(result.energy)
        sa_times.append(sa_time)
        
        if verbose:
            print(f"   Run {i+1}: E={result.energy:.4f}, t={sa_time:.4f}s")
    
    comparison = SolverComparison(
        qaoa_energies=qaoa_energies,
        sa_energies=sa_energies,
        qaoa_times=qaoa_times,
        sa_times=sa_times,
        qaoa_best=min(qaoa_energies),
        sa_best=min(sa_energies),
        optimal_energy=None
    )
    
    if verbose:
        print(f"\n" + "="*70)
        print(" Results Summary")
        print("="*70)
        print(f"\n {'Metric':<25} {'QAOA':>15} {'SA':>15}")
        print("-"*55)
        print(f" {'Best Energy':<25} {comparison.qaoa_best:>15.4f} {comparison.sa_best:>15.4f}")
        print(f" {'Mean Energy':<25} {np.mean(qaoa_energies):>15.4f} {np.mean(sa_energies):>15.4f}")
        print(f" {'Std Dev':<25} {np.std(qaoa_energies):>15.4f} {np.std(sa_energies):>15.4f}")
        print(f" {'Mean Time (s)':<25} {np.mean(qaoa_times):>15.2f} {np.mean(sa_times):>15.4f}")
        print(f" {'Success Rate':<25} {comparison.qaoa_success_rate:>14.1%} {comparison.sa_success_rate:>14.1%}")
    
    return comparison


def plot_comparison(
    comparison: SolverComparison,
    save_path: Optional[str] = None
):
    """Plot comparison results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Energy distribution
    ax1 = axes[0]
    ax1.boxplot([comparison.qaoa_energies, comparison.sa_energies], labels=['QAOA', 'SA'])
    ax1.set_ylabel('Energy')
    ax1.set_title('Energy Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Time comparison
    ax2 = axes[1]
    ax2.bar(['QAOA', 'SA'], 
            [np.mean(comparison.qaoa_times), np.mean(comparison.sa_times)],
            color=['blue', 'orange'], alpha=0.7)
    ax2.set_ylabel('Time (s)')
    ax2.set_title('Average Solve Time')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Success rate
    ax3 = axes[2]
    ax3.bar(['QAOA', 'SA'], 
            [comparison.qaoa_success_rate * 100, comparison.sa_success_rate * 100],
            color=['blue', 'orange'], alpha=0.7)
    ax3.set_ylabel('Success Rate (%)')
    ax3.set_title('Success Rate (Finding Best)')
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    
    plt.show()


# =============================================================================
# Demo
# =============================================================================

def run_quantum_solve_demo():
    """Run the first quantum solve demo."""
    
    print("\n" + "="*70)
    print(" First Quantum Solve: Execution QUBO with QAOA")  
    print("="*70)
    
    # Create small execution QUBO
    config = QUBOConfig(
        total_shares=400,
        num_time_slices=4,
        num_venues=1,
        quantity_levels=[0, 100, 200],  # 3 levels = 12 qubits
        equality_penalty=100.0,
        capacity_penalty=50.0
    )
    
    qubo = ExecutionQUBO(config)
    Q = qubo.build_qubo_matrix()
    
    print(f"\n Problem: Execute {config.total_shares} shares over {config.num_time_slices} slices")
    print(f" QUBO size: {Q.shape[0]} variables")
    
    # Run comparison
    comparison = compare_qaoa_vs_sa(
        Q=Q,
        num_runs=5,  # Fewer runs for demo
        qaoa_p=2,
        qaoa_maxiter=30,
        sa_sweeps=500,
        verbose=True
    )
    
    # Interpret best solution
    print("\n" + "="*70)
    print(" Best Solution Interpretation")
    print("="*70)
    
    # Get best from SA (usually more reliable)
    sa_solver = SimulatedAnnealingSolver(num_sweeps=1000, seed=42)
    sa_result = sa_solver.solve(Q, verbose=False)
    
    schedule = qubo.interpret_solution(sa_result.solution)
    print(f"\n Execution Schedule:")
    print(schedule.to_string(index=False))
    
    costs = qubo.calculate_solution_cost(sa_result.solution)
    print(f"\n Total shares: {int(costs['total_shares'])}")
    print(f" Target shares: {int(costs['target_shares'])}")
    
    # Plot
    print("\n Generating comparison plot...")
    plot_comparison(comparison, save_path="qaoa_vs_sa_comparison.png")
    
    # Document challenges
    print("\n" + "="*70)
    print(" Challenges & Limitations")
    print("="*70)
    print("""
 1. QUBIT SCALING: QAOA requires O(n) qubits, limiting problem size
    - Current: 12 qubits handles 4 slices × 3 levels
    - Real execution: 100+ slices would need 300+ qubits
    
 2. OPTIMIZATION LANDSCAPE: QAOA's parameter landscape is non-convex
    - Local minima issues, sensitive to initialization
    - More layers (p) help but increase circuit depth
    
 3. SHOT NOISE: Finite measurements add variance
    - Need ~1000+ shots for reliable expectation values
    - Tradeoff: more shots = more time
    
 4. CLASSICAL OVERHEAD: Classical optimization dominates time
    - QAOA: ~10s per run (mostly optimizer iterations)
    - SA: ~0.01s per run (1000x faster for this size)
    
 5. CURRENT ADVANTAGE: SA outperforms for small problems
    - QAOA may have advantage for larger, structured problems
    - Hybrid approaches (QAOA-inspired SA) could combine benefits
""")
    
    return comparison


if __name__ == "__main__":
    comparison = run_quantum_solve_demo()
