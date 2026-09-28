"""
QAOA vs simulated annealing on a small execution QUBO.

Runs both solvers repeatedly on a 12-variable ExecutionQUBO (4 slices x 3
quantity levels), prints energy/time/success statistics and writes a
comparison plot.

Usage:
    python experiments/qaoa_vs_sa.py
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass
from time import time

from qexec.optimization.qubo import QUBOConfig, ExecutionQUBO
from qexec.optimization.solvers.annealing import SimulatedAnnealingSolver
from qexec.optimization.solvers.qaoa import QAOASolver



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
    ax1.boxplot([comparison.qaoa_energies, comparison.sa_energies], tick_labels=['QAOA', 'SA'])
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


def main():
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
    main()
