"""
Hardware Comparison Benchmark

Comprehensive benchmark comparing:
- Quantum Simulators: QASM, Statevector
- Real Quantum Hardware: IBM (placeholder)
- Classical Solvers: Simulated Annealing, Brute Force, Gurobi (if available)

Problem sizes: 4, 6, 8 qubits (execution slices)
Metrics: Solution quality, time, cost, reliability
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from time import time
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Benchmark Result
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    solver_name: str
    solver_type: str  # 'quantum_sim', 'quantum_hw', 'classical'
    problem_size: int
    solution: np.ndarray
    energy: float
    optimal_energy: float
    optimality_gap: float  # Percentage gap from optimal
    solve_time: float  # Seconds
    num_runs: int
    success_rate: float  # Rate of finding optimal
    metadata: Dict = field(default_factory=dict)
    
    @property
    def is_optimal(self) -> bool:
        return abs(self.energy - self.optimal_energy) < 1e-6


# =============================================================================
# Abstract Solver Interface
# =============================================================================

class Solver(ABC):
    """Abstract base class for all solvers."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def solver_type(self) -> str:
        pass
    
    @abstractmethod
    def solve(self, Q: np.ndarray, **kwargs) -> Tuple[np.ndarray, float]:
        """Solve QUBO and return (solution, energy)."""
        pass


# =============================================================================
# Classical Solvers
# =============================================================================

class BruteForceSolver(Solver):
    """Brute force enumeration (optimal but exponential)."""
    
    @property
    def name(self) -> str:
        return "BruteForce"
    
    @property
    def solver_type(self) -> str:
        return "classical"
    
    def solve(self, Q: np.ndarray, **kwargs) -> Tuple[np.ndarray, float]:
        n = Q.shape[0]
        best_x = None
        best_e = float('inf')
        
        for i in range(2**n):
            x = np.array([(i >> j) & 1 for j in range(n)])
            e = x @ Q @ x
            if e < best_e:
                best_e = e
                best_x = x
        
        return best_x, best_e


class SABenchmarkSolver(Solver):
    """Simulated Annealing solver for benchmark."""
    
    def __init__(self, num_sweeps: int = 500, seed: int = 42):
        self.num_sweeps = num_sweeps
        self.seed = seed
    
    @property
    def name(self) -> str:
        return f"SA-{self.num_sweeps}"
    
    @property
    def solver_type(self) -> str:
        return "classical"
    
    def solve(self, Q: np.ndarray, **kwargs) -> Tuple[np.ndarray, float]:
        from .qubo_solvers import SimulatedAnnealingSolver
        
        solver = SimulatedAnnealingSolver(num_sweeps=self.num_sweeps, seed=self.seed)
        result = solver.solve(Q, verbose=False)
        return result.solution, result.energy


class GurobiSolver(Solver):
    """Gurobi solver (if available)."""
    
    _available = None
    
    @classmethod
    def is_available(cls) -> bool:
        if cls._available is None:
            try:
                import gurobipy
                cls._available = True
            except ImportError:
                cls._available = False
        return cls._available
    
    @property
    def name(self) -> str:
        return "Gurobi"
    
    @property
    def solver_type(self) -> str:
        return "classical"
    
    def solve(self, Q: np.ndarray, **kwargs) -> Tuple[np.ndarray, float]:
        if not self.is_available():
            raise RuntimeError("Gurobi not available")
        
        import gurobipy as gp
        from gurobipy import GRB
        
        n = Q.shape[0]
        
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            
            with gp.Model(env=env) as model:
                x = model.addVars(n, vtype=GRB.BINARY, name="x")
                
                obj = gp.quicksum(
                    Q[i, j] * x[i] * x[j]
                    for i in range(n)
                    for j in range(n)
                )
                model.setObjective(obj, GRB.MINIMIZE)
                model.optimize()
                
                solution = np.array([x[i].X for i in range(n)])
                energy = model.objVal
        
        return solution, energy


# =============================================================================
# Quantum Simulators
# =============================================================================

class QASMSimulatorSolver(Solver):
    """QASM simulator (shot-based)."""
    
    def __init__(self, shots: int = 4000, p: int = 1):
        self.shots = shots
        self.p = p
    
    @property
    def name(self) -> str:
        return f"QASM-p{self.p}"
    
    @property
    def solver_type(self) -> str:
        return "quantum_sim"
    
    def solve(self, Q: np.ndarray, **kwargs) -> Tuple[np.ndarray, float]:
        from qiskit_aer import AerSimulator
        from qiskit import transpile
        from .qubo_to_ising import qubo_to_ising, build_qaoa_circuit_from_ising
        
        n = Q.shape[0]
        
        # Pre-optimized parameters (simple heuristic)
        gamma = 0.5 + 0.1 * self.p
        beta = 0.3 + 0.05 * self.p
        
        ising = qubo_to_ising(Q)
        qc = build_qaoa_circuit_from_ising(ising, gamma, beta, self.p)
        
        simulator = AerSimulator()
        compiled = transpile(qc, simulator)
        result = simulator.run(compiled, shots=self.shots).result()
        counts = result.get_counts()
        
        # Find best
        best_x = None
        best_e = float('inf')
        
        for bitstring in counts:
            x = np.array([int(b) for b in bitstring[::-1]])
            if len(x) == n:
                e = x @ Q @ x
                if e < best_e:
                    best_e = e
                    best_x = x
        
        return best_x if best_x is not None else np.zeros(n), best_e


class StatevectorSolver(Solver):
    """Statevector simulator (exact)."""
    
    def __init__(self, p: int = 1):
        self.p = p
    
    @property
    def name(self) -> str:
        return f"Statevector-p{self.p}"
    
    @property
    def solver_type(self) -> str:
        return "quantum_sim"
    
    def solve(self, Q: np.ndarray, **kwargs) -> Tuple[np.ndarray, float]:
        from qiskit_aer import AerSimulator
        from qiskit import transpile
        from .qubo_to_ising import qubo_to_ising, build_qaoa_circuit_from_ising
        
        n = Q.shape[0]
        
        gamma = 0.5 + 0.1 * self.p
        beta = 0.3 + 0.05 * self.p
        
        ising = qubo_to_ising(Q)
        qc = build_qaoa_circuit_from_ising(ising, gamma, beta, self.p)
        qc.remove_final_measurements()
        qc.save_statevector()
        
        simulator = AerSimulator(method='statevector')
        compiled = transpile(qc, simulator)
        result = simulator.run(compiled).result()
        
        statevector = result.get_statevector()
        probs = np.abs(statevector.data) ** 2
        
        # Find best
        best_x = None
        best_e = float('inf')
        
        for i, prob in enumerate(probs):
            if prob > 0.01:
                x = np.array([(i >> j) & 1 for j in range(n)])
                e = x @ Q @ x
                if e < best_e:
                    best_e = e
                    best_x = x
        
        return best_x if best_x is not None else np.zeros(n), best_e


# =============================================================================
# Benchmark Framework
# =============================================================================

class Benchmark:
    """Benchmark framework for comparing solvers."""
    
    def __init__(self, problem_sizes: List[int] = [4, 6, 8]):
        self.problem_sizes = problem_sizes
        self.results: List[BenchmarkResult] = []
    
    def generate_qubo(self, n: int, seed: int = 42) -> np.ndarray:
        """Generate random QUBO for benchmarking."""
        np.random.seed(seed)
        Q = np.random.randn(n, n)
        Q = (Q + Q.T) / 2  # Symmetric
        return Q
    
    def find_optimal(self, Q: np.ndarray) -> float:
        """Find optimal solution via brute force."""
        solver = BruteForceSolver()
        _, energy = solver.solve(Q)
        return energy
    
    def run_solver(
        self,
        solver: Solver,
        Q: np.ndarray,
        optimal_energy: float,
        num_runs: int = 5
    ) -> BenchmarkResult:
        """Run solver multiple times and collect statistics."""
        n = Q.shape[0]
        
        energies = []
        times = []
        solutions = []
        
        for i in range(num_runs):
            start = time()
            try:
                solution, energy = solver.solve(Q)
                solve_time = time() - start
                
                energies.append(energy)
                times.append(solve_time)
                solutions.append(solution)
            except Exception as e:
                logger.error(f"{solver.name} run {i+1} failed: {e}")
        
        if not energies:
            return BenchmarkResult(
                solver_name=solver.name,
                solver_type=solver.solver_type,
                problem_size=n,
                solution=np.zeros(n),
                energy=float('inf'),
                optimal_energy=optimal_energy,
                optimality_gap=float('inf'),
                solve_time=0.0,
                num_runs=0,
                success_rate=0.0,
                metadata={'error': 'All runs failed'}
            )
        
        best_idx = np.argmin(energies)
        best_energy = energies[best_idx]
        best_solution = solutions[best_idx]
        
        # Calculate metrics
        gap = ((best_energy - optimal_energy) / abs(optimal_energy) * 100 
               if abs(optimal_energy) > 1e-10 else 0.0)
        success_rate = sum(1 for e in energies if abs(e - optimal_energy) < 1e-6) / len(energies)
        
        return BenchmarkResult(
            solver_name=solver.name,
            solver_type=solver.solver_type,
            problem_size=n,
            solution=best_solution,
            energy=best_energy,
            optimal_energy=optimal_energy,
            optimality_gap=gap,
            solve_time=np.mean(times),
            num_runs=len(energies),
            success_rate=success_rate,
            metadata={
                'all_energies': energies,
                'all_times': times
            }
        )
    
    def run_benchmark(
        self,
        solvers: List[Solver],
        num_runs: int = 5,
        verbose: bool = True
    ) -> pd.DataFrame:
        """Run full benchmark across all problem sizes."""
        
        if verbose:
            print("\n" + "="*70)
            print(" Hardware/Solver Benchmark")
            print("="*70)
        
        self.results = []
        
        for size in self.problem_sizes:
            if verbose:
                print(f"\n Problem size: {size} qubits")
                print("-" * 50)
            
            Q = self.generate_qubo(size)
            
            # Find optimal
            if verbose:
                print(f" Finding optimal...")
            optimal_energy = self.find_optimal(Q)
            if verbose:
                print(f" Optimal energy: {optimal_energy:.4f}")
            
            # Run each solver
            for solver in solvers:
                if verbose:
                    print(f" Running {solver.name}...", end=" ")
                
                try:
                    result = self.run_solver(solver, Q, optimal_energy, num_runs)
                    self.results.append(result)
                    
                    if verbose:
                        status = "✓" if result.is_optimal else f"gap={result.optimality_gap:.1f}%"
                        print(f"{status} ({result.solve_time:.3f}s)")
                except Exception as e:
                    if verbose:
                        print(f"✗ Error: {e}")
        
        return self.to_dataframe()
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        data = []
        for r in self.results:
            data.append({
                'Solver': r.solver_name,
                'Type': r.solver_type,
                'Size': r.problem_size,
                'Energy': r.energy,
                'Optimal': r.optimal_energy,
                'Gap (%)': r.optimality_gap,
                'Time (s)': r.solve_time,
                'Success Rate': r.success_rate,
                'Optimal?': r.is_optimal
            })
        return pd.DataFrame(data)


# =============================================================================
# Visualization
# =============================================================================

def create_benchmark_report(
    df: pd.DataFrame,
    save_path: str = "benchmark_report.png"
) -> None:
    """Create visualization report from benchmark results."""
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # =========================================================================
    # Plot 1: Optimality Gap by Solver
    # =========================================================================
    ax1 = axes[0, 0]
    solvers = df['Solver'].unique()
    sizes = sorted(df['Size'].unique())
    
    x = np.arange(len(solvers))
    width = 0.25
    
    for i, size in enumerate(sizes):
        gaps = []
        for solver in solvers:
            subset = df[(df['Solver'] == solver) & (df['Size'] == size)]
            gap = subset['Gap (%)'].values[0] if len(subset) > 0 else 0
            gaps.append(max(0, gap))  # Ensure non-negative
        
        ax1.bar(x + i * width, gaps, width, label=f'{size} qubits')
    
    ax1.set_xlabel('Solver')
    ax1.set_ylabel('Optimality Gap (%)')
    ax1.set_title('Solution Quality by Solver')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(solvers, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # =========================================================================
    # Plot 2: Time to Solution
    # =========================================================================
    ax2 = axes[0, 1]
    
    for i, size in enumerate(sizes):
        times = []
        for solver in solvers:
            subset = df[(df['Solver'] == solver) & (df['Size'] == size)]
            t = subset['Time (s)'].values[0] if len(subset) > 0 else 0
            times.append(t)
        
        ax2.bar(x + i * width, times, width, label=f'{size} qubits')
    
    ax2.set_xlabel('Solver')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Time to Solution')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(solvers, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_yscale('log')
    
    # =========================================================================
    # Plot 3: Success Rate
    # =========================================================================
    ax3 = axes[1, 0]
    
    for i, size in enumerate(sizes):
        rates = []
        for solver in solvers:
            subset = df[(df['Solver'] == solver) & (df['Size'] == size)]
            rate = subset['Success Rate'].values[0] * 100 if len(subset) > 0 else 0
            rates.append(rate)
        
        ax3.bar(x + i * width, rates, width, label=f'{size} qubits')
    
    ax3.set_xlabel('Solver')
    ax3.set_ylabel('Success Rate (%)')
    ax3.set_title('Reliability (Finding Optimal)')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(solvers, rotation=45, ha='right')
    ax3.legend()
    ax3.set_ylim(0, 105)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # =========================================================================
    # Plot 4: Summary Table
    # =========================================================================
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Aggregate by solver
    summary = df.groupby('Solver').agg({
        'Gap (%)': 'mean',
        'Time (s)': 'mean',
        'Success Rate': 'mean'
    }).reset_index()
    
    table_data = []
    for _, row in summary.iterrows():
        table_data.append([
            row['Solver'],
            f"{row['Gap (%)']:.1f}%",
            f"{row['Time (s)']:.4f}",
            f"{row['Success Rate']*100:.0f}%"
        ])
    
    table = ax4.table(
        cellText=table_data,
        colLabels=['Solver', 'Avg Gap', 'Avg Time', 'Success'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    ax4.set_title('Overall Performance Summary', pad=20)
    
    plt.suptitle('Quantum-Classical Solver Benchmark Report', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Report saved to: {save_path}")
    plt.show()


# =============================================================================
# Demo
# =============================================================================

def run_benchmark_demo():
    """Run benchmark demo."""
    
    print("\n" + "="*70)
    print(" Hardware/Solver Comparison Benchmark")
    print("="*70)
    
    # Define solvers
    solvers = [
        BruteForceSolver(),
        SABenchmarkSolver(num_sweeps=500),
        QASMSimulatorSolver(shots=2000, p=1),
        QASMSimulatorSolver(shots=2000, p=2),
        StatevectorSolver(p=1),
    ]
    
    # Check Gurobi
    if GurobiSolver.is_available():
        solvers.insert(2, GurobiSolver())
        print(" Gurobi: Available")
    else:
        print(" Gurobi: Not available")
    
    # Run benchmark
    benchmark = Benchmark(problem_sizes=[4, 6, 8])
    df = benchmark.run_benchmark(solvers, num_runs=3, verbose=True)
    
    # Print results
    print("\n" + "="*70)
    print(" Results Summary")
    print("="*70)
    print(df.to_string(index=False))
    
    # Create visualization
    print("\n Generating report...")
    create_benchmark_report(df, "benchmark_report.png")
    
    return df


if __name__ == "__main__":
    df = run_benchmark_demo()
