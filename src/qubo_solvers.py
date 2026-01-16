"""
QUBO Solvers

Classical solvers for QUBO (Quadratic Unconstrained Binary Optimization):
1. Brute-Force - Exact solution via enumeration (exponential complexity)
2. Simulated Annealing - Heuristic with probabilistic exploration

These serve as baselines for comparison with quantum solvers.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Callable
from time import time


# =============================================================================
# Solution Result
# =============================================================================

@dataclass
class QUBOResult:
    """
    Result of QUBO optimization.
    
    Attributes:
        solution: Binary solution vector
        energy: Objective value (x^T Q x)
        num_evaluations: Number of solutions evaluated
        solve_time: Time taken to solve (seconds)
        solver_name: Name of solver used
        iterations: Number of iterations (for iterative solvers)
        history: Energy history during optimization
    """
    solution: np.ndarray
    energy: float
    num_evaluations: int
    solve_time: float
    solver_name: str
    iterations: int = 0
    history: Optional[List[float]] = None
    
    def __repr__(self) -> str:
        return (
            f"QUBOResult({self.solver_name})\n"
            f"  Energy: {self.energy:.4f}\n"
            f"  Evaluations: {self.num_evaluations:,}\n"
            f"  Time: {self.solve_time:.3f}s\n"
            f"  Solution: {self.solution[:10]}..." if len(self.solution) > 10 
            else f"  Solution: {self.solution}"
        )


# =============================================================================
# Brute-Force Solver
# =============================================================================

class BruteForceSolver:
    """
    Exact QUBO solver via exhaustive enumeration.
    
    Enumerates all 2^n binary solutions and returns the one with
    minimum objective value. Guaranteed to find global optimum but
    only feasible for small problems (n ≤ 20).
    
    Time Complexity: O(n^2 * 2^n)
    Space Complexity: O(n^2) for Q matrix
    
    Example:
        >>> solver = BruteForceSolver()
        >>> result = solver.solve(Q)
        >>> print(f"Optimal energy: {result.energy}")
    """
    
    def __init__(self, max_variables: int = 25):
        """
        Initialize brute-force solver.
        
        Args:
            max_variables: Maximum number of variables to allow
                          (safety limit to prevent accidental huge runs)
        """
        self.max_variables = max_variables
    
    def solve(self, Q: np.ndarray, verbose: bool = False) -> QUBOResult:
        """
        Solve QUBO by enumerating all solutions.
        
        Args:
            Q: QUBO matrix (n x n symmetric)
            verbose: Print progress updates
            
        Returns:
            QUBOResult with optimal solution
        """
        n = Q.shape[0]
        
        # Safety check
        if n > self.max_variables:
            raise ValueError(
                f"Problem size {n} exceeds max_variables={self.max_variables}. "
                f"Use simulated annealing for larger problems."
            )
        
        total_solutions = 2 ** n
        if verbose:
            print(f"Brute-force: Enumerating {total_solutions:,} solutions...")
        
        start_time = time()
        
        best_solution = None
        best_energy = float('inf')
        num_evaluations = 0
        
        # Enumerate all 2^n binary vectors
        for i in range(total_solutions):
            # Convert integer to binary vector
            x = self._int_to_binary(i, n)
            
            # Evaluate objective: x^T Q x
            energy = self._evaluate(x, Q)
            num_evaluations += 1
            
            if energy < best_energy:
                best_energy = energy
                best_solution = x.copy()
            
            # Progress update
            if verbose and (i + 1) % (total_solutions // 10) == 0:
                print(f"  Progress: {100 * (i + 1) / total_solutions:.0f}%")
        
        solve_time = time() - start_time
        
        if verbose:
            print(f"Brute-force complete: {solve_time:.3f}s")
        
        return QUBOResult(
            solution=best_solution,
            energy=best_energy,
            num_evaluations=num_evaluations,
            solve_time=solve_time,
            solver_name="BruteForce"
        )
    
    def _int_to_binary(self, i: int, n: int) -> np.ndarray:
        """Convert integer to n-bit binary array."""
        return np.array([(i >> bit) & 1 for bit in range(n)], dtype=np.int8)
    
    def _evaluate(self, x: np.ndarray, Q: np.ndarray) -> float:
        """Evaluate QUBO objective x^T Q x."""
        return float(x @ Q @ x)


# =============================================================================
# Simulated Annealing Solver
# =============================================================================

class SimulatedAnnealingSolver:
    """
    QUBO solver using simulated annealing.
    
    A probabilistic metaheuristic that explores the solution space
    by accepting worse solutions with probability exp(-ΔE/T), where
    T is a temperature that decreases over time.
    
    This allows escaping local minima while gradually converging
    to a good solution.
    
    Key parameters:
    - initial_temp: Starting temperature (higher = more exploration)
    - final_temp: Ending temperature (lower = more exploitation)
    - cooling_rate: How fast temperature decreases
    - num_sweeps: Number of complete passes over all variables
    
    Example:
        >>> solver = SimulatedAnnealingSolver(num_sweeps=1000)
        >>> result = solver.solve(Q)
        >>> print(f"Best energy found: {result.energy}")
    """
    
    def __init__(
        self,
        initial_temp: float = 10.0,
        final_temp: float = 0.01,
        cooling_rate: float = 0.95,
        num_sweeps: int = 1000,
        seed: Optional[int] = None
    ):
        """
        Initialize simulated annealing solver.
        
        Args:
            initial_temp: Starting temperature
            final_temp: Minimum temperature before stopping
            cooling_rate: Multiplicative factor for cooling (< 1)
            num_sweeps: Number of sweeps over all variables
            seed: Random seed for reproducibility
        """
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.num_sweeps = num_sweeps
        self.rng = np.random.default_rng(seed)
    
    def solve(
        self, 
        Q: np.ndarray, 
        initial_solution: Optional[np.ndarray] = None,
        verbose: bool = False
    ) -> QUBOResult:
        """
        Solve QUBO using simulated annealing.
        
        Args:
            Q: QUBO matrix (n x n symmetric)
            initial_solution: Starting solution (random if None)
            verbose: Print progress updates
            
        Returns:
            QUBOResult with best solution found
        """
        n = Q.shape[0]
        start_time = time()
        
        # Initialize solution
        if initial_solution is not None:
            x = initial_solution.copy()
        else:
            x = self.rng.integers(0, 2, size=n, dtype=np.int8)
        
        # Precompute diagonal for efficient delta calculation
        diag = np.diag(Q)
        
        # Current state
        current_energy = self._evaluate(x, Q)
        best_solution = x.copy()
        best_energy = current_energy
        
        # Statistics
        num_evaluations = 1
        accepted_moves = 0
        history = [current_energy]
        
        # Temperature schedule
        temp = self.initial_temp
        
        if verbose:
            print(f"Simulated Annealing: n={n}, sweeps={self.num_sweeps}")
            print(f"  Initial energy: {current_energy:.4f}")
        
        iteration = 0
        
        while temp > self.final_temp and iteration < self.num_sweeps:
            # One sweep: try flipping each variable once
            for i in self.rng.permutation(n):
                # Calculate energy change from flipping bit i
                # ΔE = Q[i,i] * (1 - 2*x[i]) + 2 * sum_{j≠i} Q[i,j] * x[j] * (1 - 2*x[i])
                delta_e = self._delta_energy(x, Q, i)
                num_evaluations += 1
                
                # Accept or reject
                if delta_e < 0:
                    # Always accept improvements
                    x[i] = 1 - x[i]
                    current_energy += delta_e
                    accepted_moves += 1
                elif self.rng.random() < np.exp(-delta_e / temp):
                    # Accept worse solution with probability exp(-ΔE/T)
                    x[i] = 1 - x[i]
                    current_energy += delta_e
                    accepted_moves += 1
                
                # Track best
                if current_energy < best_energy:
                    best_solution = x.copy()
                    best_energy = current_energy
            
            # Cool down
            temp *= self.cooling_rate
            iteration += 1
            history.append(best_energy)
            
            if verbose and iteration % (self.num_sweeps // 10) == 0:
                print(f"  Iter {iteration}: T={temp:.4f}, E={current_energy:.4f}, Best={best_energy:.4f}")
        
        solve_time = time() - start_time
        
        if verbose:
            print(f"SA complete: {solve_time:.3f}s, accepted {accepted_moves:,} moves")
        
        return QUBOResult(
            solution=best_solution,
            energy=best_energy,
            num_evaluations=num_evaluations,
            solve_time=solve_time,
            solver_name="SimulatedAnnealing",
            iterations=iteration,
            history=history
        )
    
    def _evaluate(self, x: np.ndarray, Q: np.ndarray) -> float:
        """Evaluate QUBO objective x^T Q x."""
        return float(x @ Q @ x)
    
    def _delta_energy(self, x: np.ndarray, Q: np.ndarray, i: int) -> float:
        """
        Calculate energy change from flipping bit i.
        
        Uses the identity:
        ΔE = (1 - 2*x[i]) * (Q[i,i] + 2 * sum_{j≠i} Q[i,j] * x[j])
        
        This is O(n) instead of O(n^2) for full re-evaluation.
        """
        flip = 1 - 2 * x[i]  # +1 if x[i]=0, -1 if x[i]=1
        
        # Contribution from diagonal
        delta = Q[i, i] * flip
        
        # Contribution from off-diagonal (interaction with other variables)
        row_sum = 2 * (Q[i, :] @ x - Q[i, i] * x[i])
        delta += row_sum * flip
        
        return delta


# =============================================================================
# Greedy Solver (Bonus)
# =============================================================================

class GreedySolver:
    """
    Simple greedy QUBO solver for fast baseline.
    
    Iteratively flips the bit that gives the largest improvement
    until no improvement is possible.
    """
    
    def __init__(self, max_iterations: int = 1000, seed: Optional[int] = None):
        """
        Initialize greedy solver.
        
        Args:
            max_iterations: Maximum iterations without improvement
            seed: Random seed
        """
        self.max_iterations = max_iterations
        self.rng = np.random.default_rng(seed)
    
    def solve(
        self, 
        Q: np.ndarray,
        initial_solution: Optional[np.ndarray] = None,
        verbose: bool = False
    ) -> QUBOResult:
        """
        Solve QUBO using greedy descent.
        
        Args:
            Q: QUBO matrix
            initial_solution: Starting solution (random if None)
            verbose: Print progress (unused, for API compatibility)
            
        Returns:
            QUBOResult with locally optimal solution
        """
        n = Q.shape[0]
        start_time = time()
        
        # Initialize
        if initial_solution is not None:
            x = initial_solution.copy()
        else:
            x = self.rng.integers(0, 2, size=n, dtype=np.int8)
        
        current_energy = float(x @ Q @ x)
        num_evaluations = 1
        
        for iteration in range(self.max_iterations):
            improved = False
            
            # Try flipping each bit
            for i in range(n):
                # Calculate delta
                flip = 1 - 2 * x[i]
                delta = Q[i, i] * flip + 2 * flip * (Q[i, :] @ x - Q[i, i] * x[i])
                num_evaluations += 1
                
                if delta < -1e-10:  # Improvement found
                    x[i] = 1 - x[i]
                    current_energy += delta
                    improved = True
            
            if not improved:
                break  # Local optimum reached
        
        solve_time = time() - start_time
        
        return QUBOResult(
            solution=x,
            energy=current_energy,
            num_evaluations=num_evaluations,
            solve_time=solve_time,
            solver_name="Greedy",
            iterations=iteration + 1
        )


# =============================================================================
# Solver Comparison Utility
# =============================================================================

def compare_solvers(
    Q: np.ndarray,
    solvers: Optional[List] = None,
    verbose: bool = True
) -> List[QUBOResult]:
    """
    Compare multiple QUBO solvers on the same problem.
    
    Args:
        Q: QUBO matrix
        solvers: List of solver instances (default: all available)
        verbose: Print comparison table
        
    Returns:
        List of QUBOResult from each solver
    """
    n = Q.shape[0]
    
    if solvers is None:
        solvers = []
        
        # Add brute-force if small enough
        if n <= 20:
            solvers.append(BruteForceSolver())
        
        solvers.extend([
            GreedySolver(seed=42),
            SimulatedAnnealingSolver(num_sweeps=500, seed=42),
            SimulatedAnnealingSolver(num_sweeps=2000, seed=42),
        ])
    
    results = []
    for solver in solvers:
        result = solver.solve(Q, verbose=False)
        results.append(result)
    
    if verbose:
        print("\n" + "=" * 70)
        print(" QUBO Solver Comparison")
        print("=" * 70)
        print(f" Problem size: {n} variables\n")
        
        print("{:<25} {:>12} {:>12} {:>12}".format(
            "Solver", "Energy", "Evaluations", "Time (s)"))
        print("-" * 70)
        
        for result in results:
            print("{:<25} {:>12.4f} {:>12,} {:>12.4f}".format(
                result.solver_name,
                result.energy,
                result.num_evaluations,
                result.solve_time
            ))
        
        # Find best
        best = min(results, key=lambda r: r.energy)
        print("-" * 70)
        print(f" Best: {best.solver_name} with energy {best.energy:.4f}")
    
    return results
