"""Exact QUBO solver by exhaustive enumeration (reference optimum for small n)."""

import numpy as np
from time import time

from qexec.optimization.solvers.result import QUBOResult


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
