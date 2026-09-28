"""Greedy single-bit-flip QUBO solver (fast baseline)."""

import numpy as np
from typing import Optional
from time import time

from qexec.optimization.solvers.result import QUBOResult


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
