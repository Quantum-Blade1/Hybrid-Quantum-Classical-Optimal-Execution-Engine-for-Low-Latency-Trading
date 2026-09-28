"""
Simulated annealing QUBO solver.

The canonical classical heuristic used throughout the project (slow path,
benchmarks, figure generation).
"""

import numpy as np
from typing import Optional
from time import time

from qexec.optimization.solvers.result import QUBOResult


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
