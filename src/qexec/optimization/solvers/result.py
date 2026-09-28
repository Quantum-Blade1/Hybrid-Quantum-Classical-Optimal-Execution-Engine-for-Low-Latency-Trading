"""Common result type returned by the classical QUBO solvers."""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List


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
