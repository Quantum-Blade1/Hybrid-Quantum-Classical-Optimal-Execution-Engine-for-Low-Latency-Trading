"""Run several QUBO solvers on the same instance and tabulate the results."""

import numpy as np
from typing import List, Optional

from qexec.optimization.solvers.result import QUBOResult
from qexec.optimization.solvers.exact import BruteForceSolver
from qexec.optimization.solvers.annealing import SimulatedAnnealingSolver
from qexec.optimization.solvers.greedy import GreedySolver


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
