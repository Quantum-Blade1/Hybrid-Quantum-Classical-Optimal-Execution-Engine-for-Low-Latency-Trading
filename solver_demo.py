"""
QUBO Solver Demo

Demonstrates brute-force and simulated annealing solvers
on the execution QUBO with N=8 time slices.
"""

import sys
import numpy as np

sys.path.insert(0, ".")

from src.qubo_execution import QUBOConfig, ExecutionQUBO
from src.qubo_solvers import (
    BruteForceSolver,
    SimulatedAnnealingSolver,
    GreedySolver,
    compare_solvers
)


def print_header(text: str):
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70)


def print_section(text: str):
    print(f"\n--- {text} ---")


def run_demo():
    """Run the QUBO solver demo."""
    
    print_header("QUBO Solver Demo - Execution Optimization")
    
    # =========================================================================
    # Step 1: Create a small execution QUBO (N=8 slices)
    # =========================================================================
    print_section("1. Creating Execution QUBO (N=8 slices, 1 venue)")
    
    config = QUBOConfig(
        total_shares=1000,
        num_time_slices=8,
        num_venues=1,  # Keep small for brute-force
        quantity_levels=[0, 100, 200, 300],  # 4 levels
        impact_weight=0.4,
        timing_weight=0.3,
        transaction_weight=0.3,
        equality_penalty=100.0,
        capacity_penalty=50.0,
        max_shares_per_slice=400
    )
    
    qubo = ExecutionQUBO(config)
    Q = qubo.build_qubo_matrix()
    
    n = config.num_variables
    print(f"Problem size: {n} binary variables")
    print(f"Solution space: 2^{n} = {2**n:,} possibilities")
    print(f"Q matrix: {Q.shape}, {np.count_nonzero(Q):,} non-zeros")
    
    # =========================================================================
    # Step 2: Solve with Brute-Force (exact)
    # =========================================================================
    print_section("2. Brute-Force Solver (Exact)")
    
    if n <= 20:
        bf_solver = BruteForceSolver()
        bf_result = bf_solver.solve(Q, verbose=True)
        
        print(f"\nOptimal solution found!")
        print(f"Energy: {bf_result.energy:.4f}")
        print(f"Solution: {bf_result.solution}")
        
        # Interpret solution
        bf_schedule = qubo.interpret_solution(bf_result.solution)
        print(f"\nExecution Schedule:")
        print(bf_schedule.to_string(index=False))
        
        bf_costs = qubo.calculate_solution_cost(bf_result.solution)
        print(f"\nTotal shares: {int(bf_costs['total_shares'])}")
    else:
        print(f"Problem too large for brute-force (n={n} > 20)")
        bf_result = None
    
    # =========================================================================
    # Step 3: Solve with Greedy
    # =========================================================================
    print_section("3. Greedy Solver")
    
    greedy_solver = GreedySolver(seed=42)
    greedy_result = greedy_solver.solve(Q)
    
    print(f"Energy: {greedy_result.energy:.4f}")
    print(f"Iterations: {greedy_result.iterations}")
    print(f"Time: {greedy_result.solve_time:.4f}s")
    
    greedy_schedule = qubo.interpret_solution(greedy_result.solution)
    print(f"\nExecution Schedule:")
    print(greedy_schedule.to_string(index=False))
    
    # =========================================================================
    # Step 4: Solve with Simulated Annealing
    # =========================================================================
    print_section("4. Simulated Annealing Solver")
    
    sa_solver = SimulatedAnnealingSolver(
        initial_temp=10.0,
        final_temp=0.01,
        cooling_rate=0.95,
        num_sweeps=1000,
        seed=42
    )
    
    sa_result = sa_solver.solve(Q, verbose=True)
    
    print(f"\nBest solution found!")
    print(f"Energy: {sa_result.energy:.4f}")
    
    sa_schedule = qubo.interpret_solution(sa_result.solution)
    print(f"\nExecution Schedule:")
    print(sa_schedule.to_string(index=False))
    
    sa_costs = qubo.calculate_solution_cost(sa_result.solution)
    print(f"\nTotal shares: {int(sa_costs['total_shares'])}")
    
    # =========================================================================
    # Step 5: Compare All Solvers
    # =========================================================================
    print_section("5. Solver Comparison")
    
    solvers = [
        GreedySolver(seed=42),
        SimulatedAnnealingSolver(num_sweeps=100, seed=42),
        SimulatedAnnealingSolver(num_sweeps=500, seed=42),
        SimulatedAnnealingSolver(num_sweeps=2000, seed=42),
    ]
    
    if n <= 20:
        solvers.insert(0, BruteForceSolver())
    
    results = compare_solvers(Q, solvers, verbose=True)
    
    # =========================================================================
    # Step 6: Validate Best Solution
    # =========================================================================
    print_section("6. Solution Validation")
    
    best_result = min(results, key=lambda r: r.energy)
    validation = qubo.validate_solution(best_result.solution)
    
    print(f"\nBest solver: {best_result.solver_name}")
    print(f"Constraints:")
    for constraint, satisfied in validation.items():
        status = "✓" if satisfied else "✗"
        print(f"  {status} {constraint}")
    
    # =========================================================================
    # Step 7: Cost Breakdown
    # =========================================================================
    print_section("7. Cost Breakdown (Best Solution)")
    
    costs = qubo.calculate_solution_cost(best_result.solution)
    
    print(f"\n{'Component':<25} {'Value':>12}")
    print("-" * 40)
    print(f"{'Market Impact':<25} {costs['impact_cost']:>12.4f}")
    print(f"{'Timing Risk':<25} {costs['timing_cost']:>12.4f}")
    print(f"{'Transaction Cost':<25} {costs['transaction_cost']:>12.4f}")
    print(f"{'Constraint Penalty':<25} {costs['constraint_penalty']:>12.4f}")
    print("-" * 40)
    print(f"{'TOTAL':<25} {costs['total_cost']:>12.4f}")
    print(f"\n{'Shares Executed':<25} {int(costs['total_shares']):>12,}")
    print(f"{'Target Shares':<25} {int(costs['target_shares']):>12,}")
    
    print_section("Demo Complete")
    
    return results


if __name__ == "__main__":
    results = run_demo()
