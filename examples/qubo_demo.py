"""
QUBO Execution Demo

Demonstrates the QUBO formulation for optimal execution and
shows how to interpret solutions.
"""

import sys
import numpy as np

import os
# Add src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.qubo_execution import (
    QUBOConfig,
    ExecutionQUBO,
    create_random_binary_solution,
    create_uniform_solution,
    print_qubo_summary
)


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70)


def print_section(text: str):
    """Print a section header."""
    print(f"\n--- {text} ---")


def run_demo():
    """Run the QUBO formulation demo."""
    
    print_header("QUBO Execution Optimization Demo")
    
    # =========================================================================
    # Step 1: Configure the problem
    # =========================================================================
    print_section("1. Problem Setup")
    
    config = QUBOConfig(
        total_shares=10_000,
        num_time_slices=10,
        num_venues=2,
        quantity_levels=[0, 500, 1000, 1500, 2000],
        
        # Cost weights
        impact_weight=0.4,
        timing_weight=0.3,
        transaction_weight=0.3,
        
        # Market parameters
        volatility=0.02,
        avg_spread_bps=5.0,
        impact_coefficient=0.1,
        
        # Constraints
        max_shares_per_slice=3000,
        equality_penalty=1000.0,
        capacity_penalty=500.0
    )
    
    print(f"Order: Execute {config.total_shares:,} shares")
    print(f"Time Slices: {config.num_time_slices}")
    print(f"Venues: {config.num_venues} (Primary, Dark Pool)")
    print(f"Quantity Options: {config.quantity_levels}")
    print(f"Binary Variables: {config.num_variables}")
    
    # =========================================================================
    # Step 2: Build the QUBO matrix
    # =========================================================================
    print_section("2. Building QUBO Matrix")
    
    qubo = ExecutionQUBO(config)
    Q = qubo.build_qubo_matrix()
    
    print(f"Q Matrix Shape: {Q.shape}")
    print(f"Non-zero entries: {np.count_nonzero(Q)}")
    print(f"Matrix density: {np.count_nonzero(Q) / Q.size:.1%}")
    
    # =========================================================================
    # Step 3: Compare uniform vs random solutions
    # =========================================================================
    print_section("3. Solution Comparison")
    
    # Uniform (TWAP-like) solution
    x_uniform = create_uniform_solution(config)
    uniform_costs = qubo.calculate_solution_cost(x_uniform)
    uniform_schedule = qubo.interpret_solution(x_uniform)
    
    print("\nUniform Solution (TWAP-like):")
    print(uniform_schedule.to_string(index=False))
    print(f"\nTotal shares: {int(uniform_costs['total_shares']):,}")
    print(f"Total QUBO cost: {uniform_costs['total_cost']:.4f}")
    
    # Random solution
    x_random = create_random_binary_solution(config, seed=42)
    random_costs = qubo.calculate_solution_cost(x_random)
    random_schedule = qubo.interpret_solution(x_random)
    
    print("\nRandom Solution:")
    print(random_schedule.to_string(index=False))
    print(f"\nTotal shares: {int(random_costs['total_shares']):,}")
    print(f"Total QUBO cost: {random_costs['total_cost']:.4f}")
    
    # =========================================================================
    # Step 4: Cost breakdown
    # =========================================================================
    print_section("4. Cost Breakdown Comparison")
    
    print("\n{:<25} {:>15} {:>15}".format("Cost Component", "Uniform", "Random"))
    print("-" * 55)
    print("{:<25} {:>15.4f} {:>15.4f}".format(
        "Market Impact", uniform_costs['impact_cost'], random_costs['impact_cost']))
    print("{:<25} {:>15.4f} {:>15.4f}".format(
        "Timing Risk", uniform_costs['timing_cost'], random_costs['timing_cost']))
    print("{:<25} {:>15.4f} {:>15.4f}".format(
        "Transaction Cost", uniform_costs['transaction_cost'], random_costs['transaction_cost']))
    print("{:<25} {:>15.4f} {:>15.4f}".format(
        "Constraint Penalty", uniform_costs['constraint_penalty'], random_costs['constraint_penalty']))
    print("-" * 55)
    print("{:<25} {:>15.4f} {:>15.4f}".format(
        "TOTAL", uniform_costs['total_cost'], random_costs['total_cost']))
    
    # =========================================================================
    # Step 5: Constraint validation
    # =========================================================================
    print_section("5. Constraint Validation")
    
    print("\nUniform Solution:")
    uniform_valid = qubo.validate_solution(x_uniform)
    for constraint, satisfied in uniform_valid.items():
        status = "✓" if satisfied else "✗"
        print(f"  {status} {constraint}")
    
    print("\nRandom Solution:")
    random_valid = qubo.validate_solution(x_random)
    for constraint, satisfied in random_valid.items():
        status = "✓" if satisfied else "✗"
        print(f"  {status} {constraint}")
    
    # =========================================================================
    # Step 6: QUBO dictionary format (for solvers)
    # =========================================================================
    print_section("6. QUBO Dictionary Format (sample)")
    
    qubo_dict = qubo.get_qubo_dict()
    print(f"\nTotal dictionary entries: {len(qubo_dict)}")
    print("\nFirst 10 entries:")
    for i, ((row, col), value) in enumerate(list(qubo_dict.items())[:10]):
        print(f"  Q[{row:2d}, {col:2d}] = {value:>10.4f}")
    
    # =========================================================================
    # Step 7: Print full summary
    # =========================================================================
    print_qubo_summary(qubo)
    
    print_section("Demo Complete")
    print("\nThe QUBO matrix Q can now be sent to:")
    print("  - IBM Qiskit (QAOA, VQE)")
    print("  - D-Wave quantum annealer")
    print("  - Classical QUBO solvers (simulated annealing)")
    
    return qubo, uniform_costs, random_costs


if __name__ == "__main__":
    qubo, uniform, random = run_demo()
