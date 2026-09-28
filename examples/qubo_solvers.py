"""
Execution QUBO: build it, solve it three ways, read the schedule back.

An order of 800 shares over 4 time slices with quantity levels
{0, 100, 200, 300} gives a 16-variable QUBO, small enough for exhaustive
search, so the simulated-annealing and greedy energies can be compared
with the exact optimum.

    python examples/qubo_solvers.py
"""

from qexec.optimization.qubo import ExecutionQUBO, QUBOConfig
from qexec.optimization.solvers.compare import compare_solvers
from qexec.optimization.solvers.exact import BruteForceSolver
from qexec.optimization.solvers.annealing import SimulatedAnnealingSolver
from qexec.optimization.solvers.greedy import GreedySolver

SEED = 42


def main() -> None:
    config = QUBOConfig(
        total_shares=800,
        num_time_slices=4,
        num_venues=1,
        quantity_levels=[0, 100, 200, 300],
        max_shares_per_slice=400,
    )
    qubo = ExecutionQUBO(config)
    Q = qubo.build_qubo_matrix()
    print(f"QUBO: {Q.shape[0]} binary variables "
          f"({config.num_time_slices} slices x {config.num_quantity_levels} levels)")

    results = compare_solvers(
        Q,
        solvers=[
            BruteForceSolver(),
            SimulatedAnnealingSolver(num_sweeps=1000, seed=SEED),
            GreedySolver(seed=SEED),
        ],
        verbose=True,
    )

    best = min(results, key=lambda r: r.energy)
    print(f"\nSchedule from {best.solver_name} (energy {best.energy:.4f}):")
    print(qubo.interpret_solution(best.solution).to_string(index=False))
    print("\nConstraint check:", qubo.validate_solution(best.solution))


if __name__ == "__main__":
    main()
