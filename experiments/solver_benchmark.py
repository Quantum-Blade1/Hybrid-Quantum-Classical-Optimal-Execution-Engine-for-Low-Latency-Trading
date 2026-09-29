"""Exact, SA and greedy QUBO solvers over families, sizes and seeds (paper fig05, fig06, fig09)."""

from dataclasses import dataclass

import pandas as pd

from experiments.common import Experiment, seed_range, summarize_groups
from experiments.problems import execution_qubo, instance, slice_program
from qexec.experiment import ExperimentRecorder
from qexec.optimization.solvers.annealing import SimulatedAnnealingSolver
from qexec.optimization.solvers.exact import BruteForceSolver
from qexec.optimization.solvers.greedy import GreedySolver
from qexec.optimization.solvers.metrics import (
    OPTIMUM_TOL,
    approximation_ratio,
    energy_bounds,
    optimality_gap,
)
from qexec.optimization.solvers.result import QUBOResult


@dataclass(frozen=True)
class Config:
    toy_sizes: tuple[int, ...] = (4, 6, 8, 10, 12, 14, 16, 18, 20)
    execution_sizes: tuple[int, ...] = (6, 9, 12, 15, 18)
    slice_sizes: tuple[int, ...] = (4, 6, 8, 9, 10, 12, 15, 16, 18, 20)
    random_sizes: tuple[int, ...] = (4, 6, 8, 10, 12, 14, 16, 18, 20)
    sa_sweeps: int = 1000
    sa_restarts: int = 16
    convergence_sweeps: tuple[int, ...] = (100, 300, 500, 1000, 2000)
    convergence_size: int = 12
    seed: int = 0
    num_seeds: int = 10


FULL = Config()
QUICK = Config(
    toy_sizes=(4, 8),
    execution_sizes=(6, 9),
    slice_sizes=(6, 9),
    random_sizes=(4, 8),
    convergence_sweeps=(100, 500),
    num_seeds=2,
)


def _row(family: str, n: int, seed: int, name: str, *, result: QUBOResult, bounds) -> dict:
    return {
        "family": family,
        "n": n,
        "seed": seed,
        "solver": name,
        "energy": result.energy,
        "min_energy": bounds.min_energy,
        "max_energy": bounds.max_energy,
        "approx_ratio": approximation_ratio(result.energy, bounds),
        "optimality_gap": optimality_gap(result.energy, bounds),
        "optimal": bool(result.energy <= bounds.min_energy + OPTIMUM_TOL),
        "time_s": result.solve_time,
        "iterations": result.iterations,
    }


def run(config: Config, rec: ExperimentRecorder) -> None:
    rows = []
    families = [
        ("toy", config.toy_sizes),
        ("execution", config.execution_sizes),
        ("slice", config.slice_sizes),
        ("random", config.random_sizes),
    ]
    for family, sizes in families:
        for n in sizes:
            for seed in seed_range(config):
                Q = instance(family, n, seed)
                bounds = energy_bounds(Q)
                # Non-random instances ignore the seed: above n = 16, solve exactly only once.
                if family == "random" or seed == config.seed or n <= 16:
                    exact = BruteForceSolver().solve(Q)
                    rows.append(_row(family, n, seed, "BruteForce", result=exact, bounds=bounds))
                single = SimulatedAnnealingSolver(num_sweeps=config.sa_sweeps, seed=seed).solve(Q)
                rows.append(_row(family, n, seed, "SA_1", result=single, bounds=bounds))
                sa = SimulatedAnnealingSolver(
                    num_sweeps=config.sa_sweeps, num_restarts=config.sa_restarts, seed=seed
                ).solve(Q)
                row = _row(family, n, seed, "SA", result=sa, bounds=bounds)
                if family == "execution":
                    qubo = execution_qubo(n)
                    row["fill_rate"] = qubo.slice_quantities(sa.solution).sum() / (100 * (n // 3))
                if family == "slice":
                    counts = slice_program(n).decode(sa.solution)
                    row["fill_rate"] = counts.sum() / slice_program(n).units
                rows.append(row)
                rows.append(
                    _row(
                        family,
                        n,
                        seed,
                        "Greedy",
                        result=GreedySolver(seed=seed).solve(Q),
                        bounds=bounds,
                    )
                )
    runs = pd.DataFrame(rows)
    rec.write_table("runs", runs)
    rec.write_table(
        "summary",
        summarize_groups(
            runs,
            ["family", "n", "solver"],
            ["approx_ratio", "optimality_gap", "optimal", "time_s", "iterations", "fill_rate"],
        ),
    )

    Q = instance("execution", config.convergence_size, config.seed)
    bounds = energy_bounds(Q)
    history_rows, final_rows = [], []
    for sweeps in config.convergence_sweeps:
        for seed in seed_range(config):
            result = SimulatedAnnealingSolver(num_sweeps=sweeps, seed=seed).solve(Q)
            final_rows.append(
                {
                    "num_sweeps": sweeps,
                    "seed": seed,
                    "energy": result.energy,
                    "sweeps_run": result.iterations,
                    "min_energy": bounds.min_energy,
                    "optimal": bool(result.energy <= bounds.min_energy + OPTIMUM_TOL),
                }
            )
            if seed == config.seed:
                history_rows.extend(
                    {"num_sweeps": sweeps, "iteration": i, "best_energy": e}
                    for i, e in enumerate(result.history)
                )
    rec.write_table("sa_convergence_history", pd.DataFrame(history_rows))
    rec.write_table("sa_convergence_final", pd.DataFrame(final_rows))


EXPERIMENT = Experiment(
    "solver_benchmark", FULL, QUICK, run, seed_range, description=__doc__.splitlines()[0]
)

if __name__ == "__main__":
    EXPERIMENT.main()
