"""QAOA (ideal and noisy Aer) vs simulated annealing vs uniform random sampling, over sizes,
depths and seeds (paper fig07, fig_hw_*, tables tab:sim_results / tab:config).

This is the simulator part of the hardware benchmark. Families (experiments.problems):
toy (the IBM-hardware problem), random Gaussian QUBOs, the 12-variable execution QUBO
of fig07, and the Phase 7 exact binary encoding of the execution cost model ("slice").
For each QAOA run we record the best-of-shots energy (which saturates when the shot
budget is comparable to 2^n, claims audit F2), the probability mass on the optimal set,
the <H>-based approximation ratio, and a uniform-random baseline given the same total
shot budget (optimisation shots + final shots). Noisy runs use qexec.hardware.noise
(not calibrated to a device).

Usage:
    python -m experiments.qaoa_benchmark [--quick] [--results-dir results] [--seed 0]
"""

import time
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from qiskit_aer import AerSimulator

from experiments.common import Experiment, seed_range, summarize_groups
from experiments.problems import instance
from qexec.experiment import ExperimentRecorder
from qexec.hardware.noise import noisy_aer_backend
from qexec.optimization.solvers.annealing import SimulatedAnnealingSolver
from qexec.optimization.solvers.metrics import (
    OPTIMUM_TOL,
    EnergyBounds,
    approximation_ratio,
    counts_quality,
    energy_bounds,
    optimality_gap,
    random_sampling_baseline,
)
from qexec.optimization.solvers.qaoa import aer_sampler, run_qaoa

METRICS = (
    "approx_ratio_best",
    "approx_ratio_mean",
    "optimality_gap_best",
    "optimal_found",
    "success_probability",
    "random_success_probability",
    "random_approx_ratio_best",
    "random_approx_ratio_mean",
    "time_s",
)


@dataclass(frozen=True)
class Config:
    toy_sizes: tuple[int, ...] = (4, 6, 8, 10, 12)
    random_sizes: tuple[int, ...] = (4, 6, 8, 10, 12)
    slice_sizes: tuple[int, ...] = (4, 6, 8, 9, 10, 12)
    fig07: bool = True
    depths: tuple[int, ...] = (1, 2, 3)
    noisy_max_n: int = 10
    noise_level: float = 0.02
    shots: int = 1000
    final_shots: int = 5000
    maxiter: int = 50
    sa_sweeps: int = 1000
    sa_restarts: int = 16
    count_table_n: int = 4
    seed: int = 0
    num_seeds: int = 5


FULL = Config()
QUICK = Config(
    toy_sizes=(4, 6),
    random_sizes=(4,),
    slice_sizes=(4,),
    fig07=False,
    depths=(1,),
    noisy_max_n=4,
    num_seeds=2,
)


def _quality_row(energy: float, bounds: EnergyBounds, suffix: str) -> dict[str, float]:
    return {
        f"energy_{suffix}": energy,
        f"approx_ratio_{suffix}": approximation_ratio(energy, bounds),
        f"optimality_gap_{suffix}": optimality_gap(energy, bounds),
    }


def qaoa_row(
    Q: np.ndarray, bounds: EnergyBounds, *, noisy: bool, p: int, seed: int, config: Config
) -> tuple[dict[str, object], dict[str, int]]:
    backend = noisy_aer_backend(config.noise_level) if noisy else AerSimulator()
    start = time.perf_counter()
    result = run_qaoa(
        Q,
        p=p,
        sample=aer_sampler(backend, seed=seed),
        shots=config.shots,
        maxiter=config.maxiter,
        final_shots=config.final_shots,
        rng=np.random.default_rng(seed),
    )
    elapsed = time.perf_counter() - start
    quality = counts_quality(result.counts, Q, bounds)
    budget = config.shots * result.num_iterations + config.final_shots
    base = random_sampling_baseline(Q, budget, np.random.default_rng([seed, 1]))
    row: dict[str, object] = {
        "solver": "QAOA_Noisy" if noisy else "QAOA_Ideal",
        "p": p,
        "iterations": result.num_iterations,
        "total_shots": budget,
        "time_s": elapsed,
        **_quality_row(quality.best_energy, bounds, "best"),
        **_quality_row(quality.mean_energy, bounds, "mean"),
        "optimal_found": bool(quality.best_energy <= bounds.min_energy + OPTIMUM_TOL),
        "success_probability": quality.success_probability,
        "random_success_probability": base.success_probability,
        "random_approx_ratio_best": approximation_ratio(base.best_energy, bounds),
        "random_approx_ratio_mean": approximation_ratio(base.mean_energy, bounds),
        "random_optimal_found": bool(base.best_energy <= bounds.min_energy + OPTIMUM_TOL),
        "num_optimal": base.num_optimal,
    }
    return row, result.counts


def run(config: Config, rec: ExperimentRecorder) -> None:
    jobs = (
        [("toy", n) for n in config.toy_sizes]
        + [("random", n) for n in config.random_sizes]
        + [("slice", n) for n in config.slice_sizes]
    )
    if config.fig07:
        jobs.append(("fig07", 12))
    rows, count_rows = [], []
    for family, n in jobs:
        for seed in seed_range(config):
            Q = instance(family, n, seed)
            bounds = energy_bounds(Q)
            common = {"family": family, "n": n, "seed": seed}
            start = time.perf_counter()
            sa = SimulatedAnnealingSolver(
                num_sweeps=config.sa_sweeps, num_restarts=config.sa_restarts, seed=seed
            ).solve(Q)
            rows.append(
                {
                    **common,
                    "solver": "SA",
                    "p": 0,
                    "time_s": time.perf_counter() - start,
                    **_quality_row(sa.energy, bounds, "best"),
                    "optimal_found": bool(sa.energy <= bounds.min_energy + OPTIMUM_TOL),
                    "min_energy": bounds.min_energy,
                    "max_energy": bounds.max_energy,
                }
            )
            for p in config.depths:
                for noisy in (False, True):
                    if noisy and n > config.noisy_max_n:
                        continue
                    row, counts = qaoa_row(Q, bounds, noisy=noisy, p=p, seed=seed, config=config)
                    rows.append(
                        {
                            **common,
                            **row,
                            "min_energy": bounds.min_energy,
                            "max_energy": bounds.max_energy,
                        }
                    )
                    if family == "toy" and n == config.count_table_n and seed == config.seed:
                        for bitstring, count in Counter(counts).most_common(15):
                            x = np.array([int(b) for b in bitstring[::-1]], dtype=float)
                            count_rows.append(
                                {
                                    "solver": row["solver"],
                                    "p": p,
                                    "bitstring": bitstring,
                                    "count": count,
                                    "energy": float(x @ Q @ x),
                                    "optimal": bool(
                                        float(x @ Q @ x) <= bounds.min_energy + OPTIMUM_TOL
                                    ),
                                }
                            )
    runs = pd.DataFrame(rows)
    rec.write_table("runs", runs)
    rec.write_table("top_counts", pd.DataFrame(count_rows))
    rec.write_table(
        "summary",
        summarize_groups(runs, ["family", "n", "solver", "p"], [m for m in METRICS if m in runs]),
    )


EXPERIMENT = Experiment(
    "qaoa_benchmark", FULL, QUICK, run, seed_range, description=__doc__.splitlines()[0]
)

if __name__ == "__main__":
    EXPERIMENT.main()
