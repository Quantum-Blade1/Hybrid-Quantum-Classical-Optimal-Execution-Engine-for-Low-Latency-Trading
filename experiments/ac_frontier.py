"""Almgren-Chriss efficient frontier and schedule shapes vs the SA-QUBO schedule (paper
fig19, fig20).

    frontier    E[C] and V[C] of the AC trajectory over 20 risk aversions (10k shares)
    schedules   TWAP, VWAP on the simulator's expected volume curve, AC (lambda = 1e-4),
                and the SA-QUBO schedule both as solved (`qubo_raw`, may not sum to the
                order) and repaired to the order (`qubo_repaired`); 10k shares, 15 slices
    ac_vs_qubo  RMSE between the runtime's SA-QUBO schedule and AC at lambda = 1e-9 and
                1e-4 (50k shares, 20 steps), over seeds. The execution QUBO has no
                risk-aversion term, so it can only be compared with near risk-neutral AC.

Usage:
    python -m experiments.ac_frontier [--quick] [--results-dir results] [--seed 0]
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from experiments.common import Experiment, seed_range, summarize_groups
from qexec.execution.strategies.almgren_chriss import ACConfig, AlmgrenChrissSolver
from qexec.experiment import ExperimentRecorder
from qexec.market.simulator import VolumeProfileGenerator
from qexec.optimization.qubo import ExecutionQUBO, QUBOConfig
from qexec.optimization.schedule import optimize_schedule, repair_schedule, slice_level_config
from qexec.optimization.solvers.annealing import SimulatedAnnealingSolver


@dataclass(frozen=True)
class Config:
    frontier_points: int = 20
    frontier_shares: int = 10_000
    frontier_steps: int = 15
    schedule_shares: int = 10_000
    schedule_steps: int = 15
    comparison_shares: int = 50_000
    comparison_steps: int = 20
    comparison_days: float = 1 / 26
    comparison_risk_aversions: tuple[float, ...] = (1e-9, 1e-4)
    seed: int = 0
    num_seeds: int = 10


FULL = Config()
QUICK = Config(frontier_points=4, num_seeds=2)


def frontier(config: Config) -> pd.DataFrame:
    rows = []
    for lam in np.logspace(-10, -3, config.frontier_points):
        solver = AlmgrenChrissSolver(
            ACConfig(
                total_shares=config.frontier_shares,
                n_steps=config.frontier_steps,
                risk_aversion=lam,
            )
        )
        traj = solver.compute_trajectory()
        rows.append(
            {
                "risk_aversion": lam,
                "expected_cost": solver.calculate_expected_cost(traj),
                "variance": solver.calculate_variance(traj),
                "first_step_fraction": traj["shares_to_trade"].iloc[0] / config.frontier_shares,
            }
        )
    return pd.DataFrame(rows)


def schedules(config: Config) -> pd.DataFrame:
    total, steps = config.schedule_shares, config.schedule_steps
    twap = np.full(steps, total / steps)
    weights = VolumeProfileGenerator._volume_profile_weights(steps)
    vwap = weights / weights.sum() * total
    ac = (
        AlmgrenChrissSolver(ACConfig(total_shares=total, n_steps=steps, risk_aversion=1e-4))
        .compute_trajectory()["shares_to_trade"]
        .to_numpy()
    )
    qubo = ExecutionQUBO(
        QUBOConfig(
            total_shares=total,
            num_time_slices=steps,
            num_venues=1,
            quantity_levels=[0, total // (steps * 2), total // steps],
            equality_penalty=100.0,
            impact_coefficient=0.1,
        )
    )
    raw, _ = optimize_schedule(qubo, SimulatedAnnealingSolver(num_sweeps=500, seed=config.seed))
    repaired = repair_schedule(raw, total)
    return pd.DataFrame(
        {
            "step": np.arange(steps),
            "twap": twap,
            "vwap_expected_profile": vwap,
            "almgren_chriss_1e-4": ac,
            "qubo_raw": raw,
            "qubo_repaired": repaired,
        }
    )


def ac_vs_qubo(config: Config) -> pd.DataFrame:
    rows = []
    qubo = ExecutionQUBO(slice_level_config(config.comparison_shares, config.comparison_steps))
    Q = qubo.build_qubo_matrix()
    for lam in config.comparison_risk_aversions:
        ac = (
            AlmgrenChrissSolver(
                ACConfig(
                    total_shares=config.comparison_shares,
                    n_days=config.comparison_days,
                    n_steps=config.comparison_steps,
                    risk_aversion=lam,
                )
            )
            .compute_trajectory()["shares_to_trade"]
            .to_numpy()
        )
        for seed in seed_range(config):
            # Same QUBO and SA settings as the runtime's slow path (AsyncOptimizer).
            raw, _ = optimize_schedule(qubo, SimulatedAnnealingSolver(num_sweeps=200, seed=seed), Q)
            sched = repair_schedule(raw, config.comparison_shares)
            rows.append(
                {
                    "risk_aversion": lam,
                    "seed": seed,
                    "rmse_shares": float(np.sqrt(np.mean((sched - ac) ** 2))),
                    "rmse_vs_twap_shares": float(
                        np.sqrt(np.mean((sched - config.comparison_shares / len(ac)) ** 2))
                    ),
                    "ac_first_step": float(ac[0]),
                    "qubo_first_step": float(sched[0]),
                }
            )
    return pd.DataFrame(rows)


def run(config: Config, rec: ExperimentRecorder) -> None:
    rec.write_table("frontier", frontier(config))
    rec.write_table("schedules", schedules(config))
    comparison = ac_vs_qubo(config)
    rec.write_table("ac_vs_qubo", comparison)
    rec.write_table(
        "ac_vs_qubo_summary",
        summarize_groups(comparison, ["risk_aversion"], ["rmse_shares", "rmse_vs_twap_shares"]),
    )


EXPERIMENT = Experiment(
    "ac_frontier", FULL, QUICK, run, seed_range, description=__doc__.splitlines()[0]
)

if __name__ == "__main__":
    EXPERIMENT.main()
