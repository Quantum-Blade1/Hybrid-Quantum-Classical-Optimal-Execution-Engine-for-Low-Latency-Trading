"""Almgren-Chriss optimal trajectories vs the SA-QUBO schedule of the async runtime.

The execution QUBO has no explicit risk-aversion term, so it is compared against AC at
a near risk-neutral lambda (1e-9, effectively TWAP) and at lambda = 1e-4.

Usage:
    python experiments/ac_comparison.py [--seed 42] [--output-dir results]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qexec.execution.strategies.almgren_chriss import ACConfig, AlmgrenChrissSolver
from qexec.optimization.qubo import ExecutionQUBO
from qexec.optimization.schedule import optimize_schedule, slice_level_config
from qexec.optimization.solvers.annealing import SimulatedAnnealingSolver

TOTAL_SHARES = 50_000
N_STEPS = 20
HORIZON_DAYS = 1 / 26
RISK_AVERSIONS = (1e-9, 1e-4)


def run_comparison(risk_aversion: float, seed: int, output_dir: Path) -> None:
    ac_config = ACConfig(
        total_shares=TOTAL_SHARES,
        n_days=HORIZON_DAYS,
        n_steps=N_STEPS,
        risk_aversion=risk_aversion,
    )
    ac_schedule = AlmgrenChrissSolver(ac_config).compute_trajectory()["shares_to_trade"].to_numpy()

    # Same QUBO and SA settings as the runtime's slow-path optimizer (AsyncOptimizer).
    qubo = ExecutionQUBO(slice_level_config(TOTAL_SHARES, N_STEPS))
    hybrid_schedule, _ = optimize_schedule(
        qubo, SimulatedAnnealingSolver(num_sweeps=200, seed=seed)
    )
    rmse = float(np.sqrt(np.mean((hybrid_schedule - ac_schedule) ** 2)))
    print(f"\nlambda = {risk_aversion:.1e}: RMSE (SA-QUBO vs AC) = {rmse:.1f} shares")

    steps = np.arange(N_STEPS)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, ac_schedule, "b-o", label="Almgren-Chriss")
    ax.plot(steps, hybrid_schedule, "r--s", label="SA-QUBO")
    ax.axhline(TOTAL_SHARES / N_STEPS, color="g", linestyle=":", label="TWAP")
    ax.set_title(f"Trajectory Comparison ($\\lambda={risk_aversion:.1e}$)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Shares Traded")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = output_dir / f"ac_vs_hybrid_lambda_{risk_aversion:.1e}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")

    table = pd.DataFrame(
        {
            "Step": steps,
            "AC": ac_schedule,
            "SA-QUBO": hybrid_schedule,
            "Diff": hybrid_schedule - ac_schedule,
        }
    )
    print(table.head().to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for risk_aversion in RISK_AVERSIONS:
        run_comparison(risk_aversion, args.seed, args.output_dir)


if __name__ == "__main__":
    main()
