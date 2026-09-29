"""Exact p=1 QAOA energy landscape <H>(gamma, beta) of a 4-variable execution QUBO (paper
fig08), from the statevector (no shot noise).

Usage:
    python -m experiments.qaoa_landscape [--quick] [--results-dir results]
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from qiskit.quantum_info import Statevector

from experiments.common import Experiment, single_seed
from qexec.experiment import ExperimentRecorder
from qexec.optimization.ising import build_qaoa_circuit_from_ising, qubo_to_ising
from qexec.optimization.qubo import ExecutionQUBO, QUBOConfig
from qexec.optimization.solvers.metrics import energy_bounds, enumerate_energies


@dataclass(frozen=True)
class Config:
    grid: int = 41
    seed: int = 0


FULL = Config()
QUICK = Config(grid=5)


def run(config: Config, rec: ExperimentRecorder) -> None:
    qubo = ExecutionQUBO(
        QUBOConfig(
            total_shares=200,
            num_time_slices=2,
            num_venues=1,
            quantity_levels=[0, 100],
            equality_penalty=100.0,
        )
    )
    Q = qubo.build_qubo_matrix()
    n = Q.shape[0]
    energies = enumerate_energies(Q)
    bounds = energy_bounds(Q)
    ising = qubo_to_ising(Q)
    optimal = energies <= bounds.min_energy + 1e-6
    rows = []
    for gamma in np.linspace(0, 2 * np.pi, config.grid):
        for beta in np.linspace(0, np.pi, config.grid):
            qc = build_qaoa_circuit_from_ising(ising, [gamma], [beta], 1)
            qc.remove_final_measurements()
            # Statevector index i has bit k = qubit k, matching enumerate_energies.
            probs = Statevector(qc).probabilities()
            rows.append(
                {
                    "gamma": gamma,
                    "beta": beta,
                    "expected_energy": float(probs @ energies),
                    "success_probability": float(probs[optimal].sum()),
                }
            )
    grid = pd.DataFrame(rows)
    rec.write_table("landscape", grid)
    best = grid.loc[grid["expected_energy"].idxmin()]
    rec.write_json(
        "summary",
        {
            "n": n,
            "min_energy": bounds.min_energy,
            "max_energy": bounds.max_energy,
            "uniform_mean_energy": float(energies.mean()),
            "uniform_success_probability": float(optimal.mean()),
            "best_grid_point": best.to_dict(),
        },
    )


EXPERIMENT = Experiment(
    "qaoa_landscape", FULL, QUICK, run, single_seed, description=__doc__.splitlines()[0]
)

if __name__ == "__main__":
    EXPERIMENT.main()
