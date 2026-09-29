"""QUBO formulation checks and sensitivities (paper fig03, fig04, fig18, fig27, fig28)."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from qiskit import transpile

from experiments.common import Experiment, seed_range, summarize_groups
from qexec.experiment import ExperimentRecorder
from qexec.optimization.hft_qubo import HFTExecutionQUBO, HFTQUBOConfig
from qexec.optimization.ising import binary_to_spins, build_qaoa_circuit_from_ising, qubo_to_ising
from qexec.optimization.solvers.annealing import SimulatedAnnealingSolver

COST_TERMS = (
    "impact_cost",
    "timing_cost",
    "transaction_cost",
    "adverse_selection_cost",
    "inventory_risk_cost",
    "information_leakage_cost",
)
BASIS_GATES = ["cx", "rz", "sx", "x"]


@dataclass(frozen=True)
class Config:
    ising_sizes: tuple[int, ...] = (4, 6, 8, 10, 12, 14, 16, 18, 20)
    ising_samples: int = 200
    sensitivity_points: int = 15
    circuit_sizes: tuple[int, ...] = (4, 6, 8, 10, 12)
    circuit_depths: tuple[int, ...] = (1, 2, 3)
    venue_ticks: int = 15
    sa_sweeps: int = 500
    seed: int = 0
    num_seeds: int = 10


FULL = Config()
QUICK = Config(
    ising_sizes=(4, 8),
    ising_samples=20,
    sensitivity_points=3,
    circuit_sizes=(4, 6),
    venue_ticks=5,
    num_seeds=2,
)


def cost_breakdown(config: Config) -> pd.DataFrame:
    qubo = HFTExecutionQUBO(
        HFTQUBOConfig(
            total_shares=2000,
            num_tick_slices=10,
            num_venues=3,
            quantity_levels=[0, 100, 250, 500],
            kyle_lambda=0.001,
            vpin=0.4,
            adverse_selection_cost=0.0002,
        )
    )
    Q = qubo.build_qubo_matrix()
    rows = []
    for seed in seed_range(config):
        result = SimulatedAnnealingSolver(num_sweeps=config.sa_sweeps, seed=seed).solve(Q)
        costs = qubo.calculate_cost_breakdown(result.solution)
        total_abs = sum(abs(costs[t]) for t in COST_TERMS)
        for term in COST_TERMS:
            rows.append(
                {
                    "seed": seed,
                    "term": term,
                    "cost": costs[term],
                    "share_of_abs_total": abs(costs[term]) / total_abs if total_abs else 0.0,
                    "energy": result.energy,
                    "selected_shares": float(qubo.slice_quantities(result.solution).sum()),
                }
            )
    return pd.DataFrame(rows)


def ising_check(config: Config) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    rows = []
    for n in config.ising_sizes:
        A = np.random.default_rng([config.seed, n]).standard_normal((n, n))
        Q = (A + A.T) / 2
        ising = qubo_to_ising(Q)
        abs_err, rel_err = 0.0, 0.0
        for _ in range(config.ising_samples):
            x = rng.integers(0, 2, n)
            qubo_val = float(x @ Q @ x)
            err = abs(qubo_val - ising.evaluate(binary_to_spins(x)))
            abs_err = max(abs_err, err)
            rel_err = max(rel_err, err / max(abs(qubo_val), 1e-12))
        terms = int(np.sum(np.abs(ising.h) > 1e-10) + np.sum(np.abs(ising.J) > 1e-10))
        rows.append(
            {
                "n": n,
                "samples": config.ising_samples,
                "max_abs_error": abs_err,
                "max_rel_error": rel_err,
                "hamiltonian_terms": terms,
                "max_terms": n + n * (n - 1) // 2,
            }
        )
    return pd.DataFrame(rows)


def sensitivity(config: Config) -> pd.DataFrame:
    rows = []
    sweeps = [
        ("impact_weight_multiplier", np.linspace(0.2, 5.0, config.sensitivity_points)),
        ("vpin", np.linspace(0.0, 0.9, config.sensitivity_points)),
    ]
    for parameter, values in sweeps:
        for value in values:
            kwargs = {"vpin": 0.3}
            if parameter == "vpin":
                kwargs = {"vpin": float(value)}
            else:
                kwargs["impact_weight"] = 0.25 * float(value)
            qubo = HFTExecutionQUBO(
                HFTQUBOConfig(
                    total_shares=1000,
                    num_tick_slices=5,
                    num_venues=2,
                    quantity_levels=[0, 100, 250, 500],
                    **kwargs,
                )
            )
            Q = qubo.build_qubo_matrix()
            for seed in seed_range(config):
                result = SimulatedAnnealingSolver(num_sweeps=300, seed=seed).solve(Q)
                rows.append(
                    {"parameter": parameter, "value": value, "seed": seed, "energy": result.energy}
                )
    return pd.DataFrame(rows)


def venue_routing(config: Config) -> pd.DataFrame:
    ticks = config.venue_ticks
    qubo = HFTExecutionQUBO(
        HFTQUBOConfig(
            total_shares=200 * ticks,
            num_tick_slices=ticks,
            num_venues=3,
            quantity_levels=[0, 100, 250, 500],
            kyle_lambda=0.001,
            vpin=0.4,
        )
    )
    Q = qubo.build_qubo_matrix()
    rows = []
    for seed in seed_range(config):
        result = SimulatedAnnealingSolver(num_sweeps=config.sa_sweeps, seed=seed).solve(Q)
        for entry in qubo.interpret_solution(result.solution)["schedule"]:
            rows.append(
                {
                    "seed": seed,
                    "tick": entry["tick"],
                    "venue": entry["venue"],
                    "shares": entry["quantity"],
                }
            )
    return pd.DataFrame(rows)


def circuit_scaling(config: Config) -> pd.DataFrame:
    rows = []
    for n in config.circuit_sizes:
        A = np.random.default_rng([config.seed, n]).standard_normal((n, n))
        ising = qubo_to_ising((A + A.T) / 2)
        for p in config.circuit_depths:
            qc = build_qaoa_circuit_from_ising(ising, np.full(p, 0.5), np.full(p, 0.3), p)
            compiled = transpile(
                qc, basis_gates=BASIS_GATES, optimization_level=1, seed_transpiler=config.seed
            )
            ops = compiled.count_ops()
            rows.append(
                {
                    "n": n,
                    "p": p,
                    "logical_depth": qc.depth(),
                    "logical_gates": sum(qc.count_ops().values()),
                    "transpiled_depth": compiled.depth(),
                    "transpiled_gates": sum(ops.values()),
                    "transpiled_cx": ops.get("cx", 0),
                }
            )
    return pd.DataFrame(rows)


def run(config: Config, rec: ExperimentRecorder) -> None:
    costs = cost_breakdown(config)
    rec.write_table("cost_breakdown", costs)
    rec.write_table(
        "cost_breakdown_summary", summarize_groups(costs, ["term"], ["share_of_abs_total", "cost"])
    )
    rec.write_table("ising_check", ising_check(config))
    rec.write_table("sensitivity", sensitivity(config))
    rec.write_table("venue_routing", venue_routing(config))
    rec.write_table("circuit_scaling", circuit_scaling(config))
    rec.note("circuit_basis_gates", BASIS_GATES)


EXPERIMENT = Experiment(
    "formulation", FULL, QUICK, run, seed_range, description=__doc__.splitlines()[0]
)

if __name__ == "__main__":
    EXPERIMENT.main()
