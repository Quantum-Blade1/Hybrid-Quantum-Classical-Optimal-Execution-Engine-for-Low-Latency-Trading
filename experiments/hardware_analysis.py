"""Analysis of recovered IBM ibm_fez raw counts for the toy execution QUBO (PROTOCOL.md §9).

Input: `results/ibm_fez_recovered.jsonl`, one JSON object per job (job_id, status,
created, num_bits, shots, counts in Qiskit little-endian bit order, or error), written by
the user's recovery script. Energies are recomputed from the counts with
`toy_execution_qubo(n)`; nothing reported by the original run is trusted. Final-sampling
and COBYLA-loop jobs are told apart by their shot counts (`qexec.hardware.jobs`). Each job
is compared with uniform random sampling at the same number of shots and, per n, with the
ideal and noisy Aer QAOA results of `qaoa_benchmark` (toy family), when available.

Without the input file the experiment is skipped (no hardware claim is made). `--quick`
runs on `tests/fixtures/ibm_jobs_sample.jsonl`, a SYNTHETIC fixture (not hardware data).

Usage:
    python -m experiments.hardware_analysis [--quick] [--results-dir results]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from experiments.common import Experiment, single_seed
from qexec.experiment import ExperimentRecorder
from qexec.hardware.jobs import read_jobs, screen_and_analyse

SIM_SOLVERS = ("QAOA_Ideal", "QAOA_Noisy")
SIM_METRICS = ("success_probability", "approx_ratio_mean", "approx_ratio_best", "optimal_found")


@dataclass(frozen=True)
class Config:
    jobs_path: str = "results/ibm_fez_recovered.jsonl"
    simulator_runs: str = "results/qaoa_benchmark/runs.csv"
    synthetic: bool = False
    seed: int = 0


def missing_jobs(config: Config) -> str | None:
    if Path(config.jobs_path).exists():
        return None
    return (
        f"{config.jobs_path} not found; recover the ibm_fez job counts first "
        "(no hardware claim is made without them)"
    )


def simulator_reference(path: Path) -> pd.DataFrame:
    """Mean over seeds of the toy-family Aer QAOA metrics per (n, solver, p)."""
    if not path.exists():
        return pd.DataFrame()
    runs = pd.read_csv(path)
    runs = runs[(runs["family"] == "toy") & runs["solver"].isin(SIM_SOLVERS)]
    if runs.empty:
        return pd.DataFrame()
    runs = runs.assign(optimal_found=runs["optimal_found"].astype(float))
    return (
        runs.groupby(["n", "solver", "p"], as_index=False)[list(SIM_METRICS)]
        .mean()
        .rename(columns={m: f"sim_{m}" for m in SIM_METRICS})
    )


def summarise(jobs: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Final-sampling jobs per n next to the uniform baseline and the Aer references."""
    final = jobs[jobs["role"] == "final"]
    if final.empty:
        return pd.DataFrame()
    cols = [
        "success_probability",
        "uniform_success_probability",
        "approx_ratio_mean",
        "uniform_approx_ratio_mean",
        "approx_ratio_best",
        "optimal_found",
        "uniform_prob_optimum_in_shots",
        "counted_shots",
    ]
    summary = final.assign(optimal_found=final["optimal_found"].astype(float))
    summary = summary.groupby("n", as_index=False).agg(
        num_final_jobs=("job_id", "count"), **{c: (c, "mean") for c in cols}
    )
    summary["success_ratio_vs_uniform"] = (
        summary["success_probability"] / summary["uniform_success_probability"]
    )
    if not reference.empty:
        for solver in SIM_SOLVERS:
            # Depth 1 is the only depth common to every size of the Aer benchmark.
            ref = reference[(reference["solver"] == solver) & (reference["p"] == 1)]
            prefix = f"{solver.lower()}_p1_"
            ref = ref.drop(columns=["solver", "p"]).rename(
                columns={c: c.replace("sim_", prefix) for c in ref.columns}
            )
            summary = summary.merge(ref, on="n", how="left")
    return summary


def run(config: Config, rec: ExperimentRecorder) -> None:
    screening = screen_and_analyse(read_jobs(config.jobs_path), seed=config.seed)
    jobs = pd.DataFrame([a.as_dict() for a in screening.analysed])
    excluded = pd.DataFrame(screening.excluded, columns=["job_id", "num_bits", "shots", "reason"])
    if not jobs.empty:
        jobs = jobs.sort_values(["n", "role", "created", "job_id"]).reset_index(drop=True)
    reference = simulator_reference(Path(config.simulator_runs))
    rec.write_table("jobs", jobs)
    rec.write_table("excluded", excluded)
    rec.write_table("summary", summarise(jobs, reference) if not jobs.empty else pd.DataFrame())
    if not reference.empty:
        rec.write_table("simulator_reference", reference)
    rec.note("input", config.jobs_path)
    rec.note(
        "data",
        "SYNTHETIC test fixture, not hardware data" if config.synthetic else "IBM ibm_fez counts",
    )
    rec.note("job_ids", jobs["job_id"].tolist() if not jobs.empty else [])
    rec.note("excluded_job_ids", excluded["job_id"].tolist())


EXPERIMENT = Experiment(
    "hardware",
    Config(),
    Config(
        jobs_path="tests/fixtures/ibm_jobs_sample.jsonl",
        simulator_runs="build/quick/results/qaoa_benchmark/runs.csv",
        synthetic=True,
    ),
    run,
    single_seed,
    description="Recomputed metrics of recovered IBM ibm_fez counts (toy execution QUBO)",
    precheck=missing_jobs,
)

if __name__ == "__main__":
    EXPERIMENT.main()
