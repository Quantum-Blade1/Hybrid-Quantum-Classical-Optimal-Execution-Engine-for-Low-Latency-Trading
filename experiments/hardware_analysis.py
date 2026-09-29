"""Analysis of recovered IBM ibm_fez raw counts for the toy execution QUBO (PROTOCOL.md §9).

Input: `results/ibm_fez_recovered.jsonl`, one JSON object per job (job_id, status,
created, num_bits, shots, counts in Qiskit little-endian bit order, or error), written by
the user's recovery script. Duplicate job records are collapsed (latest status wins).
Energies are recomputed from the counts with `toy_execution_qubo(n)`; nothing reported by
the original run is trusted. Final-sampling and COBYLA-loop jobs are told apart by their
shot counts, and loop jobs are assigned to runs by creation time per n
(`qexec.hardware.jobs`).

Outputs (results/hardware/):
  jobs.csv                 every analysed job, with run and position in the run
  excluded.csv             jobs not analysed, with the reason
  final_tests.csv          per final job: Wilson 95% CI of P(optimum), one-sided exact
                           binomial tests against the uniform mass on the optimum (the
                           pre-specified 'greater', and 'less' for description), and a
                           shot-level bootstrap 95% CI of (mean-energy ratio - uniform)
  runs.csv                 per run: loop jobs, first/best/last loop ratio, final ratio,
                           span of job creation times (not QPU time)
  summary.csv              per n: per-run mean and range (runs are not pooled), number of
                           runs significant, a sign count across runs, and the ideal and
                           noisy Aer p=1 references (mean and range over seeds)
  simulator_reference.csv  toy-family Aer QAOA metrics per (n, solver, p)
  manifest.json            provenance, and the analysed / excluded / incomplete-run job IDs

Without the input file the experiment is skipped (no hardware claim is made). `--quick`
runs on `tests/fixtures/ibm_jobs_sample.jsonl`, a SYNTHETIC fixture (not hardware data).

Usage:
    python -m experiments.hardware_analysis [--quick] [--results-dir results]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.common import Experiment, single_seed
from qexec.experiment import ExperimentRecorder
from qexec.hardware.jobs import assign_runs, read_jobs, screen_and_analyse
from qexec.hardware.stats import (
    binomial_test_greater,
    binomial_test_less,
    bootstrap_ratio_advantage,
    count_energies,
    sign_test_greater,
    wilson_interval,
)
from qexec.optimization.solvers.metrics import OPTIMUM_TOL, energy_bounds
from qexec.optimization.toy import toy_execution_qubo

SIM_SOLVERS = ("QAOA_Ideal", "QAOA_Noisy")
SIM_METRICS = ("success_probability", "approx_ratio_mean", "approx_ratio_best", "optimal_found")
SIM_RANGE_METRICS = ("success_probability", "approx_ratio_mean")
ALPHA = 0.05


@dataclass(frozen=True)
class Config:
    jobs_path: str = "results/ibm_fez_recovered.jsonl"
    simulator_runs: str = "results/qaoa_benchmark/runs.csv"
    synthetic: bool = False
    seed: int = 0
    bootstrap_resamples: int = 10_000


def missing_jobs(config: Config) -> str | None:
    if Path(config.jobs_path).exists():
        return None
    return (
        f"{config.jobs_path} not found; recover the ibm_fez job counts first "
        "(no hardware claim is made without them)"
    )


def simulator_reference(path: Path) -> pd.DataFrame:
    """Toy-family Aer QAOA metrics per (n, solver, p): mean over seeds (`sim_<metric>`),
    and min / max over seeds for the success probability and mean-energy ratio."""
    if not path.exists():
        return pd.DataFrame()
    runs = pd.read_csv(path)
    runs = runs[(runs["family"] == "toy") & runs["solver"].isin(SIM_SOLVERS)]
    if runs.empty:
        return pd.DataFrame()
    runs = runs.assign(optimal_found=runs["optimal_found"].astype(float))
    groups = runs.groupby(["n", "solver", "p"], as_index=False)
    agg: dict[str, Any] = {f"sim_{m}": (m, "mean") for m in SIM_METRICS}
    for m in SIM_RANGE_METRICS:
        agg[f"sim_{m}_min"] = (m, "min")
        agg[f"sim_{m}_max"] = (m, "max")
    agg["sim_num_seeds"] = ("success_probability", "size")
    return groups.agg(**agg)


def final_tests(
    jobs: pd.DataFrame, counts: dict[str, dict[str, int]], config: Config
) -> pd.DataFrame:
    """Per final job of a complete run: Wilson CI and binomial test of P(optimum) against
    the uniform mass on the optimum; bootstrap CI of the mean-energy ratio advantage."""
    rows = []
    final = jobs[(jobs["role"] == "final") & jobs["run_complete"]]
    for _, job in final.iterrows():
        n = int(job["n"])
        Q = toy_execution_qubo(n)
        bounds = energy_bounds(Q)
        energies, weights = count_energies(counts[job["job_id"]], Q)
        shots = int(weights.sum())
        hits = int(weights[energies <= bounds.min_energy + OPTIMUM_TOL].sum())
        p0 = float(job["uniform_success_probability"])
        low, high = wilson_interval(hits, shots)
        rng = np.random.default_rng([config.seed, n, int(job["run"])])
        adv = bootstrap_ratio_advantage(
            energies,
            weights,
            bounds,
            float(job["uniform_approx_ratio_mean"]),
            resamples=config.bootstrap_resamples,
            rng=rng,
        )
        p_value = binomial_test_greater(hits, shots, p0)
        rows.append(
            {
                "n": n,
                "run": int(job["run"]),
                "job_id": job["job_id"],
                "created": job["created"],
                "shots": shots,
                "optimum_hits": hits,
                "success_probability": hits / shots,
                "success_ci_low": low,
                "success_ci_high": high,
                "uniform_success_probability": p0,
                "success_ratio_vs_uniform": hits / shots / p0,
                "binom_p_greater": p_value,
                "success_significant": p_value < ALPHA,
                "binom_p_less": binomial_test_less(hits, shots, p0),
                "approx_ratio_mean": adv.ratio,
                "uniform_approx_ratio_mean": adv.uniform_ratio,
                "ratio_advantage": adv.difference,
                "ratio_advantage_ci_low": adv.ci_low,
                "ratio_advantage_ci_high": adv.ci_high,
                "ratio_advantage_significant": adv.ci_low > 0,
                "bootstrap_resamples": adv.resamples,
                "optimal_found": bool(job["optimal_found"]),
            }
        )
    return pd.DataFrame(rows)


def run_table(jobs: pd.DataFrame) -> pd.DataFrame:
    """Per run: loop jobs and their mean-energy ratios, the final job, and the span of job
    creation times (queueing + execution + classical overhead; not QPU time)."""
    rows = []
    for (n, run), group in jobs.groupby(["n", "run"]):
        g = group.sort_values("iteration")
        loop = g[g["role"] == "optimization"]
        final = g[g["role"] == "final"]
        created = pd.to_datetime(g["created"])
        rows.append(
            {
                "n": int(n),
                "run": int(run),
                "complete": bool(g["run_complete"].iloc[0]),
                "num_loop_jobs": len(loop),
                "loop_shots": int(loop["shots"].iloc[0]) if len(loop) else 0,
                "final_job_id": final["job_id"].iloc[0] if len(final) else "",
                "first_created": g["created"].iloc[0],
                "last_created": g["created"].iloc[-1],
                "creation_span_s": (created.max() - created.min()).total_seconds(),
                "median_creation_interval_s": float(
                    created.sort_values().diff().dt.total_seconds().median()
                ),
                "loop_ratio_first": float(loop["approx_ratio_mean"].iloc[0])
                if len(loop)
                else np.nan,
                "loop_ratio_first5_mean": float(loop["approx_ratio_mean"].head(5).mean())
                if len(loop)
                else np.nan,
                "loop_ratio_last5_mean": float(loop["approx_ratio_mean"].tail(5).mean())
                if len(loop)
                else np.nan,
                "loop_ratio_best": float(loop["approx_ratio_mean"].max()) if len(loop) else np.nan,
                "final_ratio": float(final["approx_ratio_mean"].iloc[0]) if len(final) else np.nan,
                "final_success_probability": float(final["success_probability"].iloc[0])
                if len(final)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarise(tests: pd.DataFrame, runs: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Per n: mean and range over complete runs (not pooled), counts of significant runs, a
    sign count across runs, and the Aer p=1 references."""
    if tests.empty:
        return pd.DataFrame()
    rows = []
    for n, g in tests.groupby("n"):
        r = runs[runs["n"] == n]
        k = len(g)
        above_p = int((g["success_probability"] > g["uniform_success_probability"]).sum())
        above_r = int((g["ratio_advantage"] > 0).sum())
        rows.append(
            {
                "n": int(n),
                "num_runs": k,
                "final_shots": int(g["shots"].iloc[0]),
                "loop_shots": int(r[r["complete"]]["loop_shots"].max()),
                "num_loop_jobs": int(r[r["complete"]]["num_loop_jobs"].sum()),
                "loop_jobs_min": int(r[r["complete"]]["num_loop_jobs"].min()),
                "loop_jobs_max": int(r[r["complete"]]["num_loop_jobs"].max()),
                "incomplete_loop_jobs": int(r[~r["complete"]]["num_loop_jobs"].sum()),
                "success_probability": g["success_probability"].mean(),
                "success_probability_min": g["success_probability"].min(),
                "success_probability_max": g["success_probability"].max(),
                "uniform_success_probability": g["uniform_success_probability"].iloc[0],
                "success_ratio_vs_uniform": g["success_probability"].mean()
                / g["uniform_success_probability"].iloc[0],
                "runs_success_above_uniform": above_p,
                "runs_success_significant": int(g["success_significant"].sum()),
                "runs_success_below_significant": int((g["binom_p_less"] < ALPHA).sum()),
                "sign_p_success": sign_test_greater(above_p, k),
                "approx_ratio_mean": g["approx_ratio_mean"].mean(),
                "approx_ratio_mean_min": g["approx_ratio_mean"].min(),
                "approx_ratio_mean_max": g["approx_ratio_mean"].max(),
                "uniform_approx_ratio_mean": g["uniform_approx_ratio_mean"].iloc[0],
                "ratio_advantage_min": g["ratio_advantage"].min(),
                "ratio_advantage_max": g["ratio_advantage"].max(),
                "runs_ratio_above_uniform": above_r,
                "runs_ratio_significant": int(g["ratio_advantage_significant"].sum()),
                "sign_p_ratio": sign_test_greater(above_r, k),
                "optimal_found_runs": int(g["optimal_found"].sum()),
                "creation_span_s_min": r[r["complete"]]["creation_span_s"].min(),
                "creation_span_s_max": r[r["complete"]]["creation_span_s"].max(),
            }
        )
    summary = pd.DataFrame(rows)
    if not reference.empty:
        for solver in SIM_SOLVERS:
            # Depth 1 is the only depth common to every size of the Aer benchmark, and the
            # configured depth of the hardware runs.
            ref = reference[(reference["solver"] == solver) & (reference["p"] == 1)]
            prefix = f"{solver.lower()}_p1_"
            ref = ref.drop(columns=["solver", "p"]).rename(
                columns={c: c.replace("sim_", prefix) for c in ref.columns}
            )
            summary = summary.merge(ref, on="n", how="left")
    return summary


def run(config: Config, rec: ExperimentRecorder) -> None:
    records = read_jobs(config.jobs_path)
    counts = {str(r.get("job_id")): r.get("counts") or {} for r in records}
    screening = screen_and_analyse(records, seed=config.seed)
    jobs = pd.DataFrame([a.as_dict() for a in screening.analysed])
    excluded = pd.DataFrame(screening.excluded, columns=["job_id", "num_bits", "shots", "reason"])
    reference = simulator_reference(Path(config.simulator_runs))
    tests = runs = summary = pd.DataFrame()
    if not jobs.empty:
        labels = assign_runs(
            zip(jobs["job_id"], jobs["n"], jobs["role"], jobs["created"], strict=True)
        )
        jobs["run"] = [labels[j].run for j in jobs["job_id"]]
        jobs["iteration"] = [labels[j].iteration for j in jobs["job_id"]]
        jobs["run_complete"] = [labels[j].complete for j in jobs["job_id"]]
        jobs = jobs.sort_values(["n", "run", "iteration"]).reset_index(drop=True)
        tests = final_tests(jobs, counts, config)
        runs = run_table(jobs)
        summary = summarise(tests, runs, reference)
    rec.write_table("jobs", jobs)
    rec.write_table("excluded", excluded)
    rec.write_table("final_tests", tests)
    rec.write_table("runs", runs)
    rec.write_table("summary", summary)
    if not reference.empty:
        rec.write_table("simulator_reference", reference)
    rec.note("input", config.jobs_path)
    rec.note(
        "data",
        "SYNTHETIC test fixture, not hardware data" if config.synthetic else "IBM ibm_fez counts",
    )
    lines = Path(config.jobs_path).read_text().splitlines()
    rec.note("records_in_file", sum(1 for line in lines if line.strip()))
    rec.note("unique_jobs", len(records))
    rec.note(
        "run_assignment",
        "per n, by creation time: loop jobs belong to the next final job of the same n; "
        "loop jobs after the last final job form an incomplete run (no final job)",
    )
    rec.note(
        "tests",
        "per final job: Wilson 95% CI; one-sided exact binomial test vs num_optimal/2^n; "
        f"shot-level multinomial bootstrap ({config.bootstrap_resamples} resamples) 95% "
        "percentile CI of mean-energy ratio minus uniform; runs not pooled",
    )
    rec.note("job_ids", jobs["job_id"].tolist() if not jobs.empty else [])
    rec.note(
        "final_job_ids",
        tests["job_id"].tolist() if not tests.empty else [],
    )
    rec.note(
        "incomplete_run_job_ids",
        jobs[~jobs["run_complete"]]["job_id"].tolist() if not jobs.empty else [],
    )
    rec.note("excluded_job_ids", excluded["job_id"].tolist())


EXPERIMENT = Experiment(
    "hardware",
    Config(),
    Config(
        jobs_path="tests/fixtures/ibm_jobs_sample.jsonl",
        simulator_runs="build/quick/results/qaoa_benchmark/runs.csv",
        synthetic=True,
        bootstrap_resamples=1_000,
    ),
    run,
    single_seed,
    description="Recomputed metrics and tests of recovered IBM ibm_fez counts (toy QUBO)",
    precheck=missing_jobs,
)

if __name__ == "__main__":
    EXPERIMENT.main()
