"""Analysis of recovered IBM Quantum job records (raw counts) for the toy execution QUBO.

Input records (one JSON object per line) carry `job_id`, `status`, `created`,
`num_bits`, `shots` and `counts`; count keys are Qiskit bitstrings (little-endian: qubit 0
is the rightmost character), as `qexec.optimization.solvers.metrics.counts_quality`
expects.

Job roles. The original hardware script (`qexec.hardware.ibm.run_qaoa_on_hardware`)
sampled `shots` per COBYLA evaluation and `5 * shots` for the final distribution. The
configuration of the ibm_fez run said shots = 2000 (final 10,000); because the record of
which configuration was actually used is not verifiable (claims audit F10), the loop sizes
{1000, 2000, 4000} and final sizes {5000, 10000, 20000} are all admitted. The two sets
do not overlap, so the role follows from the shot count alone: `final` if shots in
FINAL_SHOTS, `optimization` if shots in LOOP_SHOTS, `unknown` otherwise (excluded and
reported, never guessed). When a record has no `shots`, the total of its counts is used.
QAOA depth p and the COBYLA iteration are not recoverable from counts; `p` is carried
through only if the record has it. Several final jobs for one n (depths, retries) are
analysed separately. Only completed jobs whose number of measured bits is a benchmark size
are analysed.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from qexec.optimization.solvers.metrics import (
    OPTIMUM_TOL,
    approximation_ratio,
    counts_quality,
    energy_bounds,
    enumerate_energies,
)
from qexec.optimization.toy import toy_execution_qubo

SIZES = (4, 6, 8, 10)
FINAL_SHOTS = frozenset({5_000, 10_000, 20_000})
LOOP_SHOTS = frozenset({1_000, 2_000, 4_000})
DONE_STATUSES = frozenset({"DONE", "COMPLETED", "JobStatus.DONE"})


def job_role(shots: int) -> str:
    if shots in FINAL_SHOTS:
        return "final"
    if shots in LOOP_SHOTS:
        return "optimization"
    return "unknown"


def read_jobs(path: str | Path) -> list[dict[str, Any]]:
    """Records of a JSONL file; blank lines are skipped, malformed lines raise."""
    records = []
    for number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: not JSON ({exc})") from exc
    return records


@dataclass(frozen=True)
class JobAnalysis:
    """Quality of one job's counts on toy_execution_qubo(n) and the uniform baseline."""

    job_id: str
    created: str
    n: int
    role: str
    shots: int
    counted_shots: int
    success_probability: float
    mean_energy: float
    approx_ratio_mean: float
    best_energy: float
    approx_ratio_best: float
    optimal_found: bool
    min_energy: float
    max_energy: float
    num_optimal: int
    uniform_success_probability: float
    uniform_mean_energy: float
    uniform_approx_ratio_mean: float
    uniform_prob_optimum_in_shots: float
    uniform_best_energy_sample: float
    p: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def analyse_counts(
    job_id: str,
    created: str,
    n: int,
    shots: int,
    counts: dict[str, int],
    *,
    seed: int = 0,
    p: int | None = None,
) -> JobAnalysis:
    Q = toy_execution_qubo(n)
    bounds = energy_bounds(Q)
    quality = counts_quality(counts, Q, bounds)
    energies = enumerate_energies(Q)
    num_optimal = int(np.sum(energies <= bounds.min_energy + OPTIMUM_TOL))
    p_uniform = num_optimal / 2**n
    total = quality.shots
    sample = np.random.default_rng([seed, n, total]).integers(0, 2**n, size=total)
    return JobAnalysis(
        job_id=job_id,
        created=created,
        n=n,
        role=job_role(shots),
        shots=shots,
        counted_shots=total,
        success_probability=quality.success_probability,
        mean_energy=quality.mean_energy,
        approx_ratio_mean=approximation_ratio(quality.mean_energy, bounds),
        best_energy=quality.best_energy,
        approx_ratio_best=approximation_ratio(quality.best_energy, bounds),
        optimal_found=bool(quality.best_energy <= bounds.min_energy + OPTIMUM_TOL),
        min_energy=bounds.min_energy,
        max_energy=bounds.max_energy,
        num_optimal=num_optimal,
        uniform_success_probability=p_uniform,
        uniform_mean_energy=float(energies.mean()),
        uniform_approx_ratio_mean=approximation_ratio(float(energies.mean()), bounds),
        uniform_prob_optimum_in_shots=float(1 - (1 - p_uniform) ** total),
        uniform_best_energy_sample=float(energies[sample].min()),
        p=p,
    )


@dataclass(frozen=True)
class Screening:
    """Records split into analysed jobs and exclusions (with reasons)."""

    analysed: list[JobAnalysis]
    excluded: list[dict[str, Any]]


def screen_and_analyse(records: Iterable[dict[str, Any]], seed: int = 0) -> Screening:
    analysed, excluded = [], []
    for record in records:
        job_id = str(record.get("job_id", "?"))
        status = str(record.get("status", ""))
        n = int(record.get("num_bits") or 0)
        counts = record.get("counts") or {}
        shots = int(record.get("shots") or sum(int(v) for v in counts.values()))
        depth = record.get("p")
        reason = None
        if status.upper() not in {s.upper() for s in DONE_STATUSES}:
            reason = f"status {status!r}"
        elif n not in SIZES:
            reason = f"num_bits {n} not a benchmark size"
        elif not counts:
            reason = "no counts"
        elif any(len(k) != n or set(k) - {"0", "1"} for k in counts):
            reason = "count keys do not match num_bits"
        elif job_role(shots) == "unknown":
            reason = f"shots {shots} match neither final nor optimisation jobs"
        if reason is not None:
            excluded.append({"job_id": job_id, "num_bits": n, "shots": shots, "reason": reason})
            continue
        analysed.append(
            analyse_counts(
                job_id,
                str(record.get("created", "")),
                n,
                shots,
                {str(k): int(v) for k, v in counts.items()},
                seed=seed,
                p=int(depth) if depth is not None else None,
            )
        )
    return Screening(analysed, excluded)
