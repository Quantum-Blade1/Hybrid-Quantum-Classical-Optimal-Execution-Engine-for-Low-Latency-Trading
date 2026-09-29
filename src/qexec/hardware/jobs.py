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
# Disjoint shot sets, so a job's role follows from its shot count (claims audit F10).
FINAL_SHOTS = frozenset({5_000, 10_000, 20_000})
LOOP_SHOTS = frozenset({1_000, 2_000, 4_000})
DONE_STATUSES = frozenset({"DONE", "COMPLETED", "JobStatus.DONE"})


def job_role(shots: int) -> str:
    if shots in FINAL_SHOTS:
        return "final"
    if shots in LOOP_SHOTS:
        return "optimization"
    return "unknown"


def latest_by_job(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """One record per job_id: the last one seen wins, kept at its first position."""
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        latest[str(record.get("job_id", "?"))] = record
    return list(latest.values())


def read_jobs(path: str | Path) -> list[dict[str, Any]]:
    """`latest_by_job` records of a JSONL file; blank lines are skipped, malformed lines raise."""
    records = []
    for number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: not JSON ({exc})") from exc
    return latest_by_job(records)


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


@dataclass(frozen=True)
class RunLabel:
    """Run number within its n and position in that run (both 1-based; the final job is last)."""

    run: int
    iteration: int
    complete: bool


def assign_runs(jobs: Iterable[tuple[str, int, str, str]]) -> dict[str, RunLabel]:
    """Loop jobs join the next final job of their n by `created`; trailing ones are incomplete."""
    by_n: dict[int, list[tuple[str, str, str]]] = {}
    for job_id, n, role, created in jobs:
        by_n.setdefault(int(n), []).append((created, job_id, role))
    labels: dict[str, RunLabel] = {}
    for items in by_n.values():
        items.sort()
        pending: list[str] = []
        run = 1
        for _, job_id, role in items:
            pending.append(job_id)
            if role == "final":
                for k, jid in enumerate(pending, start=1):
                    labels[jid] = RunLabel(run, k, True)
                pending, run = [], run + 1
        for k, jid in enumerate(pending, start=1):
            labels[jid] = RunLabel(run, k, False)
    return labels
