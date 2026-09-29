"""QAOA on IBM Quantum hardware for the toy execution QUBO (needs credentials; not in run_all)."""

import argparse
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common import DEFAULT_RESULTS_DIR, seed_range
from experiments.problems import instance
from qexec.experiment import ExperimentRecorder
from qexec.hardware.ibm import backend_properties, connect_service, run_qaoa_on_hardware
from qexec.optimization.solvers.metrics import (
    OPTIMUM_TOL,
    approximation_ratio,
    counts_quality,
    energy_bounds,
    random_sampling_baseline,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    sizes: tuple[int, ...] = (4, 6, 8, 10)
    depths: tuple[int, ...] = (1, 2)
    shots: int = 1000
    maxiter: int = 30
    backend_name: str = "ibm_fez"
    counts_log: str = "results/hw_jobs.jsonl"
    seed: int = 0
    num_seeds: int = 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default=Config.backend_name)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    token = os.environ.get("IBM_QUANTUM_TOKEN") or os.environ.get("IBM_CLOUD_API_KEY")
    if not token:
        raise SystemExit("Set IBM_QUANTUM_TOKEN, or IBM_CLOUD_API_KEY and IBM_CLOUD_CRN.")
    config = Config(backend_name=args.backend)
    service = connect_service(token=token, instance=os.environ.get("IBM_CLOUD_CRN"))
    backend = service.backend(config.backend_name)
    rows = []
    with ExperimentRecorder(
        "hardware_benchmark", config, seed_range(config), root=args.results_dir
    ) as rec:
        rec.note("backend_properties", backend_properties(backend))
        for n in config.sizes:
            Q = instance("toy", n, 0)
            bounds = energy_bounds(Q)
            for p in config.depths:
                for seed in seed_range(config):
                    start = time.perf_counter()
                    # External failures (queue, quota, network) are logged; other runs go on.
                    try:
                        result, job_ids = run_qaoa_on_hardware(
                            Q,
                            backend,
                            p=p,
                            shots=config.shots,
                            maxiter=config.maxiter,
                            seed=seed,
                            log_path=config.counts_log,
                            metadata={"experiment": "hardware_benchmark", "seed": seed},
                        )
                    except Exception:
                        logger.exception("hardware run failed (n=%d, p=%d)", n, p)
                        continue
                    quality = counts_quality(result.counts, Q, bounds)
                    budget = config.shots * result.num_iterations + 5 * config.shots
                    base = random_sampling_baseline(Q, budget, np.random.default_rng([seed, 1]))
                    rows.append(
                        {
                            "family": "toy",
                            "n": n,
                            "p": p,
                            "seed": seed,
                            "solver": "QAOA_Hardware",
                            "backend": config.backend_name,
                            "job_ids": " ".join(job_ids),
                            "time_s": time.perf_counter() - start,
                            "total_shots": budget,
                            "approx_ratio_best": approximation_ratio(quality.best_energy, bounds),
                            "approx_ratio_mean": approximation_ratio(quality.mean_energy, bounds),
                            "optimal_found": quality.best_energy <= bounds.min_energy + OPTIMUM_TOL,
                            "success_probability": quality.success_probability,
                            "random_success_probability": base.success_probability,
                        }
                    )
        rec.write_table("runs", pd.DataFrame(rows))


if __name__ == "__main__":
    main()
