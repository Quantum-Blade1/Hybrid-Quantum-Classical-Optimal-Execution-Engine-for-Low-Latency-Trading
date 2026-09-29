"""Run every experiment that feeds the paper (IBM hardware excluded).

Usage:
    python -m experiments.run_all [--quick] [--results-dir DIR] [--only NAME ...]

Full runs write to results/; --quick uses tiny sizes and writes to build/quick/results/
unless --results-dir is given. Exits non-zero if any experiment fails.
"""

import argparse
import importlib
import logging
import sys
import time
import traceback
from pathlib import Path

from experiments.common import DEFAULT_RESULTS_DIR, QUICK_RESULTS_DIR, Experiment

# Order: cheap deterministic experiments first, QAOA last.
EXPERIMENTS = (
    "formulation",
    "qaoa_landscape",
    "ac_frontier",
    "microstructure",
    "regime",
    "solver_benchmark",
    "is_comparison",
    "strategy_comparison",
    "stress_test",
    "walk_forward",
    "latency",
    "load_test",
    "qaoa_benchmark",
)


def load(name: str) -> Experiment:
    experiment: Experiment = importlib.import_module(f"experiments.{name}").EXPERIMENT
    return experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--only", nargs="*", choices=EXPERIMENTS, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    results_dir = args.results_dir or (QUICK_RESULTS_DIR if args.quick else DEFAULT_RESULTS_DIR)

    failures = []
    total = time.perf_counter()
    for name in args.only or EXPERIMENTS:
        # Keep running the remaining experiments if one fails; the exit code reports it.
        try:
            elapsed = load(name).execute(quick=args.quick, results_dir=results_dir)
        except Exception:
            traceback.print_exc()
            failures.append(name)
            print(f"{name:<22} FAILED", flush=True)
            continue
        print(f"{name:<22} {elapsed:7.1f}s", flush=True)
    print(f"total {time.perf_counter() - total:.1f}s -> {results_dir}/")
    if failures:
        print(f"failed: {', '.join(failures)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
