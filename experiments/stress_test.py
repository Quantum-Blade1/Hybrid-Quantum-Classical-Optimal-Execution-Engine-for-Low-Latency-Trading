"""VWAP vs hybrid (decision layer + SA-QUBO) under four synthetic stress scenarios.

Scenario definitions: qexec.analysis.stress.

Usage:
    python experiments/stress_test.py [--seed 42] [--shares 50000]
"""

import argparse
import logging

from qexec.analysis.stress import StressRunner, results_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shares", type=int, default=50_000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    results = StressRunner(total_shares=args.shares, seed=args.seed).run_suite()
    print(results_table(results).to_string(index=False))
    for r in results:
        if r.crashed:
            mode = "hybrid" if r.is_hybrid else "classical"
            print(f"Crashed: {r.scenario_name} ({mode}): {r.error_msg}")


if __name__ == "__main__":
    main()
