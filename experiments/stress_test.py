"""
Stress-test suite: VWAP vs hybrid (decision layer + SA-QUBO) under four
synthetic scenarios (flash crash, liquidity crisis, volatility spike,
market outage). See qexec.analysis.stress for the scenario definitions.

Usage:
    python experiments/stress_test.py
"""

from qexec.analysis.stress import StressRunner


def main() -> None:
    StressRunner().run_suite()


if __name__ == "__main__":
    main()
