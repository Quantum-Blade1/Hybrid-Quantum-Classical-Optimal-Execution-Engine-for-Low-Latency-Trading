"""Real-data evaluation on held-out test days, run once per docs/PROTOCOL.md.

See experiments/real_data.py.

Usage:
    python -m experiments.real_data_test [--quick] [--results-dir results]
"""

from experiments.real_data import TEST as EXPERIMENT

__all__ = ["EXPERIMENT"]

if __name__ == "__main__":
    EXPERIMENT.main()
