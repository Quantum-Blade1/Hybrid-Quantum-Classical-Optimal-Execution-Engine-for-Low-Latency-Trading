"""Real-data evaluation on development days (in-sample); see experiments/real_data.py.

Usage:
    python -m experiments.real_data_dev [--quick] [--results-dir results]
"""

from experiments.real_data import DEV as EXPERIMENT

__all__ = ["EXPERIMENT"]

if __name__ == "__main__":
    EXPERIMENT.main()
