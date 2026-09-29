"""Real-data evaluation on development days (in-sample); see experiments/real_data.py."""

from experiments.real_data import DEV as EXPERIMENT

__all__ = ["EXPERIMENT"]

if __name__ == "__main__":
    EXPERIMENT.main()
