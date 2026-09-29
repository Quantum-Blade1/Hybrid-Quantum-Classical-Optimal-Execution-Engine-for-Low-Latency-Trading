"""Held-out real-data evaluation; run once, per docs/PROTOCOL.md."""

from experiments.real_data import TEST as EXPERIMENT

__all__ = ["EXPERIMENT"]

if __name__ == "__main__":
    EXPERIMENT.main()
