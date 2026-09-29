import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SPRINGER_RC = {
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    # Reproducible PDFs: no creation date in the metadata.
    "pdf.compression": 6,
}

STRATEGY_COLORS = {
    "TWAP": "#9E9E9E",
    "VWAP": "#2196F3",
    "SA-QUBO": "#FF9800",
    "Hybrid": "#D32F2F",
    "Static": "#9E9E9E",
    "Adaptive": "#2196F3",
}
SOLVER_COLORS = {
    "SA": "#2196F3",
    "QAOA_Ideal": "#4CAF50",
    "QAOA_Noisy": "#FF9800",
    "Random": "#757575",
    "BruteForce": "#D32F2F",
    "Greedy": "#F57C00",
}


class MissingResultsError(FileNotFoundError):
    pass


@dataclass(frozen=True)
class Results:
    """Tables written by the experiments under `root/<experiment>/`."""

    root: Path

    def path(self, experiment: str, name: str) -> Path:
        path = self.root / experiment / name
        if not path.exists():
            raise MissingResultsError(f"{path} (run: python -m experiments.{experiment})")
        return path

    def table(self, experiment: str, name: str) -> pd.DataFrame:
        return pd.read_csv(self.path(experiment, f"{name}.csv"))

    def json(self, experiment: str, name: str) -> Any:
        return json.loads(self.path(experiment, f"{name}.json").read_text())

    def manifest(self, experiment: str) -> dict[str, Any]:
        data: dict[str, Any] = self.json(experiment, "manifest")
        return data


def metric(summary: pd.DataFrame, name: str, **where: Any) -> pd.DataFrame:
    """Rows of a `summarize_groups` table for one metric, filtered by column values."""
    df = summary[summary["metric"] == name]
    for column, value in where.items():
        df = df[df[column] == value]
    return df


def ci_errorbars(df: pd.DataFrame) -> list[list[float]]:
    """Asymmetric error bars (mean - ci_low, ci_high - mean) for matplotlib."""
    return [
        (df["mean"] - df["ci_low"]).clip(lower=0).tolist(),
        (df["ci_high"] - df["mean"]).clip(lower=0).tolist(),
    ]


def new_figure(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
    plt.rcParams.update(SPRINGER_RC)
    return plt.subplots(*args, **kwargs)
