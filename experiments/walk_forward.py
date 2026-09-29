"""Seeded walk-forward backtest: TWAP, static VWAP, adaptive VWAP and SA-QUBO ("Hybrid")
over rolling windows, repeated over seeds (paper fig21).

Each seed simulates `total_days` chained days; each window trains on `train_days` and
tests on the next day (qexec.analysis.walk_forward). The unit of the paired statistics
is the per-seed mean over windows (windows of one seed share a price history).

Usage:
    python -m experiments.walk_forward [--quick] [--results-dir results] [--seed 0]
"""

from dataclasses import dataclass

import pandas as pd

from experiments.common import Experiment, paired_vs_baselines, seed_range, summarize_groups
from qexec.analysis.walk_forward import WalkForwardAnalyzer
from qexec.experiment import ExperimentRecorder


@dataclass(frozen=True)
class Config:
    total_days: int = 13
    train_days: int = 3
    daily_shares: int = 500_000
    baselines: tuple[str, ...] = ("TWAP", "Static", "Adaptive")
    seed: int = 3000
    num_seeds: int = 30


FULL = Config()
QUICK = Config(total_days=5, num_seeds=3)


def run(config: Config, rec: ExperimentRecorder) -> None:
    rows = []
    for seed in seed_range(config):
        analyzer = WalkForwardAnalyzer(
            total_days=config.total_days,
            train_days=config.train_days,
            test_days=1,
            daily_shares=config.daily_shares,
            seed=seed,
        )
        for window in analyzer.run():
            for strategy, shortfall in window.shortfall_bps.items():
                rows.append(
                    {
                        "seed": seed,
                        "window": window.window_id,
                        "train_volatility": window.train_volatility,
                        "strategy": strategy,
                        "shortfall_bps": shortfall,
                        "fill_rate": window.fill_rate[strategy],
                    }
                )
    windows = pd.DataFrame(rows)
    per_seed = windows.groupby(["seed", "strategy"], as_index=False, sort=False)[
        ["shortfall_bps", "fill_rate"]
    ].mean()
    wins = (
        windows.loc[windows.groupby(["seed", "window"])["shortfall_bps"].idxmin(), "strategy"]
        .value_counts()
        .rename_axis("strategy")
        .reset_index(name="windows_won")
    )
    rec.write_table("windows", windows)
    rec.write_table("per_seed", per_seed)
    rec.write_table(
        "summary", summarize_groups(per_seed, ["strategy"], ["shortfall_bps", "fill_rate"])
    )
    rec.write_table(
        "paired",
        paired_vs_baselines(
            per_seed,
            unit_col="seed",
            strategy_col="strategy",
            value_col="shortfall_bps",
            baselines=config.baselines,
        ),
    )
    rec.write_table("wins", wins)


EXPERIMENT = Experiment(
    "walk_forward", FULL, QUICK, run, seed_range, description=__doc__.splitlines()[0]
)

if __name__ == "__main__":
    EXPERIMENT.main()
