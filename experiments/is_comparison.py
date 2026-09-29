"""Implementation-shortfall decomposition of TWAP, VWAP, SA-QUBO and Hybrid (paper fig22)."""

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from experiments.common import Experiment, paired_vs_baselines, seed_range, summarize_groups
from qexec.analysis.runners import STRATEGIES, run_strategy
from qexec.experiment import ExperimentRecorder
from qexec.market.simulator import MarketDataSimulator, MarketParams

COST_METRICS = (
    "shortfall_bps",
    "execution_cost_bps",
    "spread_cost_bps",
    "impact_cost_bps",
    "timing_cost_bps",
    "opportunity_cost_bps",
    "fill_rate",
    "optimization_invocations",
    "optimization_time_s",
)


@dataclass(frozen=True)
class Config:
    total_shares: int = 100_000
    minutes: int = 60
    initial_price: float = 100.0
    annual_volatility: float = 0.30
    daily_volume: int = 50_000_000
    strategies: tuple[str, ...] = STRATEGIES
    baselines: tuple[str, ...] = ("TWAP", "VWAP")
    seed: int = 0
    num_seeds: int = 30


FULL = Config()
QUICK = Config(num_seeds=3)


def market(config: Config, seed: int) -> pd.DataFrame:
    params = MarketParams(
        initial_price=config.initial_price, annual_volatility=config.annual_volatility
    )
    sim = MarketDataSimulator(params, total_daily_volume=config.daily_volume, seed=seed)
    return sim.generate(datetime(2024, 1, 2), num_minutes=config.minutes)


def strategy_runs(
    config: Config, data: pd.DataFrame, seed: int, **labels: object
) -> list[dict[str, object]]:
    rows = []
    for name in config.strategies:
        result = run_strategy(
            name, data, config.total_shares, seed=seed, daily_volume=config.daily_volume
        )
        rows.append({**labels, "seed": seed, **result.metrics()})
    return rows


def run(config: Config, rec: ExperimentRecorder) -> None:
    rows = []
    for seed in seed_range(config):
        rows.extend(strategy_runs(config, market(config, seed), seed))
    runs = pd.DataFrame(rows)
    rec.write_table("runs", runs)
    rec.write_table("summary", summarize_groups(runs, ["strategy"], COST_METRICS))
    paired = pd.concat(
        [
            paired_vs_baselines(
                runs,
                unit_col="seed",
                strategy_col="strategy",
                value_col=metric,
                baselines=config.baselines,
            )
            for metric in ("shortfall_bps", "impact_cost_bps", "timing_cost_bps")
        ]
    )
    rec.write_table("paired", paired)


EXPERIMENT = Experiment(
    "is_comparison", FULL, QUICK, run, seed_range, description=__doc__.splitlines()[0]
)

if __name__ == "__main__":
    EXPERIMENT.main()
