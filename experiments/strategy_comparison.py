"""TWAP, VWAP, SA-QUBO and Hybrid on a simulated day at three order sizes, over seeds."""

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from experiments.common import Experiment, paired_vs_baselines, seed_range, summarize_groups
from experiments.is_comparison import COST_METRICS
from qexec.analysis.runners import STRATEGIES, run_strategy
from qexec.experiment import ExperimentRecorder
from qexec.market.simulator import TRADING_MINUTES_PER_DAY, MarketDataSimulator, MarketParams


@dataclass(frozen=True)
class Config:
    order_fractions_of_adv: tuple[float, ...] = (0.001, 0.01, 0.05)
    minutes: int = TRADING_MINUTES_PER_DAY
    initial_price: float = 175.0
    annual_volatility: float = 0.22
    daily_volume: int = 60_000_000
    strategies: tuple[str, ...] = STRATEGIES
    baselines: tuple[str, ...] = ("TWAP", "VWAP")
    seed: int = 1000
    num_seeds: int = 30


FULL = Config()
QUICK = Config(num_seeds=3, minutes=60, order_fractions_of_adv=(0.001, 0.05))


def run(config: Config, rec: ExperimentRecorder) -> None:
    params = MarketParams(
        symbol="AAPL",
        initial_price=config.initial_price,
        annual_volatility=config.annual_volatility,
    )
    rows = []
    for seed in seed_range(config):
        sim = MarketDataSimulator(params, total_daily_volume=config.daily_volume, seed=seed)
        data = sim.generate(datetime(2024, 1, 15), num_minutes=config.minutes)
        for fraction in config.order_fractions_of_adv:
            shares = int(fraction * config.daily_volume)
            for name in config.strategies:
                result = run_strategy(
                    name, data, shares, seed=seed, daily_volume=config.daily_volume
                )
                rows.append({"order_fraction_of_adv": fraction, "seed": seed, **result.metrics()})
    runs = pd.DataFrame(rows)
    rec.write_table("runs", runs)
    rec.write_table(
        "summary", summarize_groups(runs, ["order_fraction_of_adv", "strategy"], COST_METRICS)
    )
    paired = pd.concat(
        [
            paired_vs_baselines(
                runs,
                unit_col="seed",
                strategy_col="strategy",
                value_col=metric,
                baselines=config.baselines,
                group_cols=["order_fraction_of_adv"],
            )
            for metric in ("shortfall_bps", "impact_cost_bps")
        ]
    )
    rec.write_table("paired", paired)


EXPERIMENT = Experiment(
    "strategy_comparison", FULL, QUICK, run, seed_range, description=__doc__.splitlines()[0]
)

if __name__ == "__main__":
    EXPERIMENT.main()
