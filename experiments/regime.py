"""Volatility-regime detection and adaptive risk aversion on the microstructure ticks, over
seeds (paper fig15, fig16, fig17).

Stores the first seed's per-tick series (fast/slow EWMA volatility, ratio, regime, lambda)
and, over all seeds, the regime distribution and the lambda range (claims audit P26, P27).

Usage:
    python -m experiments.regime [--quick] [--results-dir results] [--seed 0]
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from experiments.common import Experiment, seed_range, summarize_groups
from experiments.microstructure import Config as TickConfig
from experiments.microstructure import synthetic_ticks
from qexec.experiment import ExperimentRecorder
from qexec.microstructure.regime import AdaptiveRiskManager, VolatilityEstimator

REGIMES = ("low", "normal", "high", "extreme")


@dataclass(frozen=True)
class Config:
    series_ticks: int = 600
    distribution_ticks: int = 1000
    fast_alpha: float = 0.06
    slow_alpha: float = 0.01
    base_lambda: float = 1.0
    seed: int = 0
    num_seeds: int = 30


FULL = Config()
QUICK = Config(num_seeds=3)


def regime_series(ticks: pd.DataFrame, config: Config) -> pd.DataFrame:
    manager = AdaptiveRiskManager(base_lambda=config.base_lambda)
    vol = VolatilityEstimator(fast_alpha=config.fast_alpha, slow_alpha=config.slow_alpha)
    volumes = ticks["volume"].to_numpy(dtype=float)
    rows = []
    for i, row in enumerate(ticks.itertuples()):
        fast, slow = vol.update(row.price)
        state = manager.update(
            row.price, row.bid, row.ask, row.volume, float(np.mean(volumes[: max(1, i)]))
        )
        rows.append(
            {
                "tick": i,
                "price": row.price,
                "fast_vol": fast,
                "slow_vol": slow,
                "vol_ratio": vol.vol_ratio,
                "regime": state.vol_regime.value,
                "lambda": state.lambda_value,
            }
        )
    return pd.DataFrame(rows)


def run(config: Config, rec: ExperimentRecorder) -> None:
    dist_rows = []
    for seed in seed_range(config):
        tick_cfg = replace(TickConfig(), num_ticks=config.distribution_ticks)
        series = regime_series(synthetic_ticks(tick_cfg, seed), config)
        if seed == config.seed:
            short = replace(TickConfig(), num_ticks=config.series_ticks)
            rec.write_table("series", regime_series(synthetic_ticks(short, seed), config))
            rec.write_table("series_1000", series)
        counts = series["regime"].value_counts(normalize=True)
        dist_rows.append(
            {
                "seed": seed,
                **{f"frac_{r}": float(counts.get(r, 0.0)) for r in REGIMES},
                "lambda_min": series["lambda"].min(),
                "lambda_max": series["lambda"].max(),
                "lambda_max_over_base": series["lambda"].max() / config.base_lambda,
            }
        )
    per_seed = pd.DataFrame(dist_rows)
    rec.write_table("per_seed", per_seed)
    per_seed["group"] = "all"
    metrics = [c for c in per_seed.columns if c not in ("seed", "group")]
    rec.write_table("summary", summarize_groups(per_seed, ["group"], metrics).drop(columns="group"))


EXPERIMENT = Experiment("regime", FULL, QUICK, run, seed_range, description=__doc__.splitlines()[0])

if __name__ == "__main__":
    EXPERIMENT.main()
