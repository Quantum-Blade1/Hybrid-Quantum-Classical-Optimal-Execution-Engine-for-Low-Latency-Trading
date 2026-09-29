"""Kyle's lambda, VPIN, adverse selection and spread on synthetic ticks with a
high-volatility block (ticks 200-399), over seeds (paper fig10, fig11, fig12).

The synthetic trade side is sign(return), so lambda's rise in the stress block is partly
built into the data (claims audit P29). The series of the first seed is stored for the
figures; the ratios are summarised over all seeds.

Usage:
    python -m experiments.microstructure [--quick] [--results-dir results] [--seed 0]
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from experiments.common import Experiment, seed_range, summarize_groups
from qexec.experiment import ExperimentRecorder
from qexec.microstructure.analyzer import MicrostructureAnalyzer
from qexec.microstructure.kyle import KyleLambdaEstimator
from qexec.microstructure.vpin import VPINEstimator

STRESS_START, STRESS_END = 200, 400
VPIN_THRESHOLD = 0.7


@dataclass(frozen=True)
class Config:
    num_ticks: int = 600
    volatilities: tuple[float, float, float] = (0.001, 0.005, 0.0015)
    kyle_window: int = 50
    vpin_bucket_size: int = 200
    vpin_buckets: int = 20
    seed: int = 0
    num_seeds: int = 30


FULL = Config()
QUICK = Config(num_seeds=3)


def synthetic_ticks(config: Config, seed: int) -> pd.DataFrame:
    """Ticks with volatility v0 before tick 200, v1 on 200-399 and v2 after."""
    rng = np.random.default_rng(seed)
    price = 100.0
    rows = []
    for i in range(config.num_ticks):
        vol = config.volatilities[0 if i < STRESS_START else 1 if i < STRESS_END else 2]
        ret = rng.normal(0, vol)
        price *= 1 + ret
        spread = max(0.01, abs(ret) * price * 2 + 0.01)
        rows.append(
            {
                "tick": i,
                "price": price,
                "bid": price - spread / 2,
                "ask": price + spread / 2,
                "volume": max(10, int(rng.exponential(500))),
                "side": "buy" if ret >= 0 else "sell",
            }
        )
    return pd.DataFrame(rows)


def estimate(config: Config, ticks: pd.DataFrame) -> pd.DataFrame:
    kyle = KyleLambdaEstimator(window_size=config.kyle_window)
    vpin = VPINEstimator(bucket_size=config.vpin_bucket_size, num_buckets=config.vpin_buckets)
    analyzer = MicrostructureAnalyzer(
        kyle_window=config.kyle_window,
        vpin_bucket_size=config.vpin_bucket_size,
        vpin_buckets=config.vpin_buckets,
    )
    prices = ticks["price"].to_numpy()
    out = []
    for i, row in enumerate(ticks.itertuples()):
        prev = prices[i - 1] if i > 0 else prices[i]
        if i > 0:
            signed = row.volume if row.side == "buy" else -row.volume
            kyle.update(prices[i] - prev, signed)
        state = analyzer.process_tick(
            row.price, row.volume, row.bid, row.ask, row.side, timestamp_ns=i * 100_000
        )
        out.append(
            {
                "kyle_lambda": kyle.lambda_value if i > 0 else np.nan,
                "vpin": vpin.update(row.price, row.volume, prev),
                "analyzer_kyle_lambda": state.kyle_lambda,
                "analyzer_vpin": state.vpin,
                "adverse_selection_cost": state.adverse_selection_cost,
                "spread_bps": (row.ask - row.bid) / ((row.ask + row.bid) / 2) * 10_000,
            }
        )
    return pd.concat([ticks, pd.DataFrame(out)], axis=1)


def run(config: Config, rec: ExperimentRecorder) -> None:
    summary_rows = []
    for seed in seed_range(config):
        series = estimate(config, synthetic_ticks(config, seed))
        if seed == config.seed:
            rec.write_table("series", series)
        calm = series.iloc[config.kyle_window : STRESS_START]
        stress = series.iloc[STRESS_START + config.kyle_window : STRESS_END]
        summary_rows.append(
            {
                "seed": seed,
                "kyle_lambda_calm": calm["kyle_lambda"].mean(),
                "kyle_lambda_stress": stress["kyle_lambda"].mean(),
                "kyle_lambda_ratio": stress["kyle_lambda"].mean() / calm["kyle_lambda"].mean(),
                "vpin_above_threshold_frac": float((series["vpin"] > VPIN_THRESHOLD).mean()),
                "vpin_mean_calm": calm["vpin"].mean(),
                "vpin_mean_stress": stress["vpin"].mean(),
                "spread_bps_calm": calm["spread_bps"].mean(),
                "spread_bps_stress": stress["spread_bps"].mean(),
            }
        )
    per_seed = pd.DataFrame(summary_rows)
    per_seed["group"] = "all"
    rec.write_table("per_seed", per_seed.drop(columns="group"))
    metrics = [c for c in per_seed.columns if c not in ("seed", "group")]
    rec.write_table("summary", summarize_groups(per_seed, ["group"], metrics).drop(columns="group"))
    rec.note(
        "windows",
        {
            "calm": [config.kyle_window, STRESS_START],
            "stress": [STRESS_START + config.kyle_window, STRESS_END],
        },
    )


EXPERIMENT = Experiment(
    "microstructure", FULL, QUICK, run, seed_range, description=__doc__.splitlines()[0]
)

if __name__ == "__main__":
    EXPERIMENT.main()
