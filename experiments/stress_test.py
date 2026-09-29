"""TWAP, VWAP, SA-QUBO and Hybrid under four synthetic stress scenarios, over seeds
(paper fig23).

Scenarios (qexec.analysis.stress): flash crash (50% linear drop over 5 minutes, then a
slow recovery), liquidity crisis (spread 10x, volume -90% for 20 minutes), volatility
spike (N(0, $5) price jumps for 30 minutes) and market outage (10 minutes with zero
volume). They differ from the scenarios described in the paper (claims audit P14).
Nothing fills in a zero-volume bar; shares carry forward; unfilled shares count as
opportunity cost.

Usage:
    python -m experiments.stress_test [--quick] [--results-dir results] [--seed 0]
"""

from dataclasses import dataclass

import pandas as pd

from experiments.common import Experiment, paired_vs_baselines, seed_range, summarize_groups
from qexec.analysis.runners import STRATEGIES
from qexec.analysis.stress import SCENARIOS, StressRunner, scenario_data
from qexec.experiment import ExperimentRecorder

METRICS = ("shortfall_bps", "execution_cost_bps", "opportunity_cost_bps", "fill_rate")


@dataclass(frozen=True)
class Config:
    total_shares: int = 50_000
    scenarios: tuple[str, ...] = SCENARIOS
    strategies: tuple[str, ...] = STRATEGIES
    baselines: tuple[str, ...] = ("TWAP", "VWAP")
    seed: int = 2000
    num_seeds: int = 30


FULL = Config()
QUICK = Config(num_seeds=3)


def run(config: Config, rec: ExperimentRecorder) -> None:
    rows = []
    for seed in seed_range(config):
        runner = StressRunner(config.total_shares, seed=seed, strategies=config.strategies)
        for name in config.scenarios:
            rows.extend(vars(r) for r in runner.run_scenario(name, scenario_data(name, seed)))
    runs = pd.DataFrame(rows).rename(columns={"scenario_name": "scenario"})
    crashed = runs[runs["crashed"]]
    rec.write_table("runs", runs)
    rec.write_table("summary", summarize_groups(runs, ["scenario", "strategy"], METRICS))
    paired = pd.concat(
        [
            paired_vs_baselines(
                runs,
                unit_col="seed",
                strategy_col="strategy",
                value_col=metric,
                baselines=config.baselines,
                group_cols=["scenario"],
            )
            for metric in ("shortfall_bps", "fill_rate")
        ]
    )
    rec.write_table("paired", paired)
    rec.note("crashed_runs", len(crashed))


EXPERIMENT = Experiment(
    "stress_test", FULL, QUICK, run, seed_range, description=__doc__.splitlines()[0]
)

if __name__ == "__main__":
    EXPERIMENT.main()
