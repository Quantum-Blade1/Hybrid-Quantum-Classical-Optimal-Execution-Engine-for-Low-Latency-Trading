import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from qexec.analysis.runners import STRATEGIES, run_strategy
from qexec.market.simulator import MarketDataSimulator, MarketParams

logger = logging.getLogger(__name__)

SCENARIO_MINUTES = 60
BASE_VOLATILITY = 0.0002


@dataclass(frozen=True)
class StressResult:
    """One strategy on one scenario and seed; costs in bps of arrival notional."""

    scenario_name: str
    strategy: str
    seed: int
    fill_rate: float
    shortfall_bps: float
    execution_cost_bps: float
    opportunity_cost_bps: float
    crashed: bool = False
    error_msg: str = ""


def _base_data(num_minutes: int, seed: int, params: MarketParams | None = None) -> pd.DataFrame:
    data = MarketDataSimulator(params=params, seed=seed).generate(num_minutes=num_minutes)
    data["volatility"] = BASE_VOLATILITY
    data["is_stress"] = False
    return data


class StressGenerator:
    @staticmethod
    def flash_crash(
        num_minutes: int = SCENARIO_MINUTES,
        crash_start: int = 20,
        crash_duration: int = 5,
        drop_pct: float = 0.50,
        seed: int = 42,
    ) -> pd.DataFrame:
        data = _base_data(num_minutes, seed, MarketParams(initial_price=100.0))
        peak_price = data.iloc[crash_start]["price"]
        target_price = peak_price * (1 - drop_pct)
        crash_step = (peak_price - target_price) / crash_duration
        for i in range(crash_duration):
            idx = crash_start + i
            if idx < len(data):
                data.at[idx, "price"] = peak_price - crash_step * (i + 1)
                data.at[idx, "volatility"] = 0.05
                data.at[idx, "spread"] = 0.50

        recovery_start = crash_start + crash_duration
        for i in range(recovery_start, len(data)):
            data.at[i, "price"] = target_price * (1 + 0.005 * (i - recovery_start))
            data.at[i, "volatility"] = 0.01

        data.loc[crash_start : crash_start + crash_duration + 10, "is_stress"] = True
        return data

    @staticmethod
    def liquidity_crisis(num_minutes: int = SCENARIO_MINUTES, seed: int = 42) -> pd.DataFrame:
        data = _base_data(num_minutes, seed)
        start, end = 20, 40
        data["spread"] = 0.02
        data.loc[start:end, "spread"] = 0.20
        data.loc[start:end, "volume"] = (data.loc[start:end, "volume"] * 0.1).astype(int)
        data.loc[start:end, "is_stress"] = True
        return data

    @staticmethod
    def volatility_spike(num_minutes: int = SCENARIO_MINUTES, seed: int = 42) -> pd.DataFrame:
        data = _base_data(num_minutes, seed)
        rng = np.random.default_rng(seed)
        start, end = 15, 45
        data.loc[start : end - 1, "price"] += rng.normal(0, 5.0, end - start)
        data.loc[start : end - 1, "volatility"] = 0.02
        data.loc[start:end, "is_stress"] = True
        return data

    @staticmethod
    def market_outage(num_minutes: int = SCENARIO_MINUTES, seed: int = 42) -> pd.DataFrame:
        data = _base_data(num_minutes, seed)
        start, end = 25, 35
        data.loc[start:end, "volume"] = 0
        data.loc[start:end, "spread"] = 1000.0
        data.loc[start:end, "price"] = data.loc[start - 1, "price"]
        data.loc[start:end, "is_stress"] = True
        return data


SCENARIOS = ("Flash Crash", "Liquidity Crisis", "Volatility Spike", "Market Outage")


def scenario_data(name: str, seed: int) -> pd.DataFrame:
    generators: dict[str, Callable[..., pd.DataFrame]] = {
        "Flash Crash": StressGenerator.flash_crash,
        "Liquidity Crisis": StressGenerator.liquidity_crisis,
        "Volatility Spike": StressGenerator.volatility_spike,
        "Market Outage": StressGenerator.market_outage,
    }
    return generators[name](seed=seed)


class StressRunner:
    """All strategies on a (scenario, seed) share the price path and the per-minute books."""

    def __init__(
        self,
        total_shares: int = 50_000,
        seed: int = 42,
        strategies: tuple[str, ...] = STRATEGIES,
    ) -> None:
        self.total_shares = total_shares
        self.seed = seed
        self.strategies = strategies

    def run_scenario(self, name: str, data: pd.DataFrame) -> list[StressResult]:
        results = []
        for strategy in self.strategies:
            # Stress data can break any stage of a pipeline; record it as a crash.
            try:
                res = run_strategy(strategy, data, self.total_shares, seed=self.seed)
            except Exception as e:
                logger.exception("%s: %s crashed", name, strategy)
                results.append(
                    StressResult(name, strategy, self.seed, 0.0, 0.0, 0.0, 0.0, True, str(e))
                )
                continue
            m = res.metrics()
            results.append(
                StressResult(
                    scenario_name=name,
                    strategy=strategy,
                    seed=self.seed,
                    fill_rate=res.fill_rate,
                    shortfall_bps=res.shortfall_bps,
                    execution_cost_bps=res.execution_cost_bps,
                    opportunity_cost_bps=float(m["opportunity_cost_bps"]),
                )
            )
        return results

    def run_suite(self) -> list[StressResult]:
        results = []
        for name in SCENARIOS:
            results.extend(self.run_scenario(name, scenario_data(name, self.seed)))
        return results


def results_table(results: list[StressResult]) -> pd.DataFrame:
    return pd.DataFrame([vars(r) for r in results])
