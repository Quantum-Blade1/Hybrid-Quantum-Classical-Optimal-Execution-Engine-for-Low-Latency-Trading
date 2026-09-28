"""Stress scenarios on 60 simulated minutes and a VWAP-vs-hybrid runner.

Scenarios: flash crash (50% linear drop over 5 minutes from minute 20, then a slow
recovery), liquidity crisis (spread 10x and volume -90% over minutes 20-40), volatility
spike (N(0, $5) price jumps over minutes 15-44) and market outage (minutes 25-35 with
zero volume, frozen price and a $1000 spread).
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from qexec.analysis.runners import ExecutionResult, run_hybrid_execution, run_vwap_execution
from qexec.market.simulator import MarketDataSimulator, MarketParams

logger = logging.getLogger(__name__)

SCENARIO_MINUTES = 60
BASE_VOLATILITY = 0.0002


@dataclass(frozen=True)
class StressResult:
    scenario_name: str
    is_hybrid: bool
    total_cost: float
    slippage_bps: float
    filled_shares: int
    fill_rate: float
    crashed: bool = False
    error_msg: str = ""


def _base_data(num_minutes: int, seed: int, params: MarketParams | None = None) -> pd.DataFrame:
    data = MarketDataSimulator(params=params, seed=seed).generate(num_minutes=num_minutes)
    data["volatility"] = BASE_VOLATILITY
    data["is_stress"] = False
    return data


class StressGenerator:
    """Scenario market data; every generator is deterministic given `seed`."""

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


class StressRunner:
    """Runs VWAP and the hybrid runner on each scenario and tabulates fills and costs."""

    def __init__(self, total_shares: int = 50_000, seed: int = 42) -> None:
        self.total_shares = total_shares
        self.seed = seed

    def scenarios(self) -> list[tuple[str, pd.DataFrame]]:
        return [
            ("Flash Crash", StressGenerator.flash_crash(seed=self.seed)),
            ("Liquidity Crisis", StressGenerator.liquidity_crisis(seed=self.seed)),
            ("Volatility Spike", StressGenerator.volatility_spike(seed=self.seed)),
            ("Market Outage", StressGenerator.market_outage(seed=self.seed)),
        ]

    def _result(self, name: str, is_hybrid: bool, res: ExecutionResult) -> StressResult:
        return StressResult(
            scenario_name=name,
            is_hybrid=is_hybrid,
            total_cost=res.total_cost,
            slippage_bps=res.slippage_bps,
            filled_shares=res.executed_shares,
            fill_rate=res.executed_shares / self.total_shares,
        )

    def run_scenario(self, name: str, data: pd.DataFrame) -> list[StressResult]:
        """One result per mode; a mode that raises is recorded as crashed."""
        results = []
        modes = [
            (False, lambda: run_vwap_execution(data, self.total_shares, seed=self.seed)),
            (
                True,
                lambda: run_hybrid_execution(
                    data, self.total_shares, lambda_tradeoff=0.5, seed=self.seed
                ),
            ),
        ]
        for is_hybrid, run in modes:
            mode = "hybrid" if is_hybrid else "classical"
            # Stress data can break any stage of either pipeline; record it as a crash.
            try:
                results.append(self._result(name, is_hybrid, run()))
            except Exception as e:
                logger.exception("%s: %s run crashed", name, mode)
                results.append(
                    StressResult(
                        scenario_name=name,
                        is_hybrid=is_hybrid,
                        total_cost=0.0,
                        slippage_bps=0.0,
                        filled_shares=0,
                        fill_rate=0.0,
                        crashed=True,
                        error_msg=str(e),
                    )
                )
        return results

    def run_suite(self) -> list[StressResult]:
        results = []
        for name, data in self.scenarios():
            results.extend(self.run_scenario(name, data))
        return results


def results_table(results: list[StressResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Scenario": r.scenario_name,
                "Mode": "Hybrid" if r.is_hybrid else "Classical",
                "Crashed": "YES" if r.crashed else "No",
                "Fill %": f"{r.fill_rate * 100:.1f}%",
                "Slippage": f"{r.slippage_bps:.1f}",
                "Cost": f"${r.total_cost:,.0f}",
            }
            for r in results
        ]
    )
