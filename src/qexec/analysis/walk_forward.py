import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from qexec.execution.engine import ExecutionEngine, ExecutionReport, OrderSide, ParentOrder
from qexec.execution.strategies.base import BaseStrategy
from qexec.execution.strategies.fixed import FixedScheduleStrategy
from qexec.execution.strategies.vwap import VWAPStrategy
from qexec.market.order_book import OrderBook
from qexec.market.simulator import TRADING_DAYS_PER_YEAR, MarketDataSimulator, MarketParams
from qexec.optimization.qubo import ExecutionQUBO
from qexec.optimization.schedule import (
    optimize_schedule,
    repair_schedule,
    slice_level_config,
    spread_over_minutes,
)
from qexec.optimization.solvers.annealing import SimulatedAnnealingSolver

logger = logging.getLogger(__name__)

MINUTES_PER_DAY = 390
_MAX_QUBO_SLICES = 20
_SA_SWEEPS = 300
_MIN_DAILY_VOLATILITY = 0.001


WALK_FORWARD_STRATEGIES = ("TWAP", "Static", "Adaptive", "Hybrid")


@dataclass(frozen=True)
class WindowResult:
    """Mean shortfall (bps, incl. opportunity cost) and fill rate over one window's test days."""

    window_id: int
    shortfall_bps: dict[str, float]
    fill_rate: dict[str, float]
    train_volatility: float

    @property
    def winner(self) -> str:
        return min(self.shortfall_bps, key=lambda name: self.shortfall_bps[name])


class WalkForwardAnalyzer:
    """Rolling train/test windows; all strategies on a test day share one book seed."""

    def __init__(
        self,
        total_days: int = 10,
        train_days: int = 3,
        test_days: int = 1,
        daily_shares: int = 50_000,
        seed: int = 42,
    ) -> None:
        self.total_days = total_days
        self.train_days = train_days
        self.test_days = test_days
        self.daily_shares = daily_shares
        self.seed = seed
        self.market_params = MarketParams(initial_price=100.0, annual_volatility=0.40)

    def generate_full_dataset(self) -> list[pd.DataFrame]:
        """`total_days` days; each opens at the previous day's last price."""
        sim = MarketDataSimulator(self.market_params, seed=self.seed)
        days = []
        price = self.market_params.initial_price
        for day_id in range(self.total_days):
            day = sim.generate(num_minutes=MINUTES_PER_DAY, initial_price=price)
            day["day"] = day_id
            days.append(day)
            price = float(day.iloc[-1]["price"])
        return days

    def run(self) -> list[WindowResult]:
        all_days = self.generate_full_dataset()
        book_seeds = np.random.SeedSequence(self.seed).spawn(self.total_days)
        num_windows = (self.total_days - self.train_days) // self.test_days
        results = []
        for w in range(num_windows):
            start = w * self.test_days
            train_end = start + self.train_days
            test_end = train_end + self.test_days
            if test_end > len(all_days):
                break
            seeds = [int(s.generate_state(1)[0]) for s in book_seeds[train_end:test_end]]
            result = self._process_window(
                w, all_days[start:train_end], all_days[train_end:test_end], seeds
            )
            logger.info(
                "window %d: vol %.1f%% shortfall %s (%s)",
                w,
                100 * result.train_volatility,
                {k: round(v, 2) for k, v in result.shortfall_bps.items()},
                result.winner,
            )
            results.append(result)
        return results

    def _hybrid_schedule(self, num_minutes: int, train_vol: float) -> NDArray[np.float64]:
        num_slices = min(_MAX_QUBO_SLICES, num_minutes)
        config = slice_level_config(
            self.daily_shares,
            num_slices,
            volatility=max(_MIN_DAILY_VOLATILITY, train_vol / np.sqrt(TRADING_DAYS_PER_YEAR)),
            impact_coefficient=0.1,
        )
        slice_qty, _ = optimize_schedule(
            ExecutionQUBO(config), SimulatedAnnealingSolver(num_sweeps=_SA_SWEEPS, seed=self.seed)
        )
        repaired = repair_schedule(slice_qty, self.daily_shares).astype(float)
        return spread_over_minutes(repaired, num_minutes)

    def _shortfall(
        self, day: pd.DataFrame, strategy: BaseStrategy, book_seed: int
    ) -> ExecutionReport:
        engine = ExecutionEngine(OrderBook(seed=book_seed))
        return engine.process_order(self._create_order(), day, strategy)

    def _process_window(
        self,
        window_id: int,
        train_data: list[pd.DataFrame],
        test_data: list[pd.DataFrame],
        book_seeds: list[int],
    ) -> WindowResult:
        avg_volume_profile = np.mean([d["volume"].to_numpy() for d in train_data], axis=0)
        static_profile = train_data[0]["volume"].to_numpy(dtype=np.float64)
        returns = pd.concat(train_data)["price"].pct_change().dropna()
        train_vol = float(returns.std() * np.sqrt(MINUTES_PER_DAY * TRADING_DAYS_PER_YEAR))

        shortfalls: dict[str, list[float]] = {name: [] for name in WALK_FORWARD_STRATEGIES}
        fills: dict[str, list[float]] = {name: [] for name in WALK_FORWARD_STRATEGIES}
        for day, book_seed in zip(test_data, book_seeds, strict=True):
            strategies: dict[str, BaseStrategy] = {
                "TWAP": FixedScheduleStrategy(np.ones(len(day))),
                "Static": VWAPStrategy(historical_profile=static_profile, seed=book_seed),
                "Adaptive": VWAPStrategy(historical_profile=avg_volume_profile, seed=book_seed),
                "Hybrid": FixedScheduleStrategy(self._hybrid_schedule(len(day), train_vol)),
            }
            for name, strategy in strategies.items():
                report = self._shortfall(day, strategy, book_seed)
                shortfalls[name].append(report.implementation_shortfall_bps)
                fills[name].append(report.fill_rate)

        return WindowResult(
            window_id=window_id,
            shortfall_bps={k: float(np.mean(v)) for k, v in shortfalls.items()},
            fill_rate={k: float(np.mean(v)) for k, v in fills.items()},
            train_volatility=train_vol,
        )

    def _create_order(self) -> ParentOrder:
        return ParentOrder(
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=self.daily_shares,
            time_horizon_minutes=MINUTES_PER_DAY,
        )
