import logging
from time import perf_counter
from typing import Any

import pandas as pd

from qexec.runtime.engine import AsyncExecutionEngine
from qexec.runtime.latency import LatencyMonitor
from qexec.runtime.optimizer import AsyncOptimizer
from qexec.runtime.policy import ExecutionPolicy, PolicyQueue, uniform_schedule

logger = logging.getLogger(__name__)


class HybridController:
    """Runs one order: TWAP fallback, background optimizer, and a `num_slices`-tick loop."""

    def __init__(
        self,
        optimizer_type: str = "sa",
        optimizer_interval: float = 0.5,
        engine_tick_interval: float = 0.1,
        seed: int | None = None,
        latency_monitor: LatencyMonitor | None = None,
    ) -> None:
        self.policy_queue = PolicyQueue()
        self.latency = latency_monitor
        self.optimizer = AsyncOptimizer(
            policy_queue=self.policy_queue,
            optimizer_type=optimizer_type,
            update_interval=optimizer_interval,
            seed=seed,
            latency_monitor=latency_monitor,
        )
        self.engine = AsyncExecutionEngine(
            policy_queue=self.policy_queue,
            tick_interval=engine_tick_interval,
            latency_monitor=latency_monitor,
        )

    def execute_order(self, total_shares: int, num_slices: int) -> dict[str, Any]:
        """Blocks until the tick loop finishes; returns fill and optimizer statistics."""
        logger.info("Starting hybrid execution: %d shares, %d slices", total_shares, num_slices)
        start = perf_counter()

        self.engine.set_fallback_policy(
            ExecutionPolicy(
                schedule=uniform_schedule(total_shares, num_slices).astype(float),
                optimizer_name="fallback_uniform",
            )
        )
        self.optimizer.start(total_shares, num_slices)
        self.engine.start(num_slices, total_shares)
        self.engine.wait_complete()
        self.optimizer.stop()

        num_optimizations = self.optimizer.num_optimizations
        return {
            "total_shares": total_shares,
            "executed_shares": self.engine.executed_shares,
            "fill_rate": self.engine.executed_shares / total_shares,
            "num_slices": num_slices,
            "num_optimizations": num_optimizations,
            "avg_optimization_time": (
                self.optimizer.total_optimization_time / max(1, num_optimizations)
            ),
            "total_time": perf_counter() - start,
            "execution_log": self.engine.execution_log,
        }

    def get_execution_report(self) -> pd.DataFrame:
        return pd.DataFrame(self.engine.execution_log)
