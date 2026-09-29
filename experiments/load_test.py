"""Concurrent HybridController orders on a thread pool: wall time, per-order duration, fill
rate and memory.

Order sizes and slice counts are drawn from a seeded generator. Timings depend on the
machine and Python's GIL; they measure this implementation, not a production system.

Usage:
    python -m experiments.load_test [--quick] [--results-dir results] [--seed 0]
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
import psutil

from experiments.common import Experiment, single_seed
from qexec.experiment import ExperimentRecorder
from qexec.runtime.controller import HybridController


@dataclass(frozen=True)
class Config:
    num_orders: int = 100
    concurrency: int = 20
    min_shares: int = 100
    max_shares: int = 10_000
    min_slices: int = 10
    max_slices: int = 50
    tick_interval_s: float = 0.05
    optimizer_interval_s: float = 1.0
    seed: int = 0


FULL = Config()
QUICK = Config(num_orders=6, concurrency=3, max_slices=15)


def run_order(order_id: int, shares: int, slices: int, seed: int, config: Config) -> dict:
    controller = HybridController(
        optimizer_type="sa",
        optimizer_interval=config.optimizer_interval_s,
        engine_tick_interval=config.tick_interval_s,
        seed=seed,
    )
    start = time.perf_counter()
    # A failed order is recorded and counted, not allowed to stop the load test.
    try:
        res = controller.execute_order(total_shares=shares, num_slices=slices)
    except Exception as e:
        return {"order_id": order_id, "shares": shares, "slices": slices, "error": str(e)}
    duration = time.perf_counter() - start
    return {
        "order_id": order_id,
        "shares": shares,
        "slices": slices,
        "executed_shares": res["executed_shares"],
        "fill_rate": res["fill_rate"],
        "optimizations": res["num_optimizations"],
        "duration_s": duration,
        "min_duration_s": slices * config.tick_interval_s,
        "error": "",
    }


def run(config: Config, rec: ExperimentRecorder) -> None:
    logging.getLogger("qexec").setLevel(logging.WARNING)
    rng = np.random.default_rng(config.seed)
    orders = [
        (
            i,
            int(rng.integers(config.min_shares, config.max_shares + 1)),
            int(rng.integers(config.min_slices, config.max_slices + 1)),
            int(rng.integers(2**31)),
        )
        for i in range(config.num_orders)
    ]
    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss / 1024**2
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
        rows = list(pool.map(lambda o: run_order(*o, config), orders))
    wall = time.perf_counter() - start
    df = pd.DataFrame(rows).sort_values("order_id")
    rec.write_table("orders", df)
    ok = df[df["error"] == ""]
    rec.write_json(
        "summary",
        {
            "orders": config.num_orders,
            "concurrency": config.concurrency,
            "wall_time_s": wall,
            "orders_per_s": config.num_orders / wall,
            "success_rate": len(ok) / config.num_orders,
            "orders_fully_filled": int((ok["fill_rate"] == 1.0).sum()),
            "duration_s": ok["duration_s"].describe(percentiles=[0.5, 0.95, 0.99]).to_dict(),
            "overhead_vs_tick_schedule_s": (ok["duration_s"] - ok["min_duration_s"])
            .describe(percentiles=[0.5, 0.95])
            .to_dict(),
            "rss_mb_before": rss_before,
            "rss_mb_after": process.memory_info().rss / 1024**2,
        },
    )


EXPERIMENT = Experiment(
    "load_test", FULL, QUICK, run, single_seed, description=__doc__.splitlines()[0]
)

if __name__ == "__main__":
    EXPERIMENT.main()
