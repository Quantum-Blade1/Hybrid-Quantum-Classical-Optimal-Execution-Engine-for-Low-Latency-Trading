"""Concurrent HybridController orders on a thread pool: wall time, per-order latency, memory.

Order sizes and slice counts are drawn from a seeded generator. Timings depend on the
machine and Python's GIL; they measure this implementation, not a production system.

Usage:
    python experiments/load_test.py [--orders 100] [--concurrency 20] [--seed 42]
"""

import argparse
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import pandas as pd
import psutil

from qexec.runtime.controller import HybridController


@dataclass(frozen=True)
class LoadTestConfig:
    num_orders: int = 100
    concurrency: int = 20
    min_shares: int = 100
    max_shares: int = 10_000
    min_slices: int = 10
    max_slices: int = 50


@dataclass(frozen=True)
class OrderSpec:
    order_id: int
    total_shares: int
    num_slices: int
    seed: int


@dataclass(frozen=True)
class OrderResult:
    order_id: int
    total_shares: int
    executed_shares: int
    num_slices: int
    duration: float
    throughput: float
    optimizations: int
    success: bool
    error: str = ""


def draw_orders(config: LoadTestConfig, rng: np.random.Generator) -> list[OrderSpec]:
    return [
        OrderSpec(
            order_id=i,
            total_shares=int(rng.integers(config.min_shares, config.max_shares + 1)),
            num_slices=int(rng.integers(config.min_slices, config.max_slices + 1)),
            seed=int(rng.integers(2**31)),
        )
        for i in range(config.num_orders)
    ]


def run_single_order(spec: OrderSpec) -> OrderResult:
    controller = HybridController(
        optimizer_type="sa", optimizer_interval=1.0, engine_tick_interval=0.05, seed=spec.seed
    )
    start = time.perf_counter()
    # A failed order is recorded and counted, not allowed to stop the load test.
    try:
        res = controller.execute_order(total_shares=spec.total_shares, num_slices=spec.num_slices)
    except Exception as e:
        return OrderResult(
            spec.order_id, spec.total_shares, 0, spec.num_slices, 0, 0, 0, False, str(e)
        )
    duration = time.perf_counter() - start
    return OrderResult(
        order_id=spec.order_id,
        total_shares=spec.total_shares,
        executed_shares=res["executed_shares"],
        num_slices=spec.num_slices,
        duration=duration,
        throughput=res["executed_shares"] / max(0.001, duration),
        optimizations=res["num_optimizations"],
        success=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--orders", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    config = LoadTestConfig(num_orders=args.orders, concurrency=args.concurrency)
    orders = draw_orders(config, np.random.default_rng(args.seed))
    process = psutil.Process(os.getpid())
    initial_mem = process.memory_info().rss / 1024**2
    print(f"{config.num_orders} orders, concurrency {config.concurrency}")

    start = time.perf_counter()
    results: list[OrderResult] = []
    with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
        futures = [executor.submit(run_single_order, spec) for spec in orders]
        for done, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if done % 10 == 0:
                print(f"  {done}/{config.num_orders} orders completed")
    total_duration = time.perf_counter() - start
    final_mem = process.memory_info().rss / 1024**2

    df = pd.DataFrame([vars(r) for r in results])
    print(f"\nWall time:      {total_duration:.2f}s")
    print(f"Success rate:   {df['success'].mean() * 100:.1f}%")
    print(f"Total volume:   {df['executed_shares'].sum():,.0f} shares")
    print(f"Orders/sec:     {config.num_orders / total_duration:.2f}")
    print(f"Memory (RSS):   {initial_mem:.1f} MB -> {final_mem:.1f} MB")
    print(
        "Per-order duration (s): "
        + ", ".join(
            f"{name} {value:.2f}"
            for name, value in [
                ("mean", df["duration"].mean()),
                ("p50", df["duration"].median()),
                ("p95", df["duration"].quantile(0.95)),
                ("p99", df["duration"].quantile(0.99)),
            ]
        )
    )
    print(f"Mean throughput: {df['throughput'].mean():.1f} shares/s per order")
    failed = df[~df["success"]]
    if not failed.empty:
        print("\nErrors:")
        print(failed[["order_id", "error"]].to_string(index=False))


if __name__ == "__main__":
    main()
