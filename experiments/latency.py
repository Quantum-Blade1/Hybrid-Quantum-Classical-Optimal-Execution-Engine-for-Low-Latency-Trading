"""Measured fast-path and slow-path latency of the Python runtime (paper fig24).

    runtime   `HybridController` orders (AsyncExecutionEngine + AsyncOptimizer): each tick's
              work (policy poll, re-plan, execute; not the sleep between ticks) is
              `fast_path`; each SA solve is `slow_path_optimize`; publication-to-application
              delay is `policy_propagation`; how late each tick starts versus its 5 ms
              schedule is `tick_lateness` (the pure-Python SA thread holds the GIL, so the
              tick thread wakes late even though its own work takes microseconds)
    pipeline  `HFTQuantumPipeline` ticks (microstructure + regime estimator updates, QUBO
              config update, policy application) are `fast_path`; the SA solve of the
              HFT QUBO is `slow_path_optimize`

Latencies are CPython thread timings (time.monotonic_ns) on the machine recorded in the
manifest, with the GIL shared by the fast and slow paths. They are not kernel-bypass
numbers and must not be reported as such (claims audit P15, P16).

Usage:
    python -m experiments.latency [--quick] [--results-dir results] [--seed 0]
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from experiments.common import Experiment, single_seed
from experiments.microstructure import Config as TickConfig
from experiments.microstructure import synthetic_ticks
from qexec.experiment import ExperimentRecorder
from qexec.runtime.controller import HybridController
from qexec.runtime.hft_pipeline import HFTPipelineConfig, HFTQuantumPipeline
from qexec.runtime.latency import LatencyMonitor

COMPONENTS = (
    LatencyMonitor.FAST_PATH,
    LatencyMonitor.SLOW_PATH_OPTIMIZE,
    LatencyMonitor.POLICY_PROPAGATION,
    LatencyMonitor.TICK_LATENESS,
)


@dataclass(frozen=True)
class Config:
    runtime_orders: int = 20
    ticks_per_order: int = 100
    shares_per_order: int = 10_000
    tick_interval_s: float = 0.005
    optimizer_interval_s: float = 0.05
    pipeline_ticks: int = 10_000
    pipeline_order_shares: int = 5_000
    pipeline_solver_sweeps: int = 100
    seed: int = 0


FULL = Config()
QUICK = Config(runtime_orders=3, ticks_per_order=30, pipeline_ticks=300)


def runtime_latency(config: Config) -> tuple[LatencyMonitor, int]:
    monitor = LatencyMonitor(max_records=10**6)
    completed = 0
    for k in range(config.runtime_orders):
        controller = HybridController(
            optimizer_type="sa",
            optimizer_interval=config.optimizer_interval_s,
            engine_tick_interval=config.tick_interval_s,
            seed=config.seed + k,
            latency_monitor=monitor,
        )
        result = controller.execute_order(config.shares_per_order, config.ticks_per_order)
        completed += int(result["executed_shares"] == config.shares_per_order)
    return monitor, completed


def pipeline_latency(config: Config) -> tuple[LatencyMonitor, int]:
    """Replay `pipeline_ticks` ticks through one pipeline, one order after another."""
    ticks = synthetic_ticks(TickConfig(num_ticks=config.pipeline_ticks), config.seed)
    pipeline = HFTQuantumPipeline(
        HFTPipelineConfig(
            total_shares=config.pipeline_order_shares,
            solver_sweeps=config.pipeline_solver_sweeps,
            seed=config.seed,
        )
    )
    arrays = [ticks[c].to_numpy(dtype=float) for c in ("price", "bid", "ask", "volume")]
    start, orders = 0, 0
    while start < config.pipeline_ticks:
        result = pipeline.execute(*(a[start:] for a in arrays))
        start += max(1, result.num_ticks)
        orders += 1
    return pipeline.latency, orders


def _samples(source: str, monitor: LatencyMonitor) -> pd.DataFrame:
    frames = []
    for component in COMPONENTS:
        records = list(monitor._records.get(component, ()))
        if records:
            frames.append(
                pd.DataFrame(
                    {
                        "source": source,
                        "component": component,
                        "duration_us": np.round([r.duration_us for r in records], 3),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def run(config: Config, rec: ExperimentRecorder) -> None:
    logging.getLogger("qexec").setLevel(logging.WARNING)
    runtime, completed = runtime_latency(config)
    pipeline, orders = pipeline_latency(config)
    samples = pd.concat([_samples("runtime", runtime), _samples("pipeline", pipeline)])
    rec.write_table("samples", samples, float_format="%.3f")
    stats = (
        samples.groupby(["source", "component"])["duration_us"]
        .describe(percentiles=[0.5, 0.9, 0.99, 0.999])
        .reset_index()
    )
    rec.write_table("summary", stats)
    rec.note("runtime_orders_completed", f"{completed}/{config.runtime_orders}")
    rec.note("pipeline_orders", orders)


EXPERIMENT = Experiment(
    "latency", FULL, QUICK, run, single_seed, description=__doc__.splitlines()[0]
)

if __name__ == "__main__":
    EXPERIMENT.main()
