"""
Latency-decoupled runtime: the fast path never waits for the optimizer.

1. HybridController: a 1,000-share order in 20 ticks of 100 ms while the
   SA-QUBO optimizer republishes a policy every 500 ms in the background.
2. HFTQuantumPipeline: synthetic ticks with a volatility regime change;
   microstructure and regime estimates re-parameterise the HFT QUBO that
   the slow path keeps re-solving.

Timings depend on the machine and are for illustration only.

    python examples/hybrid_runtime.py
"""

import logging

import numpy as np

from qexec.runtime.controller import HybridController
from qexec.runtime.hft_pipeline import HFTPipelineConfig, HFTQuantumPipeline

SEED = 42


def synthetic_ticks(n_ticks: int, rng: np.random.Generator):
    """Prices/bids/asks/volumes with a high-volatility segment in the middle."""
    price = 100.0
    prices, bids, asks, volumes = [], [], [], []
    for i in range(n_ticks):
        vol = 0.001 if i < 40 else 0.004 if i < 70 else 0.0015
        ret = rng.normal(0, vol)
        price *= 1 + ret
        spread = max(0.01, abs(ret) * price * 2 + 0.01)
        prices.append(price)
        bids.append(price - spread / 2)
        asks.append(price + spread / 2)
        volumes.append(max(10, int(rng.exponential(500))))
    return np.array(prices), np.array(bids), np.array(asks), np.array(volumes, dtype=float)


def main() -> None:
    logging.getLogger("qexec").setLevel(logging.WARNING)

    controller = HybridController(
        optimizer_type="sa", optimizer_interval=0.5, engine_tick_interval=0.1, seed=SEED
    )
    result = controller.execute_order(total_shares=1000, num_slices=20)
    print("HybridController")
    print(f"  executed {result['executed_shares']}/{result['total_shares']} shares "
          f"in {result['total_time']:.2f}s, {result['num_optimizations']} policy updates, "
          f"mean solve {result['avg_optimization_time'] * 1e3:.1f} ms")
    log = controller.get_execution_report()
    print(f"  policies applied per tick: {log['policy_id'].tolist()}")

    config = HFTPipelineConfig(
        total_shares=3000, num_tick_slices=15, num_venues=3,
        tick_interval_ms=100.0, optimizer_interval_ms=300.0,
        solver_sweeps=200, seed=SEED,
    )
    pipeline = HFTQuantumPipeline(config)
    hft = pipeline.execute(*synthetic_ticks(100, np.random.default_rng(SEED)))
    print("\nHFTQuantumPipeline")
    print(f"  executed {hft.total_shares_executed}/{hft.target_shares} shares in {hft.num_ticks} ticks")
    print(f"  {hft.num_optimizations} optimizations, mean {hft.avg_optimization_ms:.1f} ms; "
          f"mean fast path {hft.avg_fast_path_us:.1f} us")
    print(f"  regime transitions: {hft.regime_transitions}, final regime {hft.final_regime}, "
          f"lambda {hft.final_lambda:.3f}")


if __name__ == "__main__":
    main()
