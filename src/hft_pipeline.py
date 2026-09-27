"""
Integrated HFT + Quantum Pipeline

The complete end-to-end pipeline that combines:
1. Market microstructure analysis (tick-level)
2. Adaptive risk aversion (regime detection)
3. HFT-specific QUBO formulation (microstructure-aware)
4. Quantum/classical optimization (SA/QAOA)
5. Latency-decoupled execution (Fast/Slow path)
6. Nanosecond latency monitoring

This is the system-level integration that no existing paper provides.
Prior work either does:
- Quantum optimization without microstructure (Rosenberg 2016)
- Microstructure without quantum (Cartea, Jaimungal)
- Neither with a latency-decoupled async architecture

Architecture:
    Market Ticks
        |
        v
    [MicrostructureAnalyzer] --> MicrostructureState
        |                              |
        v                              v
    [AdaptiveRiskManager]     [HFTQUBOConfig update]
        |                              |
        v                              v
    RegimeState (lambda)        HFT QUBO Matrix
        |                              |
        +------> [AsyncOptimizer] <----+
                       |
                   (Slow Path: 100ms-5s)
                       |
                       v
                 [PolicyQueue]
                       |
                   (Non-blocking)
                       |
                       v
              [AsyncExecutionEngine]
                   (Fast Path: <1ms per tick)
                       |
                       v
                  Execution Orders
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from threading import Thread, Event, Lock
import logging

from .hft_microstructure import MicrostructureAnalyzer, MicrostructureState
from .adaptive_risk import AdaptiveRiskManager, RegimeState
from .hft_qubo import HFTQUBOConfig, HFTExecutionQUBO
from .latency_monitor import LatencyMonitor, LatencySpan, get_latency_monitor
from .hybrid_async import PolicyQueue, ExecutionPolicy

logger = logging.getLogger(__name__)


@dataclass
class HFTPipelineConfig:
    total_shares: int = 5000
    num_tick_slices: int = 20
    num_venues: int = 3
    tick_interval_ms: float = 100.0
    optimizer_interval_ms: float = 500.0
    base_lambda: float = 1.0
    solver_sweeps: int = 300
    seed: Optional[int] = None


@dataclass
class HFTExecutionResult:
    total_shares_executed: int = 0
    target_shares: int = 0
    fill_rate: float = 0.0
    num_ticks: int = 0
    num_optimizations: int = 0
    avg_optimization_ms: float = 0.0
    avg_fast_path_us: float = 0.0
    avg_policy_staleness_ms: float = 0.0
    final_regime: str = "normal"
    final_lambda: float = 1.0
    venues_used: List[str] = field(default_factory=list)
    cost_breakdown: Dict[str, float] = field(default_factory=dict)
    execution_log: List[Dict] = field(default_factory=list)
    regime_transitions: int = 0


class HFTQuantumPipeline:
    """
    Full HFT + Quantum execution pipeline.

    Integrates market microstructure analysis, adaptive risk management,
    and quantum-optimized execution in a latency-decoupled architecture.
    """

    def __init__(self, config: HFTPipelineConfig):
        self.config = config

        self.microstructure = MicrostructureAnalyzer(
            kyle_window=50,
            vpin_bucket_size=max(100, config.total_shares // 50),
            vpin_buckets=20,
            tick_size=0.01
        )

        self.risk_manager = AdaptiveRiskManager(
            base_lambda=config.base_lambda,
            lambda_min=0.1,
            lambda_max=10.0
        )

        self.latency = LatencyMonitor()
        self.policy_queue = PolicyQueue()

        self._optimizer_thread: Optional[Thread] = None
        self._stop_event = Event()
        self._qubo_config_lock = Lock()
        self._current_qubo_config: Optional[HFTQUBOConfig] = None
        self._num_optimizations = 0
        self._total_opt_time = 0.0

    def execute(
        self,
        prices: np.ndarray,
        bids: np.ndarray,
        asks: np.ndarray,
        volumes: np.ndarray
    ) -> HFTExecutionResult:
        """
        Execute the full HFT pipeline on tick data.

        Args:
            prices: Array of trade prices (tick-level)
            bids: Array of best bid prices
            asks: Array of best ask prices
            volumes: Array of trade volumes per tick
        """
        n_ticks = len(prices)
        result = HFTExecutionResult(target_shares=self.config.total_shares)
        shares_remaining = self.config.total_shares
        prev_regime = None
        current_policy: Optional[ExecutionPolicy] = None
        policy_birth_time = time.monotonic()

        self._initialize_qubo_config()
        self._stop_event.clear()
        self._start_optimizer()

        try:
            for tick in range(n_ticks):
                if shares_remaining <= 0:
                    break

                tick_start = time.monotonic()

                with LatencySpan("fast_path", self.latency):
                    # 1. Update microstructure state
                    with LatencySpan("tick_to_decision", self.latency):
                        side = "buy" if tick == 0 or prices[tick] >= prices[tick - 1] else "sell"
                        micro_state = self.microstructure.process_tick(
                            prices[tick], int(volumes[tick]),
                            bids[tick], asks[tick], side,
                            timestamp_ns=int(tick * self.config.tick_interval_ms * 1e6)
                        )

                        regime_state = self.risk_manager.update(
                            prices[tick], bids[tick], asks[tick],
                            volumes[tick], float(np.mean(volumes[:max(1, tick)]))
                        )

                    # 2. Update QUBO config for optimizer
                    self._update_qubo_config(micro_state, regime_state)

                    # 3. Check for new policy (non-blocking)
                    new_policy = self.policy_queue.poll()
                    if new_policy is not None:
                        staleness = (time.monotonic() - policy_birth_time) * 1000
                        self.latency.record_staleness(staleness, new_policy.policy_id)
                        current_policy = new_policy
                        policy_birth_time = time.monotonic()

                    # 4. Execute according to policy
                    shares_this_tick = 0
                    if current_policy is not None:
                        tick_in_policy = tick % len(current_policy.schedule)
                        shares_this_tick = min(
                            int(current_policy.schedule[tick_in_policy]),
                            shares_remaining
                        )
                    else:
                        shares_this_tick = min(
                            self.config.total_shares // self.config.num_tick_slices,
                            shares_remaining
                        )

                    if shares_this_tick > 0:
                        shares_remaining -= shares_this_tick
                        result.total_shares_executed += shares_this_tick
                        result.execution_log.append({
                            "tick": tick,
                            "shares": shares_this_tick,
                            "price": prices[tick],
                            "remaining": shares_remaining,
                            "regime": regime_state.vol_regime.value,
                            "lambda": regime_state.lambda_value,
                            "vpin": micro_state.vpin,
                            "kyle_lambda": micro_state.kyle_lambda,
                            "policy_id": current_policy.policy_id if current_policy else 0
                        })

                # Track regime transitions
                if prev_regime is not None and regime_state.vol_regime != prev_regime:
                    result.regime_transitions += 1
                prev_regime = regime_state.vol_regime

                result.num_ticks = tick + 1

        finally:
            self._stop_optimizer()

        # Compile results
        result.fill_rate = result.total_shares_executed / self.config.total_shares
        result.num_optimizations = self._num_optimizations
        result.avg_optimization_ms = (
            self._total_opt_time / max(1, self._num_optimizations) * 1000
        )
        result.final_regime = regime_state.vol_regime.value
        result.final_lambda = regime_state.lambda_value

        fast_stats = self.latency.get_stats("fast_path")
        if fast_stats:
            result.avg_fast_path_us = fast_stats.mean_us

        staleness = self.latency.analyze_staleness()
        if staleness:
            result.avg_policy_staleness_ms = staleness.mean_staleness_ms

        return result

    def _initialize_qubo_config(self) -> None:
        with self._qubo_config_lock:
            self._current_qubo_config = HFTQUBOConfig(
                total_shares=self.config.total_shares,
                num_tick_slices=self.config.num_tick_slices,
                num_venues=self.config.num_venues,
                tick_duration_ms=self.config.tick_interval_ms
            )

    def _update_qubo_config(self, micro: MicrostructureState, regime: RegimeState) -> None:
        with self._qubo_config_lock:
            if self._current_qubo_config is None:
                return
            self._current_qubo_config.kyle_lambda = micro.kyle_lambda
            self._current_qubo_config.vpin = micro.vpin
            self._current_qubo_config.adverse_selection_cost = micro.adverse_selection_cost
            self._current_qubo_config.impact_weight = max(0.1, 0.25 * regime.lambda_value)

    def _start_optimizer(self) -> None:
        self._optimizer_thread = Thread(target=self._optimizer_loop, daemon=True)
        self._optimizer_thread.start()

    def _stop_optimizer(self) -> None:
        self._stop_event.set()
        if self._optimizer_thread:
            self._optimizer_thread.join(timeout=5.0)

    def _optimizer_loop(self) -> None:
        from .qubo_solvers import SimulatedAnnealingSolver

        solver = SimulatedAnnealingSolver(
            num_sweeps=self.config.solver_sweeps,
            seed=self.config.seed
        )

        while not self._stop_event.is_set():
            opt_start = time.monotonic()

            try:
                with self._qubo_config_lock:
                    config_snapshot = HFTQUBOConfig(
                        total_shares=self._current_qubo_config.total_shares,
                        num_tick_slices=self._current_qubo_config.num_tick_slices,
                        num_venues=self._current_qubo_config.num_venues,
                        kyle_lambda=self._current_qubo_config.kyle_lambda,
                        vpin=self._current_qubo_config.vpin,
                        adverse_selection_cost=self._current_qubo_config.adverse_selection_cost,
                        impact_weight=self._current_qubo_config.impact_weight,
                        tick_duration_ms=self._current_qubo_config.tick_duration_ms
                    )

                with LatencySpan("slow_path_optimize", self.latency):
                    with LatencySpan("slow_path_qubo_build", self.latency):
                        qubo = HFTExecutionQUBO(config_snapshot)
                        Q = qubo.build_qubo_matrix()

                    with LatencySpan("slow_path_solve", self.latency):
                        result = solver.solve(Q, verbose=False)

                solution = qubo.interpret_solution(result.solution)
                schedule = np.zeros(config_snapshot.num_tick_slices)
                for entry in solution["schedule"]:
                    t = entry["tick"]
                    if t < len(schedule):
                        schedule[t] += entry["quantity"]

                sched_sum = schedule.sum()
                if sched_sum > 0:
                    schedule = schedule * (config_snapshot.total_shares / sched_sum)

                policy = ExecutionPolicy(
                    schedule=schedule,
                    optimizer_name="hft_qubo_sa",
                    optimization_time=time.monotonic() - opt_start,
                    energy=result.energy
                )

                prop_start = time.monotonic()
                self.policy_queue.publish(policy)
                prop_ns = int((time.monotonic() - prop_start) * 1e9)
                self.latency.record_latency("policy_propagation", prop_ns)

                self._num_optimizations += 1
                self._total_opt_time += time.monotonic() - opt_start

            except Exception as e:
                logger.error(f"Optimizer error: {e}")

            wait_s = self.config.optimizer_interval_ms / 1000.0
            elapsed = time.monotonic() - opt_start
            remaining = max(0, wait_s - elapsed)
            self._stop_event.wait(remaining)


def run_hft_pipeline_demo():
    """Full demonstration of the HFT + Quantum pipeline."""
    np.random.seed(42)

    config = HFTPipelineConfig(
        total_shares=3000,
        num_tick_slices=15,
        num_venues=3,
        tick_interval_ms=100.0,
        optimizer_interval_ms=300.0,
        base_lambda=1.0,
        solver_sweeps=200,
        seed=42
    )

    pipeline = HFTQuantumPipeline(config)

    # Generate synthetic tick data with regime change
    n_ticks = 100
    price = 100.0
    prices, bids, asks, volumes = [], [], [], []

    for i in range(n_ticks):
        if i < 40:
            vol = 0.001
        elif i < 70:
            vol = 0.004
        else:
            vol = 0.0015

        ret = np.random.normal(0, vol)
        price *= (1 + ret)
        spread = max(0.01, abs(ret) * price * 2 + 0.01)
        prices.append(price)
        bids.append(price - spread / 2)
        asks.append(price + spread / 2)
        volumes.append(max(10, int(np.random.exponential(500))))

    prices = np.array(prices)
    bids = np.array(bids)
    asks = np.array(asks)
    volumes = np.array(volumes, dtype=float)

    print("\n" + "=" * 70)
    print(" HFT + Quantum Pipeline Demo")
    print("=" * 70)
    print(f"  Target: {config.total_shares} shares over {n_ticks} ticks")
    print(f"  Venues: {config.num_venues} (Lit, Dark, ECN)")
    print(f"  Optimizer: SA with {config.solver_sweeps} sweeps")

    result = pipeline.execute(prices, bids, asks, volumes)

    print(f"\n  Results:")
    print(f"    Shares executed:     {result.total_shares_executed} / {result.target_shares}")
    print(f"    Fill rate:           {result.fill_rate:.1%}")
    print(f"    Ticks used:          {result.num_ticks}")
    print(f"    Optimizations:       {result.num_optimizations}")
    print(f"    Avg opt time:        {result.avg_optimization_ms:.1f} ms")
    print(f"    Avg fast path:       {result.avg_fast_path_us:.1f} us")
    print(f"    Policy staleness:    {result.avg_policy_staleness_ms:.1f} ms")
    print(f"    Regime transitions:  {result.regime_transitions}")
    print(f"    Final regime:        {result.final_regime}")
    print(f"    Final lambda:        {result.final_lambda:.3f}")

    pipeline.latency.print_report()

    print("=" * 70)
    print("  Key: Microstructure-aware QUBO with adaptive risk aversion")
    print("       and latency-decoupled execution -- no prior work combines these.")
    print("=" * 70)

    return result


if __name__ == "__main__":
    run_hft_pipeline_demo()
