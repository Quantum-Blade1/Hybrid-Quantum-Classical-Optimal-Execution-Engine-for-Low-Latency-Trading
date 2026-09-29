import logging
import time
from dataclasses import dataclass, field
from threading import Event, Lock, Thread
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qexec.microstructure.analyzer import MicrostructureAnalyzer, MicrostructureState
from qexec.microstructure.regime import AdaptiveRiskManager, RegimeState
from qexec.optimization.hft_qubo import HFTExecutionQUBO, HFTQUBOConfig
from qexec.optimization.schedule import optimize_schedule, repair_schedule
from qexec.optimization.solvers.annealing import SimulatedAnnealingSolver
from qexec.runtime.latency import LatencyMonitor, LatencySpan
from qexec.runtime.policy import ExecutionPolicy, PolicyQueue

logger = logging.getLogger(__name__)

_MIN_IMPACT_WEIGHT = 0.1
_IMPACT_WEIGHT_PER_LAMBDA = 0.25


@dataclass(frozen=True)
class HFTPipelineConfig:
    total_shares: int = 5000
    num_tick_slices: int = 20
    num_venues: int = 3
    tick_interval_ms: float = 100.0
    optimizer_interval_ms: float = 500.0
    base_lambda: float = 1.0
    solver_sweeps: int = 300
    seed: int | None = None


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
    execution_log: list[dict[str, Any]] = field(default_factory=list)
    regime_transitions: int = 0


class HFTQuantumPipeline:
    """Tick loop on the newest policy while a background SA thread re-solves the HFT QUBO."""

    def __init__(self, config: HFTPipelineConfig) -> None:
        self.config = config
        self.microstructure = MicrostructureAnalyzer(
            kyle_window=50,
            vpin_bucket_size=max(100, config.total_shares // 50),
            vpin_buckets=20,
        )
        self.risk_manager = AdaptiveRiskManager(
            base_lambda=config.base_lambda, lambda_min=0.1, lambda_max=10.0
        )
        self.latency = LatencyMonitor()
        self.policy_queue = PolicyQueue()

        self._optimizer_thread: Thread | None = None
        self._stop_event = Event()
        self._qubo_config_lock = Lock()
        self._qubo_config = self._initial_qubo_config()
        self._num_optimizations = 0
        self._total_opt_time = 0.0

    def execute(
        self,
        prices: NDArray[np.float64],
        bids: NDArray[np.float64],
        asks: NDArray[np.float64],
        volumes: NDArray[np.float64],
    ) -> HFTExecutionResult:
        if len(prices) == 0:
            raise ValueError("execute needs at least one tick")
        result = HFTExecutionResult(target_shares=self.config.total_shares)
        shares_remaining = self.config.total_shares
        current_policy: ExecutionPolicy | None = None
        policy_birth_time = time.monotonic()
        regime_state = self.risk_manager.current_state
        prev_regime = None

        with self._qubo_config_lock:
            self._qubo_config = self._initial_qubo_config()
        self._stop_event.clear()
        self._start_optimizer()
        try:
            for tick in range(len(prices)):
                if shares_remaining <= 0:
                    break
                with LatencySpan(LatencyMonitor.FAST_PATH, self.latency):
                    with LatencySpan(LatencyMonitor.TICK_TO_DECISION, self.latency):
                        micro_state, regime_state = self._update_estimators(
                            tick, prices, bids, asks, volumes
                        )

                    self._update_qubo_config(micro_state, regime_state)

                    new_policy = self.policy_queue.poll()
                    if new_policy is not None:
                        staleness_ms = (time.monotonic() - policy_birth_time) * 1000
                        self.latency.record_staleness(staleness_ms, new_policy.policy_id)
                        current_policy = new_policy
                        policy_birth_time = time.monotonic()

                    if current_policy is not None:
                        slot = tick % len(current_policy.schedule)
                        planned = int(current_policy.schedule[slot])
                    else:
                        planned = self.config.total_shares // self.config.num_tick_slices
                    shares_this_tick = min(planned, shares_remaining)

                    if shares_this_tick > 0:
                        shares_remaining -= shares_this_tick
                        result.total_shares_executed += shares_this_tick
                        result.execution_log.append(
                            {
                                "tick": tick,
                                "shares": shares_this_tick,
                                "price": prices[tick],
                                "remaining": shares_remaining,
                                "regime": regime_state.vol_regime.value,
                                "lambda": regime_state.lambda_value,
                                "vpin": micro_state.vpin,
                                "kyle_lambda": micro_state.kyle_lambda,
                                "policy_id": current_policy.policy_id if current_policy else 0,
                            }
                        )

                if prev_regime is not None and regime_state.vol_regime != prev_regime:
                    result.regime_transitions += 1
                prev_regime = regime_state.vol_regime
                result.num_ticks = tick + 1
        finally:
            self._stop_optimizer()

        self._summarise(result, regime_state)
        return result

    def _update_estimators(
        self,
        tick: int,
        prices: NDArray[np.float64],
        bids: NDArray[np.float64],
        asks: NDArray[np.float64],
        volumes: NDArray[np.float64],
    ) -> tuple[MicrostructureState, RegimeState]:
        side = "buy" if tick == 0 or prices[tick] >= prices[tick - 1] else "sell"
        micro_state = self.microstructure.process_tick(
            prices[tick],
            int(volumes[tick]),
            bids[tick],
            asks[tick],
            side,
            timestamp_ns=int(tick * self.config.tick_interval_ms * 1e6),
        )
        regime_state = self.risk_manager.update(
            prices[tick],
            bids[tick],
            asks[tick],
            volumes[tick],
            float(np.mean(volumes[: max(1, tick)])),
        )
        return micro_state, regime_state

    def _summarise(self, result: HFTExecutionResult, regime_state: RegimeState) -> None:
        result.fill_rate = result.total_shares_executed / self.config.total_shares
        result.num_optimizations = self._num_optimizations
        result.avg_optimization_ms = self._total_opt_time / max(1, self._num_optimizations) * 1000
        result.final_regime = regime_state.vol_regime.value
        result.final_lambda = regime_state.lambda_value
        fast_stats = self.latency.get_stats(LatencyMonitor.FAST_PATH)
        if fast_stats is not None:
            result.avg_fast_path_us = fast_stats.mean_us
        staleness = self.latency.analyze_staleness()
        if staleness is not None:
            result.avg_policy_staleness_ms = staleness.mean_staleness_ms

    def _initial_qubo_config(self) -> HFTQUBOConfig:
        return HFTQUBOConfig(
            total_shares=self.config.total_shares,
            num_tick_slices=self.config.num_tick_slices,
            num_venues=self.config.num_venues,
            tick_duration_ms=self.config.tick_interval_ms,
        )

    def _update_qubo_config(self, micro: MicrostructureState, regime: RegimeState) -> None:
        with self._qubo_config_lock:
            self._qubo_config.kyle_lambda = micro.kyle_lambda
            self._qubo_config.vpin = micro.vpin
            self._qubo_config.adverse_selection_cost = micro.adverse_selection_cost
            self._qubo_config.impact_weight = max(
                _MIN_IMPACT_WEIGHT, _IMPACT_WEIGHT_PER_LAMBDA * regime.lambda_value
            )

    def _snapshot_qubo_config(self) -> HFTQUBOConfig:
        """Copy of the fields the tick loop updates, on default venue and weight settings."""
        with self._qubo_config_lock:
            c = self._qubo_config
            return HFTQUBOConfig(
                total_shares=c.total_shares,
                num_tick_slices=c.num_tick_slices,
                num_venues=c.num_venues,
                kyle_lambda=c.kyle_lambda,
                vpin=c.vpin,
                adverse_selection_cost=c.adverse_selection_cost,
                impact_weight=c.impact_weight,
                tick_duration_ms=c.tick_duration_ms,
            )

    def _start_optimizer(self) -> None:
        self._optimizer_thread = Thread(target=self._optimizer_loop, daemon=True)
        self._optimizer_thread.start()

    def _stop_optimizer(self) -> None:
        self._stop_event.set()
        if self._optimizer_thread is not None:
            self._optimizer_thread.join(timeout=5.0)

    def _solve_once(self, solver: SimulatedAnnealingSolver, opt_start: float) -> None:
        config = self._snapshot_qubo_config()
        with LatencySpan(LatencyMonitor.SLOW_PATH_OPTIMIZE, self.latency):
            with LatencySpan(LatencyMonitor.SLOW_PATH_QUBO_BUILD, self.latency):
                qubo = HFTExecutionQUBO(config)
                Q = qubo.build_qubo_matrix()
            with LatencySpan(LatencyMonitor.SLOW_PATH_SOLVE, self.latency):
                schedule, result = optimize_schedule(qubo, solver, Q)

        policy = ExecutionPolicy(
            schedule=repair_schedule(schedule, config.total_shares).astype(np.float64),
            optimizer_name="hft_qubo_sa",
            optimization_time=time.monotonic() - opt_start,
            energy=result.energy,
        )
        prop_start = time.monotonic()
        self.policy_queue.publish(policy)
        self.latency.record_latency(
            LatencyMonitor.POLICY_PROPAGATION, int((time.monotonic() - prop_start) * 1e9)
        )
        self._num_optimizations += 1
        self._total_opt_time += time.monotonic() - opt_start

    def _optimizer_loop(self) -> None:
        solver = SimulatedAnnealingSolver(
            num_sweeps=self.config.solver_sweeps, seed=self.config.seed
        )
        interval_s = self.config.optimizer_interval_ms / 1000.0
        while not self._stop_event.is_set():
            opt_start = time.monotonic()
            # Background thread: log any solver failure and keep the fast path's current policy.
            try:
                self._solve_once(solver, opt_start)
            except Exception:
                logger.exception("HFT optimizer iteration failed")
            self._stop_event.wait(max(0.0, interval_s - (time.monotonic() - opt_start)))
