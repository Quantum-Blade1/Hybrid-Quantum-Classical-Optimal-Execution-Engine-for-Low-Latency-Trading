"""Slow path: background thread that re-solves the execution QUBO and publishes policies."""

import logging
import time
from threading import Event, Lock, Thread
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from qexec.optimization.qubo import ExecutionQUBO
from qexec.optimization.schedule import optimize_schedule, repair_schedule, slice_level_config
from qexec.optimization.solvers.annealing import SimulatedAnnealingSolver
from qexec.runtime.latency import LatencyMonitor
from qexec.runtime.policy import ExecutionPolicy, PolicyQueue, uniform_schedule
from qexec.runtime.resilience import OptimizerResilience, ResilienceConfig, validate_schedule

logger = logging.getLogger(__name__)

SUPPORTED_OPTIMIZERS = ("sa", "uniform")
_SA_SWEEPS = 200


class AsyncOptimizer:
    """Every `update_interval` seconds, solves for the current order and publishes a policy.

    Each policy plans the whole order; the fast path rescales its tail to the shares still
    unexecuted. With a `latency_monitor`, each solve is recorded as `slow_path_optimize`.

    `optimizer_type` is "sa" (SA on `slice_level_config`) or "uniform" (TWAP). QAOA is
    offline-only (`qexec.optimization.solvers.qaoa`). Failures fall back to TWAP.
    """

    def __init__(
        self,
        policy_queue: PolicyQueue,
        optimizer_type: str = "sa",
        update_interval: float = 1.0,
        seed: int | None = None,
        latency_monitor: LatencyMonitor | None = None,
    ) -> None:
        if optimizer_type not in SUPPORTED_OPTIMIZERS:
            raise ValueError(
                f"Unsupported optimizer_type {optimizer_type!r}; the async runtime supports "
                f"{SUPPORTED_OPTIMIZERS}. QAOA is available offline via "
                "qexec.optimization.solvers.qaoa."
            )
        self.policy_queue = policy_queue
        self.optimizer_type = optimizer_type
        self.update_interval = update_interval
        self.seed = seed
        self.latency = latency_monitor

        self._thread: Thread | None = None
        self._stop_event = Event()
        self._running = False
        self._context_lock = Lock()
        self._current_order_size = 0
        self._num_slices = 10

        self.num_optimizations = 0
        self.total_optimization_time = 0.0
        self.resilience = OptimizerResilience(ResilienceConfig(timeout_seconds=5.0, max_retries=3))

    def start(self, order_size: int, num_slices: int) -> None:
        if self._running:
            logger.warning("Optimizer already running")
            return
        with self._context_lock:
            self._current_order_size = order_size
            self._num_slices = num_slices
        self._stop_event.clear()
        self._thread = Thread(target=self._optimization_loop, daemon=True)
        self._thread.start()
        self._running = True
        logger.info("Optimizer started (%s)", self.optimizer_type)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._running = False
        logger.info("Optimizer stopped")

    def _optimization_loop(self) -> None:
        while not self._stop_event.is_set():
            start = perf_counter()
            start_ns = time.monotonic_ns()
            # Keep the background thread alive whatever the solver or fallback raises.
            try:
                policy = self._run_optimization()
            except Exception:
                logger.exception("Optimization failed; keeping the previous policy")
                policy = None
            if policy is not None and self.latency is not None:
                self.latency.end_span(LatencyMonitor.SLOW_PATH_OPTIMIZE, start_ns)
            if policy is not None:
                self.policy_queue.publish(policy)
                self.num_optimizations += 1
                self.total_optimization_time += policy.optimization_time
            self._stop_event.wait(max(0.0, self.update_interval - (perf_counter() - start)))

    def _run_optimization(self) -> ExecutionPolicy | None:
        with self._context_lock:
            order_size = self._current_order_size
            num_slices = self._num_slices
        if order_size <= 0:
            return None

        start = perf_counter()

        def optimize() -> NDArray[np.float64]:
            if self.optimizer_type == "sa":
                return self._optimize_sa(order_size, num_slices)
            return uniform_schedule(order_size, num_slices).astype(np.float64)

        def fallback() -> NDArray[np.float64]:
            logger.warning("Using fallback execution strategy")
            return uniform_schedule(order_size, num_slices).astype(np.float64)

        schedule = self.resilience.execute(
            optimize,
            fallback_func=fallback,
            validation_func=lambda s: validate_schedule(s, order_size),
        )
        return ExecutionPolicy(
            schedule=schedule,
            optimizer_name=self.optimizer_type,
            optimization_time=perf_counter() - start,
        )

    def _optimize_sa(self, order_size: int, num_slices: int) -> NDArray[np.float64]:
        qubo = ExecutionQUBO(slice_level_config(order_size, num_slices))
        solver = SimulatedAnnealingSolver(num_sweeps=_SA_SWEEPS, seed=self.seed)
        schedule, _ = optimize_schedule(qubo, solver)
        # Discrete quantity levels rarely sum to the order exactly; repair to the order size.
        return repair_schedule(schedule, order_size).astype(np.float64)
