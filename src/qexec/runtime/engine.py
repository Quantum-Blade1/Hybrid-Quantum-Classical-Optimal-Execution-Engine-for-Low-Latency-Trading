"""Fast path: tick loop that applies the newest policy without waiting for the optimizer."""

import logging
import time
from collections.abc import Callable
from datetime import datetime
from threading import Event, Thread
from time import perf_counter, sleep
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qexec.optimization.schedule import repair_schedule
from qexec.runtime.latency import LatencyMonitor
from qexec.runtime.policy import ExecutionPolicy, PolicyQueue

logger = logging.getLogger(__name__)


class AsyncExecutionEngine:
    """Each tick polls the `PolicyQueue` (non-blocking) and executes the current plan's slice.

    The plan starts as the fallback policy's schedule. A policy that arrives at tick t
    plans the whole order, so its schedule from t on is rescaled to the shares still
    unexecuted (`repair_schedule`; uniform if its tail is empty) and replaces the plan
    from t on. The order therefore completes by the last tick after any number of policy
    switches, and never overfills. With a `latency_monitor`, each tick's work (poll,
    re-plan, execute; not the sleep) is recorded as `fast_path` and the delay between a
    policy's publication and its application as `policy_propagation`.
    """

    def __init__(
        self,
        policy_queue: PolicyQueue,
        tick_interval: float = 0.1,
        latency_monitor: LatencyMonitor | None = None,
    ) -> None:
        self.policy_queue = policy_queue
        self.tick_interval = tick_interval
        self.latency = latency_monitor
        self._plan: NDArray[np.int_] | None = None
        self._total_shares = 0
        self._current_policy: ExecutionPolicy | None = None
        self._fallback_policy: ExecutionPolicy | None = None
        self.current_time_idx = 0
        self.executed_shares = 0
        self.execution_log: list[dict[str, Any]] = []
        self._thread: Thread | None = None
        self._stop_event = Event()
        self._running = False
        self._on_execute: Callable[[dict[str, Any]], None] | None = None

    @property
    def current_policy(self) -> ExecutionPolicy | None:
        return self._current_policy

    def set_fallback_policy(self, policy: ExecutionPolicy) -> None:
        self._fallback_policy = policy
        if self._current_policy is None:
            self._current_policy = policy

    def set_on_execute(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register a callback that receives each execution log entry."""
        self._on_execute = callback

    def start(self, total_ticks: int, total_shares: int | None = None) -> None:
        """Run `total_ticks` ticks for an order of `total_shares` (default: the fallback's)."""
        if self._running:
            return
        if total_shares is None:
            if self._fallback_policy is None:
                raise ValueError("start needs total_shares or a fallback policy")
            total_shares = self._fallback_policy.total_shares
        self._total_shares = total_shares
        self._plan = self._initial_plan(total_ticks)
        self._stop_event.clear()
        self._thread = Thread(target=self._execution_loop, args=(total_ticks,), daemon=True)
        self._thread.start()
        self._running = True
        logger.info("Execution engine started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._running = False
        logger.info("Execution engine stopped")

    def wait_complete(self) -> None:
        if self._thread is not None:
            self._thread.join()

    @property
    def plan(self) -> NDArray[np.int_] | None:
        """Current per-tick plan (executed ticks keep what was planned for them)."""
        return None if self._plan is None else self._plan.copy()

    def _initial_plan(self, total_ticks: int) -> NDArray[np.int_]:
        policy = self._current_policy or self._fallback_policy
        if policy is None:
            return repair_schedule(np.ones(total_ticks), self._total_shares)
        schedule = np.zeros(total_ticks)
        head = np.asarray(policy.schedule, dtype=float)[:total_ticks]
        schedule[: len(head)] = head
        return repair_schedule(schedule, self._total_shares)

    def _replan(self, tick: int, policy: ExecutionPolicy) -> None:
        """Rescale the policy's schedule from `tick` on to the unexecuted shares."""
        assert self._plan is not None
        tail = np.zeros(len(self._plan) - tick)
        schedule = np.asarray(policy.schedule, dtype=float)[tick : len(self._plan)]
        tail[: len(schedule)] = schedule
        remaining = self._total_shares - self.executed_shares
        self._plan[tick:] = repair_schedule(tail, max(0, remaining))

    def _tick(self, tick: int) -> None:
        new_policy = self.policy_queue.poll()
        if new_policy is not None:
            if self.latency is not None and new_policy.published_ns:
                self.latency.record_latency(
                    LatencyMonitor.POLICY_PROPAGATION,
                    time.monotonic_ns() - new_policy.published_ns,
                    {"policy_id": new_policy.policy_id},
                )
            self._current_policy = new_policy
            self._replan(tick, new_policy)
            logger.info("Tick %d: policy %d applied", tick, new_policy.policy_id)
        assert self._plan is not None
        remaining = self._total_shares - self.executed_shares
        self._execute(tick, min(int(self._plan[tick]), remaining))

    def _execution_loop(self, total_ticks: int) -> None:
        for tick in range(total_ticks):
            if self._stop_event.is_set():
                break
            tick_start = perf_counter()
            start_ns = time.monotonic_ns()
            self._tick(tick)
            if self.latency is not None:
                self.latency.end_span(LatencyMonitor.FAST_PATH, start_ns)
            self.current_time_idx = tick + 1
            sleep(max(0.0, self.tick_interval - (perf_counter() - tick_start)))

        logger.info("Execution complete: %d shares", self.executed_shares)

    def _execute(self, tick: int, shares: int) -> None:
        if shares <= 0:
            return
        self.executed_shares += shares
        log_entry = {
            "tick": tick,
            "shares": shares,
            "cumulative": self.executed_shares,
            "timestamp": datetime.now(),
            "policy_id": self._current_policy.policy_id if self._current_policy else 0,
        }
        self.execution_log.append(log_entry)
        if self._on_execute is not None:
            self._on_execute(log_entry)
