"""Fast path: tick loop that applies the newest policy without waiting for the optimizer."""

import logging
from collections.abc import Callable
from datetime import datetime
from threading import Event, Thread
from time import perf_counter, sleep
from typing import Any

from qexec.runtime.policy import ExecutionPolicy, PolicyQueue

logger = logging.getLogger(__name__)


class AsyncExecutionEngine:
    """Each tick polls the `PolicyQueue` (non-blocking) and executes the current policy's slice.

    Before the first published policy, the fallback policy is used.
    """

    def __init__(self, policy_queue: PolicyQueue, tick_interval: float = 0.1) -> None:
        self.policy_queue = policy_queue
        self.tick_interval = tick_interval
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

    def start(self, total_ticks: int) -> None:
        if self._running:
            return
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

    def _execution_loop(self, total_ticks: int) -> None:
        for tick in range(total_ticks):
            if self._stop_event.is_set():
                break
            tick_start = perf_counter()

            new_policy = self.policy_queue.poll()
            if new_policy is not None:
                self._current_policy = new_policy
                logger.info("Tick %d: policy %d applied", tick, new_policy.policy_id)

            policy = self._current_policy or self._fallback_policy
            if policy is None:
                logger.warning("Tick %d: no policy available", tick)
            else:
                self._execute(tick, policy.get_slice(tick))

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
