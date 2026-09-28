"""Timeout, retry with exponential backoff, validation and fallback around an optimizer call."""

import logging
import queue
import time
from collections.abc import Callable
from dataclasses import dataclass
from threading import Thread
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

T = TypeVar("T")


class OptimizerTimeoutError(Exception):
    pass


class InvalidSolutionError(Exception):
    pass


@dataclass(frozen=True)
class ResilienceConfig:
    timeout_seconds: float = 5.0
    max_retries: int = 3
    base_backoff_seconds: float = 0.5
    validate_solution: bool = True


class OptimizerResilience:
    """Runs a callable with a timeout; the k-th retry waits base * 2^(k-1) seconds first."""

    def __init__(self, config: ResilienceConfig | None = None) -> None:
        self.config = config or ResilienceConfig()

    def execute(
        self,
        func: Callable[[], T],
        fallback_func: Callable[[], T] | None = None,
        validation_func: Callable[[T], bool] | None = None,
    ) -> T:
        """Return the first valid result of `func`, else `fallback_func()`, else re-raise."""
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            if attempt > 0:
                backoff = self.config.base_backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "Retry %d/%d after %.1fs backoff", attempt, self.config.max_retries, backoff
                )
                time.sleep(backoff)
            try:
                result = self._run_with_timeout(func)
                if (
                    self.config.validate_solution
                    and validation_func is not None
                    and not validation_func(result)
                ):
                    raise InvalidSolutionError("Solution failed validation")
                return result
            # The optimizer is arbitrary user code: any failure should trigger a retry.
            except Exception as e:
                last_error = e
                logger.error(
                    "Attempt %d failed: %s: %s", attempt + 1, type(e).__name__, e, exc_info=True
                )

        logger.error("All retries failed. Last error: %s", last_error)
        if fallback_func is not None:
            logger.info("Using fallback")
            return fallback_func()
        assert last_error is not None
        raise last_error

    def _run_with_timeout(self, func: Callable[[], T]) -> T:
        """Run `func` in a daemon thread; on timeout the thread is abandoned, not killed."""
        results: queue.Queue[T] = queue.Queue()
        errors: queue.Queue[Exception] = queue.Queue()

        def wrapper() -> None:
            # Forward any failure to the calling thread, where it is re-raised.
            try:
                results.put(func())
            except Exception as e:
                errors.put(e)

        worker = Thread(target=wrapper, daemon=True)
        worker.start()
        worker.join(timeout=self.config.timeout_seconds)
        if not errors.empty():
            raise errors.get()
        if results.empty():
            raise OptimizerTimeoutError(
                f"Optimization timed out after {self.config.timeout_seconds}s"
            )
        return results.get()


def validate_schedule(schedule: NDArray[np.float64], total_shares: int) -> bool:
    """Non-negative numpy schedule summing to `total_shares` within max(1, 1%)."""
    if not isinstance(schedule, np.ndarray):
        logger.debug("Invalid schedule: not a numpy array")
        return False
    if np.any(schedule < 0):
        logger.debug("Invalid schedule: negative quantities")
        return False
    current_sum = float(np.sum(schedule))
    if abs(current_sum - total_shares) > max(1, total_shares * 0.01):
        logger.debug("Invalid schedule: sum %s != %s", current_sum, total_shares)
        return False
    return True
