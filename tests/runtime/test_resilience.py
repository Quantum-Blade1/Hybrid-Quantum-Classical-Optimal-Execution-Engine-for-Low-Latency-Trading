"""Optimizer resilience wrapper: timeout, retry, validation and fallback."""

import time

import numpy as np
import pytest

from qexec.runtime.resilience import (
    OptimizerResilience,
    OptimizerTimeoutError,
    ResilienceConfig,
    validate_schedule,
)


def wrapper(max_retries: int = 2, timeout: float = 0.05) -> OptimizerResilience:
    return OptimizerResilience(
        ResilienceConfig(timeout_seconds=timeout, max_retries=max_retries, base_backoff_seconds=0)
    )


def counting(func):
    calls = []

    def wrapped():
        calls.append(1)
        return func(len(calls))

    return wrapped, calls


def fallback() -> np.ndarray:
    return np.array([50.0, 50.0])


def test_valid_result_is_returned_without_retry():
    func, calls = counting(lambda _: np.array([60.0, 40.0]))
    result = wrapper().execute(func, fallback, lambda s: validate_schedule(s, 100))
    np.testing.assert_array_equal(result, [60.0, 40.0])
    assert len(calls) == 1


def test_timeout_falls_back_without_waiting_for_the_optimizer():
    def slow(_):
        time.sleep(0.5)
        return np.array([100.0, 0.0])

    func, calls = counting(slow)
    start = time.perf_counter()
    result = wrapper(max_retries=1).execute(func, fallback)
    assert time.perf_counter() - start < 0.3  # two 0.05 s timeouts, not 2 x 0.5 s
    np.testing.assert_array_equal(result, fallback())
    assert len(calls) == 2


def test_transient_exception_is_retried():
    def flaky(attempt):
        if attempt < 3:
            raise RuntimeError("solver crashed")
        return np.array([100.0])

    func, calls = counting(flaky)
    np.testing.assert_array_equal(wrapper(max_retries=2).execute(func, fallback), [100.0])
    assert len(calls) == 3


@pytest.mark.parametrize(
    "bad_schedule",
    [np.array([100.0, 100.0]), np.array([110.0, -10.0]), [50, 50]],
    ids=["wrong-total", "negative", "not-an-array"],
)
def test_invalid_schedule_falls_back_after_all_retries(bad_schedule):
    func, calls = counting(lambda _: bad_schedule)
    result = wrapper(max_retries=2).execute(func, fallback, lambda s: validate_schedule(s, 100))
    np.testing.assert_array_equal(result, fallback())
    assert len(calls) == 3


def test_without_fallback_the_last_error_is_raised():
    func, _ = counting(lambda _: time.sleep(0.5))
    with pytest.raises(OptimizerTimeoutError):
        wrapper(max_retries=0).execute(func)


@pytest.mark.parametrize(
    ("schedule", "valid"),
    [([100, 0], True), ([99.5, 0], True), ([98, 0], False), ([101, -1], False)],
)
def test_validate_schedule_tolerance_is_one_share_or_one_percent(schedule, valid):
    assert validate_schedule(np.array(schedule, dtype=float), 100) is valid
