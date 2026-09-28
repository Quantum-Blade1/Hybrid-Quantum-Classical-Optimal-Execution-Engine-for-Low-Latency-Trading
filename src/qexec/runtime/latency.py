"""Thread-safe monotonic-clock latency records, per-component statistics and policy staleness."""

import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from types import TracebackType
from typing import Any

import numpy as np

TRADING_MS_PER_YEAR = 252 * 6.5 * 60 * 1000
_MIN_STALENESS_RECORDS = 5
# Assumed market autocorrelation time, in ticks, for the stale-regime flag.
_AUTOCORRELATION_TICKS = 10


@dataclass(frozen=True)
class LatencyRecord:
    component: str
    start_ns: int
    end_ns: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns

    @property
    def duration_us(self) -> float:
        return self.duration_ns / 1_000.0

    @property
    def duration_ms(self) -> float:
        return self.duration_ns / 1_000_000.0


@dataclass(frozen=True)
class LatencyStats:
    """Summary of one component's durations in microseconds; jitter is std of successive diffs."""

    component: str
    count: int
    mean_us: float
    median_us: float
    p95_us: float
    p99_us: float
    min_us: float
    max_us: float
    stddev_us: float
    jitter_us: float

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "component": self.component,
            "count": self.count,
            **{
                name: round(getattr(self, name), 2)
                for name in (
                    "mean_us",
                    "median_us",
                    "p95_us",
                    "p99_us",
                    "min_us",
                    "max_us",
                    "stddev_us",
                    "jitter_us",
                )
            },
        }


@dataclass(frozen=True)
class StalenessAnalysis:
    """Age of each policy when replaced, and a heuristic regret scale sigma_ms sqrt(T_solve)."""

    mean_staleness_ms: float
    max_staleness_ms: float
    p95_staleness_ms: float
    staleness_at_use: list[float]
    regret_bound: float
    autocorrelation_time_ms: float
    is_stale_regime: bool


class LatencyMonitor:
    """Stores the last `max_records` latency records per component."""

    FAST_PATH = "fast_path"
    TICK_TO_DECISION = "tick_to_decision"
    SLOW_PATH_OPTIMIZE = "slow_path_optimize"
    SLOW_PATH_QUBO_BUILD = "slow_path_qubo_build"
    SLOW_PATH_SOLVE = "slow_path_solve"
    POLICY_PROPAGATION = "policy_propagation"
    POLICY_STALENESS = "policy_staleness"

    def __init__(self, max_records: int = 100_000) -> None:
        self._lock = Lock()
        self._records: dict[str, deque[LatencyRecord]] = {}
        self._max_records = max_records

    def _append(self, record: LatencyRecord) -> None:
        with self._lock:
            if record.component not in self._records:
                self._records[record.component] = deque(maxlen=self._max_records)
            self._records[record.component].append(record)

    def end_span(
        self, component: str, start_ns: int, metadata: dict[str, Any] | None = None
    ) -> LatencyRecord:
        """Record a span that started at `start_ns` (time.monotonic_ns) and ends now."""
        record = LatencyRecord(component, start_ns, time.monotonic_ns(), metadata or {})
        self._append(record)
        return record

    def record_latency(
        self, component: str, duration_ns: int, metadata: dict[str, Any] | None = None
    ) -> None:
        now = time.monotonic_ns()
        self._append(LatencyRecord(component, now - duration_ns, now, metadata or {}))

    def record_staleness(self, policy_age_ms: float, policy_id: int) -> None:
        self.record_latency(
            self.POLICY_STALENESS, int(policy_age_ms * 1_000_000), {"policy_id": policy_id}
        )

    def get_stats(self, component: str) -> LatencyStats | None:
        """Statistics for `component`, or None with fewer than two records."""
        with self._lock:
            records = self._records.get(component)
            if not records or len(records) < 2:
                return None
            durations_us = np.array([r.duration_us for r in records])

        return LatencyStats(
            component=component,
            count=len(durations_us),
            mean_us=float(np.mean(durations_us)),
            median_us=float(np.median(durations_us)),
            p95_us=float(np.percentile(durations_us, 95)),
            p99_us=float(np.percentile(durations_us, 99)),
            min_us=float(np.min(durations_us)),
            max_us=float(np.max(durations_us)),
            stddev_us=float(np.std(durations_us)),
            jitter_us=float(np.std(np.diff(durations_us))) if len(durations_us) > 2 else 0.0,
        )

    def get_all_stats(self) -> dict[str, LatencyStats]:
        with self._lock:
            components = list(self._records)
        stats = {c: self.get_stats(c) for c in components}
        return {c: s for c, s in stats.items() if s is not None}

    def analyze_staleness(
        self, market_volatility: float = 0.02, tick_interval_ms: float = 100.0
    ) -> StalenessAnalysis | None:
        """Staleness statistics once at least five policies have been replaced.

        `market_volatility` is annualised; the regret scale is a heuristic, not a proven bound.
        """
        with self._lock:
            staleness_records = list(self._records.get(self.POLICY_STALENESS, ()))
            optimize_records = list(self._records.get(self.SLOW_PATH_OPTIMIZE, ()))
        if len(staleness_records) < _MIN_STALENESS_RECORDS:
            return None

        staleness_ms = [r.duration_ms for r in staleness_records]
        avg_solve_time_ms = (
            float(np.mean([r.duration_ms for r in optimize_records])) if optimize_records else 0.0
        )
        autocorr_time_ms = tick_interval_ms * _AUTOCORRELATION_TICKS
        sigma_per_ms = market_volatility / np.sqrt(TRADING_MS_PER_YEAR)
        regret_bound = float(sigma_per_ms * np.sqrt(avg_solve_time_ms))

        return StalenessAnalysis(
            mean_staleness_ms=float(np.mean(staleness_ms)),
            max_staleness_ms=float(np.max(staleness_ms)),
            p95_staleness_ms=float(np.percentile(staleness_ms, 95)),
            staleness_at_use=staleness_ms,
            regret_bound=regret_bound,
            autocorrelation_time_ms=autocorr_time_ms,
            is_stale_regime=bool(np.mean(staleness_ms) > autocorr_time_ms),
        )


class LatencySpan:
    """Context manager that records the duration of its block in `monitor`."""

    def __init__(
        self, component: str, monitor: LatencyMonitor, metadata: dict[str, Any] | None = None
    ) -> None:
        self._component = component
        self._monitor = monitor
        self._metadata = metadata or {}
        self._start_ns = 0
        self.record: LatencyRecord | None = None

    def __enter__(self) -> "LatencySpan":
        self._start_ns = time.monotonic_ns()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.record = self._monitor.end_span(self._component, self._start_ns, self._metadata)
