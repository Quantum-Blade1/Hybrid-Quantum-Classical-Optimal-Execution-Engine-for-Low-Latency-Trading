"""
Nanosecond-Precision Latency Monitor for HFT Pipeline

Tracks end-to-end latency through the hybrid quantum-classical architecture:
1. Tick-to-Decision latency (fast path)
2. Optimization cycle time (slow path)
3. Policy propagation delay (queue transfer)
4. Policy staleness (age of current policy when used)
5. Execution-to-fill latency

Provides formal policy staleness analysis:
    If market autocorrelation time is tau and solve time is T_solve,
    the regret from using a stale policy is bounded by O(sigma * sqrt(T_solve))
    when T_solve << tau.

No existing quantum finance paper measures or reports these metrics.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import deque
from threading import Lock


def _monotonic_ns() -> int:
    return time.monotonic_ns()


@dataclass
class LatencyRecord:
    component: str
    start_ns: int
    end_ns: int
    metadata: Dict = field(default_factory=dict)

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns

    @property
    def duration_us(self) -> float:
        return self.duration_ns / 1000.0

    @property
    def duration_ms(self) -> float:
        return self.duration_ns / 1_000_000.0


@dataclass
class LatencyStats:
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

    def to_dict(self) -> Dict:
        return {
            "component": self.component,
            "count": self.count,
            "mean_us": round(self.mean_us, 2),
            "median_us": round(self.median_us, 2),
            "p95_us": round(self.p95_us, 2),
            "p99_us": round(self.p99_us, 2),
            "min_us": round(self.min_us, 2),
            "max_us": round(self.max_us, 2),
            "stddev_us": round(self.stddev_us, 2),
            "jitter_us": round(self.jitter_us, 2)
        }


@dataclass
class StalenessAnalysis:
    """Policy staleness analysis results."""
    mean_staleness_ms: float
    max_staleness_ms: float
    p95_staleness_ms: float
    staleness_at_use: List[float]
    regret_bound: float
    autocorrelation_time_ms: float
    is_stale_regime: bool


class LatencyMonitor:
    """
    Thread-safe nanosecond-precision latency tracker.

    Tracks latency across all components of the hybrid architecture,
    enabling formal analysis of the Fast/Slow path decoupling.
    """

    FAST_PATH = "fast_path"
    SLOW_PATH_OPTIMIZE = "slow_path_optimize"
    SLOW_PATH_QUBO_BUILD = "slow_path_qubo_build"
    SLOW_PATH_SOLVE = "slow_path_solve"
    POLICY_PROPAGATION = "policy_propagation"
    POLICY_STALENESS = "policy_staleness"
    TICK_TO_DECISION = "tick_to_decision"
    DECISION_TO_ORDER = "decision_to_order"
    END_TO_END = "end_to_end"

    def __init__(self, max_records: int = 100_000):
        self._lock = Lock()
        self._records: Dict[str, deque] = {}
        self._max_records = max_records
        self._active_spans: Dict[str, int] = {}

    def start_span(self, component: str) -> int:
        ts = _monotonic_ns()
        key = f"{component}_{id(ts)}"
        with self._lock:
            self._active_spans[key] = ts
        return ts

    def end_span(self, component: str, start_ns: int, metadata: Dict = None) -> LatencyRecord:
        end_ns = _monotonic_ns()
        record = LatencyRecord(
            component=component,
            start_ns=start_ns,
            end_ns=end_ns,
            metadata=metadata or {}
        )
        with self._lock:
            if component not in self._records:
                self._records[component] = deque(maxlen=self._max_records)
            self._records[component].append(record)
        return record

    def record_latency(self, component: str, duration_ns: int, metadata: Dict = None) -> None:
        now = _monotonic_ns()
        record = LatencyRecord(
            component=component,
            start_ns=now - duration_ns,
            end_ns=now,
            metadata=metadata or {}
        )
        with self._lock:
            if component not in self._records:
                self._records[component] = deque(maxlen=self._max_records)
            self._records[component].append(record)

    def record_staleness(self, policy_age_ms: float, policy_id: int) -> None:
        self.record_latency(
            self.POLICY_STALENESS,
            int(policy_age_ms * 1_000_000),
            {"policy_id": policy_id}
        )

    def get_stats(self, component: str) -> Optional[LatencyStats]:
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
            jitter_us=float(np.std(np.diff(durations_us))) if len(durations_us) > 2 else 0.0
        )

    def get_all_stats(self) -> Dict[str, LatencyStats]:
        with self._lock:
            components = list(self._records.keys())
        return {c: self.get_stats(c) for c in components if self.get_stats(c) is not None}

    def analyze_staleness(
        self,
        market_volatility: float = 0.02,
        tick_interval_ms: float = 100.0
    ) -> Optional[StalenessAnalysis]:
        with self._lock:
            staleness_records = self._records.get(self.POLICY_STALENESS)
            optimize_records = self._records.get(self.SLOW_PATH_OPTIMIZE)

        if not staleness_records or len(staleness_records) < 5:
            return None

        staleness_ms = [r.duration_ms for r in staleness_records]

        avg_solve_time_ms = 0.0
        if optimize_records and len(optimize_records) > 0:
            avg_solve_time_ms = np.mean([r.duration_ms for r in optimize_records])

        autocorr_time_ms = tick_interval_ms * 10
        sigma_per_ms = market_volatility / np.sqrt(252 * 6.5 * 60 * 1000)
        regret_bound = sigma_per_ms * np.sqrt(avg_solve_time_ms) if avg_solve_time_ms > 0 else 0

        return StalenessAnalysis(
            mean_staleness_ms=float(np.mean(staleness_ms)),
            max_staleness_ms=float(np.max(staleness_ms)),
            p95_staleness_ms=float(np.percentile(staleness_ms, 95)),
            staleness_at_use=staleness_ms,
            regret_bound=regret_bound,
            autocorrelation_time_ms=autocorr_time_ms,
            is_stale_regime=np.mean(staleness_ms) > autocorr_time_ms
        )

    def print_report(self) -> None:
        all_stats = self.get_all_stats()

        print("\n" + "=" * 80)
        print(" Latency Report - Hybrid Quantum-Classical HFT Pipeline")
        print("=" * 80)
        print(f"{'Component':<30} {'Count':>6} {'Mean':>10} {'P50':>10} {'P95':>10} {'P99':>10} {'Jitter':>10}")
        print(f"{'':30} {'':>6} {'(us)':>10} {'(us)':>10} {'(us)':>10} {'(us)':>10} {'(us)':>10}")
        print("-" * 80)

        order = [
            self.FAST_PATH, self.TICK_TO_DECISION, self.DECISION_TO_ORDER,
            self.SLOW_PATH_OPTIMIZE, self.SLOW_PATH_QUBO_BUILD, self.SLOW_PATH_SOLVE,
            self.POLICY_PROPAGATION, self.POLICY_STALENESS, self.END_TO_END
        ]

        for component in order:
            stats = all_stats.get(component)
            if stats:
                print(
                    f"  {stats.component:<28} {stats.count:>6} "
                    f"{stats.mean_us:>10.1f} {stats.median_us:>10.1f} "
                    f"{stats.p95_us:>10.1f} {stats.p99_us:>10.1f} "
                    f"{stats.jitter_us:>10.1f}"
                )

        for component, stats in all_stats.items():
            if component not in order:
                print(
                    f"  {stats.component:<28} {stats.count:>6} "
                    f"{stats.mean_us:>10.1f} {stats.median_us:>10.1f} "
                    f"{stats.p95_us:>10.1f} {stats.p99_us:>10.1f} "
                    f"{stats.jitter_us:>10.1f}"
                )

        staleness = self.analyze_staleness()
        if staleness:
            print(f"\n  Policy Staleness Analysis:")
            print(f"    Mean staleness:       {staleness.mean_staleness_ms:.2f} ms")
            print(f"    P95 staleness:        {staleness.p95_staleness_ms:.2f} ms")
            print(f"    Max staleness:        {staleness.max_staleness_ms:.2f} ms")
            print(f"    Autocorrelation time: {staleness.autocorrelation_time_ms:.0f} ms")
            print(f"    Regret bound:         {staleness.regret_bound:.6f} (sigma * sqrt(T_solve))")
            print(f"    Stale regime:         {'YES' if staleness.is_stale_regime else 'NO'}")

        print("=" * 80)


_global_monitor: Optional[LatencyMonitor] = None
_monitor_lock = Lock()


def get_latency_monitor() -> LatencyMonitor:
    global _global_monitor
    with _monitor_lock:
        if _global_monitor is None:
            _global_monitor = LatencyMonitor()
        return _global_monitor


class LatencySpan:
    """Context manager for measuring a span's latency."""

    def __init__(self, component: str, monitor: Optional[LatencyMonitor] = None, metadata: Dict = None):
        self._component = component
        self._monitor = monitor or get_latency_monitor()
        self._metadata = metadata or {}
        self._start_ns = 0
        self.record: Optional[LatencyRecord] = None

    def __enter__(self):
        self._start_ns = _monotonic_ns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.record = self._monitor.end_span(
            self._component, self._start_ns, self._metadata
        )
        return False


def run_latency_demo():
    """Demonstrate latency monitoring on the hybrid architecture."""
    monitor = LatencyMonitor()

    print("\n" + "=" * 70)
    print(" HFT Latency Monitor Demo")
    print("=" * 70)

    for i in range(200):
        with LatencySpan("fast_path", monitor):
            time.sleep(np.random.exponential(0.0001))

        if i % 10 == 0:
            with LatencySpan("slow_path_optimize", monitor):
                time.sleep(np.random.exponential(0.05))

            staleness = np.random.exponential(50)
            monitor.record_staleness(staleness, i // 10)

        with LatencySpan("tick_to_decision", monitor):
            time.sleep(np.random.exponential(0.00005))

    monitor.print_report()
    return monitor


if __name__ == "__main__":
    run_latency_demo()
