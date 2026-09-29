"""Tick-level HFT pipeline and its latency bookkeeping."""

import numpy as np
import pytest

from qexec.runtime.hft_pipeline import HFTPipelineConfig, HFTQuantumPipeline
from qexec.runtime.latency import LatencyMonitor, LatencySpan


def ticks(rng: np.random.Generator, n: int):
    prices = 100 * np.exp(np.cumsum(2e-4 * rng.standard_normal(n)))
    return prices, prices - 0.01, prices + 0.01, rng.integers(100, 1000, n).astype(float)


@pytest.mark.parametrize("total_shares", [500, 777])
def test_pipeline_fills_exactly_the_order(rng, total_shares):
    config = HFTPipelineConfig(
        total_shares=total_shares,
        num_tick_slices=4,
        num_venues=2,
        optimizer_interval_ms=5.0,
        solver_sweeps=50,
        seed=0,
    )
    result = HFTQuantumPipeline(config).execute(*ticks(rng, 300))
    log = result.execution_log
    assert result.total_shares_executed == total_shares
    assert result.fill_rate == 1.0
    assert sum(entry["shares"] for entry in log) == total_shares
    assert all(entry["shares"] > 0 for entry in log)
    assert [entry["remaining"] for entry in log] == list(
        total_shares - np.cumsum([e["shares"] for e in log])
    )
    assert result.num_ticks == log[-1]["tick"] + 1  # stops as soon as the order is done


def test_pipeline_rejects_empty_tick_stream():
    with pytest.raises(ValueError, match="at least one tick"):
        HFTQuantumPipeline(HFTPipelineConfig()).execute(*(np.array([]),) * 4)


def test_latency_statistics_match_recorded_durations():
    monitor = LatencyMonitor()
    durations_us = np.array([5, 1, 4, 2, 3, 100], dtype=float)
    for d in durations_us:
        monitor.record_latency("x", int(d * 1000))
    stats = monitor.get_stats("x")
    assert stats is not None
    assert stats.count == 6
    assert stats.mean_us == pytest.approx(durations_us.mean())
    assert stats.median_us == pytest.approx(3.5)
    assert stats.p99_us == pytest.approx(np.percentile(durations_us, 99))
    assert (stats.min_us, stats.max_us) == (1.0, 100.0)
    assert monitor.get_stats("missing") is None


def test_latency_span_records_its_block():
    monitor = LatencyMonitor()
    for _ in range(2):
        with LatencySpan("block", monitor) as span:
            sum(range(1000))
        assert span.record is not None and span.record.duration_ns > 0
    stats = monitor.get_all_stats()
    assert list(stats) == ["block"] and stats["block"].count == 2
