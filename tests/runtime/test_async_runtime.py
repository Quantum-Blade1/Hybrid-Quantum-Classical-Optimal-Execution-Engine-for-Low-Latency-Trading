"""Fast/slow path runtime: the tick loop never waits for the optimizer."""

import threading
import time

import numpy as np
import pytest

from qexec.runtime.controller import HybridController
from qexec.runtime.engine import AsyncExecutionEngine
from qexec.runtime.optimizer import AsyncOptimizer
from qexec.runtime.policy import ExecutionPolicy, PolicyQueue, uniform_schedule


def policy(schedule, name: str = "test") -> ExecutionPolicy:
    return ExecutionPolicy(schedule=np.asarray(schedule, dtype=float), optimizer_name=name)


def test_policy_queue_delivers_only_the_newest_policy_once():
    queue = PolicyQueue()
    assert queue.poll() is None
    for k in range(3):
        queue.publish(policy([k]))
    newest = queue.poll()
    assert newest is not None
    assert newest.policy_id == 3 and newest.get_slice(0) == 2
    assert queue.poll() is None  # already consumed
    assert not queue.has_update
    queue.publish(policy([9]))
    assert queue.has_update


def test_policy_queue_is_consistent_under_concurrent_publishers():
    queue = PolicyQueue()

    def publish_many():
        for _ in range(200):
            queue.publish(policy([1]))

    threads = [threading.Thread(target=publish_many) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    latest = queue.poll()
    assert latest is not None and latest.policy_id == 800


@pytest.mark.parametrize(("total", "slices"), [(1000, 10), (1003, 10), (7, 10), (0, 3)])
def test_uniform_schedule_splits_order_exactly(total, slices):
    schedule = uniform_schedule(total, slices)
    assert schedule.sum() == total
    assert schedule.max() - schedule.min() <= 1


def test_fast_path_does_not_block_on_a_slow_optimizer(monkeypatch):
    # The optimizer takes 1 s per solve; 20 ticks at 1 ms must finish long before that,
    # all on the fallback policy.
    def slow_solve(self, order_size, num_slices):
        time.sleep(1.0)
        return uniform_schedule(order_size, num_slices).astype(float)

    monkeypatch.setattr(AsyncOptimizer, "_optimize_sa", slow_solve)
    queue = PolicyQueue()
    optimizer = AsyncOptimizer(queue, optimizer_type="sa", update_interval=0.0)
    engine = AsyncExecutionEngine(queue, tick_interval=0.001)
    engine.set_fallback_policy(policy(uniform_schedule(2000, 20), "fallback"))

    start = time.perf_counter()
    optimizer.start(order_size=2000, num_slices=20)
    engine.start(total_ticks=20)
    engine.wait_complete()
    elapsed = time.perf_counter() - start
    optimizer.stop()

    assert elapsed < 0.5
    assert engine.executed_shares == 2000
    assert {entry["policy_id"] for entry in engine.execution_log} == {0}  # fallback only


def test_fast_path_switches_to_published_policy_without_overfilling():
    # Fallback trades 10 per tick; after tick 1 the optimizer publishes a back-loaded
    # whole-order schedule. Following it blindly would trade 10 + 10 + 20 + 20 = 60 of a
    # 40-share order; its tail [20, 20] is rescaled to the 20 shares left.
    queue = PolicyQueue()
    engine = AsyncExecutionEngine(queue, tick_interval=0.0)
    engine.set_fallback_policy(policy([10, 10, 10, 10], "fallback"))

    def publish_after_tick_one(entry):
        if entry["tick"] == 1:
            queue.publish(policy([0, 0, 20, 20], "sa"))

    engine.set_on_execute(publish_after_tick_one)
    engine.start(total_ticks=4)
    engine.wait_complete()

    assert [e["shares"] for e in engine.execution_log] == [10, 10, 10, 10]
    assert [e["policy_id"] for e in engine.execution_log] == [0, 0, 1, 1]
    assert engine.executed_shares == 40


def test_policy_switch_replans_the_remaining_shares_so_the_order_completes():
    # Regression (claims audit T5): a front-loaded policy arriving after tick 1 has nothing
    # left in its tail; the runtime used to follow it (0 shares) and stop at 20 of 40.
    # Now the remaining 20 shares are re-planned over the remaining ticks.
    queue = PolicyQueue()
    engine = AsyncExecutionEngine(queue, tick_interval=0.0)
    engine.set_fallback_policy(policy([10, 10, 10, 10], "fallback"))

    def publish_after_tick_one(entry):
        if entry["tick"] == 1:
            queue.publish(policy([40, 0, 0, 0], "sa"))

    engine.set_on_execute(publish_after_tick_one)
    engine.start(total_ticks=4)
    engine.wait_complete()
    assert engine.executed_shares == 40
    assert [e["shares"] for e in engine.execution_log] == [10, 10, 10, 10]


def test_replan_keeps_the_shape_of_the_new_policy_tail():
    queue = PolicyQueue()
    engine = AsyncExecutionEngine(queue, tick_interval=0.0)
    engine.set_fallback_policy(policy([25, 25, 25, 25], "fallback"))
    queue_after = {0: [0, 60, 30, 10]}

    def publish(entry):
        if entry["tick"] in queue_after:
            queue.publish(policy(queue_after[entry["tick"]], "sa"))

    engine.set_on_execute(publish)
    engine.start(total_ticks=4)
    engine.wait_complete()
    # After tick 0 (25 done), the tail [60, 30, 10] is rescaled to 75: [45, 22.5, 7.5]
    # -> largest remainder [45, 23, 7].
    assert [e["shares"] for e in engine.execution_log] == [25, 45, 23, 7]


@pytest.mark.parametrize("seed", range(5))
def test_random_policy_switches_always_complete_the_order(seed):
    rng = np.random.default_rng(seed)
    total, ticks = int(rng.integers(50, 5000)), 12
    queue = PolicyQueue()
    engine = AsyncExecutionEngine(queue, tick_interval=0.0)
    engine.set_fallback_policy(policy(uniform_schedule(total, ticks), "fallback"))
    switch_ticks = set(rng.choice(ticks, size=4, replace=False).tolist())

    def publish(entry):
        if entry["tick"] in switch_ticks:
            queue.publish(policy(rng.integers(0, 3, ticks) * rng.integers(0, total, ticks)))

    engine.set_on_execute(publish)
    engine.start(total_ticks=ticks)
    engine.wait_complete()
    assert engine.executed_shares == total


def test_unsupported_optimizer_is_rejected():
    # QAOA is offline-only; the runtime must not silently relabel SA as QAOA (claims audit R7).
    with pytest.raises(ValueError, match="Unsupported optimizer_type"):
        AsyncOptimizer(PolicyQueue(), optimizer_type="qaoa")


def test_hybrid_controller_applies_optimizer_policies_and_completes_the_order():
    # Policies plan the whole order; the fast path re-plans the remainder on each switch.
    controller = HybridController(
        optimizer_type="sa", optimizer_interval=0.01, engine_tick_interval=0.05, seed=0
    )
    result = controller.execute_order(total_shares=1000, num_slices=10)
    log = result["execution_log"]
    assert result["num_optimizations"] >= 1
    assert any(entry["policy_id"] > 0 for entry in log)
    assert sum(entry["shares"] for entry in log) == result["executed_shares"] == 1000
    assert [entry["cumulative"] for entry in log] == list(np.cumsum([e["shares"] for e in log]))


def test_latency_monitor_records_fast_path_slow_path_and_propagation():
    from qexec.runtime.latency import LatencyMonitor

    monitor = LatencyMonitor()
    controller = HybridController(
        optimizer_type="sa",
        optimizer_interval=0.005,
        engine_tick_interval=0.01,
        seed=0,
        latency_monitor=monitor,
    )
    controller.execute_order(total_shares=2000, num_slices=40)
    stats = monitor.get_all_stats()
    assert stats[LatencyMonitor.FAST_PATH].count == 40
    assert stats[LatencyMonitor.SLOW_PATH_OPTIMIZE].count >= 2
    # The fast path's recorded work excludes the tick sleep.
    assert stats[LatencyMonitor.FAST_PATH].median_us < 10_000
    assert LatencyMonitor.POLICY_PROPAGATION in stats


def test_tick_lateness_is_recorded_for_every_tick_after_the_first():
    from qexec.runtime.latency import LatencyMonitor

    monitor = LatencyMonitor()
    queue = PolicyQueue()
    engine = AsyncExecutionEngine(queue, tick_interval=0.001, latency_monitor=monitor)
    engine.set_fallback_policy(policy(uniform_schedule(100, 10), "fallback"))
    engine.start(total_ticks=10)
    engine.wait_complete()
    stats = monitor.get_stats(LatencyMonitor.TICK_LATENESS)
    assert stats is not None and stats.count == 9 and stats.min_us >= 0
