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
    # schedule. Following it blindly would trade 10 + 10 + 20 + 20 = 60 of a 40-share order.
    queue = PolicyQueue()
    engine = AsyncExecutionEngine(queue, tick_interval=0.0)
    engine.set_fallback_policy(policy([10, 10, 10, 10], "fallback"))

    def publish_after_tick_one(entry):
        if entry["tick"] == 1:
            queue.publish(policy([0, 0, 20, 20], "sa"))

    engine.set_on_execute(publish_after_tick_one)
    engine.start(total_ticks=4)
    engine.wait_complete()

    assert [e["shares"] for e in engine.execution_log] == [10, 10, 20]
    assert [e["policy_id"] for e in engine.execution_log] == [0, 0, 1]
    assert engine.executed_shares == 40


def test_unsupported_optimizer_is_rejected():
    # QAOA is offline-only; the runtime must not silently relabel SA as QAOA (claims audit R7).
    with pytest.raises(ValueError, match="Unsupported optimizer_type"):
        AsyncOptimizer(PolicyQueue(), optimizer_type="qaoa")


def test_hybrid_controller_applies_optimizer_policies_without_overfilling():
    # Policies plan the whole order, so switching mid-order can leave shares unexecuted
    # (a known limitation: the runtime does not re-plan the remainder), but never overfills.
    controller = HybridController(
        optimizer_type="sa", optimizer_interval=0.01, engine_tick_interval=0.05, seed=0
    )
    result = controller.execute_order(total_shares=1000, num_slices=10)
    log = result["execution_log"]
    assert result["num_optimizations"] >= 1
    assert any(entry["policy_id"] > 0 for entry in log)
    assert sum(entry["shares"] for entry in log) == result["executed_shares"] <= 1000
    assert [entry["cumulative"] for entry in log] == list(np.cumsum([e["shares"] for e in log]))
