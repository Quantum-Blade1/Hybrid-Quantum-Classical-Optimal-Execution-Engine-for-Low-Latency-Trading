import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class ExecutionPolicy:
    """Shares per time step; `policy_id` and `published_ns` (monotonic) are set on publish."""

    schedule: NDArray[np.float64]
    timestamp: datetime = field(default_factory=datetime.now)
    policy_id: int = 0
    optimizer_name: str = "default"
    optimization_time: float = 0.0
    energy: float = float("inf")
    published_ns: int = 0

    @property
    def total_shares(self) -> int:
        return int(np.sum(self.schedule))

    def get_slice(self, time_idx: int) -> int:
        if 0 <= time_idx < len(self.schedule):
            return int(self.schedule[time_idx])
        return 0


class PolicyQueue:
    """Single-slot, lock-protected mailbox: the consumer only ever sees the newest policy."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._latest_policy: ExecutionPolicy | None = None
        self._policy_count = 0
        self._last_read_id = -1

    def publish(self, policy: ExecutionPolicy) -> None:
        with self._lock:
            self._policy_count += 1
            policy.policy_id = self._policy_count
            policy.published_ns = time.monotonic_ns()
            self._latest_policy = policy
        logger.debug("Published policy %d", policy.policy_id)

    def poll(self) -> ExecutionPolicy | None:
        """Newest policy if not returned before, else None; never waits for the producer."""
        with self._lock:
            if self._latest_policy is None:
                return None
            if self._latest_policy.policy_id <= self._last_read_id:
                return None
            self._last_read_id = self._latest_policy.policy_id
            return self._latest_policy

    @property
    def has_update(self) -> bool:
        with self._lock:
            if self._latest_policy is None:
                return False
            return self._latest_policy.policy_id > self._last_read_id


def uniform_schedule(total_shares: int, num_slices: int) -> NDArray[np.int_]:
    schedule = np.full(num_slices, total_shares // num_slices)
    schedule[: total_shares % num_slices] += 1
    return schedule
