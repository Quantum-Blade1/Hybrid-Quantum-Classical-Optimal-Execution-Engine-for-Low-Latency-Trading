"""
Execution policies and the queue that carries them from the slow path to the fast path.

The optimizer (slow path) publishes immutable ExecutionPolicy objects; the
execution engine (fast path) polls the PolicyQueue without blocking and
only ever sees the latest policy.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from threading import Lock
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ExecutionPolicy:
    """
    Execution policy specifying how to slice an order.
    
    This is the output from the optimizer that guides the execution engine.
    Policies are immutable once created - new optimizations produce new policies.
    """
    schedule: np.ndarray  # Shares to execute at each time step
    timestamp: datetime = field(default_factory=datetime.now)
    policy_id: int = 0
    optimizer_name: str = "default"
    optimization_time: float = 0.0
    energy: float = float('inf')
    
    @property
    def total_shares(self) -> int:
        return int(np.sum(self.schedule))
    
    def get_slice(self, time_idx: int) -> int:
        """Get shares to execute at given time index."""
        if 0 <= time_idx < len(self.schedule):
            return int(self.schedule[time_idx])
        return 0


class PolicyQueue:
    """
    Thread-safe queue for policy updates.
    
    Single-producer (optimizer) / single-consumer (engine) pattern.
    Engine can poll without blocking; only latest policy matters.
    """
    
    def __init__(self):
        self._lock = Lock()
        self._latest_policy: Optional[ExecutionPolicy] = None
        self._policy_count = 0
        self._last_read_id = -1
    
    def publish(self, policy: ExecutionPolicy) -> None:
        """Publish new policy (called by optimizer thread)."""
        with self._lock:
            self._policy_count += 1
            policy.policy_id = self._policy_count
            self._latest_policy = policy
            logger.debug(f"Published policy {policy.policy_id}")
    
    def poll(self) -> Optional[ExecutionPolicy]:
        """
        Poll for new policy (called by engine thread).
        
        Returns:
            New policy if available, None if no update
        """
        with self._lock:
            if self._latest_policy is None:
                return None
            if self._latest_policy.policy_id <= self._last_read_id:
                return None  # Already seen this policy
            
            self._last_read_id = self._latest_policy.policy_id
            return self._latest_policy
    
    @property
    def has_update(self) -> bool:
        """Check if new policy available without consuming it."""
        with self._lock:
            if self._latest_policy is None:
                return False
            return self._latest_policy.policy_id > self._last_read_id
