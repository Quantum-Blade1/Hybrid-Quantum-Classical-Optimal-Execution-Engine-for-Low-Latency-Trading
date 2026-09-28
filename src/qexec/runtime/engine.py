"""
Asynchronous execution engine (fast path).

Runs the tick loop, polls the PolicyQueue without blocking and executes
the current policy's slice; it never waits for the optimizer.
"""

from typing import Dict, List, Optional, Callable
from datetime import datetime
from time import time, sleep
from threading import Thread, Event
import logging

from qexec.runtime.policy import ExecutionPolicy, PolicyQueue

logger = logging.getLogger(__name__)


class AsyncExecutionEngine:
    """
    Asynchronous execution engine that NEVER waits for optimizer.
    
    Runs the main execution loop, polling for policy updates.
    If no optimization available, uses current/fallback policy.
    """
    
    def __init__(
        self,
        policy_queue: PolicyQueue,
        tick_interval: float = 0.1  # Seconds between ticks
    ):
        self.policy_queue = policy_queue
        self.tick_interval = tick_interval
        
        # Current state
        self._current_policy: Optional[ExecutionPolicy] = None
        self._fallback_policy: Optional[ExecutionPolicy] = None
        
        # Execution tracking
        self.current_time_idx = 0
        self.executed_shares = 0
        self.execution_log: List[Dict] = []
        
        # Thread control
        self._thread: Optional[Thread] = None
        self._stop_event = Event()
        self._running = False
        
        # Callbacks
        self._on_execute: Optional[Callable] = None
    
    def set_fallback_policy(self, policy: ExecutionPolicy) -> None:
        """Set fallback policy used when no optimization available."""
        self._fallback_policy = policy
        if self._current_policy is None:
            self._current_policy = policy
    
    def set_on_execute(self, callback: Callable) -> None:
        """Set callback for execution events."""
        self._on_execute = callback
    
    def start(self, total_ticks: int) -> None:
        """Start execution engine."""
        if self._running:
            return
        
        self._stop_event.clear()
        self._thread = Thread(
            target=self._execution_loop,
            args=(total_ticks,),
            daemon=True
        )
        self._thread.start()
        self._running = True
        logger.info("Execution engine started")
    
    def stop(self) -> None:
        """Stop execution engine."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        self._running = False
        logger.info("Execution engine stopped")
    
    def wait_complete(self) -> None:
        """Wait for execution to complete."""
        if self._thread:
            self._thread.join()
    
    def _execution_loop(self, total_ticks: int) -> None:
        """Main execution loop."""
        for tick in range(total_ticks):
            if self._stop_event.is_set():
                break
            
            tick_start = time()
            
            # Poll for policy update (NON-BLOCKING)
            new_policy = self.policy_queue.poll()
            if new_policy is not None:
                self._current_policy = new_policy
                logger.info(f"Tick {tick}: New policy {new_policy.policy_id} applied")
            
            # Get policy (use fallback if none)
            policy = self._current_policy or self._fallback_policy
            
            if policy is None:
                # No policy - skip execution
                logger.warning(f"Tick {tick}: No policy available")
            else:
                # Execute according to policy
                shares_to_execute = policy.get_slice(tick)
                self._execute(tick, shares_to_execute)
            
            self.current_time_idx = tick + 1
            
            # Wait for next tick
            elapsed = time() - tick_start
            wait_time = max(0, self.tick_interval - elapsed)
            sleep(wait_time)
        
        logger.info(f"Execution complete: {self.executed_shares} shares")
    
    def _execute(self, tick: int, shares: int) -> None:
        """Execute shares at current tick."""
        if shares <= 0:
            return
        
        self.executed_shares += shares
        
        log_entry = {
            'tick': tick,
            'shares': shares,
            'cumulative': self.executed_shares,
            'timestamp': datetime.now(),
            'policy_id': self._current_policy.policy_id if self._current_policy else 0
        }
        self.execution_log.append(log_entry)
        
        if self._on_execute:
            self._on_execute(log_entry)
        
        logger.debug(f"Tick {tick}: Executed {shares} shares")
