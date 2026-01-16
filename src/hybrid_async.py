"""
Hybrid Async Execution Architecture

This module implements a hybrid quantum-classical execution system where:
1. Classical engine runs the main execution loop (FAST PATH - never waits)
2. Quantum/classical optimizer runs in separate thread (SLOW PATH)
3. Communication via thread-safe policy queue
4. Engine polls for updated execution policy without blocking

Key Requirement: TRADING NEVER WAITS FOR OPTIMIZER
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from time import time, sleep
from threading import Thread, Lock, Event
from queue import Queue, Empty
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Execution Policy
# =============================================================================

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


# =============================================================================
# Policy Queue (Thread-Safe Communication)
# =============================================================================

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


# =============================================================================
# Async Optimizer (Slow Path)
# =============================================================================

class AsyncOptimizer:
    """
    Asynchronous optimizer that runs in a background thread.
    
    Continuously optimizes execution schedule based on market conditions
    and publishes updated policies to the queue.
    """
    
    def __init__(
        self,
        policy_queue: PolicyQueue,
        optimizer_type: str = 'sa',  # 'sa' or 'qaoa'
        update_interval: float = 1.0,  # Seconds between optimizations
        seed: Optional[int] = None
    ):
        self.policy_queue = policy_queue
        self.optimizer_type = optimizer_type
        self.update_interval = update_interval
        self.seed = seed
        
        # Thread control
        self._thread: Optional[Thread] = None
        self._stop_event = Event()
        self._running = False
        
        # Current optimization context
        self._current_order_size: int = 0
        self._num_slices: int = 10
        self._market_data: Optional[pd.DataFrame] = None
        self._context_lock = Lock()
        
        # Statistics
        self.num_optimizations = 0
        self.total_optimization_time = 0.0
        
        # Resilience
        from .optimizer_resilience import OptimizerResilience, ResilienceConfig
        self.resilience = OptimizerResilience(ResilienceConfig(
            timeout_seconds=5.0,
            max_retries=3
        ))
    
    def start(self, order_size: int, num_slices: int) -> None:
        """Start the optimizer thread."""
        if self._running:
            logger.warning("Optimizer already running")
            return
        
        with self._context_lock:
            self._current_order_size = order_size
            self._num_slices = num_slices
        
        self._stop_event.clear()
        self._thread = Thread(target=self._optimization_loop, daemon=True)
        self._thread.start()
        self._running = True
        logger.info(f"Optimizer started ({self.optimizer_type})")
    
    def stop(self) -> None:
        """Stop the optimizer thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._running = False
        logger.info("Optimizer stopped")
    
    def update_market_data(self, market_data: pd.DataFrame) -> None:
        """Update market data for next optimization."""
        with self._context_lock:
            self._market_data = market_data.copy()
    
    def _optimization_loop(self) -> None:
        """Main optimization loop running in background thread."""
        while not self._stop_event.is_set():
            start = time()
            
            try:
                policy = self._run_optimization()
                if policy is not None:
                    self.policy_queue.publish(policy)
                    self.num_optimizations += 1
                    self.total_optimization_time += policy.optimization_time
            except Exception as e:
                logger.error(f"Optimization error: {e}")
            
            # Wait for next interval
            elapsed = time() - start
            wait_time = max(0, self.update_interval - elapsed)
            self._stop_event.wait(wait_time)
    
    def _run_optimization(self) -> Optional[ExecutionPolicy]:
        """Run single optimization iteration."""
        with self._context_lock:
            order_size = self._current_order_size
            num_slices = self._num_slices
        
        if order_size <= 0:
            return None
        
        start = time()
        
        def optimize_task():
            if self.optimizer_type == 'sa':
                return self._optimize_sa(order_size, num_slices)
            elif self.optimizer_type == 'qaoa':
                return self._optimize_qaoa(order_size, num_slices)
            else:
                return self._optimize_uniform(order_size, num_slices)
        
        def validate(schedule):
            from .optimizer_resilience import validate_schedule
            return validate_schedule(schedule, order_size)
        
        try:
            # Execute with resilience
            schedule = self.resilience.execute(
                optimize_task,
                fallback_func=lambda: self._optimize_fallback(order_size, num_slices),
                validation_func=validate
            )
            
            opt_time = time() - start
            
            return ExecutionPolicy(
                schedule=schedule,
                optimizer_name=self.optimizer_type,
                optimization_time=opt_time
            )
            
        except Exception as e:
            logger.error(f"Optimization failed permanently: {e}")
            return None

    def _optimize_fallback(self, order_size: int, num_slices: int) -> np.ndarray:
        """Fallback optimization (TWAP)."""
        logger.warning("Using fallback execution strategy")
        return self._optimize_uniform(order_size, num_slices)
    
    def _optimize_uniform(self, order_size: int, num_slices: int) -> np.ndarray:
        """Simple uniform distribution (TWAP-like)."""
        base = order_size // num_slices
        remainder = order_size % num_slices
        schedule = np.full(num_slices, base)
        schedule[:remainder] += 1
        return schedule
    
    def _optimize_sa(self, order_size: int, num_slices: int) -> np.ndarray:
        """Optimize using simulated annealing on QUBO."""
        from .qubo_execution import QUBOConfig, ExecutionQUBO
        from .qubo_solvers import SimulatedAnnealingSolver
        
        # Build QUBO (simplified for speed)
        config = QUBOConfig(
            total_shares=order_size,
            num_time_slices=num_slices,
            num_venues=1,
            quantity_levels=[0, order_size // (num_slices * 2), order_size // num_slices],
            equality_penalty=100.0
        )
        
        qubo = ExecutionQUBO(config)
        Q = qubo.build_qubo_matrix()
        
        solver = SimulatedAnnealingSolver(num_sweeps=200, seed=self.seed)
        result = solver.solve(Q, verbose=False)
        
        # Convert solution to schedule
        solution_df = qubo.interpret_solution(result.solution)
        schedule = np.zeros(num_slices)
        for _, row in solution_df.iterrows():
            t = int(row["time_slice"])
            if t < num_slices:
                schedule[t] += row["quantity"]
        
        return schedule
    
    def _optimize_qaoa(self, order_size: int, num_slices: int) -> np.ndarray:
        """Optimize using QAOA (expensive, use sparingly)."""
        # For now, fall back to SA (QAOA too slow for real-time)
        return self._optimize_sa(order_size, num_slices)


# =============================================================================
# Async Execution Engine (Fast Path)
# =============================================================================

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


# =============================================================================
# Hybrid Controller
# =============================================================================

class HybridController:
    """
    Coordinates execution engine and optimizer.
    
    Ensures:
    1. Engine runs fast path without blocking
    2. Optimizer runs slow path in background
    3. Policy updates flow via queue
    4. Clean startup/shutdown
    """
    
    def __init__(
        self,
        optimizer_type: str = 'sa',
        optimizer_interval: float = 0.5,
        engine_tick_interval: float = 0.1,
        seed: Optional[int] = None
    ):
        # Create shared queue
        self.policy_queue = PolicyQueue()
        
        # Create optimizer
        self.optimizer = AsyncOptimizer(
            policy_queue=self.policy_queue,
            optimizer_type=optimizer_type,
            update_interval=optimizer_interval,
            seed=seed
        )
        
        # Create execution engine
        self.engine = AsyncExecutionEngine(
            policy_queue=self.policy_queue,
            tick_interval=engine_tick_interval
        )
        
        self._running = False
    
    def execute_order(
        self,
        total_shares: int,
        num_slices: int,
        duration_seconds: Optional[float] = None
    ) -> Dict:
        """
        Execute order using hybrid async architecture.
        
        Args:
            total_shares: Total shares to execute
            num_slices: Number of execution slices
            duration_seconds: Total execution duration (optional)
            
        Returns:
            Execution summary dict
        """
        logger.info(f"Starting hybrid execution: {total_shares} shares, {num_slices} slices")
        
        start_time = time()
        
        # Create fallback policy (uniform)
        fallback_schedule = np.full(num_slices, total_shares // num_slices)
        remainder = total_shares % num_slices
        fallback_schedule[:remainder] += 1
        
        fallback_policy = ExecutionPolicy(
            schedule=fallback_schedule,
            optimizer_name="fallback_uniform"
        )
        self.engine.set_fallback_policy(fallback_policy)
        
        # Start optimizer (background thread)
        self.optimizer.start(total_shares, num_slices)
        
        # Start execution engine
        self.engine.start(num_slices)
        
        self._running = True
        
        # Wait for execution to complete
        self.engine.wait_complete()
        
        # Stop optimizer
        self.optimizer.stop()
        
        self._running = False
        
        total_time = time() - start_time
        
        # Compile results
        return {
            'total_shares': total_shares,
            'executed_shares': self.engine.executed_shares,
            'fill_rate': self.engine.executed_shares / total_shares,
            'num_slices': num_slices,
            'num_optimizations': self.optimizer.num_optimizations,
            'avg_optimization_time': (
                self.optimizer.total_optimization_time / max(1, self.optimizer.num_optimizations)
            ),
            'total_time': total_time,
            'execution_log': self.engine.execution_log
        }
    
    def get_execution_report(self) -> pd.DataFrame:
        """Get execution log as DataFrame."""
        if not self.engine.execution_log:
            return pd.DataFrame()
        return pd.DataFrame(self.engine.execution_log)


# =============================================================================
# Demo
# =============================================================================

def run_hybrid_demo():
    """Demonstrate hybrid async architecture."""
    
    print("\n" + "="*70)
    print(" Hybrid Async Execution Demo")
    print("="*70)
    print(" Architecture:")
    print("   - Execution Engine (FAST): Runs every 100ms, never waits")
    print("   - Optimizer (SLOW): Runs SA every 500ms in background")
    print("   - Policy Queue: Thread-safe updates")
    
    controller = HybridController(
        optimizer_type='sa',
        optimizer_interval=0.5,  # Optimize every 500ms
        engine_tick_interval=0.1,  # Execute every 100ms
        seed=42
    )
    
    # Execute order
    result = controller.execute_order(
        total_shares=1000,
        num_slices=20
    )
    
    print(f"\n Results:")
    print(f"   Total shares: {result['total_shares']}")
    print(f"   Executed: {result['executed_shares']}")
    print(f"   Fill rate: {result['fill_rate']:.1%}")
    print(f"   Optimizations: {result['num_optimizations']}")
    print(f"   Avg opt time: {result['avg_optimization_time']*1000:.1f}ms")
    print(f"   Total time: {result['total_time']:.2f}s")
    
    # Show execution log
    log_df = controller.get_execution_report()
    if not log_df.empty:
        print(f"\n Execution Log (first 5):")
        print(log_df.head().to_string(index=False))
    
    print("\n" + "="*70)
    print(" Key: Trading NEVER waited for optimizer!")
    print("="*70)
    
    return result


if __name__ == "__main__":
    result = run_hybrid_demo()
