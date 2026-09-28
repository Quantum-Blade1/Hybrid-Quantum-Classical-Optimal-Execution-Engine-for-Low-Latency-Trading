"""
Asynchronous optimizer (slow path).

Runs in a background thread, re-solves the execution QUBO at a fixed
interval and publishes the resulting schedule to a PolicyQueue.
"""

import numpy as np
import pandas as pd
from typing import Optional
from time import time
from threading import Thread, Lock, Event
import logging

from qexec.runtime.policy import ExecutionPolicy, PolicyQueue

logger = logging.getLogger(__name__)


class AsyncOptimizer:
    """
    Asynchronous optimizer that runs in a background thread.
    
    Continuously optimizes execution schedule based on market conditions
    and publishes updated policies to the queue.
    """
    
    def __init__(
        self,
        policy_queue: PolicyQueue,
        optimizer_type: str = 'sa',  # 'sa' or 'uniform' (QAOA is offline-only: see qaoa_solver.py)
        update_interval: float = 1.0,  # Seconds between optimizations
        seed: Optional[int] = None
    ):
        if optimizer_type not in ('sa', 'uniform'):
            raise ValueError(
                f"Unsupported optimizer_type {optimizer_type!r}; the async runtime "
                "supports 'sa' or 'uniform'. QAOA is available offline via qaoa_solver.py."
            )
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
        from qexec.runtime.resilience import OptimizerResilience, ResilienceConfig
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
            else:
                return self._optimize_uniform(order_size, num_slices)
        
        def validate(schedule):
            from qexec.runtime.resilience import validate_schedule
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
        from qexec.optimization.qubo import QUBOConfig, ExecutionQUBO
        from qexec.optimization.solvers.annealing import SimulatedAnnealingSolver
        
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
