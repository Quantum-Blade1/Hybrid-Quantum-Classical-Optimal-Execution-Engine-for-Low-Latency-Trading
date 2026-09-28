"""
Hybrid controller: wires the optimizer (slow path) and execution engine
(fast path) together through a shared PolicyQueue.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from time import time
import logging

from qexec.runtime.policy import ExecutionPolicy, PolicyQueue
from qexec.runtime.optimizer import AsyncOptimizer
from qexec.runtime.engine import AsyncExecutionEngine

logger = logging.getLogger(__name__)


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
