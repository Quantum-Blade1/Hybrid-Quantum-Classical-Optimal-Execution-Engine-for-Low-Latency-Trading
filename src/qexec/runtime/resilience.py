"""
Optimizer Resilience Module

Provides robust error handling for quantum optimization:
1. Timeout protection
2. Retry logic with exponential backoff
3. Solution validation
4. Classical fallback

Usage:
    resilience = OptimizerResilience(timeout=5.0, max_retries=3)
    result = resilience.execute(optimizer_func, args...)
"""

import time
import logging
import numpy as np
from typing import Callable, Optional, Any, Tuple, List
from dataclasses import dataclass
from threading import Thread
import queue

logger = logging.getLogger(__name__)

class OptimizerTimeoutError(Exception):
    pass

class InvalidSolutionError(Exception):
    pass

@dataclass
class ResilienceConfig:
    timeout_seconds: float = 5.0
    max_retries: int = 3
    base_backoff_seconds: float = 0.5
    validate_solution: bool = True

class OptimizerResilience:
    """Manages resilient execution of optimization functions."""
    
    def __init__(self, config: Optional[ResilienceConfig] = None):
        self.config = config or ResilienceConfig()
    
    def execute(
        self,
        func: Callable,
        *args,
        fallback_func: Optional[Callable] = None,
        validation_func: Optional[Callable[[np.ndarray], bool]] = None,
        **kwargs
    ) -> Any:
        """
        Execute function with resilience logic.
        
        Args:
            func: Primary optimization function
            fallback_func: Function to call if primary fails/times out
            validation_func: Function to validate result
            
        Returns:
            Result from func or fallback_func
        """
        attempt = 0
        last_error = None
        
        while attempt <= self.config.max_retries:
            try:
                # Calculate backoff
                if attempt > 0:
                    backoff = self.config.base_backoff_seconds * (2 ** (attempt - 1))
                    logger.warning(f"Retry {attempt}/{self.config.max_retries} after {backoff:.1f}s backoff")
                    time.sleep(backoff)
                
                # Execute with timeout mechanism
                result = self._run_with_timeout(func, *args, **kwargs)
                
                # Validate
                if self.config.validate_solution and validation_func:
                    if not validation_func(result):
                        raise InvalidSolutionError("Solution failed validation")
                
                return result
                
            except Exception as e:
                last_error = e
                logger.error(f"Attempt {attempt + 1} failed: {type(e).__name__}: {e}")
                attempt += 1
        
        # All retries failed
        logger.error(f"All retries failed. Last error: {last_error}")
        
        if fallback_func:
            logger.info("Engaging fallback mechanism...")
            try:
                return fallback_func(*args, **kwargs)
            except Exception as fe:
                logger.error(f"Fallback failed: {fe}")
                raise fe
        
        raise last_error
    
    def _run_with_timeout(self, func: Callable, *args, **kwargs) -> Any:
        """Run function in separate thread to enforce timeout."""
        result_queue = queue.Queue()
        
        def wrapper():
            try:
                out = func(*args, **kwargs)
                result_queue.put(('success', out))
            except Exception as e:
                result_queue.put(('error', e))
        
        t = Thread(target=wrapper, daemon=True)
        t.start()
        
        try:
            status, value = result_queue.get(timeout=self.config.timeout_seconds)
            if status == 'error':
                raise value
            return value
        except queue.Empty:
            # Thread is still running but we ignore it
            raise OptimizerTimeoutError(f"Optimization timed out after {self.config.timeout_seconds}s")

# =============================================================================
# Validation Utils
# =============================================================================

def validate_schedule(schedule: np.ndarray, total_shares: int, verbose: bool = False) -> bool:
    """Validate that schedule conforms to constraints."""
    
    # Check 1: Output is numpy array
    if not isinstance(schedule, np.ndarray):
        if verbose: print("Fail: Not a numpy array")
        return False
        
    # Check 2: No negative values
    if np.any(schedule < 0):
        if verbose: print("Fail: Negative values")
        return False
        
    # Check 3: Sum constraint (within 1% tolerance for floating point)
    current_sum = np.sum(schedule)
    tolerance = max(1, total_shares * 0.01)
    if abs(current_sum - total_shares) > tolerance:
        if verbose: print(f"Fail: Sum {current_sum} != {total_shares}")
        return False
        
    return True
