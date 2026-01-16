
"""
Test Failure Modes for Optimizer Resilience
"""

import time
import unittest
import numpy as np
from src.optimizer_resilience import (
    OptimizerResilience, ResilienceConfig, 
    OptimizerTimeoutError, InvalidSolutionError,
    validate_schedule
)

class TestOptimizerResilience(unittest.TestCase):
    
    def setUp(self):
        # Fast config for testing
        self.config = ResilienceConfig(
            timeout_seconds=1.0,
            max_retries=2,
            base_backoff_seconds=0.1,
            validate_solution=True
        )
        self.resilience = OptimizerResilience(self.config)
        
    def test_successful_execution(self):
        """Test normal successful execution."""
        def success_func(x):
            return np.array([x, x])
        
        result = self.resilience.execute(
            success_func, 5,
            validation_func=lambda res: np.sum(res) == 10
        )
        self.assertTrue(np.array_equal(result, np.array([5, 5])))
        
    def test_timeout_fallback(self):
        """Test that timeout triggers fallback."""
        def slow_func():
            time.sleep(2.0)
            return "Too late"
            
        def fallback_func():
            return "Fallback Active"
            
        result = self.resilience.execute(
            slow_func,
            fallback_func=fallback_func
        )
        self.assertEqual(result, "Fallback Active")
        
    def test_retry_success(self):
        """Test that it retries and eventually succeeds."""
        self.attempts = 0
        
        def flaky_func():
            self.attempts += 1
            if self.attempts < 2:
                raise ValueError("Random failure")
            return "Success"
            
        result = self.resilience.execute(flaky_func)
        self.assertEqual(result, "Success")
        self.assertEqual(self.attempts, 2)
        
    def test_validation_failure(self):
        """Test validation logic."""
        def invalid_func():
            return np.array([100]) # Wrong sum
            
        def fallback_func():
            return np.array([50]) # Valid backup
            
        def validator(res):
            return np.sum(res) == 50
            
        result = self.resilience.execute(
            invalid_func,
            fallback_func=fallback_func,
            validation_func=validator
        )
        self.assertEqual(result[0], 50)
        
    def test_max_retries_exceeded(self):
        """Test that it fails if retries exhausted and no fallback."""
        def fail_func():
            raise ValueError("Always fails")
            
        with self.assertRaises(ValueError):
            self.resilience.execute(fail_func)

    def test_validate_schedule_helper(self):
        """Test the helper function."""
        valid_schedule = np.array([50, 50])
        self.assertTrue(validate_schedule(valid_schedule, 100))
        
        invalid_sum = np.array([50, 40])
        self.assertFalse(validate_schedule(invalid_sum, 100))
        
        negative = np.array([110, -10])
        self.assertFalse(validate_schedule(negative, 100))

if __name__ == '__main__':
    unittest.main()
