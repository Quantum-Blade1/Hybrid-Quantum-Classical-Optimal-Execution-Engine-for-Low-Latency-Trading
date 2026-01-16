"""
Tests for QUBO Execution Formulation

Verifies QUBO construction, solution interpretation, and validation.
"""

import pytest
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(__file__).replace("\\tests\\test_qubo_execution.py", ""))

from src.qubo_execution import (
    QUBOConfig,
    ExecutionQUBO,
    create_random_binary_solution,
    create_uniform_solution
)


class TestQUBOConfig:
    """Tests for QUBOConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = QUBOConfig()
        
        assert config.total_shares == 10_000
        assert config.num_time_slices == 10
        assert config.num_venues == 2
        assert config.num_variables == 10 * 2 * 5  # T * V * K
    
    def test_variable_indexing(self):
        """Test variable index conversion."""
        config = QUBOConfig(num_time_slices=3, num_venues=2, 
                           quantity_levels=[0, 100, 200])
        
        # Index calculation: t * V * K + v * K + k
        assert config.variable_index(0, 0, 0) == 0
        assert config.variable_index(0, 0, 2) == 2
        assert config.variable_index(0, 1, 0) == 3
        assert config.variable_index(1, 0, 0) == 6
    
    def test_decode_index(self):
        """Test decoding flat index to (t, v, k)."""
        config = QUBOConfig(num_time_slices=3, num_venues=2,
                           quantity_levels=[0, 100, 200])
        
        # Check round-trip
        for t in range(config.num_time_slices):
            for v in range(config.num_venues):
                for k in range(config.num_quantity_levels):
                    i = config.variable_index(t, v, k)
                    t2, v2, k2 = config.decode_index(i)
                    assert (t, v, k) == (t2, v2, k2)


class TestExecutionQUBO:
    """Tests for ExecutionQUBO builder."""
    
    @pytest.fixture
    def qubo(self):
        """Create a small QUBO instance for testing."""
        config = QUBOConfig(
            total_shares=1000,
            num_time_slices=5,
            num_venues=2,
            quantity_levels=[0, 100, 200, 300]
        )
        return ExecutionQUBO(config)
    
    def test_build_qubo_matrix(self, qubo):
        """Test that Q matrix is built correctly."""
        Q = qubo.build_qubo_matrix()
        
        n = qubo.config.num_variables
        assert Q.shape == (n, n)
        assert np.allclose(Q, Q.T)  # Symmetric
    
    def test_matrix_has_nonzero_entries(self, qubo):
        """Test that Q matrix has meaningful entries."""
        Q = qubo.build_qubo_matrix()
        
        assert np.count_nonzero(Q) > 0
        assert np.abs(Q).max() > 0
    
    def test_interpret_solution(self, qubo):
        """Test solution interpretation."""
        qubo.build_qubo_matrix()
        
        # Create a simple solution
        x = create_uniform_solution(qubo.config)
        schedule = qubo.interpret_solution(x)
        
        assert isinstance(schedule, pd.DataFrame)
        assert "time_slice" in schedule.columns
        assert "quantity" in schedule.columns
    
    def test_calculate_solution_cost(self, qubo):
        """Test cost calculation."""
        qubo.build_qubo_matrix()
        
        x = create_random_binary_solution(qubo.config, seed=42)
        costs = qubo.calculate_solution_cost(x)
        
        assert "total_cost" in costs
        assert "impact_cost" in costs
        assert "timing_cost" in costs
        assert "transaction_cost" in costs
    
    def test_validate_solution(self, qubo):
        """Test solution validation."""
        qubo.build_qubo_matrix()
        
        x = create_uniform_solution(qubo.config)
        validation = qubo.validate_solution(x)
        
        assert "total_shares_satisfied" in validation
        assert "capacity_satisfied" in validation
        assert "binary_satisfied" in validation
    
    def test_get_qubo_dict(self, qubo):
        """Test dictionary conversion."""
        qubo.build_qubo_matrix()
        
        qubo_dict = qubo.get_qubo_dict()
        
        assert isinstance(qubo_dict, dict)
        assert all(isinstance(k, tuple) and len(k) == 2 for k in qubo_dict.keys())
    
    def test_linear_quadratic_separation(self, qubo):
        """Test separation of linear and quadratic terms."""
        Q = qubo.build_qubo_matrix()
        
        linear, quadratic = qubo.get_linear_and_quadratic()
        
        assert linear.shape == (qubo.config.num_variables,)
        assert quadratic.shape == Q.shape
        assert np.allclose(np.diag(quadratic), 0)  # No diagonal


class TestHelperFunctions:
    """Tests for helper functions."""
    
    def test_random_solution_shape(self):
        """Test random solution has correct shape."""
        config = QUBOConfig(num_time_slices=5, num_venues=2)
        x = create_random_binary_solution(config, seed=42)
        
        assert len(x) == config.num_variables
        assert all(val in [0, 1] for val in x)
    
    def test_uniform_solution_shape(self):
        """Test uniform solution has correct shape."""
        config = QUBOConfig(num_time_slices=5, num_venues=2)
        x = create_uniform_solution(config)
        
        assert len(x) == config.num_variables
        assert all(val in [0, 1] for val in x)
    
    def test_reproducibility(self):
        """Test that same seed gives same result."""
        config = QUBOConfig()
        
        x1 = create_random_binary_solution(config, seed=42)
        x2 = create_random_binary_solution(config, seed=42)
        
        assert np.array_equal(x1, x2)


class TestConstraints:
    """Tests for constraint formulation."""
    
    def test_equality_constraint_satisfiable(self):
        """Test that target shares can theoretically be achieved."""
        config = QUBOConfig(
            total_shares=1000,
            num_time_slices=10,
            quantity_levels=[0, 50, 100, 150, 200]
        )
        
        # 10 slices * 100 shares = 1000 (exact match)
        # So a valid solution should exist
        qubo = ExecutionQUBO(config)
        Q = qubo.build_qubo_matrix()
        
        # Create a perfectly balanced solution
        x = np.zeros(config.num_variables)
        for t in range(config.num_time_slices):
            # Choose 100 shares at venue 0
            k = config.quantity_levels.index(100)
            i = config.variable_index(t, 0, k)
            x[i] = 1
        
        validation = qubo.validate_solution(x)
        assert validation["total_shares_satisfied"]
    
    def test_capacity_violation_penalized(self):
        """Test that capacity violations increase cost."""
        config = QUBOConfig(
            total_shares=1000,
            num_time_slices=5,
            quantity_levels=[0, 500, 1000],
            max_shares_per_slice=500
        )
        
        qubo = ExecutionQUBO(config)
        Q = qubo.build_qubo_matrix()
        
        # Solution within capacity
        x_good = np.zeros(config.num_variables)
        for t in range(5):
            k = 1  # 500 shares
            i = config.variable_index(t, 0, k)
            x_good[i] = 1
        
        # Solution violating capacity (executing at both venues)
        x_bad = x_good.copy()
        i2 = config.variable_index(0, 1, 1)  # Add venue 1 at time 0
        x_bad[i2] = 1  # Now time 0 has 1000 shares, exceeds 500 limit
        
        cost_good = float(x_good @ Q @ x_good)
        cost_bad = float(x_bad @ Q @ x_bad)
        
        # Bad solution should have higher cost due to penalty
        assert cost_bad > cost_good


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
