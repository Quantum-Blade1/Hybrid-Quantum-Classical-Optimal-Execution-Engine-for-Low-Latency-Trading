"""
Tests for QUBO Solvers

Verifies brute-force and simulated annealing solvers.
"""

import pytest
import numpy as np

import sys
sys.path.insert(0, str(__file__).replace("\\tests\\test_qubo_solvers.py", ""))

from src.qubo_solvers import (
    QUBOResult,
    BruteForceSolver,
    SimulatedAnnealingSolver,
    GreedySolver,
    compare_solvers
)


def create_simple_qubo(n: int = 5) -> np.ndarray:
    """Create a simple test QUBO matrix."""
    # Diagonal-dominant matrix (easy to solve)
    Q = np.eye(n) * 2  # Prefer 0s
    Q[0, 0] = -5  # Except first variable, prefer 1
    Q[1, 1] = -3  # And second
    return Q


def create_random_qubo(n: int = 10, seed: int = 42) -> np.ndarray:
    """Create a random QUBO matrix."""
    rng = np.random.default_rng(seed)
    Q = rng.normal(0, 1, (n, n))
    Q = (Q + Q.T) / 2  # Make symmetric
    return Q


class TestBruteForceSolver:
    """Tests for BruteForceSolver."""
    
    def test_solves_simple_qubo(self):
        """Test solving a simple QUBO."""
        Q = create_simple_qubo(5)
        solver = BruteForceSolver()
        
        result = solver.solve(Q)
        
        assert isinstance(result, QUBOResult)
        assert result.solver_name == "BruteForce"
        assert result.num_evaluations == 2 ** 5
    
    def test_finds_optimal(self):
        """Test that optimal solution is found."""
        # Create QUBO where x=[1,1,0,0,0] is optimal
        Q = np.diag([1, 1, 2, 2, 2]).astype(float)
        Q[0, 0] = -10  # Strong preference for x[0]=1
        Q[1, 1] = -10  # Strong preference for x[1]=1
        
        solver = BruteForceSolver()
        result = solver.solve(Q)
        
        # Check first two bits are 1
        assert result.solution[0] == 1
        assert result.solution[1] == 1
    
    def test_rejects_large_problems(self):
        """Test that large problems are rejected."""
        Q = np.zeros((30, 30))
        solver = BruteForceSolver(max_variables=25)
        
        with pytest.raises(ValueError):
            solver.solve(Q)
    
    def test_records_time(self):
        """Test that solving time is recorded."""
        Q = create_simple_qubo(8)
        solver = BruteForceSolver()
        
        result = solver.solve(Q)
        
        assert result.solve_time > 0
        assert result.solve_time < 10  # Should be fast for 8 vars


class TestSimulatedAnnealingSolver:
    """Tests for SimulatedAnnealingSolver."""
    
    def test_solves_qubo(self):
        """Test that SA solver runs without error."""
        Q = create_random_qubo(20)
        solver = SimulatedAnnealingSolver(num_sweeps=100, seed=42)
        
        result = solver.solve(Q)
        
        assert isinstance(result, QUBOResult)
        assert result.solver_name == "SimulatedAnnealing"
    
    def test_accepts_initial_solution(self):
        """Test using initial solution."""
        Q = create_random_qubo(10)
        x0 = np.ones(10, dtype=np.int8)
        
        solver = SimulatedAnnealingSolver(num_sweeps=50, seed=42)
        result = solver.solve(Q, initial_solution=x0)
        
        assert result.solution is not None
    
    def test_finds_good_solution(self):
        """Test that SA finds a reasonably good solution."""
        # Simple problem where x=[1,0,0,...] is optimal
        n = 10
        Q = np.eye(n) * 5  # All positive diagonal
        Q[0, 0] = -100  # Except first
        
        solver = SimulatedAnnealingSolver(num_sweeps=500, seed=42)
        result = solver.solve(Q)
        
        # Should find that x[0]=1 is good
        assert result.solution[0] == 1
        assert result.energy < 0  # Negative energy possible
    
    def test_reproducible_with_seed(self):
        """Test that same seed gives same result."""
        Q = create_random_qubo(15)
        
        solver1 = SimulatedAnnealingSolver(num_sweeps=100, seed=42)
        solver2 = SimulatedAnnealingSolver(num_sweeps=100, seed=42)
        
        result1 = solver1.solve(Q)
        result2 = solver2.solve(Q)
        
        assert np.array_equal(result1.solution, result2.solution)
    
    def test_records_history(self):
        """Test that energy history is recorded."""
        Q = create_random_qubo(10)
        solver = SimulatedAnnealingSolver(num_sweeps=100, seed=42)
        
        result = solver.solve(Q)
        
        assert result.history is not None
        assert len(result.history) > 0


class TestGreedySolver:
    """Tests for GreedySolver."""
    
    def test_solves_qubo(self):
        """Test that greedy solver works."""
        Q = create_random_qubo(20)
        solver = GreedySolver(seed=42)
        
        result = solver.solve(Q)
        
        assert isinstance(result, QUBOResult)
        assert result.solver_name == "Greedy"
    
    def test_finds_local_optimum(self):
        """Test that result is locally optimal."""
        Q = create_simple_qubo(8)
        solver = GreedySolver(seed=42)
        
        result = solver.solve(Q)
        
        # Verify no single bit flip improves energy
        for i in range(8):
            x_flip = result.solution.copy()
            x_flip[i] = 1 - x_flip[i]
            flip_energy = float(x_flip @ Q @ x_flip)
            assert flip_energy >= result.energy - 1e-9


class TestSolverComparison:
    """Tests for solver comparison utility."""
    
    def test_compare_solvers(self):
        """Test comparing multiple solvers."""
        Q = create_random_qubo(12)
        
        results = compare_solvers(Q, verbose=False)
        
        assert len(results) > 0
        assert all(isinstance(r, QUBOResult) for r in results)
    
    def test_compare_includes_brute_force_for_small(self):
        """Test that brute force is included for small problems."""
        Q = create_random_qubo(10)
        
        results = compare_solvers(Q, verbose=False)
        
        solver_names = [r.solver_name for r in results]
        assert "BruteForce" in solver_names


class TestWithExecutionQUBO:
    """Test solvers on actual execution QUBO."""
    
    def test_solve_execution_qubo(self):
        """Test solving an execution QUBO with N=8 slices."""
        from src.qubo_execution import QUBOConfig, ExecutionQUBO
        
        config = QUBOConfig(
            total_shares=1000,
            num_time_slices=8,
            num_venues=1,  # Single venue to keep small
            quantity_levels=[0, 100, 200]  # 3 levels
        )
        
        qubo = ExecutionQUBO(config)
        Q = qubo.build_qubo_matrix()
        
        # Test with SA (brute force would be 2^24 = 16M, too slow)
        solver = SimulatedAnnealingSolver(num_sweeps=500, seed=42)
        result = solver.solve(Q)
        
        assert result.energy < float('inf')
        assert len(result.solution) == config.num_variables


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
