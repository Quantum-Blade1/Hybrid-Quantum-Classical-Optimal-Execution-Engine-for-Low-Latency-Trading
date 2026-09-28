"""
QUBO Formulation for Optimal Execution

This module formulates the optimal trade execution problem as a QUBO
(Quadratic Unconstrained Binary Optimization) problem, which can be
solved on quantum computers or quantum-inspired classical solvers.

============================================================================
PROBLEM DESCRIPTION
============================================================================

Given a large parent order of S shares, we want to find the optimal way
to split it into time slices executed at different venues to minimize:

    Total Cost = Market Impact + Timing Risk + Transaction Costs

Subject to:
    - Total shares executed = S (order completion constraint)
    - Maximum shares per time slice (capacity constraint)  
    - Inventory limits at each time step

============================================================================
QUBO FORMULATION
============================================================================

Decision Variables:
    x_{i,j,k} ∈ {0, 1}
    
    Where:
    - i = time slice index (0 to T-1)
    - j = venue index (0 to V-1)  
    - k = quantity level (representing q_k shares)
    
    x_{i,j,k} = 1 means: execute q_k shares at time i via venue j

We use binary encoding of quantities to keep the problem size manageable.
For example, with K=4 quantity levels: {500, 1000, 1500, 2000} shares.

Objective Function (to minimize):

    min  x^T Q x  +  c^T x
    
    where Q captures quadratic terms (market impact, timing correlation)
    and c captures linear terms (transaction costs, venue fees)

Constraint Penalties (added to objective):
    
    For equality constraint ∑x = S, we add penalty:
        P * (∑ q_k * x_{i,j,k} - S)^2
    
    where P is a large penalty coefficient.

============================================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import pandas as pd


# =============================================================================
# Configuration and Parameters
# =============================================================================

@dataclass
class QUBOConfig:
    """
    Configuration for the execution QUBO formulation.
    
    Attributes:
        total_shares: Total order size to execute
        num_time_slices: Number of time periods (T)
        num_venues: Number of execution venues (V)
        quantity_levels: Discrete quantity options per slice
        
        # Cost weights (tunable parameters)
        impact_weight: Weight for market impact cost
        timing_weight: Weight for timing risk
        transaction_weight: Weight for transaction costs
        
        # Market parameters
        volatility: Price volatility (affects timing risk)
        avg_spread_bps: Average bid-ask spread in basis points
        impact_coefficient: Market impact scaling factor
        
        # Penalty coefficients
        equality_penalty: Penalty for violating total shares constraint
        capacity_penalty: Penalty for exceeding slice capacity
    """
    # Problem dimensions
    total_shares: int = 10_000
    num_time_slices: int = 10
    num_venues: int = 2
    quantity_levels: List[int] = field(default_factory=lambda: [0, 500, 1000, 1500, 2000])
    
    # Cost weights (normalized to sum to 1)
    impact_weight: float = 0.4
    timing_weight: float = 0.3
    transaction_weight: float = 0.3
    
    # Market parameters
    volatility: float = 0.02  # 2% daily volatility
    avg_spread_bps: float = 5.0  # 5 basis points spread
    impact_coefficient: float = 0.1  # Almgren-Chriss impact factor
    
    # Penalty coefficients (should be >> cost terms)
    equality_penalty: float = 1000.0
    capacity_penalty: float = 500.0
    max_shares_per_slice: int = 3000
    
    @property
    def num_quantity_levels(self) -> int:
        """Number of quantity options (K)."""
        return len(self.quantity_levels)
    
    @property
    def num_variables(self) -> int:
        """Total number of binary variables: T * V * K."""
        return self.num_time_slices * self.num_venues * self.num_quantity_levels
    
    def variable_index(self, time: int, venue: int, qty_level: int) -> int:
        """
        Convert (time, venue, quantity_level) to flat variable index.
        
        Indexing: x[t * V * K + v * K + k]
        """
        K = self.num_quantity_levels
        V = self.num_venues
        return time * V * K + venue * K + qty_level
    
    def decode_index(self, idx: int) -> Tuple[int, int, int]:
        """
        Convert flat index back to (time, venue, quantity_level).
        
        Returns:
            Tuple of (time_slice, venue, quantity_level)
        """
        K = self.num_quantity_levels
        V = self.num_venues
        
        time = idx // (V * K)
        remainder = idx % (V * K)
        venue = remainder // K
        qty_level = remainder % K
        
        return time, venue, qty_level


# =============================================================================
# QUBO Builder
# =============================================================================

class ExecutionQUBO:
    """
    Builds the QUBO matrix for optimal execution.
    
    The QUBO form is: minimize x^T Q x
    where x is a binary vector and Q is a symmetric matrix.
    
    Q contains:
    - Diagonal: linear costs (transaction fees, spread costs)
    - Off-diagonal: quadratic costs (market impact, timing correlations)
    - Constraint penalties embedded in both
    
    Example:
        >>> config = QUBOConfig(total_shares=10000, num_time_slices=10)
        >>> qubo = ExecutionQUBO(config)
        >>> Q = qubo.build_qubo_matrix()
        >>> solution = some_solver.solve(Q)
        >>> shares_schedule = qubo.interpret_solution(solution)
    """
    
    def __init__(self, config: QUBOConfig):
        """
        Initialize the QUBO builder.
        
        Args:
            config: QUBO configuration parameters
        """
        self.config = config
        self.Q: Optional[np.ndarray] = None
        
        # Store individual cost matrices for analysis
        self._impact_matrix: Optional[np.ndarray] = None
        self._timing_matrix: Optional[np.ndarray] = None
        self._transaction_matrix: Optional[np.ndarray] = None
        self._constraint_matrix: Optional[np.ndarray] = None
    
    def build_qubo_matrix(self) -> np.ndarray:
        """
        Build the complete QUBO matrix Q.
        
        The matrix combines:
        1. Market impact costs (quadratic in quantity)
        2. Timing risk (correlation between time slices)
        3. Transaction costs (linear, on diagonal)
        4. Constraint penalties (quadratic)
        
        Returns:
            Q matrix of shape (num_variables, num_variables)
        """
        n = self.config.num_variables
        
        # Initialize component matrices
        self._impact_matrix = np.zeros((n, n))
        self._timing_matrix = np.zeros((n, n))
        self._transaction_matrix = np.zeros((n, n))
        self._constraint_matrix = np.zeros((n, n))
        
        # Build each component
        self._add_market_impact_cost()
        self._add_timing_risk_cost()
        self._add_transaction_cost()
        self._add_equality_constraint()
        self._add_capacity_constraint()
        
        # Combine with weights
        self.Q = (
            self.config.impact_weight * self._impact_matrix +
            self.config.timing_weight * self._timing_matrix +
            self.config.transaction_weight * self._transaction_matrix +
            self._constraint_matrix  # Penalties not weighted
        )
        
        # Make symmetric (QUBO requirement)
        self.Q = (self.Q + self.Q.T) / 2
        
        return self.Q
    
    def _add_market_impact_cost(self) -> None:
        """
        Add market impact cost to the QUBO matrix.
        
        Market impact is modeled using the Almgren-Chriss framework:
            Impact(q) = η * σ * (q / V)^α
        
        where:
            η = impact coefficient
            σ = volatility
            q = quantity executed
            V = available volume
            α ≈ 0.5-1.0 (typically square root or linear)
        
        For QUBO, we linearize: Impact ≈ η * σ * q
        
        The quadratic term captures self-impact: executing more
        in the same time slice increases impact non-linearly.
        """
        cfg = self.config
        
        # Impact coefficient per share
        impact_per_share = cfg.impact_coefficient * cfg.volatility
        
        for t in range(cfg.num_time_slices):
            for v in range(cfg.num_venues):
                for k in range(cfg.num_quantity_levels):
                    i = cfg.variable_index(t, v, k)
                    q_i = cfg.quantity_levels[k]
                    
                    # Linear impact cost (diagonal)
                    self._impact_matrix[i, i] += impact_per_share * q_i
                    
                    # Quadratic impact: executing at same time increases cost
                    # Cross-impact between different venues at same time
                    for v2 in range(cfg.num_venues):
                        if v2 != v:
                            for k2 in range(cfg.num_quantity_levels):
                                j = cfg.variable_index(t, v2, k2)
                                q_j = cfg.quantity_levels[k2]
                                
                                # Same-time cross impact (smaller than self-impact)
                                cross_impact = 0.3 * impact_per_share * (q_i * q_j) ** 0.5
                                self._impact_matrix[i, j] += cross_impact
    
    def _add_timing_risk_cost(self) -> None:
        """
        Add timing risk cost to the QUBO matrix.
        
        Timing risk captures the uncertainty in execution price due to
        price movements between time slices:
        
            Timing Risk = σ^2 * Var(execution schedule)
        
        For QUBO, we penalize concentrated execution (high variance)
        by adding costs for executing large quantities in single slices.
        
        We also add correlations between adjacent time slices to 
        encourage spreading execution across time.
        """
        cfg = self.config
        
        # Time-based risk factor (increases with time in market)
        for t in range(cfg.num_time_slices):
            # Risk increases with time (later = more accumulated vol)
            time_risk_factor = cfg.volatility * np.sqrt((t + 1) / cfg.num_time_slices)
            
            for v in range(cfg.num_venues):
                for k in range(cfg.num_quantity_levels):
                    i = cfg.variable_index(t, v, k)
                    q_i = cfg.quantity_levels[k]
                    
                    # Variance penalty: penalize large single-slice execution
                    # (encourages spreading across time)
                    self._timing_matrix[i, i] += time_risk_factor * (q_i ** 2) / cfg.total_shares
        
        # Add correlation between adjacent time slices
        # (encourage smooth execution profile)
        for t in range(cfg.num_time_slices - 1):
            for v in range(cfg.num_venues):
                for k in range(cfg.num_quantity_levels):
                    i = cfg.variable_index(t, v, k)
                    q_i = cfg.quantity_levels[k]
                    
                    # Negative correlation with next time slice
                    # (slightly favor alternating quantities)
                    for v2 in range(cfg.num_venues):
                        for k2 in range(cfg.num_quantity_levels):
                            j = cfg.variable_index(t + 1, v2, k2)
                            q_j = cfg.quantity_levels[k2]
                            
                            # Small negative term for adjacent times
                            smoothness = -0.01 * cfg.volatility * min(q_i, q_j)
                            self._timing_matrix[i, j] += smoothness
    
    def _add_transaction_cost(self) -> None:
        """
        Add transaction costs to the QUBO matrix.
        
        Transaction costs include:
        1. Bid-ask spread cost (pay half spread per share)
        2. Venue-specific fees (may vary by venue)
        3. Execution fees
        
        These are linear costs, so only diagonal elements.
        """
        cfg = self.config
        
        # Base spread cost per share (half the spread)
        spread_cost = (cfg.avg_spread_bps / 10000) / 2
        
        # Venue-specific fee multipliers (venue 0 = primary, venue 1 = dark pool)
        venue_fees = [1.0, 0.8]  # Dark pool slightly cheaper
        
        for t in range(cfg.num_time_slices):
            for v in range(cfg.num_venues):
                venue_multiplier = venue_fees[v] if v < len(venue_fees) else 1.0
                
                for k in range(cfg.num_quantity_levels):
                    i = cfg.variable_index(t, v, k)
                    q = cfg.quantity_levels[k]
                    
                    # Total transaction cost for this slice
                    cost = spread_cost * q * venue_multiplier
                    self._transaction_matrix[i, i] += cost
    
    def _add_equality_constraint(self) -> None:
        """
        Add penalty for violating total shares constraint.
        
        We require: ∑_{t,v,k} q_k * x_{t,v,k} = S (total_shares)
        
        QUBO formulation of equality constraint:
            P * (∑ q_k * x_{t,v,k} - S)^2
        
        Expanding:
            P * (∑∑ q_i*q_j*x_i*x_j - 2*S*∑q_k*x_k + S^2)
        
        The S^2 term is constant (ignored in optimization).
        
        This adds:
        - Quadratic penalties for all pairs of variables
        - Negative linear terms (-2*P*S*q_k) to push toward target
        """
        cfg = self.config
        P = cfg.equality_penalty
        S = cfg.total_shares
        
        # Build coefficient for each variable
        for i in range(cfg.num_variables):
            t_i, v_i, k_i = cfg.decode_index(i)
            q_i = cfg.quantity_levels[k_i]
            
            # Linear term: -2 * P * S * q_i (promotes selecting this variable)
            # This goes on diagonal in QUBO
            self._constraint_matrix[i, i] += P * q_i * q_i - 2 * P * S * q_i
            
            # Quadratic term: P * q_i * q_j for all pairs
            for j in range(i + 1, cfg.num_variables):
                t_j, v_j, k_j = cfg.decode_index(j)
                q_j = cfg.quantity_levels[k_j]
                
                # Add penalty for both variables being 1
                self._constraint_matrix[i, j] += 2 * P * q_i * q_j
    
    def _add_capacity_constraint(self) -> None:
        """
        Add penalty for exceeding maximum shares per time slice.
        
        Constraint: ∑_{v,k} q_k * x_{t,v,k} ≤ M for each time t
        
        For inequality constraints, we use slack variables or 
        approximate with soft penalty. Here we use soft penalty:
        
            P * max(0, ∑ q_k * x_{t,v,k} - M)^2
        
        Simplified: we add large penalty for combinations that exceed M.
        """
        cfg = self.config
        P = cfg.capacity_penalty
        M = cfg.max_shares_per_slice
        
        # For each time slice, penalize if total quantity exceeds M
        for t in range(cfg.num_time_slices):
            # Get all variable indices for this time slice
            vars_at_t = []
            for v in range(cfg.num_venues):
                for k in range(cfg.num_quantity_levels):
                    i = cfg.variable_index(t, v, k)
                    q = cfg.quantity_levels[k]
                    vars_at_t.append((i, q))
            
            # Add quadratic penalty for pairs that would exceed capacity
            for idx1, (i, q_i) in enumerate(vars_at_t):
                for idx2, (j, q_j) in enumerate(vars_at_t):
                    if idx2 > idx1:
                        # If both selected and sum > M, penalize
                        if q_i + q_j > M:
                            excess = (q_i + q_j - M)
                            self._constraint_matrix[i, j] += P * excess
    
    # =========================================================================
    # Solution Interpretation
    # =========================================================================
    
    def interpret_solution(self, x: np.ndarray) -> pd.DataFrame:
        """
        Convert binary solution vector to execution schedule.
        
        Args:
            x: Binary solution vector of length num_variables
            
        Returns:
            DataFrame with columns: time, venue, quantity
        """
        cfg = self.config
        schedule = []
        
        for i, val in enumerate(x):
            if val > 0.5:  # Binary 1
                t, v, k = cfg.decode_index(i)
                q = cfg.quantity_levels[k]
                if q > 0:
                    schedule.append({
                        "time_slice": t,
                        "venue": v,
                        "venue_name": self._venue_name(v),
                        "quantity": q
                    })
        
        df = pd.DataFrame(schedule)
        if len(df) > 0:
            df = df.sort_values("time_slice").reset_index(drop=True)
        
        return df
    
    def _venue_name(self, v: int) -> str:
        """Map venue index to name."""
        names = ["Primary", "Dark Pool", "ECN", "Venue 3"]
        return names[v] if v < len(names) else f"Venue {v}"
    
    def calculate_solution_cost(self, x: np.ndarray) -> Dict[str, float]:
        """
        Calculate the cost breakdown for a solution.
        
        Args:
            x: Binary solution vector
            
        Returns:
            Dictionary with cost components
        """
        if self.Q is None:
            self.build_qubo_matrix()
        
        # Total QUBO cost
        total_cost = float(x @ self.Q @ x)
        
        # Individual components
        impact_cost = float(x @ self._impact_matrix @ x) * self.config.impact_weight
        timing_cost = float(x @ self._timing_matrix @ x) * self.config.timing_weight
        transaction_cost = float(x @ self._transaction_matrix @ x) * self.config.transaction_weight
        constraint_penalty = float(x @ self._constraint_matrix @ x)
        
        # Total shares executed
        total_shares = sum(
            self.config.quantity_levels[self.config.decode_index(i)[2]]
            for i, val in enumerate(x) if val > 0.5
        )
        
        return {
            "total_cost": total_cost,
            "impact_cost": impact_cost,
            "timing_cost": timing_cost,
            "transaction_cost": transaction_cost,
            "constraint_penalty": constraint_penalty,
            "total_shares": total_shares,
            "target_shares": self.config.total_shares,
            "shares_difference": total_shares - self.config.total_shares
        }
    
    def validate_solution(self, x: np.ndarray) -> Dict[str, bool]:
        """
        Validate that a solution satisfies all constraints.
        
        Args:
            x: Binary solution vector
            
        Returns:
            Dictionary with constraint satisfaction status
        """
        cfg = self.config
        
        # Check total shares
        total_shares = sum(
            cfg.quantity_levels[cfg.decode_index(i)[2]]
            for i, val in enumerate(x) if val > 0.5
        )
        shares_ok = abs(total_shares - cfg.total_shares) < 0.01 * cfg.total_shares
        
        # Check capacity per time slice
        capacity_ok = True
        for t in range(cfg.num_time_slices):
            slice_total = 0
            for v in range(cfg.num_venues):
                for k in range(cfg.num_quantity_levels):
                    i = cfg.variable_index(t, v, k)
                    if x[i] > 0.5:
                        slice_total += cfg.quantity_levels[k]
            if slice_total > cfg.max_shares_per_slice:
                capacity_ok = False
                break
        
        # Check binary
        binary_ok = all(val in [0, 1] or abs(val - round(val)) < 0.1 for val in x)
        
        return {
            "total_shares_satisfied": shares_ok,
            "capacity_satisfied": capacity_ok,
            "binary_satisfied": binary_ok,
            "all_satisfied": shares_ok and capacity_ok and binary_ok
        }
    
    def get_qubo_dict(self) -> Dict[Tuple[int, int], float]:
        """
        Convert Q matrix to dictionary format used by some solvers.
        
        Returns:
            Dictionary mapping (i, j) -> Q[i,j] for non-zero entries
        """
        if self.Q is None:
            self.build_qubo_matrix()
        
        qubo_dict = {}
        n = self.config.num_variables
        
        for i in range(n):
            for j in range(i, n):
                if abs(self.Q[i, j]) > 1e-10:
                    qubo_dict[(i, j)] = float(self.Q[i, j])
        
        return qubo_dict
    
    def get_linear_and_quadratic(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract linear (diagonal) and quadratic (off-diagonal) terms.
        
        Useful for some solver formats.
        
        Returns:
            Tuple of (linear_coefficients, quadratic_matrix)
        """
        if self.Q is None:
            self.build_qubo_matrix()
        
        linear = np.diag(self.Q).copy()
        quadratic = self.Q.copy()
        np.fill_diagonal(quadratic, 0)
        
        return linear, quadratic


# =============================================================================
# Helper Functions
# =============================================================================

def create_random_binary_solution(config: QUBOConfig, seed: int = None) -> np.ndarray:
    """
    Create a random binary solution (for testing).
    
    Args:
        config: QUBO configuration
        seed: Random seed
        
    Returns:
        Random binary vector
    """
    rng = np.random.default_rng(seed)
    x = np.zeros(config.num_variables)
    
    # For each time slice, randomly select one venue and quantity
    for t in range(config.num_time_slices):
        v = rng.integers(0, config.num_venues)
        k = rng.integers(0, config.num_quantity_levels)
        i = config.variable_index(t, v, k)
        x[i] = 1
    
    return x


def create_uniform_solution(config: QUBOConfig) -> np.ndarray:
    """
    Create a uniform TWAP-like solution (for baseline comparison).
    
    Distributes shares equally across time slices.
    
    Args:
        config: QUBO configuration
        
    Returns:
        Binary vector with uniform distribution
    """
    x = np.zeros(config.num_variables)
    
    shares_per_slice = config.total_shares // config.num_time_slices
    
    # Find best quantity level for uniform distribution
    best_k = 0
    best_diff = float('inf')
    for k, q in enumerate(config.quantity_levels):
        if abs(q - shares_per_slice) < best_diff:
            best_diff = abs(q - shares_per_slice)
            best_k = k
    
    # Assign to first venue at each time
    for t in range(config.num_time_slices):
        i = config.variable_index(t, 0, best_k)
        x[i] = 1
    
    return x


def print_qubo_summary(qubo: ExecutionQUBO) -> None:
    """Print summary of QUBO formulation."""
    cfg = qubo.config
    Q = qubo.build_qubo_matrix()
    
    print("\n" + "=" * 60)
    print(" QUBO Formulation Summary")
    print("=" * 60)
    print(f"\nProblem Dimensions:")
    print(f"  Total shares to execute: {cfg.total_shares:,}")
    print(f"  Time slices (T):         {cfg.num_time_slices}")
    print(f"  Venues (V):              {cfg.num_venues}")
    print(f"  Quantity levels (K):     {cfg.num_quantity_levels}")
    print(f"  Binary variables:        {cfg.num_variables}")
    print(f"  Quantity options:        {cfg.quantity_levels}")
    
    print(f"\nCost Weights:")
    print(f"  Market impact:   {cfg.impact_weight:.1%}")
    print(f"  Timing risk:     {cfg.timing_weight:.1%}")
    print(f"  Transaction:     {cfg.transaction_weight:.1%}")
    
    print(f"\nQ Matrix Statistics:")
    print(f"  Shape:           {Q.shape}")
    print(f"  Non-zero:        {np.count_nonzero(Q)}")
    print(f"  Density:         {np.count_nonzero(Q) / Q.size:.1%}")
    print(f"  Min value:       {Q.min():.4f}")
    print(f"  Max value:       {Q.max():.4f}")
    print(f"  Diagonal sum:    {np.trace(Q):.4f}")
