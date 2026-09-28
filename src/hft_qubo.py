"""
HFT-Specific QUBO Formulation

Extends the base ExecutionQUBO with market microstructure costs that
are critical for high-frequency trading:

1. Adverse selection cost (from VPIN/Kyle's lambda)
2. Queue position opportunity cost
3. Venue toxicity routing penalty
4. Inventory risk at tick-level granularity
5. Cross-venue information leakage cost

The standard Almgren-Chriss QUBO (qubo_execution.py) optimizes over
minute-level time slices. This module operates at tick-level (sub-second)
granularity with microstructure-aware cost terms.

No existing paper combines QUBO formulation with market microstructure.
Rosenberg et al. (2016) and all subsequent papers use static cost models.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from .hft_microstructure import MicrostructureState


@dataclass
class HFTQUBOConfig:
    """
    Configuration for HFT-specific QUBO formulation.

    Extends QUBOConfig with microstructure parameters.
    """
    total_shares: int = 5000
    num_tick_slices: int = 20
    num_venues: int = 3
    quantity_levels: List[int] = field(default_factory=lambda: [0, 100, 250, 500, 1000])

    impact_weight: float = 0.25
    timing_weight: float = 0.15
    transaction_weight: float = 0.15
    adverse_selection_weight: float = 0.20
    inventory_risk_weight: float = 0.15
    information_leakage_weight: float = 0.10

    tick_duration_ms: float = 100.0
    volatility: float = 0.02
    avg_spread_bps: float = 3.0
    impact_coefficient: float = 0.1

    kyle_lambda: float = 0.0
    vpin: float = 0.0
    adverse_selection_cost: float = 0.0

    equality_penalty: float = 1000.0
    capacity_penalty: float = 500.0
    max_shares_per_tick: int = 2000

    venue_names: List[str] = field(default_factory=lambda: ["Lit", "Dark", "ECN"])
    venue_latency_us: List[float] = field(default_factory=lambda: [50.0, 200.0, 80.0])
    venue_adverse_selection: List[float] = field(default_factory=lambda: [1.0, 0.3, 0.7])
    venue_fill_probability: List[float] = field(default_factory=lambda: [0.95, 0.60, 0.85])

    @property
    def num_quantity_levels(self) -> int:
        return len(self.quantity_levels)

    @property
    def num_variables(self) -> int:
        return self.num_tick_slices * self.num_venues * self.num_quantity_levels

    def variable_index(self, tick: int, venue: int, qty_level: int) -> int:
        K = self.num_quantity_levels
        V = self.num_venues
        return tick * V * K + venue * K + qty_level

    def decode_index(self, idx: int) -> Tuple[int, int, int]:
        K = self.num_quantity_levels
        V = self.num_venues
        tick = idx // (V * K)
        remainder = idx % (V * K)
        venue = remainder // K
        qty_level = remainder % K
        return tick, venue, qty_level

    def update_from_microstructure(self, state: MicrostructureState) -> None:
        self.kyle_lambda = state.kyle_lambda
        self.vpin = state.vpin
        self.adverse_selection_cost = state.adverse_selection_cost


class HFTExecutionQUBO:
    """
    Builds HFT-specific QUBO matrix with microstructure cost terms.

    Cost function:
        C(x) = w_impact * C_impact(x)
             + w_timing * C_timing(x)
             + w_transaction * C_transaction(x)
             + w_adverse * C_adverse_selection(x)
             + w_inventory * C_inventory(x)
             + w_leakage * C_information_leakage(x)
             + P_equality * (sum - S)^2
             + P_capacity * violations
    """

    def __init__(self, config: HFTQUBOConfig):
        self.config = config
        self.Q: Optional[np.ndarray] = None

        self._impact_matrix: Optional[np.ndarray] = None
        self._timing_matrix: Optional[np.ndarray] = None
        self._transaction_matrix: Optional[np.ndarray] = None
        self._adverse_matrix: Optional[np.ndarray] = None
        self._inventory_matrix: Optional[np.ndarray] = None
        self._leakage_matrix: Optional[np.ndarray] = None
        self._constraint_matrix: Optional[np.ndarray] = None

    def build_qubo_matrix(self) -> np.ndarray:
        n = self.config.num_variables

        self._impact_matrix = np.zeros((n, n))
        self._timing_matrix = np.zeros((n, n))
        self._transaction_matrix = np.zeros((n, n))
        self._adverse_matrix = np.zeros((n, n))
        self._inventory_matrix = np.zeros((n, n))
        self._leakage_matrix = np.zeros((n, n))
        self._constraint_matrix = np.zeros((n, n))

        self._add_market_impact_cost()
        self._add_timing_risk_cost()
        self._add_transaction_cost()
        self._add_adverse_selection_cost()
        self._add_inventory_risk_cost()
        self._add_information_leakage_cost()
        self._add_equality_constraint()
        self._add_capacity_constraint()

        cfg = self.config
        self.Q = (
            cfg.impact_weight * self._impact_matrix
            + cfg.timing_weight * self._timing_matrix
            + cfg.transaction_weight * self._transaction_matrix
            + cfg.adverse_selection_weight * self._adverse_matrix
            + cfg.inventory_risk_weight * self._inventory_matrix
            + cfg.information_leakage_weight * self._leakage_matrix
            + self._constraint_matrix
        )

        self.Q = (self.Q + self.Q.T) / 2
        return self.Q

    def _add_market_impact_cost(self) -> None:
        """
        Market impact with Kyle's lambda scaling.

        Impact_i = (eta + kyle_lambda) * sigma * q_i
        Cross-impact between simultaneous venue executions is amplified
        by information leakage factor.
        """
        cfg = self.config
        base_impact = (cfg.impact_coefficient + cfg.kyle_lambda) * cfg.volatility

        for t in range(cfg.num_tick_slices):
            for v in range(cfg.num_venues):
                for k in range(cfg.num_quantity_levels):
                    i = cfg.variable_index(t, v, k)
                    q_i = cfg.quantity_levels[k]

                    self._impact_matrix[i, i] += base_impact * q_i

                    for v2 in range(cfg.num_venues):
                        if v2 != v:
                            for k2 in range(cfg.num_quantity_levels):
                                j = cfg.variable_index(t, v2, k2)
                                q_j = cfg.quantity_levels[k2]
                                cross = 0.4 * base_impact * np.sqrt(q_i * q_j)
                                self._impact_matrix[i, j] += cross

    def _add_timing_risk_cost(self) -> None:
        cfg = self.config
        tick_vol = cfg.volatility * np.sqrt(cfg.tick_duration_ms / (6.5 * 60 * 60 * 1000))

        for t in range(cfg.num_tick_slices):
            time_factor = tick_vol * np.sqrt(t + 1)

            for v in range(cfg.num_venues):
                for k in range(cfg.num_quantity_levels):
                    i = cfg.variable_index(t, v, k)
                    q_i = cfg.quantity_levels[k]
                    self._timing_matrix[i, i] += time_factor * (q_i ** 2) / cfg.total_shares

    def _add_transaction_cost(self) -> None:
        cfg = self.config
        base_spread = (cfg.avg_spread_bps / 10000) / 2

        for t in range(cfg.num_tick_slices):
            for v in range(cfg.num_venues):
                fill_prob = cfg.venue_fill_probability[v] if v < len(cfg.venue_fill_probability) else 0.9
                effective_cost = base_spread / fill_prob

                for k in range(cfg.num_quantity_levels):
                    i = cfg.variable_index(t, v, k)
                    q = cfg.quantity_levels[k]
                    self._transaction_matrix[i, i] += effective_cost * q

    def _add_adverse_selection_cost(self) -> None:
        """
        Adverse selection cost per venue.

        Lit venues have high adverse selection (informed traders see your order).
        Dark pools have lower adverse selection but uncertain fill.
        The cost scales with VPIN -- higher VPIN means more informed flow.
        """
        cfg = self.config
        vpin_multiplier = 1.0 + 2.0 * cfg.vpin

        for t in range(cfg.num_tick_slices):
            for v in range(cfg.num_venues):
                venue_as = cfg.venue_adverse_selection[v] if v < len(cfg.venue_adverse_selection) else 1.0
                as_cost = cfg.adverse_selection_cost * venue_as * vpin_multiplier

                for k in range(cfg.num_quantity_levels):
                    i = cfg.variable_index(t, v, k)
                    q = cfg.quantity_levels[k]
                    self._adverse_matrix[i, i] += as_cost * q

    def _add_inventory_risk_cost(self) -> None:
        """
        Inventory risk at tick-level granularity.

        In HFT, holding inventory (unexecuted shares) exposes you to
        adverse price moves. The risk is quadratic in remaining inventory
        and linear in time remaining.

        This penalizes schedules that leave large inventory late in execution.
        """
        cfg = self.config
        tick_vol = cfg.volatility * np.sqrt(cfg.tick_duration_ms / (6.5 * 60 * 60 * 1000))

        for t in range(cfg.num_tick_slices):
            ticks_remaining = cfg.num_tick_slices - t
            inventory_risk = tick_vol * np.sqrt(ticks_remaining)

            for v in range(cfg.num_venues):
                for k in range(cfg.num_quantity_levels):
                    i = cfg.variable_index(t, v, k)
                    q = cfg.quantity_levels[k]
                    remaining_after = max(0, cfg.total_shares - q)
                    self._inventory_matrix[i, i] += inventory_risk * (remaining_after / cfg.total_shares) * q

            for v1 in range(cfg.num_venues):
                for k1 in range(cfg.num_quantity_levels):
                    i = cfg.variable_index(t, v1, k1)
                    q_i = cfg.quantity_levels[k1]

                    for t2 in range(t + 1, cfg.num_tick_slices):
                        for v2 in range(cfg.num_venues):
                            for k2 in range(cfg.num_quantity_levels):
                                j = cfg.variable_index(t2, v2, k2)
                                q_j = cfg.quantity_levels[k2]
                                gap = t2 - t
                                inv_penalty = -0.01 * tick_vol * gap * min(q_i, q_j) / cfg.total_shares
                                self._inventory_matrix[i, j] += inv_penalty

    def _add_information_leakage_cost(self) -> None:
        """
        Information leakage cost across venues.

        Executing on lit venues reveals information that propagates
        to other venues within milliseconds. Sequential execution
        across venues within a short window suffers from this leakage.

        The QUBO penalizes executing large quantities on lit venues
        followed by execution on other venues in the next few ticks.
        """
        cfg = self.config
        leakage_decay_ticks = 3

        for t in range(cfg.num_tick_slices):
            for v_lit in range(cfg.num_venues):
                lit_as = cfg.venue_adverse_selection[v_lit] if v_lit < len(cfg.venue_adverse_selection) else 1.0
                if lit_as < 0.5:
                    continue

                for k1 in range(cfg.num_quantity_levels):
                    i = cfg.variable_index(t, v_lit, k1)
                    q_i = cfg.quantity_levels[k1]

                    for dt in range(1, min(leakage_decay_ticks + 1, cfg.num_tick_slices - t)):
                        t2 = t + dt
                        decay = np.exp(-dt / leakage_decay_ticks)

                        for v2 in range(cfg.num_venues):
                            if v2 == v_lit:
                                continue
                            for k2 in range(cfg.num_quantity_levels):
                                j = cfg.variable_index(t2, v2, k2)
                                q_j = cfg.quantity_levels[k2]
                                leakage = decay * cfg.impact_coefficient * np.sqrt(q_i * q_j)
                                self._leakage_matrix[i, j] += leakage

    def _add_equality_constraint(self) -> None:
        cfg = self.config
        P = cfg.equality_penalty
        S = cfg.total_shares

        for i in range(cfg.num_variables):
            _, _, k_i = cfg.decode_index(i)
            q_i = cfg.quantity_levels[k_i]
            self._constraint_matrix[i, i] += P * q_i * q_i - 2 * P * S * q_i

            for j in range(i + 1, cfg.num_variables):
                _, _, k_j = cfg.decode_index(j)
                q_j = cfg.quantity_levels[k_j]
                self._constraint_matrix[i, j] += 2 * P * q_i * q_j

    def _add_capacity_constraint(self) -> None:
        cfg = self.config
        P = cfg.capacity_penalty
        M = cfg.max_shares_per_tick

        for t in range(cfg.num_tick_slices):
            vars_at_t = []
            for v in range(cfg.num_venues):
                for k in range(cfg.num_quantity_levels):
                    i = cfg.variable_index(t, v, k)
                    q = cfg.quantity_levels[k]
                    vars_at_t.append((i, q))

            for idx1, (i, q_i) in enumerate(vars_at_t):
                for idx2, (j, q_j) in enumerate(vars_at_t):
                    if idx2 > idx1 and q_i + q_j > M:
                        excess = q_i + q_j - M
                        self._constraint_matrix[i, j] += P * excess

    def interpret_solution(self, x: np.ndarray) -> dict:
        cfg = self.config
        schedule = []
        total_shares = 0

        for i, val in enumerate(x):
            if val > 0.5:
                t, v, k = cfg.decode_index(i)
                q = cfg.quantity_levels[k]
                if q > 0:
                    venue_name = cfg.venue_names[v] if v < len(cfg.venue_names) else f"Venue_{v}"
                    schedule.append({
                        "tick": t,
                        "tick_time_ms": t * cfg.tick_duration_ms,
                        "venue": venue_name,
                        "venue_idx": v,
                        "quantity": q
                    })
                    total_shares += q

        return {
            "schedule": sorted(schedule, key=lambda x: x["tick"]),
            "total_shares": total_shares,
            "target_shares": cfg.total_shares,
            "fill_rate": total_shares / cfg.total_shares if cfg.total_shares > 0 else 0,
            "num_slices_used": len(schedule),
            "venues_used": list(set(s["venue"] for s in schedule))
        }

    def calculate_cost_breakdown(self, x: np.ndarray) -> Dict[str, float]:
        if self.Q is None:
            self.build_qubo_matrix()

        cfg = self.config
        return {
            "total_cost": float(x @ self.Q @ x),
            "impact_cost": float(x @ self._impact_matrix @ x) * cfg.impact_weight,
            "timing_cost": float(x @ self._timing_matrix @ x) * cfg.timing_weight,
            "transaction_cost": float(x @ self._transaction_matrix @ x) * cfg.transaction_weight,
            "adverse_selection_cost": float(x @ self._adverse_matrix @ x) * cfg.adverse_selection_weight,
            "inventory_risk_cost": float(x @ self._inventory_matrix @ x) * cfg.inventory_risk_weight,
            "information_leakage_cost": float(x @ self._leakage_matrix @ x) * cfg.information_leakage_weight,
            "constraint_penalty": float(x @ self._constraint_matrix @ x)
        }
