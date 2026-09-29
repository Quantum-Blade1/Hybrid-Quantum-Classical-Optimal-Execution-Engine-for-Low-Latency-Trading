from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qexec.optimization.solvers.result import BinaryVector

TRADING_DAY_MS = 6.5 * 60 * 60 * 1000
_CROSS_VENUE_IMPACT = 0.4
_INVENTORY_COUPLING = -0.01
_LEAKAGE_DECAY_TICKS = 3
_LIT_VENUE_AS_THRESHOLD = 0.5
_DEFAULT_FILL_PROBABILITY = 0.9


@dataclass
class HFTQUBOConfig:
    total_shares: int = 5000
    num_tick_slices: int = 20
    num_venues: int = 3
    quantity_levels: list[int] = field(default_factory=lambda: [0, 100, 250, 500, 1000])

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

    venue_names: list[str] = field(default_factory=lambda: ["Lit", "Dark", "ECN"])
    venue_adverse_selection: list[float] = field(default_factory=lambda: [1.0, 0.3, 0.7])
    venue_fill_probability: list[float] = field(default_factory=lambda: [0.95, 0.60, 0.85])

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

    def decode_index(self, idx: int) -> tuple[int, int, int]:
        K = self.num_quantity_levels
        V = self.num_venues
        tick, remainder = divmod(idx, V * K)
        venue, qty_level = divmod(remainder, K)
        return tick, venue, qty_level

    def venue_adverse_selection_at(self, venue: int) -> float:
        if venue < len(self.venue_adverse_selection):
            return self.venue_adverse_selection[venue]
        return 1.0

    @property
    def tick_volatility(self) -> float:
        """Daily volatility scaled to one tick: sigma sqrt(tick / trading day)."""
        return float(self.volatility * np.sqrt(self.tick_duration_ms / TRADING_DAY_MS))


class HFTExecutionQUBO:
    """Six weighted microstructure-aware cost terms plus equality and capacity penalties."""

    _TERMS = ("impact", "timing", "transaction", "adverse", "inventory", "leakage")

    def __init__(self, config: HFTQUBOConfig) -> None:
        self.config = config
        self.Q: NDArray[np.float64] | None = None
        n = config.num_variables
        self._matrices: dict[str, NDArray[np.float64]] = {
            name: np.zeros((n, n)) for name in (*self._TERMS, "constraint")
        }

    def _weights(self) -> dict[str, float]:
        cfg = self.config
        return {
            "impact": cfg.impact_weight,
            "timing": cfg.timing_weight,
            "transaction": cfg.transaction_weight,
            "adverse": cfg.adverse_selection_weight,
            "inventory": cfg.inventory_risk_weight,
            "leakage": cfg.information_leakage_weight,
        }

    def build_qubo_matrix(self) -> NDArray[np.float64]:
        n = self.config.num_variables
        self._matrices = {name: np.zeros((n, n)) for name in (*self._TERMS, "constraint")}

        self._add_market_impact_cost()
        self._add_timing_risk_cost()
        self._add_transaction_cost()
        self._add_adverse_selection_cost()
        self._add_inventory_risk_cost()
        self._add_information_leakage_cost()
        self._add_equality_constraint()
        self._add_capacity_constraint()

        weights = self._weights()
        m = self._matrices
        Q = (
            weights["impact"] * m["impact"]
            + weights["timing"] * m["timing"]
            + weights["transaction"] * m["transaction"]
            + weights["adverse"] * m["adverse"]
            + weights["inventory"] * m["inventory"]
            + weights["leakage"] * m["leakage"]
            + m["constraint"]
        )
        self.Q = (Q + Q.T) / 2
        return self.Q

    def _add_market_impact_cost(self) -> None:
        cfg = self.config
        M = self._matrices["impact"]
        base_impact = (cfg.impact_coefficient + cfg.kyle_lambda) * cfg.volatility
        for t in range(cfg.num_tick_slices):
            for v in range(cfg.num_venues):
                for k in range(cfg.num_quantity_levels):
                    i = cfg.variable_index(t, v, k)
                    q_i = cfg.quantity_levels[k]
                    M[i, i] += base_impact * q_i
                    for v2 in range(cfg.num_venues):
                        if v2 == v:
                            continue
                        for k2 in range(cfg.num_quantity_levels):
                            j = cfg.variable_index(t, v2, k2)
                            q_j = cfg.quantity_levels[k2]
                            M[i, j] += _CROSS_VENUE_IMPACT * base_impact * np.sqrt(q_i * q_j)

    def _add_timing_risk_cost(self) -> None:
        cfg = self.config
        M = self._matrices["timing"]
        tick_vol = cfg.tick_volatility
        for t in range(cfg.num_tick_slices):
            time_factor = tick_vol * np.sqrt(t + 1)
            for v in range(cfg.num_venues):
                for k in range(cfg.num_quantity_levels):
                    i = cfg.variable_index(t, v, k)
                    q_i = cfg.quantity_levels[k]
                    M[i, i] += time_factor * (q_i**2) / cfg.total_shares

    def _add_transaction_cost(self) -> None:
        cfg = self.config
        M = self._matrices["transaction"]
        half_spread = (cfg.avg_spread_bps / 10000) / 2
        for t in range(cfg.num_tick_slices):
            for v in range(cfg.num_venues):
                fill_prob = (
                    cfg.venue_fill_probability[v]
                    if v < len(cfg.venue_fill_probability)
                    else _DEFAULT_FILL_PROBABILITY
                )
                effective_cost = half_spread / fill_prob
                for k in range(cfg.num_quantity_levels):
                    i = cfg.variable_index(t, v, k)
                    M[i, i] += effective_cost * cfg.quantity_levels[k]

    def _add_adverse_selection_cost(self) -> None:
        cfg = self.config
        M = self._matrices["adverse"]
        vpin_multiplier = 1.0 + 2.0 * cfg.vpin
        for t in range(cfg.num_tick_slices):
            for v in range(cfg.num_venues):
                as_cost = (
                    cfg.adverse_selection_cost * cfg.venue_adverse_selection_at(v) * vpin_multiplier
                )
                for k in range(cfg.num_quantity_levels):
                    i = cfg.variable_index(t, v, k)
                    M[i, i] += as_cost * cfg.quantity_levels[k]

    def _add_inventory_risk_cost(self) -> None:
        cfg = self.config
        M = self._matrices["inventory"]
        tick_vol = cfg.tick_volatility
        for t in range(cfg.num_tick_slices):
            inventory_risk = tick_vol * np.sqrt(cfg.num_tick_slices - t)
            for v in range(cfg.num_venues):
                for k in range(cfg.num_quantity_levels):
                    i = cfg.variable_index(t, v, k)
                    q = cfg.quantity_levels[k]
                    remaining_after = max(0, cfg.total_shares - q)
                    M[i, i] += inventory_risk * (remaining_after / cfg.total_shares) * q

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
                                M[i, j] += (
                                    _INVENTORY_COUPLING
                                    * tick_vol
                                    * gap
                                    * min(q_i, q_j)
                                    / cfg.total_shares
                                )

    def _add_information_leakage_cost(self) -> None:
        cfg = self.config
        M = self._matrices["leakage"]
        for t in range(cfg.num_tick_slices):
            for v_lit in range(cfg.num_venues):
                if cfg.venue_adverse_selection_at(v_lit) < _LIT_VENUE_AS_THRESHOLD:
                    continue
                for k1 in range(cfg.num_quantity_levels):
                    i = cfg.variable_index(t, v_lit, k1)
                    q_i = cfg.quantity_levels[k1]
                    for dt in range(1, min(_LEAKAGE_DECAY_TICKS + 1, cfg.num_tick_slices - t)):
                        decay = np.exp(-dt / _LEAKAGE_DECAY_TICKS)
                        for v2 in range(cfg.num_venues):
                            if v2 == v_lit:
                                continue
                            for k2 in range(cfg.num_quantity_levels):
                                j = cfg.variable_index(t + dt, v2, k2)
                                q_j = cfg.quantity_levels[k2]
                                M[i, j] += decay * cfg.impact_coefficient * np.sqrt(q_i * q_j)

    def _add_equality_constraint(self) -> None:
        cfg = self.config
        M = self._matrices["constraint"]
        P = cfg.equality_penalty
        S = cfg.total_shares
        for i in range(cfg.num_variables):
            q_i = cfg.quantity_levels[cfg.decode_index(i)[2]]
            M[i, i] += P * q_i * q_i - 2 * P * S * q_i
            for j in range(i + 1, cfg.num_variables):
                q_j = cfg.quantity_levels[cfg.decode_index(j)[2]]
                M[i, j] += 2 * P * q_i * q_j

    def _add_capacity_constraint(self) -> None:
        cfg = self.config
        M = self._matrices["constraint"]
        P = cfg.capacity_penalty
        cap = cfg.max_shares_per_tick
        for t in range(cfg.num_tick_slices):
            vars_at_t = [
                (cfg.variable_index(t, v, k), cfg.quantity_levels[k])
                for v in range(cfg.num_venues)
                for k in range(cfg.num_quantity_levels)
            ]
            for idx1, (i, q_i) in enumerate(vars_at_t):
                for j, q_j in vars_at_t[idx1 + 1 :]:
                    if q_i + q_j > cap:
                        M[i, j] += P * (q_i + q_j - cap)

    def slice_quantities(self, x: BinaryVector) -> NDArray[np.float64]:
        cfg = self.config
        quantities = np.zeros(cfg.num_tick_slices)
        for i in np.flatnonzero(np.asarray(x) > 0.5):
            t, _, k = cfg.decode_index(int(i))
            quantities[t] += cfg.quantity_levels[k]
        return quantities

    def interpret_solution(self, x: BinaryVector) -> dict[str, Any]:
        cfg = self.config
        schedule = []
        total_shares = 0
        for i, val in enumerate(x):
            if val <= 0.5:
                continue
            t, v, k = cfg.decode_index(i)
            q = cfg.quantity_levels[k]
            if q > 0:
                venue_name = cfg.venue_names[v] if v < len(cfg.venue_names) else f"Venue_{v}"
                schedule.append(
                    {
                        "tick": t,
                        "tick_time_ms": t * cfg.tick_duration_ms,
                        "venue": venue_name,
                        "venue_idx": v,
                        "quantity": q,
                    }
                )
                total_shares += q

        return {
            "schedule": sorted(schedule, key=lambda entry: entry["tick"]),
            "total_shares": total_shares,
            "target_shares": cfg.total_shares,
            "fill_rate": total_shares / cfg.total_shares if cfg.total_shares > 0 else 0,
            "num_slices_used": len(schedule),
            "venues_used": sorted({s["venue"] for s in schedule}),
        }

    def calculate_cost_breakdown(self, x: BinaryVector) -> dict[str, float]:
        Q = self.Q if self.Q is not None else self.build_qubo_matrix()
        m = self._matrices
        w = self._weights()
        return {
            "total_cost": float(x @ Q @ x),
            "impact_cost": float(x @ m["impact"] @ x) * w["impact"],
            "timing_cost": float(x @ m["timing"] @ x) * w["timing"],
            "transaction_cost": float(x @ m["transaction"] @ x) * w["transaction"],
            "adverse_selection_cost": float(x @ m["adverse"] @ x) * w["adverse"],
            "inventory_risk_cost": float(x @ m["inventory"] @ x) * w["inventory"],
            "information_leakage_cost": float(x @ m["leakage"] @ x) * w["leakage"],
            "constraint_penalty": float(x @ m["constraint"] @ x),
        }
