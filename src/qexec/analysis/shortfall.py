"""Implementation shortfall (Perold, 1988) split into delay, impact, timing and opportunity cost.

For a buy of N shares decided at price P_d, arriving at P_0 and filling n_i at p_i when the
mid is m_i, with closing price P_T and U unfilled shares:
    delay = (P_0 - P_d) N,  impact = sum (p_i - m_i) n_i,  timing = sum (m_i - P_0) n_i,
    opportunity = (P_T - P_d) U,  total = sum of the four.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ISComponents:
    """Shortfall components in currency units."""

    decision_price: float
    arrival_price: float
    total_shares: int
    executed_shares: int
    total_shortfall: float
    delay_cost: float
    market_impact: float
    timing_risk: float
    opportunity_cost: float
    avg_exec_price: float

    def to_dict(self) -> dict[str, float]:
        return {
            "Total IS": self.total_shortfall,
            "Delay Cost": self.delay_cost,
            "Market Impact": self.market_impact,
            "Timing Risk": self.timing_risk,
            "Opportunity Cost": self.opportunity_cost,
        }


class ISAnalyzer:
    """Computes `ISComponents` for a fill log against minute-bar mid prices."""

    def __init__(self, decision_price: float, total_orders: int) -> None:
        self.decision_price = decision_price
        self.total_orders = total_orders

    def analyze(self, execution_log: pd.DataFrame, market_data: pd.DataFrame) -> ISComponents:
        """`execution_log`: timestamp, shares, price (fill); `market_data`: timestamp, price (mid).

        Each fill is matched to the nearest market-data timestamp.
        """
        if execution_log.empty:
            return ISComponents(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        df = pd.merge_asof(
            execution_log.sort_values("timestamp"),
            market_data.sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
            suffixes=("_exec", "_mkt"),
        )
        executed_shares = int(df["shares"].sum())
        unexecuted_shares = self.total_orders - executed_shares
        arrival_price = float(market_data.iloc[0]["price"])
        last_price = float(market_data.iloc[-1]["price"])

        delay_cost = (arrival_price - self.decision_price) * self.total_orders
        impact_cost = float(((df["price_exec"] - df["price_mkt"]) * df["shares"]).sum())
        timing_risk = float(((df["price_mkt"] - arrival_price) * df["shares"]).sum())
        opportunity_cost = (last_price - self.decision_price) * unexecuted_shares
        avg_exec_price = float((df["price_exec"] * df["shares"]).sum() / executed_shares)

        return ISComponents(
            decision_price=self.decision_price,
            arrival_price=arrival_price,
            total_shares=self.total_orders,
            executed_shares=executed_shares,
            total_shortfall=delay_cost + impact_cost + timing_risk + opportunity_cost,
            delay_cost=delay_cost,
            market_impact=impact_cost,
            timing_risk=timing_risk,
            opportunity_cost=opportunity_cost,
            avg_exec_price=avg_exec_price,
        )
