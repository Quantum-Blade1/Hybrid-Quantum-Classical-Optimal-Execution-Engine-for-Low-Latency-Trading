"""QUBO-optimised schedule vs VWAP and TWAP on the same simulated hour."""

from datetime import datetime

from qexec.execution.engine import OrderSide, ParentOrder
from qexec.execution.strategies.qubo import run_integrated_comparison
from qexec.market.simulator import MarketDataSimulator, MarketParams

SEED = 42
ORDER_SIZE = 5_000
MINUTES = 60


def main() -> None:
    params = MarketParams(symbol="AAPL", initial_price=175.00, annual_volatility=0.22)
    simulator = MarketDataSimulator(params=params, total_daily_volume=60_000_000, seed=SEED)
    market_data = simulator.generate(datetime(2024, 1, 15), num_minutes=MINUTES)

    order = ParentOrder(
        symbol="AAPL",
        side=OrderSide.BUY,
        total_quantity=ORDER_SIZE,
        time_horizon_minutes=MINUTES,
    )
    comparison = run_integrated_comparison(
        parent_order=order,
        market_data=market_data,
        qubo_time_slices=10,
        qubo_sa_sweeps=500,
        seed=SEED,
    )
    print(comparison.to_dataframe().to_string(index=False))
    print(f"\nLowest shortfall (incl. opportunity cost): {comparison.best_strategy}")


if __name__ == "__main__":
    main()
