"""VWAP, TWAP and QUBOStrategy on one simulated day through the ExecutionEngine, with a plot.

Usage:
    python experiments/strategy_comparison.py [--shares 20000] [--seed 42] [--output-dir results]
"""

import argparse
from datetime import datetime
from pathlib import Path

from plotting import plot_strategy_comparison

from qexec.execution.engine import OrderSide, ParentOrder
from qexec.execution.strategies.qubo import run_integrated_comparison
from qexec.market.simulator import TRADING_MINUTES_PER_DAY, MarketDataSimulator, MarketParams


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shares", type=int, default=20_000)
    parser.add_argument("--slices", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    params = MarketParams(symbol="AAPL", initial_price=175.0, annual_volatility=0.22)
    market_data = MarketDataSimulator(
        params, total_daily_volume=60_000_000, seed=args.seed
    ).generate(datetime(2024, 1, 15), num_minutes=TRADING_MINUTES_PER_DAY)
    order = ParentOrder(
        symbol="AAPL",
        side=OrderSide.BUY,
        total_quantity=args.shares,
        time_horizon_minutes=TRADING_MINUTES_PER_DAY,
    )
    comparison = run_integrated_comparison(
        order, market_data, qubo_time_slices=args.slices, qubo_sa_sweeps=1000, seed=args.seed
    )
    print(comparison.to_dataframe().to_string(index=False))
    print(f"\nLowest total cost: {comparison.best_strategy}")
    path = plot_strategy_comparison(
        comparison, market_data["price"].tolist(), args.output_dir / "strategy_comparison.png"
    )
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
