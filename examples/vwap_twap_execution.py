"""Classical baseline: execute a 10,000-share buy order with VWAP and TWAP."""

from datetime import datetime

from qexec.execution.engine import ExecutionEngine, OrderSide, ParentOrder
from qexec.execution.strategies.twap import TWAPStrategy
from qexec.execution.strategies.vwap import VWAPStrategy
from qexec.market.simulator import MarketDataSimulator, MarketParams

SEED = 42
ORDER_SIZE = 10_000
MINUTES = 390


def main() -> None:
    params = MarketParams(symbol="AAPL", initial_price=175.00, annual_volatility=0.22)
    simulator = MarketDataSimulator(params=params, total_daily_volume=60_000_000, seed=SEED)
    market_data = simulator.generate(datetime(2024, 1, 15), num_minutes=MINUTES)

    strategies = {
        "VWAP": VWAPStrategy(participation_rate=0.10, max_slice_pct=0.05, seed=SEED),
        "TWAP": TWAPStrategy(interval_minutes=1, max_slice_pct=0.05, seed=SEED),
    }
    reports = {}
    for name, strategy in strategies.items():
        order = ParentOrder(
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=ORDER_SIZE,
            time_horizon_minutes=MINUTES,
            strategy_name=name,
        )
        reports[name] = ExecutionEngine(seed=SEED).process_order(order, market_data, strategy)

    rows = [
        ("Avg exec price ($)", "average_execution_price", "{:.4f}"),
        ("Market VWAP ($)", "benchmark_vwap", "{:.4f}"),
        ("Slippage vs VWAP (bps)", "slippage_vs_vwap_bps", "{:+.2f}"),
        ("Slippage vs arrival (bps)", "slippage_vs_arrival_bps", "{:+.2f}"),
        ("Spread cost ($)", "spread_cost", "{:.2f}"),
        ("Impact cost ($)", "impact_cost", "{:.2f}"),
        ("Fill rate", "fill_rate", "{:.1%}"),
        ("Child orders", "num_child_orders", "{:d}"),
    ]
    print(f"BUY {ORDER_SIZE:,} AAPL over {MINUTES} simulated minutes\n")
    print(f"{'Metric':<28}" + "".join(f"{name:>14}" for name in reports))
    for label, attr, fmt in rows:
        print(
            f"{label:<28}"
            + "".join(f"{fmt.format(getattr(r, attr)):>14}" for r in reports.values())
        )


if __name__ == "__main__":
    main()
