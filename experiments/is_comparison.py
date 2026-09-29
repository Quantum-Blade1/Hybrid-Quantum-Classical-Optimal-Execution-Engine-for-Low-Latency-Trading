"""Implementation shortfall of VWAP and TWAP on one simulated hour, with a stacked-bar chart.

Usage:
    python experiments/is_comparison.py [--seed 42] [--output-dir results]
"""

import argparse
from pathlib import Path

import pandas as pd
from plotting import plot_is_breakdown

from qexec.analysis.shortfall import ISAnalyzer, ISComponents
from qexec.execution.strategies.base import BaseStrategy
from qexec.execution.strategies.twap import TWAPStrategy
from qexec.execution.strategies.vwap import VWAPStrategy
from qexec.market.order_book import OrderBook
from qexec.market.simulator import MarketDataSimulator, MarketParams

TOTAL_SHARES = 100_000
MINUTES = 60


def shortfall(strategy: BaseStrategy, data: pd.DataFrame, analyzer: ISAnalyzer) -> ISComponents:
    strategy.execute(TOTAL_SHARES, "buy", data)
    log = pd.DataFrame(
        [
            {"timestamp": s.timestamp, "shares": s.filled_quantity, "price": s.execution_price}
            for s in strategy.slices
        ]
    )
    return analyzer.analyze(log, data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    params = MarketParams(initial_price=100.0, annual_volatility=0.30)
    data = MarketDataSimulator(params=params, seed=args.seed).generate(num_minutes=MINUTES)
    decision_price = float(data.iloc[0]["price"])
    analyzer = ISAnalyzer(decision_price, TOTAL_SHARES)
    print(f"Decision price: ${decision_price:.2f}")

    strategies: dict[str, BaseStrategy] = {
        "VWAP": VWAPStrategy(
            participation_rate=0.1, order_book=OrderBook(seed=args.seed), seed=args.seed
        ),
        "TWAP": TWAPStrategy(interval_minutes=1, order_book=OrderBook(seed=args.seed)),
    }
    results = {name: shortfall(s, data, analyzer) for name, s in strategies.items()}

    table = pd.DataFrame({name: r.to_dict() for name, r in results.items()}).T
    print(table.to_string(float_format=lambda x: f"{x:,.0f}"))
    path = plot_is_breakdown(results, args.output_dir / "is_comparison.png")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
