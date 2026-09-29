"""Seeded walk-forward backtest: static VWAP, adaptive VWAP and SA-QUBO schedules.

Usage:
    python experiments/walk_forward.py [--days 10] [--seed 42] [--output-dir results]
"""

import argparse
from pathlib import Path

from plotting import plot_walk_forward

from qexec.analysis.walk_forward import WalkForwardAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--train-days", type=int, default=3)
    parser.add_argument("--shares", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    analyzer = WalkForwardAnalyzer(
        total_days=args.days,
        train_days=args.train_days,
        test_days=1,
        daily_shares=args.shares,
        seed=args.seed,
    )
    results = analyzer.run()
    print(
        f"{'Window':<8}{'Train vol':>10}{'Static $':>14}{'Adaptive $':>14}{'Hybrid $':>14}  Lowest"
    )
    for r in results:
        print(
            f"{r.window_id:<8}{r.train_volatility:>10.1%}{r.shortfall_static:>14,.0f}"
            f"{r.shortfall_adaptive:>14,.0f}{r.shortfall_hybrid:>14,.0f}  {r.winner}"
        )
    path = plot_walk_forward(results, args.output_dir / "walk_forward_results.png")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
