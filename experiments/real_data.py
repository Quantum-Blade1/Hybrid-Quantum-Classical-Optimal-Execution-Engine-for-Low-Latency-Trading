"""Phase 7 real-data evaluation (docs/PROTOCOL.md): TWAP, VWAP, discretized Almgren-Chriss,
the binary-encoded QUBO (SA) and the adaptive hybrid on Binance 1-minute bars.

Everything is calibrated on the development days; `real_data_dev` evaluates on the same
days (in-sample, for development) and `real_data_test` on the held-out test days (run once
per the protocol; re-running the frozen code reproduces it exactly: nothing is random
except SA, which is seeded per order). Orders execute through `ExecutionEngine` with the
`ImpactFillModel` (participation cap, carry-forward, opportunity cost at the last bar).

Without the downloaded data (`make data`), full runs stop with a message. `--quick` runs
on small synthetic bars generated here (clearly not market data) for CI.

Usage:
    python -m experiments.real_data_dev [--quick] [--results-dir results]
    python -m experiments.real_data_test [--quick] [--results-dir results]
"""

from __future__ import annotations

import os
import zlib
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from experiments import protocol
from experiments.common import Experiment, single_seed
from qexec.analysis.statistics import (
    bootstrap_ci,
    cluster_bootstrap_ci,
    holm_adjust,
    paired_comparison,
)
from qexec.execution.cost_model import CostModel
from qexec.execution.engine import ExecutionEngine, OrderSide, ParentOrder
from qexec.execution.fill import ImpactFillModel
from qexec.execution.strategies.base import BaseStrategy
from qexec.execution.strategies.fixed import FixedScheduleStrategy
from qexec.execution.strategies.model_based import (
    AdaptiveQUBOStrategy,
    QUBOSettings,
    ac_schedule,
    dp_schedule,
    proportional_schedule,
    qubo_plan,
)
from qexec.experiment import ExperimentRecorder
from qexec.market.calibration import MINUTES_PER_DAY, Calibration, calibrate, log_returns_bps

STRATEGIES = ("TWAP", "VWAP", "AC", "QUBO", "Hybrid", "DP")
PRIMARY_STRATEGIES = ("QUBO", "Hybrid")
BASELINES = ("TWAP", "VWAP", "AC")
ALPHA = 0.05
EQUIVALENCE_BPS = 0.5


# Frozen from results/real_data_tune (development days only; selection rules in
# experiments/real_data_tune.py). Changing them after the test run is a protocol deviation.
FROZEN_QUBO = QUBOSettings(num_slices=4, bits=3, units=16, sweeps=1000, restarts=16)
FROZEN_CHECKPOINTS = 1
FROZEN_CLIP = (0.5, 2.0)


@dataclass(frozen=True)
class Variant:
    """Risk aversion rule ('zero' or 'rule': lambda Var = E for TWAP), the evaluator's
    impact multiple and the optimisers' impact multiple."""

    name: str
    risk: str = "zero"
    eval_multiple: float = 1.0
    opt_multiple: float = 1.0


VARIANTS = (
    Variant("primary"),
    Variant("eval_x0.5", eval_multiple=0.5),
    Variant("eval_x2", eval_multiple=2.0),
    Variant("both_x0.5", eval_multiple=0.5, opt_multiple=0.5),
    Variant("both_x2", eval_multiple=2.0, opt_multiple=2.0),
    Variant("risk_averse", risk="rule"),
)


@dataclass(frozen=True)
class Config:
    split: str = "dev"
    symbols: tuple[str, ...] = protocol.SYMBOLS
    dev_days: tuple[str, ...] = protocol.DEV_DAYS
    eval_days: tuple[str, ...] = protocol.DEV_DAYS
    start_hours: tuple[int, ...] = protocol.START_HOURS
    sizes_pct_adv: tuple[float, ...] = protocol.SIZES_PCT_ADV
    horizons: tuple[int, ...] = protocol.HORIZONS_MIN
    participation_cap: float = protocol.PARTICIPATION_CAP
    bucket_minutes: int = protocol.BUCKET_MINUTES
    min_lots: int = protocol.MIN_LOTS_SMALLEST_ORDER
    qubo: QUBOSettings = FROZEN_QUBO
    hybrid_checkpoints: int = FROZEN_CHECKPOINTS
    hybrid_clip: tuple[float, float] = FROZEN_CLIP
    variants: tuple[Variant, ...] = VARIANTS
    robustness_seeds: int = 5
    synthetic: bool = False
    bars_root: str = str(protocol.BARS_DIR)
    workers: int = min(12, os.cpu_count() or 1)
    seed: int = 0


QUICK_QUBO = QUBOSettings(num_slices=4, bits=3, units=12, sweeps=200, restarts=4)
QUICK_BASE = {
    "dev_days": ("2026-07-22", "2026-07-23"),
    "start_hours": (0, 12),
    "horizons": (30,),
    "qubo": QUICK_QUBO,
    "variants": (VARIANTS[0], VARIANTS[2], VARIANTS[5]),
    "robustness_seeds": 2,
    "synthetic": True,
    "workers": 1,
}


# -- data ---------------------------------------------------------------------------------


def synthetic_bars(symbol: str, days: Iterable[str], seed: int) -> pd.DataFrame:
    """Fake 1-minute bars with the Binance bar columns (for quick runs and CI only)."""
    frames = []
    base = 60_000.0 if symbol.startswith("BTC") else 9.0
    scale = 10.0 if symbol.startswith("BTC") else 800.0
    for day in days:
        rng = np.random.default_rng([seed, zlib.crc32(f"{symbol}{day}".encode())])
        minute = np.arange(MINUTES_PER_DAY)
        shape = 1 + 0.5 * np.cos(2 * np.pi * (minute - 900) / MINUTES_PER_DAY)
        volume = scale * shape * rng.lognormal(0, 0.5, MINUTES_PER_DAY)
        signed = volume * rng.uniform(-0.6, 0.6, MINUTES_PER_DAY)
        r = 1.5 * signed / (scale * shape) + rng.normal(0, 4, MINUTES_PER_DAY)
        close = base * np.exp(np.cumsum(r) / 1e4)
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": pd.date_range(day, periods=MINUTES_PER_DAY, freq="min"),
                    "open": np.r_[base, close[:-1]],
                    "close": close,
                    "vwap": close,
                    "volume": volume,
                    "signed_volume": signed,
                    "half_spread_bps": np.where(rng.random(MINUTES_PER_DAY) < 0.3, 0.5, np.nan),
                }
            )
        )
        base = float(close[-1])
    return pd.concat(frames, ignore_index=True)


def load(config: Config, symbol: str, days: tuple[str, ...]) -> pd.DataFrame:
    if config.synthetic:
        return synthetic_bars(symbol, days, config.seed)
    return protocol.load_bars(symbol, days, Path(config.bars_root))


def evaluation_frame(bars: pd.DataFrame, cal: Calibration) -> pd.DataFrame:
    """Bars in lots with the columns the fill model and the hybrid read."""
    minute = (
        pd.DatetimeIndex(bars["timestamp"]).hour * 60 + pd.DatetimeIndex(bars["timestamp"]).minute
    )
    minute = np.asarray(minute)
    price = bars["vwap"].fillna(bars["close"]).to_numpy(dtype=np.float64)
    profile_h = cal.profile.half_spread_bps[minute]
    observed_h = bars["half_spread_bps"].to_numpy(dtype=np.float64)
    h = np.where(np.isnan(observed_h), profile_h, observed_h)
    return pd.DataFrame(
        {
            "timestamp": bars["timestamp"],
            "day": pd.DatetimeIndex(bars["timestamp"]).strftime("%Y-%m-%d"),
            "minute_of_day": minute,
            "open": bars["open"].to_numpy(dtype=np.float64),
            "close": bars["close"].to_numpy(dtype=np.float64),
            "price": price,
            "volume": bars["volume"].to_numpy(dtype=np.float64) / cal.lot_size,
            "expected_volume": np.maximum(cal.profile.volume[minute] / cal.lot_size, 1.0),
            "half_spread_obs": observed_h,
            "half_spread_bps": h,
            "spread": 2 * h * price / 1e4,
        }
    )


# -- orders -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    size_pct: float
    horizon: int


def order_seed(*parts: object) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode())


def risk_aversion(model: CostModel, total: int, rule: str) -> float:
    if rule == "zero":
        return 0.0
    twap = np.full(model.num_minutes, total / model.num_minutes)
    return model.expected_cost_bps(twap, total) / model.variance_bps2(twap, total)


def static_schedules(
    config: Config, model: CostModel, total: int, seed: int
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    plan = qubo_plan(model, total, config.qubo, seed)
    schedules = {
        # Integer schedules summing to `total`, so the model costs below are comparable.
        "TWAP": proportional_schedule(np.ones(model.num_minutes), total),
        "VWAP": proportional_schedule(model.expected_volume, total),
        "AC": ac_schedule(model, total),
        "QUBO": plan.schedule,
        "DP": dp_schedule(model, total, config.qubo),
    }
    info = {
        "qubo_feasible": float(plan.feasible),
        "qubo_gap_to_dp_bps": plan.gap_to_ip_optimum_bps,
        "qubo_solve_time_s": plan.solve_time_s,
    }
    return schedules, info


def execute(
    strategy: BaseStrategy,
    window: pd.DataFrame,
    total: int,
    side: OrderSide,
    fill_model: ImpactFillModel,
) -> dict[str, float]:
    engine = ExecutionEngine(fill_model=fill_model)
    arrival = float(window["open"].iloc[0])
    order = ParentOrder("X", side, total, len(window))
    report = engine.process_order(order, window, strategy, arrival_price=arrival)
    assert engine.state is not None
    sign = 1.0 if side is OrderSide.BUY else -1.0
    timing = sum(
        sign * c.filled_quantity * (c.market_price_at_execution - arrival)
        for c in engine.state.child_orders
        if c.filled_quantity > 0
    )
    notional = total * arrival / 1e4
    return {
        "shortfall_bps": report.implementation_shortfall_bps,
        "spread_cost_bps": report.spread_cost / notional,
        "impact_cost_bps": report.impact_cost / notional,
        "timing_cost_bps": timing / notional,
        "opportunity_cost_bps": report.opportunity_cost / notional,
        "fill_rate": report.fill_rate,
    }


@dataclass
class Plans:
    """Static schedules of one (start hour, cell) under one variant (cached across days)."""

    model: CostModel
    total: int
    schedules: dict[str, np.ndarray]
    gap_row: dict[str, Any]
    robustness: dict[int, np.ndarray]


def make_plans(
    config: Config, cal: Calibration, symbol: str, variant: Variant, *, hour: int, cell: Cell
) -> Plans:
    start = hour * 60
    total = round(cal.adv * cell.size_pct / 100 / cal.lot_size)
    base = cal.cost_model(start, cell.horizon, impact_multiple=variant.opt_multiple)
    model = base.with_risk_aversion(risk_aversion(base, total, variant.risk))
    seed = order_seed(symbol, hour, cell.size_pct, cell.horizon)
    schedules, info = static_schedules(config, model, total, seed)
    gap_row = {
        "symbol": symbol,
        "variant": variant.name,
        "start_hour": hour,
        "size_pct": cell.size_pct,
        "horizon": cell.horizon,
        "total_lots": total,
        "risk_aversion": model.risk_aversion,
        **info,
        **{f"objective_{n}": model.objective(sch, total) for n, sch in schedules.items()},
        **{
            f"expected_cost_{n}": model.expected_cost_bps(sch, total)
            for n, sch in schedules.items()
        },
    }
    robustness = {}
    if variant.name == "primary":
        for extra in range(config.robustness_seeds):
            plan = qubo_plan(model, total, config.qubo, seed + 1000 * (extra + 1))
            robustness[extra] = plan.schedule.astype(float)
    return Plans(model, total, schedules, gap_row, robustness)


def order_rows(
    config: Config,
    plans: Plans,
    window: pd.DataFrame,
    *,
    side: OrderSide,
    fill_model: ImpactFillModel,
    common: dict[str, Any],
    hybrid_seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for name in STRATEGIES:
        t0 = perf_counter()
        strategy: BaseStrategy
        extra: dict[str, float] = {}
        if name == "Hybrid":
            strategy = AdaptiveQUBOStrategy(
                plans.model,
                config.qubo,
                seed=hybrid_seed,
                num_checkpoints=config.hybrid_checkpoints,
                clip=config.hybrid_clip,
            )
        else:
            strategy = FixedScheduleStrategy(plans.schedules[name])
        metrics = execute(strategy, window, plans.total, side, fill_model)
        if isinstance(strategy, AdaptiveQUBOStrategy):
            extra = {
                "invocations": strategy.invocations,
                "infeasible_solutions": strategy.infeasible_solutions,
                "optimization_time_s": strategy.optimization_time,
            }
        planned = plans.schedules.get(name, plans.schedules["QUBO"])
        rows.append(
            {
                **common,
                "strategy": name,
                **metrics,
                "planned_expected_cost_bps": plans.model.expected_cost_bps(planned, plans.total),
                "wall_time_s": perf_counter() - t0,
                **extra,
            }
        )
    for extra_seed, schedule in plans.robustness.items():
        metrics = execute(FixedScheduleStrategy(schedule), window, plans.total, side, fill_model)
        rows.append({**common, "strategy": f"QUBO_seed{extra_seed}", **metrics})
    return rows


def run_symbol_variant(config: Config, symbol: str, variant: Variant) -> dict[str, list[dict]]:
    dev = load(config, symbol, config.dev_days)
    cal = calibrate(
        symbol,
        dev,
        smallest_order_pct_adv=min(config.sizes_pct_adv),
        min_lots=config.min_lots,
        bucket_minutes=config.bucket_minutes,
    )
    frame = evaluation_frame(load(config, symbol, config.eval_days), cal)
    fill_model = ImpactFillModel(
        cal.impact.beta_bps * variant.eval_multiple, config.participation_cap
    )
    cells = [Cell(p, h) for p in config.sizes_pct_adv for h in config.horizons]
    plans = {
        (hour, cell): make_plans(config, cal, symbol, variant, hour=hour, cell=cell)
        for hour in config.start_hours
        for cell in cells
    }
    rows: list[dict] = []
    for day, group in frame.groupby("day", sort=True):
        day_bars = group.reset_index(drop=True)
        for hour in config.start_hours:
            side = OrderSide.BUY if hour in protocol.BUY_HOURS else OrderSide.SELL
            for cell in cells:
                start = hour * 60
                window = day_bars.iloc[start : start + cell.horizon].reset_index(drop=True)
                plan = plans[(hour, cell)]
                common = {
                    "symbol": symbol,
                    "variant": variant.name,
                    "day": day,
                    "start_hour": hour,
                    "side": side.value,
                    "size_pct": cell.size_pct,
                    "horizon": cell.horizon,
                    "total_lots": plan.total,
                }
                seed = order_seed(symbol, day, hour, cell.size_pct, cell.horizon)
                rows.extend(
                    order_rows(
                        config,
                        plan,
                        window,
                        side=side,
                        fill_model=fill_model,
                        common=common,
                        hybrid_seed=seed,
                    )
                )
    gaps = [plan.gap_row for plan in plans.values()]
    calib = {**cal.summary(), "variant": variant.name}
    return {"orders": rows, "gaps": gaps, "calibration": [calib]}


# -- statistics -----------------------------------------------------------------------------


def window_differences(orders: pd.DataFrame, strategy: str, baseline: str) -> pd.DataFrame:
    """Per window (symbol, day, start hour): mean over cells of shortfall(strategy) -
    shortfall(baseline)."""
    keys = ["symbol", "variant", "day", "start_hour", "size_pct", "horizon"]
    wide = orders.pivot_table(index=keys, columns="strategy", values="shortfall_bps")
    diff = (wide[strategy] - wide[baseline]).rename("diff").reset_index()
    return diff.groupby(["symbol", "variant", "day", "start_hour"], as_index=False)["diff"].mean()


def compare(values: np.ndarray, days: np.ndarray) -> dict[str, float]:
    stats = paired_comparison(values, np.zeros_like(values)).as_dict()
    low, high = cluster_bootstrap_ci(values, days)
    return {
        "n_windows": stats["n"],
        "mean_diff_bps": stats["mean_diff"],
        "ci_low": stats["ci_low"],
        "ci_high": stats["ci_high"],
        "day_cluster_ci_low": low,
        "day_cluster_ci_high": high,
        "median_diff_bps": stats["median_diff"],
        "frac_strategy_cheaper": stats["frac_a_lower"],
        "wilcoxon_p": stats["wilcoxon_p"],
    }


def decide(row: pd.Series) -> str:
    if row["p_holm"] < ALPHA:
        return "beats" if row["mean_diff_bps"] < 0 else "worse"
    if row["ci_low"] >= -EQUIVALENCE_BPS and row["ci_high"] <= EQUIVALENCE_BPS:
        return "no detectable difference (practically equivalent)"
    return "no detectable difference"


def comparison_family(
    orders: pd.DataFrame, strategies: Iterable[str], baselines: Iterable[str]
) -> pd.DataFrame:
    rows = []
    for (symbol, variant), group in orders.groupby(["symbol", "variant"], sort=False):
        for strategy in strategies:
            for baseline in baselines:
                if strategy == baseline:
                    continue
                w = window_differences(group, strategy, baseline)
                rows.append(
                    {
                        "symbol": symbol,
                        "variant": variant,
                        "strategy": strategy,
                        "baseline": baseline,
                        **compare(w["diff"].to_numpy(), w["day"].to_numpy()),
                    }
                )
    df = pd.DataFrame(rows)
    df["p_holm"] = np.nan
    for _, idx in df.groupby("variant").groups.items():
        df.loc[idx, "p_holm"] = holm_adjust(df.loc[idx, "wilcoxon_p"].to_numpy())
    # Section 8: the sensitivity family (every impact-multiple variant) Holm-corrected jointly.
    sens = df["variant"].str.startswith(("eval_", "both_"))
    df["p_holm_sensitivity_family"] = np.nan
    if sens.any():
        df.loc[sens, "p_holm_sensitivity_family"] = holm_adjust(
            df.loc[sens, "wilcoxon_p"].to_numpy()
        )
    df["decision"] = df.apply(decide, axis=1)
    return df


def cell_comparisons(orders: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["symbol", "variant", "size_pct", "horizon"]
    for key, group in orders.groupby(keys, sort=False):
        wide = group.pivot_table(
            index=["day", "start_hour"], columns="strategy", values="shortfall_bps"
        )
        for strategy in ("VWAP", "AC", "QUBO", "Hybrid", "DP"):
            for baseline in BASELINES:
                if strategy == baseline or strategy not in wide or baseline not in wide:
                    continue
                stats = paired_comparison(wide[strategy], wide[baseline]).as_dict()
                rows.append(
                    {
                        **dict(zip(keys, key, strict=True)),
                        "strategy": strategy,
                        "baseline": baseline,
                        **stats,
                    }
                )
    return pd.DataFrame(rows)


def strategy_summary(orders: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "shortfall_bps",
        "spread_cost_bps",
        "impact_cost_bps",
        "timing_cost_bps",
        "opportunity_cost_bps",
        "fill_rate",
        "planned_expected_cost_bps",
    ]
    rows = []
    for key, group in orders.groupby(["symbol", "variant", "strategy"], sort=False):
        row = dict(zip(["symbol", "variant", "strategy"], key, strict=True))
        row["n_orders"] = len(group)
        for m in metrics:
            values = group[m].dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue
            low, high = bootstrap_ci(values)
            row[f"{m}_mean"] = float(values.mean())
            row[f"{m}_ci_low"], row[f"{m}_ci_high"] = low, high
        row["shortfall_std"] = float(group["shortfall_bps"].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def robustness_table(orders: pd.DataFrame) -> pd.DataFrame:
    primary = orders[orders["variant"] == "primary"]
    seeds = sorted(s for s in primary["strategy"].unique() if s.startswith("QUBO_seed"))
    rows = []
    for symbol, group in primary.groupby("symbol"):
        for s in ["QUBO", *seeds]:
            for baseline in BASELINES:
                w = window_differences(group, s, baseline)
                rows.append(
                    {
                        "symbol": symbol,
                        "strategy": s,
                        "baseline": baseline,
                        "mean_diff_bps": float(w["diff"].mean()),
                    }
                )
    return pd.DataFrame(rows)


def impact_bins(config: Config, symbol: str, num_bins: int = 20) -> pd.DataFrame:
    """Binned SV / expected volume vs mean return on development bars (fig_real_impact)."""
    dev = load(config, symbol, config.dev_days)
    cal = calibrate(
        symbol,
        dev,
        smallest_order_pct_adv=min(config.sizes_pct_adv),
        min_lots=config.min_lots,
        bucket_minutes=config.bucket_minutes,
    )
    minute = np.asarray(
        pd.DatetimeIndex(dev["timestamp"]).hour * 60 + pd.DatetimeIndex(dev["timestamp"]).minute
    )
    x = dev["signed_volume"].to_numpy(dtype=float) / cal.profile.volume[minute]
    y = log_returns_bps(dev)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    edges = np.quantile(x, np.linspace(0, 1, num_bins + 1))
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, num_bins - 1)
    return pd.DataFrame(
        {
            "symbol": symbol,
            "x_mean": [x[idx == b].mean() for b in range(num_bins)],
            "return_mean_bps": [y[idx == b].mean() for b in range(num_bins)],
            "count": [int((idx == b).sum()) for b in range(num_bins)],
            "beta_bps": cal.impact.beta_bps,
        }
    )


def _job(args: tuple[Config, str, Variant]) -> dict[str, list[dict]]:
    return run_symbol_variant(*args)


def missing_data(config: Config) -> str | None:
    """Reason to skip (bar files missing), or None."""
    if config.synthetic:
        return None
    missing = [
        str(protocol.bars_path(s, d, Path(config.bars_root)))
        for s in config.symbols
        for d in (*config.dev_days, *config.eval_days)
        if not protocol.bars_path(s, d, Path(config.bars_root)).exists()
    ]
    if missing:
        return (
            f"{len(missing)} bar files missing, e.g. {missing[0]}; run `make data` first "
            "(downloads public Binance aggTrades, ~255 MB)"
        )
    return None


def run(config: Config, rec: ExperimentRecorder) -> None:
    jobs = [(config, s, v) for s in config.symbols for v in config.variants]
    if config.workers > 1:
        with ProcessPoolExecutor(max_workers=config.workers) as pool:
            outputs = list(pool.map(_job, jobs))
    else:
        outputs = [_job(j) for j in jobs]
    orders = pd.DataFrame([r for o in outputs for r in o["orders"]])
    gaps = pd.DataFrame([r for o in outputs for r in o["gaps"]])
    calibration = pd.DataFrame([r for o in outputs for r in o["calibration"]])
    calibration = calibration[calibration["variant"] == "primary"].drop(columns="variant")

    core = orders[~orders["strategy"].str.startswith("QUBO_seed")]
    primary = comparison_family(core, PRIMARY_STRATEGIES, BASELINES)
    secondary = comparison_family(core, ("AC", "VWAP", "DP"), ("TWAP", "VWAP"))

    rec.write_table("orders", orders)
    rec.write_table("calibration", calibration)
    rec.write_table("gaps", gaps)
    rec.write_table("comparisons", primary)
    rec.write_table("comparisons_secondary", secondary)
    rec.write_table("cells", cell_comparisons(core))
    rec.write_table("strategy_summary", strategy_summary(core))
    rec.write_table("robustness", robustness_table(orders))
    if config.split == "dev":
        rec.write_table("impact_bins", pd.concat([impact_bins(config, s) for s in config.symbols]))
        profiles = []
        for symbol in config.symbols:
            dev = load(config, symbol, config.dev_days)
            cal = calibrate(
                symbol,
                dev,
                smallest_order_pct_adv=min(config.sizes_pct_adv),
                min_lots=config.min_lots,
                bucket_minutes=config.bucket_minutes,
            )
            profiles.append(cal.profile.as_frame().assign(symbol=symbol))
        rec.write_table("profiles", pd.concat(profiles))
    rec.note("data", "synthetic bars (quick mode)" if config.synthetic else "Binance aggTrades")
    rec.note("eval_days", list(config.eval_days))


def _seeds(config: Config) -> list[int]:
    return single_seed(config)


DEV = Experiment(
    "real_data_dev",
    Config(),
    Config(**QUICK_BASE, eval_days=QUICK_BASE["dev_days"]),  # type: ignore[arg-type]
    run,
    _seeds,
    description="Real-data evaluation on development days (in-sample)",
    precheck=missing_data,
)
TEST = Experiment(
    "real_data_test",
    Config(split="test", eval_days=protocol.TEST_DAYS),
    Config(**QUICK_BASE, split="test", eval_days=("2026-08-07",)),  # type: ignore[arg-type]
    run,
    _seeds,
    description="Real-data evaluation on held-out test days (run once per protocol)",
    precheck=missing_data,
)
