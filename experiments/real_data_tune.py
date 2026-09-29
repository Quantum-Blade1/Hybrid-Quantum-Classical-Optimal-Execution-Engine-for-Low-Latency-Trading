"""Tuning of the QUBO and hybrid hyperparameters on the development days only
(docs/PROTOCOL.md §5). No held-out day is read.

Stage 1 (model only, no execution): for each candidate QUBO setting (slices, bits, units,
SA sweeps/restarts) and every development cell (symbol x start hour x size x horizon, at
lambda = 0 and impact x1), solve the binary QUBO with SA and record its cost-model
objective against the discretized AC optimum and the exact integer optimum (DP), plus SA
time. Rule, fixed before looking at the numbers: among settings whose SA solution equals
the DP optimum (within 1e-6 bps) in at least 95% of cells and whose median SA time is at
most 1 s, choose the smallest mean objective gap to AC; ties go to fewer variables.

Stage 2 (development-day execution): with the chosen QUBO setting, the hybrid's number of
re-planning checkpoints in {1, 3, 6} and ratio clip in {(0.5, 2), (0.8, 1.25)} are chosen
by the lowest mean realized shortfall over all development orders (both symbols, primary
variant). The chosen values are frozen into `experiments/real_data.py` (FROZEN_*).

Usage:
    python -m experiments.real_data_tune [--quick] [--results-dir results]
"""

from __future__ import annotations

import dataclasses
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from experiments import protocol
from experiments.common import Experiment, single_seed
from experiments.real_data import (
    QUICK_BASE,
    Cell,
    evaluation_frame,
    execute,
    load,
    missing_data,
    order_seed,
)
from experiments.real_data import Config as EvalConfig
from qexec.execution.engine import OrderSide
from qexec.execution.fill import ImpactFillModel
from qexec.execution.strategies.fixed import FixedScheduleStrategy
from qexec.execution.strategies.model_based import (
    AdaptiveQUBOStrategy,
    QUBOSettings,
    ac_schedule,
    qubo_plan,
)
from qexec.experiment import ExperimentRecorder
from qexec.market.calibration import Calibration, calibrate

GAP_TOL_BPS = 1e-6
MIN_OPTIMAL_SHARE = 0.95
MAX_MEDIAN_TIME_S = 1.0

QUBO_CANDIDATES = (
    QUBOSettings(num_slices=4, bits=3, units=16),
    QUBOSettings(num_slices=5, bits=3, units=20),
    QUBOSettings(num_slices=6, bits=3, units=24),
    QUBOSettings(num_slices=8, bits=3, units=32),
    QUBOSettings(num_slices=5, bits=4, units=40),
    QUBOSettings(num_slices=8, bits=4, units=64),
    QUBOSettings(num_slices=8, bits=4, units=64, sweeps=2000, restarts=32),
)
HYBRID_CANDIDATES = ((1, (0.5, 2.0)), (3, (0.5, 2.0)), (6, (0.5, 2.0)), (3, (0.8, 1.25)))


@dataclass(frozen=True)
class Config:
    eval: EvalConfig = field(default_factory=EvalConfig)
    qubo_candidates: tuple[QUBOSettings, ...] = QUBO_CANDIDATES
    hybrid_candidates: tuple[tuple[int, tuple[float, float]], ...] = HYBRID_CANDIDATES
    workers: int = min(10, os.cpu_count() or 1)
    seed: int = 0


QUICK = Config(
    eval=EvalConfig(**QUICK_BASE, eval_days=QUICK_BASE["dev_days"]),  # type: ignore[arg-type]
    qubo_candidates=(
        QUBOSettings(num_slices=4, bits=3, units=12, sweeps=200, restarts=4),
        QUBOSettings(num_slices=3, bits=3, units=12, sweeps=200, restarts=4),
    ),
    hybrid_candidates=((1, (0.5, 2.0)), (3, (0.5, 2.0))),
    workers=1,
)


def _calibration(config: EvalConfig, symbol: str) -> tuple[Calibration, pd.DataFrame]:
    dev = load(config, symbol, config.dev_days)
    cal = calibrate(
        symbol,
        dev,
        smallest_order_pct_adv=min(config.sizes_pct_adv),
        min_lots=config.min_lots,
        bucket_minutes=config.bucket_minutes,
    )
    return cal, dev


def _cells(config: EvalConfig) -> list[Cell]:
    return [Cell(p, h) for p in config.sizes_pct_adv for h in config.horizons]


def qubo_rows(args: tuple[EvalConfig, str, int, QUBOSettings]) -> list[dict[str, Any]]:
    config, symbol, index, settings = args
    cal, _ = _calibration(config, symbol)
    rows = []
    for hour in config.start_hours:
        for cell in _cells(config):
            total = round(cal.adv * cell.size_pct / 100 / cal.lot_size)
            model = cal.cost_model(hour * 60, cell.horizon)
            plan = qubo_plan(model, total, settings, order_seed(symbol, hour, cell, "tune"))
            rows.append(
                {
                    "candidate": index,
                    **dataclasses.asdict(settings),
                    "num_variables": min(settings.num_slices, cell.horizon) * settings.bits,
                    "symbol": symbol,
                    "start_hour": hour,
                    "size_pct": cell.size_pct,
                    "horizon": cell.horizon,
                    "objective_qubo": plan.objective_bps,
                    "objective_dp": plan.optimal_objective_bps,
                    "objective_ac": model.objective(ac_schedule(model, total), total),
                    "feasible": plan.feasible,
                    "sa_time_s": plan.solve_time_s,
                }
            )
    return rows


def select_qubo(rows: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    rows = rows.assign(
        gap_to_ac=rows["objective_qubo"] - rows["objective_ac"],
        at_dp=(rows["objective_qubo"] - rows["objective_dp"]) <= GAP_TOL_BPS,
    )
    summary = rows.groupby("candidate", as_index=False).agg(
        num_slices=("num_slices", "first"),
        bits=("bits", "first"),
        units=("units", "first"),
        sweeps=("sweeps", "first"),
        restarts=("restarts", "first"),
        max_variables=("num_variables", "max"),
        mean_gap_to_ac_bps=("gap_to_ac", "mean"),
        max_gap_to_ac_bps=("gap_to_ac", "max"),
        share_at_dp=("at_dp", "mean"),
        share_feasible=("feasible", "mean"),
        median_sa_time_s=("sa_time_s", "median"),
    )
    summary["eligible"] = (summary["share_at_dp"] >= MIN_OPTIMAL_SHARE) & (
        summary["median_sa_time_s"] <= MAX_MEDIAN_TIME_S
    )
    pool = summary[summary["eligible"]] if summary["eligible"].any() else summary
    best = pool.sort_values(["mean_gap_to_ac_bps", "max_variables"]).iloc[0]
    summary["selected"] = summary["candidate"] == best["candidate"]
    return summary, int(best["candidate"])


def hybrid_rows(
    args: tuple[EvalConfig, str, int, int, tuple[float, float]],
) -> list[dict[str, Any]]:
    config, symbol, index, checkpoints, clip = args
    cal, dev = _calibration(config, symbol)
    frame = evaluation_frame(dev, cal)
    fill_model = ImpactFillModel(cal.impact.beta_bps, config.participation_cap)
    rows = []
    for day, group in frame.groupby("day", sort=True):
        bars = group.reset_index(drop=True)
        for hour in config.start_hours:
            side = OrderSide.BUY if hour in protocol.BUY_HOURS else OrderSide.SELL
            for cell in _cells(config):
                total = round(cal.adv * cell.size_pct / 100 / cal.lot_size)
                model = cal.cost_model(hour * 60, cell.horizon)
                window = bars.iloc[hour * 60 : hour * 60 + cell.horizon].reset_index(drop=True)
                seed = order_seed(symbol, day, hour, cell.size_pct, cell.horizon)
                t0 = perf_counter()
                hybrid = AdaptiveQUBOStrategy(
                    model, config.qubo, seed=seed, num_checkpoints=checkpoints, clip=clip
                )
                metrics = execute(hybrid, window, total, side, fill_model)
                elapsed = perf_counter() - t0
                twap = execute(
                    FixedScheduleStrategy(np.ones(cell.horizon)), window, total, side, fill_model
                )
                rows.append(
                    {
                        "candidate": index,
                        "checkpoints": checkpoints,
                        "clip_low": clip[0],
                        "clip_high": clip[1],
                        "symbol": symbol,
                        "day": day,
                        "start_hour": hour,
                        "size_pct": cell.size_pct,
                        "horizon": cell.horizon,
                        "shortfall_bps": metrics["shortfall_bps"],
                        "minus_twap_bps": metrics["shortfall_bps"] - twap["shortfall_bps"],
                        "wall_time_s": elapsed,
                    }
                )
    return rows


def _map(fn: Any, jobs: list[Any], workers: int) -> list[Any]:
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(fn, jobs))
    return [fn(j) for j in jobs]


def run(config: Config, rec: ExperimentRecorder) -> None:
    base = config.eval
    qjobs = [(base, s, i, q) for i, q in enumerate(config.qubo_candidates) for s in base.symbols]
    qrows = pd.DataFrame([r for rows in _map(qubo_rows, qjobs, config.workers) for r in rows])
    qsummary, chosen = select_qubo(qrows)
    settings = config.qubo_candidates[chosen]
    tuned = dataclasses.replace(base, qubo=settings)
    hjobs = [
        (tuned, s, i, k, clip)
        for i, (k, clip) in enumerate(config.hybrid_candidates)
        for s in base.symbols
    ]
    hrows = pd.DataFrame([r for rows in _map(hybrid_rows, hjobs, config.workers) for r in rows])
    hsummary = hrows.groupby(
        ["candidate", "checkpoints", "clip_low", "clip_high"], as_index=False
    ).agg(
        n_orders=("shortfall_bps", "size"),
        mean_shortfall_bps=("shortfall_bps", "mean"),
        mean_minus_twap_bps=("minus_twap_bps", "mean"),
        mean_wall_time_s=("wall_time_s", "mean"),
    )
    best = hsummary.sort_values(["mean_shortfall_bps", "checkpoints"]).iloc[0]
    hsummary["selected"] = hsummary["candidate"] == best["candidate"]
    per_symbol = hrows.groupby(["candidate", "symbol"], as_index=False)[
        ["shortfall_bps", "minus_twap_bps"]
    ].mean()

    rec.write_table("qubo_cells", qrows)
    rec.write_table("qubo_candidates", qsummary)
    rec.write_table("hybrid_candidates", hsummary)
    rec.write_table("hybrid_by_symbol", per_symbol)
    rec.note("dev_days", list(base.dev_days))
    rec.note("selected_qubo", dataclasses.asdict(settings))
    rec.note(
        "selected_hybrid",
        {
            "checkpoints": int(best["checkpoints"]),
            "clip": [float(best["clip_low"]), float(best["clip_high"])],
        },
    )
    rec.note("data", "synthetic bars (quick mode)" if base.synthetic else "Binance aggTrades")


def _precheck(config: Config) -> str | None:
    return missing_data(config.eval)


EXPERIMENT = Experiment(
    "real_data_tune",
    Config(),
    QUICK,
    run,
    single_seed,
    description="QUBO and hybrid hyperparameters tuned on development days only",
    precheck=_precheck,
)

if __name__ == "__main__":
    EXPERIMENT.main()
