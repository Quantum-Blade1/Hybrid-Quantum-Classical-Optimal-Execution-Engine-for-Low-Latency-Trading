from __future__ import annotations

import argparse
import dataclasses
import logging
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from qexec.analysis.statistics import paired_comparison, summarize
from qexec.experiment import ExperimentRecorder

DEFAULT_RESULTS_DIR = Path("results")
QUICK_RESULTS_DIR = Path("build/quick/results")


class SkipExperiment(Exception):  # noqa: N818 - a control-flow signal, not an error
    """Raised before any output is written when an experiment's inputs are unavailable."""


@dataclass(frozen=True)
class Experiment:
    name: str
    full: Any
    quick: Any
    run: Callable[[Any, ExperimentRecorder], None]
    seeds: Callable[[Any], list[int]]
    description: str = ""
    precheck: Callable[[Any], str | None] | None = None

    def execute(self, *, quick: bool, results_dir: Path, seed: int | None = None) -> float:
        """Run with the full or quick config (optionally re-seeded); returns wall time."""
        config = self.quick if quick else self.full
        if seed is not None:
            config = dataclasses.replace(config, seed=seed)
        if self.precheck is not None:
            reason = self.precheck(config)
            if reason:
                raise SkipExperiment(f"{self.name}: skipped ({reason})")
        start = time.perf_counter()
        with ExperimentRecorder(
            self.name, config, self.seeds(config), root=results_dir, extra={"quick": quick}
        ) as rec:
            self.run(config, rec)
        return time.perf_counter() - start

    def main(self, argv: Sequence[str] | None = None) -> None:
        parser = argparse.ArgumentParser(description=self.description)
        parser.add_argument("--quick", action="store_true", help="tiny sizes (CI smoke run)")
        parser.add_argument("--results-dir", type=Path, default=None)
        parser.add_argument("--seed", type=int, default=None, help="override the base seed")
        args = parser.parse_args(argv)
        logging.basicConfig(level=logging.WARNING)
        results_dir = args.results_dir or (QUICK_RESULTS_DIR if args.quick else DEFAULT_RESULTS_DIR)
        try:
            elapsed = self.execute(quick=args.quick, results_dir=results_dir, seed=args.seed)
        except SkipExperiment as exc:
            print(exc)
            return
        print(f"{self.name}: {elapsed:.1f}s -> {results_dir / self.name}/")


def seed_range(config: Any) -> list[int]:
    return list(range(config.seed, config.seed + config.num_seeds))


def single_seed(config: Any) -> list[int]:
    return [config.seed]


def summarize_groups(
    df: pd.DataFrame, group_cols: Sequence[str], value_cols: Iterable[str]
) -> pd.DataFrame:
    """One row per (group, metric): mean, std, 95% bootstrap CI, median, count."""
    rows = []
    for key, group in df.groupby(list(group_cols), sort=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        for col in value_cols:
            values = group[col].dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue
            stats = summarize(values).as_dict()
            stats["count"] = stats.pop("n")
            rows.append({**dict(zip(group_cols, key_tuple, strict=True)), "metric": col, **stats})
    return pd.DataFrame(rows)


def paired_vs_baselines(
    df: pd.DataFrame,
    *,
    unit_col: str,
    strategy_col: str,
    value_col: str,
    baselines: Sequence[str],
    group_cols: Sequence[str] = (),
) -> pd.DataFrame:
    """Paired differences (strategy - baseline) of `value_col`, matched on `unit_col`."""
    rows = []
    groups = df.groupby(list(group_cols), sort=False) if group_cols else [((), df)]
    for key, group in groups:
        key_tuple = key if isinstance(key, tuple) else (key,)
        wide = group.pivot_table(index=unit_col, columns=strategy_col, values=value_col)
        for baseline in baselines:
            if baseline not in wide:
                continue
            for strategy in wide.columns:
                if strategy == baseline:
                    continue
                pair = wide[[strategy, baseline]].dropna()
                if pair.empty:
                    continue
                stats = paired_comparison(pair[strategy], pair[baseline]).as_dict()
                stats["count"] = stats.pop("n")
                rows.append(
                    {
                        **dict(zip(group_cols, key_tuple, strict=True)),
                        "strategy": strategy,
                        "baseline": baseline,
                        "metric": value_col,
                        **stats,
                    }
                )
    return pd.DataFrame(rows)
