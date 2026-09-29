"""Write every number quoted in paper/ieee/main.tex as a LaTeX macro, from results/ only.

The manuscript never types a result by hand: it uses the macros in
`paper/ieee/numbers.tex` (e.g. `\\BtcQuboTwapMean`). This script reads the committed
tables under `results/`, derives the few statistics the paper needs that no table stores
(paired QAOA-vs-uniform comparisons, recomputed with the same seeded bootstrap and
Wilcoxon test as the experiments), and writes the macro file deterministically.

Usage:
    python -m experiments.paper_numbers [--results-dir results] [--output paper/ieee/numbers.tex]
                                        [--check]

`--check` regenerates the file in memory and exits non-zero if it differs from the
committed one (`make check-paper`).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qexec.analysis.statistics import paired_comparison

DEFAULT_OUTPUT = Path("paper/ieee/numbers.tex")
SYMBOL_NAMES = {"BTCUSDT": "Btc", "LINKUSDT": "Link"}
STRATEGY_NAMES = {
    "TWAP": "Twap",
    "VWAP": "Vwap",
    "AC": "Ac",
    "QUBO": "Qubo",
    "Hybrid": "Hybrid",
    "DP": "Dp",
    "SA-QUBO": "Saqubo",
}
_DIGITS = ("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine")


def spell(n: int) -> str:
    """Digits as words, for macro names (LaTeX control words cannot contain digits)."""
    return "".join(_DIGITS[int(d)] for d in str(int(n)))


def fmt(x: float, nd: int = 2, *, sign: bool = False) -> str:
    """Fixed-point number for text; `sign` forces a leading + (and maps -0.00 to 0.00)."""
    value = round(float(x), nd)
    if value == 0:
        value = 0.0
    text = f"{value:+.{nd}f}" if sign else f"{value:.{nd}f}"
    if sign and value == 0:
        text = f"{0:.{nd}f}"
    return text.replace("-", "$-$") if text.startswith("-") else text


def fmt_int(x: float) -> str:
    return f"{round(float(x)):,}".replace(",", "{,}")


def fmt_p(p: float) -> str:
    if p < 1e-4:
        return "$<$10$^{-4}$"
    if p >= 0.995:
        return "1.00"
    return f"{p:.2f}" if p >= 0.01 else f"{p:.3f}"


def fmt_ci(low: float, high: float, nd: int = 2) -> str:
    return f"[{fmt(low, nd)}, {fmt(high, nd)}]"


class Macros:
    def __init__(self) -> None:
        self._items: dict[str, str] = {}

    def __setitem__(self, name: str, value: str) -> None:
        if not name.isalpha():
            raise ValueError(f"macro name must be letters only: {name}")
        if name in self._items:
            raise ValueError(f"duplicate macro {name}")
        self._items[name] = value

    def render(self, header: str) -> str:
        lines = [header]
        lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in self._items.items()]
        return "\n".join(lines) + "\n"


class Tables:
    def __init__(self, root: Path) -> None:
        self.root = root

    def csv(self, experiment: str, name: str) -> pd.DataFrame:
        return pd.read_csv(self.root / experiment / f"{name}.csv")

    def json(self, experiment: str, name: str) -> Any:
        return json.loads((self.root / experiment / f"{name}.json").read_text())

    def has(self, experiment: str, name: str) -> bool:
        return (self.root / experiment / name).exists()


# --------------------------------------------------------------------------------------
# Real data
# --------------------------------------------------------------------------------------

PRIMARY_ORDER = [
    ("QUBO", "TWAP"),
    ("QUBO", "VWAP"),
    ("QUBO", "AC"),
    ("Hybrid", "TWAP"),
    ("Hybrid", "VWAP"),
    ("Hybrid", "AC"),
]
MAIN_STRATEGIES = ("TWAP", "VWAP", "AC", "QUBO", "Hybrid")


def real_data(t: Tables, m: Macros) -> None:  # noqa: PLR0915 - one flat list of numbers
    cal = t.csv("real_data_test", "calibration").set_index("symbol")
    for symbol, s in SYMBOL_NAMES.items():
        c = cal.loc[symbol]
        m[f"{s}Adv"] = fmt_int(c["adv_base"])
        m[f"{s}Lot"] = f"{c['lot_size']:g}"
        m[f"{s}Beta"] = fmt(c["beta_bps"])
        m[f"{s}BetaSe"] = fmt(c["beta_std_error"])
        m[f"{s}BetaRsq"] = fmt(c["beta_r_squared"])
        m[f"{s}HalfSpread"] = (
            fmt(c["mean_half_spread_bps"], 4)
            if c["mean_half_spread_bps"] < 0.01
            else fmt(c["mean_half_spread_bps"])
        )
        m[f"{s}Sigma"] = fmt(c["mean_sigma_bps"])
        m[f"{s}VolPerMin"] = fmt(c["mean_volume_per_minute"], 1)
        m[f"{s}ImpactBars"] = fmt_int(c["impact_num_bars"])
    m["LinkAdvMillions"] = fmt(cal.loc["LINKUSDT", "adv_base"] / 1e6, 3)
    m["NumDevDays"] = str(int(cal["num_days"].iloc[0]))

    test_manifest = t.json("real_data_test", "manifest")
    m["NumTestDays"] = str(len(test_manifest["config"]["eval_days"]))
    m["TestFirstDay"] = test_manifest["config"]["eval_days"][0]
    m["TestLastDay"] = test_manifest["config"]["eval_days"][-1]
    m["DevFirstDay"] = test_manifest["config"]["dev_days"][0]
    m["DevLastDay"] = test_manifest["config"]["dev_days"][-1]
    m["TestCommit"] = test_manifest["git"]["commit"][:7]
    m["TestCommitClean"] = "clean" if not test_manifest["git"]["dirty"] else "dirty"
    m["TestWallTime"] = fmt_int(test_manifest["wall_time_s"])

    comp = t.csv("real_data_test", "comparisons")
    primary = comp[comp["variant"] == "primary"].set_index(["symbol", "strategy", "baseline"])
    m["NumTestWindows"] = str(int(primary["n_windows"].iloc[0]))
    m["NumPrimaryComparisons"] = str(len(primary))
    m["PrimaryMinHolm"] = fmt_p(primary["p_holm"].min())
    m["PrimaryMinRawP"] = fmt_p(primary["wilcoxon_p"].min())
    m["PrimaryMinMean"] = fmt(primary["mean_diff_bps"].min(), sign=True)
    m["PrimaryMaxMean"] = fmt(primary["mean_diff_bps"].max(), sign=True)
    m["PrimaryNumSignificant"] = str(int((primary["p_holm"] < 0.05).sum()))
    btc = primary.loc["BTCUSDT"]
    m["BtcMaxAbsCi"] = fmt(max(btc["ci_low"].abs().max(), btc["ci_high"].abs().max()))
    m["BtcNumEquivalent"] = str(int(btc["decision"].str.contains("equivalent").sum()))
    link = primary.loc["LINKUSDT"]
    m["LinkMeanCiHalfWidth"] = fmt(((link["ci_high"] - link["ci_low"]) / 2).mean())

    rows = []
    for symbol, s in SYMBOL_NAMES.items():
        for k, (strategy, baseline) in enumerate(PRIMARY_ORDER):
            r = primary.loc[(symbol, strategy, baseline)]
            key = f"{s}{STRATEGY_NAMES[strategy]}{STRATEGY_NAMES[baseline]}"
            m[f"{key}Mean"] = fmt(r["mean_diff_bps"], sign=True)
            m[f"{key}Ci"] = fmt_ci(r["ci_low"], r["ci_high"])
            m[f"{key}P"] = fmt_p(r["wilcoxon_p"])
            m[f"{key}Holm"] = fmt_p(r["p_holm"])
            first = f"\\multirow{{6}}{{*}}{{{symbol[:-4]}}}" if k == 0 else ""
            rows.append(
                f"{first} & {strategy} $-$ {baseline} & {fmt(r['mean_diff_bps'], sign=True)} & "
                f"{fmt_ci(r['ci_low'], r['ci_high'])} & "
                f"{fmt_ci(r['day_cluster_ci_low'], r['day_cluster_ci_high'])} & "
                f"{fmt_p(r['wilcoxon_p'])} & {fmt_p(r['p_holm'])} \\\\"
            )
        rows.append("\\midrule")
    m["PrimaryTableRows"] = "\n".join(rows[:-1])

    # Sensitivity to the impact coefficient (section 8 of the protocol).
    sens = comp[comp["variant"].str.startswith(("eval_", "both_"))]
    m["SensMinMean"] = fmt(sens["mean_diff_bps"].min(), sign=True)
    m["SensMaxMean"] = fmt(sens["mean_diff_bps"].max(), sign=True)
    m["SensMinJointHolm"] = fmt_p(sens["p_holm_sensitivity_family"].min())
    m["SensNumComparisons"] = str(len(sens))
    excl = sens[(sens["ci_low"] > 0) | (sens["ci_high"] < 0)]
    m["SensNumCiExcludeZero"] = str(len(excl))
    if len(excl):
        worst = excl.loc[excl["mean_diff_bps"].abs().idxmax()]
        m["SensExclMean"] = fmt(worst["mean_diff_bps"], sign=True)
        m["SensExclCi"] = fmt_ci(worst["ci_low"], worst["ci_high"])

    risk = comp[comp["variant"] == "risk_averse"].set_index(["symbol", "strategy", "baseline"])
    r = risk.loc[("LINKUSDT", "QUBO", "TWAP")]
    m["RiskLinkQuboTwapMean"] = fmt(r["mean_diff_bps"], 1, sign=True)
    m["RiskLinkQuboTwapCi"] = fmt_ci(r["ci_low"], r["ci_high"], 1)
    r = risk.loc[("LINKUSDT", "QUBO", "AC")]
    m["RiskLinkQuboAcMean"] = fmt(r["mean_diff_bps"], 1, sign=True)
    m["RiskLinkQuboAcCi"] = fmt_ci(r["ci_low"], r["ci_high"], 1)
    m["RiskMinHolm"] = fmt_p(risk["p_holm"].min())

    secondary = t.csv("real_data_test", "comparisons_secondary")
    secondary = secondary[secondary["variant"] == "primary"].set_index(
        ["symbol", "strategy", "baseline"]
    )
    for symbol, s in SYMBOL_NAMES.items():
        r = secondary.loc[(symbol, "AC", "TWAP")]
        m[f"{s}AcTwapMean"] = fmt(r["mean_diff_bps"], sign=True)
        m[f"{s}AcTwapCi"] = fmt_ci(r["ci_low"], r["ci_high"])

    summary = t.csv("real_data_test", "strategy_summary")
    summary = summary[summary["variant"] == "primary"].set_index(["symbol", "strategy"])
    m["NumTestOrders"] = str(int(summary["n_orders"].iloc[0]))
    comp_rows = []
    for symbol, s in SYMBOL_NAMES.items():
        sub = summary.loc[symbol].loc[list(MAIN_STRATEGIES)]
        controllable = sub["spread_cost_bps_mean"] + sub["impact_cost_bps_mean"]
        m[f"{s}SpreadImpact"] = fmt(controllable.mean())
        m[f"{s}StdMin"] = fmt(sub["shortfall_std"].min(), 0)
        m[f"{s}StdMax"] = fmt(sub["shortfall_std"].max(), 0)
        m[f"{s}FillMin"] = fmt(100 * sub["fill_rate_mean"].min(), 0)
        m[f"{s}FillMax"] = fmt(100 * sub["fill_rate_mean"].max(), 0)
        m[f"{s}MeanShortfallMin"] = fmt(sub["shortfall_bps_mean"].min())
        m[f"{s}MeanShortfallMax"] = fmt(sub["shortfall_bps_mean"].max())
        m[f"{s}OppMin"] = fmt(sub["opportunity_cost_bps_mean"].min())
        m[f"{s}OppMax"] = fmt(sub["opportunity_cost_bps_mean"].max())
        m[f"{s}TimingMin"] = fmt(sub["timing_cost_bps_mean"].min())
        m[f"{s}TimingMax"] = fmt(sub["timing_cost_bps_mean"].max())
        m[f"{s}ModelCostMin"] = fmt(sub["planned_expected_cost_bps_mean"].min())
        m[f"{s}ModelCostMax"] = fmt(sub["planned_expected_cost_bps_mean"].max())
        for k, strategy in enumerate(MAIN_STRATEGIES):
            r = sub.loc[strategy]
            m[f"{s}{STRATEGY_NAMES[strategy]}Shortfall"] = fmt(r["shortfall_bps_mean"])
            first = f"\\multirow{{5}}{{*}}{{{symbol[:-4]}}}" if k == 0 else ""
            comp_rows.append(
                f"{first} & {strategy} & {fmt(r['shortfall_bps_mean'])} & "
                f"{fmt(r['spread_cost_bps_mean'] + r['impact_cost_bps_mean'])} & "
                f"{fmt(r['timing_cost_bps_mean'])} & {fmt(r['opportunity_cost_bps_mean'])} & "
                f"{fmt(100 * r['fill_rate_mean'], 1)} & {fmt(r['shortfall_std'], 1)} \\\\"
            )
        comp_rows.append("\\midrule")
    m["CostTableRows"] = "\n".join(comp_rows[:-1])

    gaps = t.csv("real_data_test", "gaps")
    gaps = gaps[gaps["variant"] == "primary"]
    m["QuboAtDpShare"] = fmt(100 * (gaps["qubo_gap_to_dp_bps"].abs() < 1e-9).mean(), 0)
    m["QuboFeasibleShare"] = fmt(100 * gaps["qubo_feasible"].mean(), 0)
    m["QuboSolveTimeMedian"] = fmt(gaps["qubo_solve_time_s"].median())
    for symbol, s in SYMBOL_NAMES.items():
        g = gaps[gaps["symbol"] == symbol]
        gap = g["objective_QUBO"] - g["objective_AC"]
        m[f"{s}QuboAcGapMean"] = fmt(gap.mean(), 4 if gap.mean() < 0.01 else 3)
        m[f"{s}QuboAcGapMax"] = fmt(gap.max(), 4 if gap.max() < 0.01 else 2)
        twap = g["objective_TWAP"] - g["objective_AC"]
        m[f"{s}TwapAcGapMean"] = fmt(twap.mean(), 4 if twap.mean() < 0.01 else 3)
        m[f"{s}AcModelCost"] = fmt(g["objective_AC"].mean(), 3)

    robustness = t.csv("real_data_test", "robustness")
    seeded = robustness[robustness["strategy"].str.contains("_seed")]
    spread = seeded.groupby(["symbol", "baseline"])["mean_diff_bps"].agg(np.ptp)
    m["RobustnessSpread"] = fmt(spread.max(), 3)
    m["RobustnessSeeds"] = str(seeded["strategy"].nunique())

    # Development days (tuning and in-sample evaluation).
    cand = t.csv("real_data_tune", "qubo_candidates")
    sel = cand[cand["selected"]].iloc[0]
    m["TuneNumCandidates"] = str(len(cand))
    m["TuneSlices"] = str(int(sel["num_slices"]))
    m["TuneBits"] = str(int(sel["bits"]))
    m["TuneUnits"] = str(int(sel["units"]))
    m["TuneVariables"] = str(int(sel["max_variables"]))
    m["TuneSweeps"] = fmt_int(sel["sweeps"])
    m["TuneRestarts"] = str(int(sel["restarts"]))
    m["TuneGapMean"] = fmt(sel["mean_gap_to_ac_bps"], 3)
    m["TuneGapMax"] = fmt(sel["max_gap_to_ac_bps"])
    large = cand[cand["max_variables"] > sel["max_variables"] + 3]
    m["TuneLargeMinVars"] = str(int(large["max_variables"].min()))
    m["TuneLargeMaxVars"] = str(int(large["max_variables"].max()))
    m["TuneLargeDpMin"] = fmt(100 * large["share_at_dp"].min(), 0)
    m["TuneLargeDpMax"] = fmt(100 * large["share_at_dp"].max(), 0)
    hyb = t.csv("real_data_tune", "hybrid_candidates")
    hsel = hyb[hyb["selected"]].iloc[0]
    m["TuneCheckpoints"] = str(int(hsel["checkpoints"]))
    m["TuneHybridMinusTwapMin"] = fmt(hyb["mean_minus_twap_bps"].min())
    m["TuneHybridMinusTwapMax"] = fmt(hyb["mean_minus_twap_bps"].max())

    dev = t.csv("real_data_dev", "comparisons")
    dev = dev[dev["variant"] == "primary"]
    m["NumDevWindows"] = str(int(dev["n_windows"].iloc[0]))
    m["DevMinHolm"] = fmt_p(dev["p_holm"].min())
    m["DevMinMean"] = fmt(dev["mean_diff_bps"].min(), sign=True)
    m["DevMaxMean"] = fmt(dev["mean_diff_bps"].max(), sign=True)


# --------------------------------------------------------------------------------------
# Synthetic strategy experiments
# --------------------------------------------------------------------------------------


def _paired_row(df: pd.DataFrame, **where: Any) -> pd.Series:
    d = df[df["metric"] == where.pop("metric", "shortfall_bps")]
    for column, value in where.items():
        d = d[d[column] == value]
    if len(d) != 1:
        raise ValueError(f"expected one row for {where}, got {len(d)}")
    return d.iloc[0]


def synthetic(t: Tables, m: Macros) -> None:
    def put(prefix: str, r: pd.Series, nd: int = 1) -> None:
        m[f"{prefix}Mean"] = fmt(r["mean_diff"], nd, sign=True)
        m[f"{prefix}Ci"] = fmt_ci(r["ci_low"], r["ci_high"], nd)
        m[f"{prefix}P"] = fmt_p(r["wilcoxon_p"])

    is_paired = t.csv("is_comparison", "paired")
    put("SynHourHybridTwap", _paired_row(is_paired, strategy="Hybrid", baseline="TWAP"))
    put("SynHourQuboTwap", _paired_row(is_paired, strategy="SA-QUBO", baseline="TWAP"))
    m["SynSeeds"] = str(int(_paired_row(is_paired, strategy="Hybrid", baseline="TWAP")["count"]))

    day = t.csv("strategy_comparison", "paired")
    fracs = sorted(day["order_fraction_of_adv"].unique())
    m["SynDaySizes"] = ", ".join(f"{100 * f:g}\\%" for f in fracs)
    one_pct = min(fracs, key=lambda f: abs(f - 0.01))
    for strategy, name in (("Hybrid", "Hybrid"), ("SA-QUBO", "Qubo")):
        r = _paired_row(
            day,
            metric="impact_cost_bps",
            strategy=strategy,
            baseline="TWAP",
            order_fraction_of_adv=one_pct,
        )
        put(f"SynDayImpact{name}Twap", r, 2)
    diffs = day[(day["metric"] == "shortfall_bps") & (day["baseline"] == "TWAP")]
    diffs = diffs[diffs["strategy"].isin(["Hybrid", "SA-QUBO"])]
    m["SynDayNumCiExcludeZero"] = str(int(diffs["ci_excludes_zero"].sum()))

    stress = t.csv("stress_test", "paired")
    r = _paired_row(stress, scenario="Market Outage", strategy="Hybrid", baseline="VWAP")
    put("SynOutageHybridVwap", r)
    s = stress[(stress["metric"] == "shortfall_bps") & (stress["baseline"] == "TWAP")]
    s = s[s["strategy"].isin(["Hybrid", "SA-QUBO"])]
    m["SynStressNumCiExcludeZero"] = str(int(s["ci_excludes_zero"].sum()))

    wf = t.csv("walk_forward", "paired")
    put("SynWfHybridTwap", _paired_row(wf, strategy="Hybrid", baseline="TWAP"))


# --------------------------------------------------------------------------------------
# Solvers, QAOA and formulation checks
# --------------------------------------------------------------------------------------

QAOA_FAMILIES = {"toy": "Toy", "random": "Random", "slice": "Slice", "fig07": "Instance"}
QAOA_SOLVERS = {"QAOA_Ideal": "Ideal", "QAOA_Noisy": "Noisy"}


def solvers(t: Tables, m: Macros) -> None:  # noqa: PLR0915 - one flat list of numbers
    summary = t.csv("solver_benchmark", "summary")
    opt = summary[summary["metric"] == "optimal"]
    for family, name in (("slice", "Slice"), ("toy", "Toy")):
        sa = opt[(opt["family"] == family) & (opt["solver"] == "SA")].set_index("n")["mean"]
        full = sa[sa >= 1.0 - 1e-12]
        # Largest n up to which SA is optimal in every seed at every size.
        n_all = max((n for n in sa.index if (sa.loc[:n] >= 1.0 - 1e-12).all()), default=0)
        m[f"Sa{name}AllOptimalUpTo"] = str(int(n_all))
        m[f"Sa{name}NumSizesAllOptimal"] = str(len(full))
        above = sa[sa.index > n_all]
        m[f"Sa{name}AboveMin"] = fmt(100 * above.min(), 0) if len(above) else "100"
        m[f"Sa{name}AboveMax"] = fmt(100 * above.max(), 0) if len(above) else "100"
        greedy = opt[(opt["family"] == family) & (opt["solver"] == "Greedy")]["mean"]
        m[f"Greedy{name}Max"] = fmt(100 * greedy.max(), 0)
    m["SolverMaxN"] = str(int(opt["n"].max()))
    m["SolverSeeds"] = str(int(opt["count"].max()))
    times = summary[summary["metric"] == "time_s"]
    sa_t = times[times["solver"] == "SA"]["mean"]
    m["SaTimeMin"] = fmt(sa_t.min())
    m["SaTimeMax"] = fmt(sa_t.max())
    bf = times[times["solver"] == "BruteForce"]
    m["BruteForceTimeMaxN"] = fmt(bf[bf["n"] == bf["n"].max()]["mean"].max(), 1)

    runs = t.csv("qaoa_benchmark", "runs")
    manifest = t.json("qaoa_benchmark", "manifest")["config"]
    m["QaoaShots"] = fmt_int(manifest["shots"])
    m["QaoaFinalShots"] = fmt_int(manifest["final_shots"])
    m["QaoaMaxIter"] = str(manifest["maxiter"])
    m["QaoaNoise"] = fmt(manifest["noise_level"])
    m["QaoaNoisyMaxN"] = str(manifest["noisy_max_n"])
    m["QaoaSeeds"] = str(manifest["num_seeds"])
    m["QaoaDepths"] = ", ".join(str(p) for p in manifest["depths"])
    q = runs[runs["solver"].isin(QAOA_SOLVERS)]
    m["QaoaShotBudgetMin"] = fmt_int(q["total_shots"].min())
    m["QaoaShotBudgetMax"] = fmt_int(q["total_shots"].max())
    m["QaoaMaxN"] = str(int(q["n"].max()))

    rows = []
    labels = {
        "toy": "Toy (hardware QUBO)",
        "random": "Random Gaussian",
        "slice": "Binary execution",
        "fig07": "Execution instance, $n{=}12$",
    }
    for family, fname in QAOA_FAMILIES.items():
        for solver, sname in QAOA_SOLVERS.items():
            d = q[(q["family"] == family) & (q["solver"] == solver)]
            if d.empty:
                continue
            pc = paired_comparison(d["success_probability"], d["random_success_probability"])
            pe = paired_comparison(d["approx_ratio_mean"], d["random_approx_ratio_mean"])
            key = f"Qaoa{fname}{sname}"
            m[f"{key}Runs"] = str(len(d))
            m[f"{key}PoptMean"] = fmt(pc.mean_diff, 3, sign=True)
            m[f"{key}PoptCi"] = fmt_ci(pc.ci_low, pc.ci_high, 3)
            m[f"{key}PoptP"] = fmt_p(pc.wilcoxon_p)
            m[f"{key}EnergyMean"] = fmt(pe.mean_diff, 3, sign=True)
            m[f"{key}EnergyCi"] = fmt_ci(pe.ci_low, pe.ci_high, 3)
            m[f"{key}BestOpt"] = fmt(100 * d["optimal_found"].mean(), 0)
            m[f"{key}UniformBestOpt"] = fmt(100 * d["random_optimal_found"].mean(), 0)
            rows.append(
                f"{labels[family]} & {sname.lower()} & {len(d)} & "
                f"{fmt(pc.mean_diff, 3, sign=True)} {fmt_ci(pc.ci_low, pc.ci_high, 3)} & "
                f"{fmt_p(pc.wilcoxon_p)} & {fmt(pe.mean_diff, 3, sign=True)} & "
                f"{fmt(100 * d['optimal_found'].mean(), 0)} / "
                f"{fmt(100 * d['random_optimal_found'].mean(), 0)} \\\\"
            )
    m["QaoaTableRows"] = "\n".join(rows)

    # Per-size success probability on the binary execution encoding (ideal, all depths).
    ideal = q[(q["family"] == "slice") & (q["solver"] == "QAOA_Ideal")]
    by_n = ideal.groupby("n")[["success_probability", "random_success_probability"]].mean()
    for n in by_n.index:
        m[f"QaoaSliceN{spell(n)}Popt"] = fmt(by_n.loc[n, "success_probability"], 4)
        m[f"QaoaSliceN{spell(n)}Uniform"] = fmt(by_n.loc[n, "random_success_probability"], 4)
    m["QaoaSliceBestRunPopt"] = fmt(ideal["success_probability"].max())
    toy = q[(q["family"] == "toy") & (q["solver"] == "QAOA_Ideal")]
    toy_cells = toy.groupby(["n", "p"])["success_probability"].mean()
    best = toy_cells.idxmax()
    m["QaoaToyBestCellPopt"] = fmt(toy_cells.max())
    m["QaoaToyBestCellN"] = str(int(best[0]))
    m["QaoaToyBestCellP"] = str(int(best[1]))
    m["QaoaToyBestCellUniform"] = fmt(
        toy[(toy["n"] == best[0]) & (toy["p"] == best[1])]["random_success_probability"].mean(), 4
    )

    # Noisy / ideal ratio of mean P_opt per (family, n, p).
    ratios = []
    for family in ("toy", "random", "slice"):
        f = q[q["family"] == family]
        cells = f.groupby(["solver", "n", "p"])["success_probability"].mean().unstack("solver")
        cells = cells.dropna()
        ratios.extend((cells["QAOA_Noisy"] / cells["QAOA_Ideal"]).tolist())
    m["NoiseRatioMin"] = fmt(min(ratios))
    m["NoiseRatioMax"] = fmt(max(ratios), 1)

    # The fig07 execution instance: SA vs ideal QAOA wall time.
    f7 = runs[runs["family"] == "fig07"]
    sa7 = f7[f7["solver"] == "SA"]
    q7 = f7[f7["solver"] == "QAOA_Ideal"]
    m["InstanceSaOptimal"] = f"{int(sa7['optimal_found'].sum())} of {len(sa7)}"
    m["InstanceSaTime"] = fmt(sa7["time_s"].mean())
    m["InstanceQaoaTimeMin"] = fmt(q7["time_s"].min(), 1)
    m["InstanceQaoaTimeMax"] = fmt(q7["time_s"].max(), 1)

    ising = t.csv("formulation", "ising_check")
    m["IsingMaxRelError"] = _sci(ising["max_rel_error"].max())
    m["IsingMinN"] = str(int(ising["n"].min()))
    m["IsingMaxN"] = str(int(ising["n"].max()))
    m["IsingSamples"] = str(int(ising["samples"].iloc[0]))


def _sci(x: float) -> str:
    mantissa, exponent = f"{x:.1e}".split("e")
    return f"${mantissa}\\times 10^{{{int(exponent)}}}$"


# --------------------------------------------------------------------------------------
# Latency and load
# --------------------------------------------------------------------------------------


def latency(t: Tables, m: Macros) -> None:
    lat = t.csv("latency", "summary").set_index(["source", "component"])
    cfg = t.json("latency", "manifest")["config"]
    m["TickIntervalMs"] = fmt(1000 * cfg["tick_interval_s"], 0)
    m["OptimizerIntervalMs"] = fmt(1000 * cfg["optimizer_interval_s"], 0)
    m["LatencyCpu"] = t.json("latency", "manifest")["environment"]["cpu_model"]
    m["LatencyPython"] = t.json("latency", "manifest")["environment"]["python"]

    def micro(v: float) -> str:
        return fmt(v, 2) if v < 10 else fmt(v, 0)

    def milli(v: float) -> str:
        return fmt(v / 1000, 1)

    names = {
        ("runtime", "fast_path"): ("RuntimeTick", "$\\mu$s"),
        ("pipeline", "fast_path"): ("PipelineTick", "$\\mu$s"),
        ("runtime", "tick_lateness"): ("TickLateness", "ms"),
        ("runtime", "policy_propagation"): ("Propagation", "ms"),
        ("pipeline", "policy_propagation"): ("PipelinePropagation", "$\\mu$s"),
        ("runtime", "slow_path_optimize"): ("RuntimeSolve", "ms"),
        ("pipeline", "slow_path_optimize"): ("PipelineSolve", "ms"),
    }
    rows = []
    tick_ms = fmt(1000 * cfg["tick_interval_s"], 0)
    labels = {
        "RuntimeTick": "Runtime tick work (poll, re-plan, execute)",
        "PipelineTick": "Pipeline tick work (estimators, policy)",
        "TickLateness": f"Tick start lateness vs.\\ {tick_ms} ms schedule",
        "Propagation": "Policy publish $\\to$ apply (runtime)",
        "PipelinePropagation": "Policy publish $\\to$ apply (pipeline)",
        "RuntimeSolve": "Slow path: SA solve (runtime QUBO)",
        "PipelineSolve": "Slow path: SA solve (pipeline QUBO)",
    }
    for key, (name, unit) in names.items():
        r = lat.loc[key]
        scale: Callable[[float], str] = micro if unit == "$\\mu$s" else milli
        m[f"{name}Median"] = scale(r["50%"])
        m[f"{name}Ptt"] = scale(r["99%"])
        m[f"{name}Count"] = fmt_int(r["count"])
        rows.append(
            f"{labels[name]} & {fmt_int(r['count'])} & {scale(r['50%'])} & {scale(r['99%'])} & "
            f"{unit} \\\\"
        )
    m["LatencyTableRows"] = "\n".join(rows)

    load = t.json("load_test", "summary")
    m["LoadOrders"] = str(load["orders"])
    m["LoadConcurrency"] = str(load["concurrency"])
    m["LoadWallTime"] = fmt(load["wall_time_s"], 1)
    m["LoadFilled"] = str(load["orders_fully_filled"])
    m["LoadOverheadMedian"] = fmt(load["overhead_vs_tick_schedule_s"]["50%"])


# --------------------------------------------------------------------------------------
# Hardware (only once raw counts are recovered)
# --------------------------------------------------------------------------------------


def hardware(t: Tables, m: Macros) -> None:
    recovered = t.has("hardware", "summary.csv") and t.has("hardware", "manifest.json")
    if recovered:
        manifest = t.json("hardware", "manifest")
        if "SYNTHETIC" in str(manifest.get("data", "")).upper() or manifest.get("quick"):
            recovered = False
    m["HwRecovered"] = "1" if recovered else "0"
    if not recovered:
        return
    s = t.csv("hardware", "summary")
    jobs = t.csv("hardware", "jobs")
    m["HwNumJobs"] = str(len(jobs))
    m["HwMinN"] = str(int(s["n"].min()))
    m["HwMaxN"] = str(int(s["n"].max()))
    rows = []
    for _, r in s.sort_values("n").iterrows():
        n = int(r["n"])
        m[f"HwNFor{spell(n)}Popt"] = fmt(r["success_probability"], 4)
        m[f"HwNFor{spell(n)}Uniform"] = fmt(r["uniform_success_probability"], 4)
        rows.append(
            f"{n} & {fmt(r['success_probability'], 4)} & "
            f"{fmt(r['uniform_success_probability'], 4)} \\\\"
        )
    m["HwTableRows"] = "\n".join(rows)


# --------------------------------------------------------------------------------------


def build(results_dir: Path) -> str:
    t = Tables(results_dir)
    m = Macros()
    real_data(t, m)
    synthetic(t, m)
    solvers(t, m)
    latency(t, m)
    hardware(t, m)
    header = (
        "% Generated by `python -m experiments.paper_numbers` from results/. Do not edit:\n"
        "% `make check-paper` fails if this file differs from a fresh regeneration."
    )
    return m.render(header)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="fail if the file is stale")
    args = parser.parse_args()
    text = build(args.results_dir)
    if args.check:
        current = args.output.read_text() if args.output.exists() else ""
        if current != text:
            print(f"{args.output} is stale: run `python -m experiments.paper_numbers`")
            sys.exit(1)
        print(f"{args.output} is up to date ({text.count(chr(10)) - 2} macros)")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(f"wrote {args.output} ({text.count(chr(10)) - 2} macros)")


if __name__ == "__main__":
    main()
