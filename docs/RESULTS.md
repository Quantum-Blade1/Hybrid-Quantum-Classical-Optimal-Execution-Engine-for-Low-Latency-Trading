# Results

Headline numbers from the committed `results/` (full run of `python -m experiments.run_all`, 844 s on an Apple M5, 10 cores, Python 3.12.14; every `results/<name>/manifest.json` records the config, seeds, commit and environment). Figures in `paper/figures/` are drawn from these files only (`figures/registry.py`). Intervals are 95% percentile-bootstrap CIs of the mean; "paired" differences are matched by seed (same price path and same order books) and carry a two-sided Wilcoxon signed-rank p-value. No multiple-comparison correction is applied. All market data are synthetic (`MarketDataSimulator`), not calibrated to any real dataset. The evaluation model is in `docs/MATHEMATICAL_MODEL.md`, "Evaluation Model".

## Bottom line

* **The SA-QUBO and hybrid schedules do not beat TWAP or VWAP.** In every strategy experiment (one hour, a full day at three order sizes, four stress scenarios, walk-forward) the paired shortfall difference of SA-QUBO or Hybrid against TWAP has a CI that includes zero. Against VWAP the only significant difference is *against* the hybrid (market outage: +9.5 bps worse). The QUBO schedules pay slightly *more* market impact than TWAP (+0.15 to +0.20 bps at 1% of ADV, CI excludes zero), because their discrete quantity levels make the schedule lumpier.
* **QAOA does not beat uniform random sampling at finding the optimum.** With the same total shot budget, uniform sampling found the optimal bitstring in 100% of runs at every size tested (n <= 12); ideal QAOA's best sample found it in 95% (toy) and 88% (random QUBOs). QAOA's final distribution does put modestly more mass on the optimum than uniform sampling (e.g. +0.036 [0.015, 0.064] on the toy family), and its mean energy is better (<H>-ratio +0.053 [0.043, 0.063]), but the advantage fades with n: at n = 10-12 the probability on the optimum is 0.0001-0.004, comparable to uniform (0.0002-0.001).
* **Simulated annealing as configured is weak on the penalty-dominated QUBOs.** Its geometric cooling (T0 = 10, rate 0.95, T_final = 0.01) stops after 135 sweeps whatever `num_sweeps` says; it reaches the exact optimum in only 10% of seeds on the toy QUBO at n = 8-12 and 0% on the execution QUBO at n >= 12 (100% on random Gaussian QUBOs).
* **Latency.** The fast path's own work per tick is microseconds (median 16 us, p99 28 us), but because the pure-Python SA thread holds the GIL, ticks scheduled every 5 ms start a median 35 ms late. None of this is kernel-bypass or sub-microsecond engineering.

## 1. Execution strategies (implementation shortfall, bps of arrival notional, incl. opportunity cost)

### One simulated hour, 100k shares (`is_comparison`, 30 seeds, fig22)

| Strategy | Mean shortfall [95% CI] | Paired vs TWAP [CI], p | Paired vs VWAP [CI], p |
|---|---|---|---|
| TWAP | 52.0 [-1.3, 104.2] | - | +3.3 [-4.5, 10.6], 0.40 |
| VWAP | 48.8 [-0.9, 96.6] | -3.3 [-10.6, 4.5], 0.40 | - |
| SA-QUBO | 51.2 [-2.0, 102.8] | -0.8 [-5.7, 3.9], 0.92 | +2.5 [-6.6, 11.1], 0.46 |
| Hybrid | 52.2 [-0.3, 103.6] | +0.2 [-3.2, 3.5], 0.84 | +3.5 [-4.4, 11.1], 0.31 |

All fills are complete. Half-spread and impact together are below 3 bps for every strategy; the shortfall is dominated by the price path (timing), which is common to all strategies and cancels in the paired differences.

### Full day at 0.1%, 1% and 5% of daily volume (`strategy_comparison`, 30 seeds)

| Order size | Hybrid - TWAP | SA-QUBO - TWAP | Hybrid - VWAP | Impact: Hybrid - TWAP | Impact: SA-QUBO - TWAP |
|---|---|---|---|---|---|
| 0.1% ADV | +0.69 [-0.36, 1.70] | -0.73 [-3.33, 1.91] | -6.9 [-17.3, 3.2] | +0.001 [0.001, 0.002] | +0.001 [0.001, 0.002] |
| 1% ADV | +0.74 [-0.24, 1.68] | -0.46 [-3.14, 2.19] | -6.7 [-17.0, 3.5] | **+0.15 [0.14, 0.17]** | **+0.20 [0.17, 0.22]** |
| 5% ADV | -0.42 [-2.11, 1.21] | +1.62 [-2.00, 5.07] | +0.2 [-7.2, 7.9] | +0.014 [0.008, 0.019] | -0.001 [-0.017, 0.015] |

At 5% of ADV the synthetic book's depth (10 levels, depth proportional to bar volume) binds: fill rates are TWAP 65%, VWAP 73%, SA-QUBO 64%, Hybrid 65%. The remainder is charged as a clean-up order against the final bar's book, with shares beyond its depth priced at the deepest level; results at that size therefore mostly reflect the limits of the book model.

### Stress scenarios, 50k shares in 60 minutes (`stress_test`, 30 seeds, fig23)

| Scenario | Hybrid - TWAP | SA-QUBO - TWAP | Hybrid - VWAP | Fill rate (all strategies) |
|---|---|---|---|---|
| Flash crash | -76 [-163, 9], p = 0.12 | +1 [-107, 112] | -480 [-567, -395], p < 1e-4 | 100% |
| Liquidity crisis | +2.6 [-0.9, 6.4] | +2.4 [-3.1, 7.7] | +7.9 [-1.0, 16.8], p = 0.08 | 100% |
| Volatility spike | +0.5 [-6.4, 7.4] | +4.9 [-4.0, 14.0] | +7.2 [-5.1, 19.7] | 100% |
| Market outage | +4.8 [-0.2, 10.0] | +2.3 [-3.0, 7.4] | **+9.5 [1.2, 17.8], p = 0.022** | 100% |

The flash crash is a 50% price drop: any buy schedule that trades later pays less, and VWAP's expected-volume profile front-loads into the open, before the crash. This is price-path luck, not a property of the optimiser. With the Phase 6 fill model (no fills in zero-volume bars, carry-forward), no strategy fills during the outage and all complete afterwards; the hybrid runner's former 6,185 bps outage "slippage" (fills at mid + $500) is gone.

### Walk-forward, 10 windows per seed, 500k shares per test day (`walk_forward`, 30 seeds, fig21)

| Strategy | Mean shortfall [CI] | Paired vs TWAP [CI], p | Windows won (of 300) |
|---|---|---|---|
| TWAP | 1.6 [-27.7, 32.3] | - | 60 |
| Static VWAP | -0.1 [-27.2, 28.2] | -1.7 [-9.0, 5.2], 0.95 | 81 |
| Adaptive VWAP | 0.2 [-26.9, 28.4] | -1.4 [-8.7, 5.4], 0.98 | 60 |
| Hybrid (SA-QUBO) | 1.9 [-27.7, 33.4] | +0.3 [-2.1, 2.8], 0.86 | 99 |

The hybrid wins the most individual windows but its mean shortfall is not lower than any baseline's (vs static VWAP +2.0 [-6.5, 11.1]; vs adaptive +1.8 [-6.8, 10.7]).

## 2. Solvers

### Classical solvers vs exact optimum (`solver_benchmark`, 10 seeds, fig05, fig06, fig09)

Fraction of seeds at the exact optimum:

| Family | n | SA | Greedy |
|---|---|---|---|
| random Gaussian | 4-20 | 1.0 at every n | 0.6-0.9 |
| toy (hardware problem) | 4 / 6 / 8-12 / 14-20 | 1.0 / 0.7 / 0.1 / 0.0 | 0.2 / 0.1 / 0.0 / 0.0 |
| execution (`slice_level_config`) | 6 / 9 / 12-18 | 0.6 / 0.2 / 0.0 | 0.6 / 0.2 / 0.0-0.2 |

SA's decoded schedules on the execution family always select the full order (fill 1.0) even when not optimal. Wall time on random QUBOs at n = 20: exhaustive search 1.8 s, SA 3.3 ms, greedy 0.07 ms.

### QAOA vs SA vs uniform sampling (`qaoa_benchmark`, 5 seeds, p = 1-3, fig07, fig_hw_*)

Ideal Aer, COBYLA with at most 50 evaluations of 1,000 shots, then 5,000 final shots; the uniform baseline gets the same total budget (31k-55k shots). Noisy Aer uses a generic depolarizing + readout model (p = 0.02), not a device calibration, for n <= 10.

| Family / solver | Runs | P_opt QAOA - uniform [CI], p | <H>-ratio QAOA - uniform [CI] | Best sample optimal: QAOA / uniform |
|---|---|---|---|---|
| toy, ideal | 75 | +0.036 [0.015, 0.064], 0.010 | +0.053 [0.043, 0.063] | 95% / 100% |
| toy, noisy | 60 | +0.018 [0.005, 0.034], 0.09 | +0.020 [0.014, 0.027] | 100% / 100% |
| random, ideal | 75 | +0.026 [0.011, 0.044], 0.006 | +0.046 [0.031, 0.064] | 88% / 100% |
| random, noisy | 60 | +0.006 [0.003, 0.011], 0.011 | +0.013 [0.007, 0.022] | 100% / 100% |
| fig07 execution QUBO (n = 12), ideal | 15 | +0.023 [0.006, 0.045], 0.17 | +0.040 [0.027, 0.054] | 100% / 100% |

Largest per-cell advantage: toy n = 4, p = 2, P_opt 0.34 vs 0.0625 uniform. At n = 12 the ideal P_opt is 0.0003-0.0012 (uniform 0.0002). On the fig07 instance SA (as configured) found the optimum in 1 of 5 seeds in 3.7 ms; ideal QAOA took 1.5-2.9 s. Noise does not uniformly reduce P_opt by "30-50%": the noisy/ideal ratio ranges from 0.37 (n = 4, p = 2) to 1.6 (n = 4, p = 1).

### Formulation checks (`formulation`, `qaoa_landscape`)

* QUBO -> Ising: max relative error 2.7e-13 over n = 4-20 (200 random x each); term counts match n + n(n-1)/2.
* Six-term HFT QUBO, SA solutions over 10 seeds (fig03): share of |cost| is information leakage 70% [52, 83], impact 26% [15, 41], adverse selection 2.1%, transaction 1.4%, inventory 0.5%, timing 0.1%.
* The SA-QUBO runtime schedule is about as far from Almgren-Chriss at lambda = 1e-4 as from TWAP (RMSE 1,315 vs 1,323 shares per 2,500-share step).

## 3. Microstructure and regimes (synthetic ticks, 30 seeds, fig10-fig17)

* Kyle's lambda rises 5.3x [5.1, 5.6] in the 5x-volatility block. The synthetic trade side is sign(return), so the rise is partly built into the data.
* VPIN (tick rule) exceeds 0.7 on 98% of ticks in calm and stress alike: on these data the estimator is saturated and does not separate the regimes.
* Regime shares: NORMAL 70.4% [69.4, 71.4], LOW 20.5%, HIGH 9.0%, EXTREME 0.16%. Adaptive lambda stays within 0.25x-1.28x of its base (max 1.28x [1.24, 1.32]); the 4x EXTREME multiplier is essentially never reached.

## 4. Runtime (`latency`, `load_test`, fig24)

Measured on an Apple M5, CPython 3.12.14, with `time.monotonic_ns`:

| Component | n | Median | p99 |
|---|---|---|---|
| Async runtime tick work (poll, re-plan, execute) | 2,000 | 16 us | 28 us |
| HFT pipeline tick (estimators + policy) | 10,000 | 64 us | 103 us |
| Async runtime tick lateness vs 5 ms schedule | 1,980 | 35 ms | 61 ms |
| Policy publication -> application (runtime) | 1,150 | 7.5 ms | 9.2 ms |
| Slow path: SA solve, 300-variable runtime QUBO | 1,182 | 64 ms | 65 ms |
| Slow path: SA solve, HFT QUBO | 500 | 66 ms | 67 ms |

Load test: 100 concurrent `HybridController` orders (20 threads) complete in 10.9 s; every order fills completely (the fast path now re-plans the remaining shares after a policy switch), with a median 0.31 s of overhead over the tick schedule.

## 5. What is not measured

No real market data; no IBM hardware results (the committed `results/bench_hw_*.json` are unverifiable, claims audit F10); no ablation of the adverse-selection or leakage terms; no multi-venue fill model (venue routing in fig27 is the QUBO's allocation, not an executed outcome); no permanent market impact. See `docs/CLAIMS_AUDIT.md`, section 6, for the status of each paper claim.
