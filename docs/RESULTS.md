# Results

Headline numbers from the committed `results/` (full run of `python -m experiments.run_all`, 844 s in Phase 6; in Phase 7 `solver_benchmark` and `qaoa_benchmark` were rerun (90 s and 1,242 s) and `real_data_tune`, `real_data_dev`, `real_data_test` were added; Apple M5, 10 cores, Python 3.12.14; every `results/<name>/manifest.json` records the config, seeds, commit and environment). Figures in `paper/figures/` are drawn from these files only (`figures/registry.py`). Intervals are 95% percentile-bootstrap CIs of the mean; "paired" differences are matched by seed (same price path and same order books) and carry a two-sided Wilcoxon signed-rank p-value. In sections 1-4 no multiple-comparison correction is applied and all market data are synthetic (`MarketDataSimulator`), not calibrated to any real dataset. Section 0 (Phase 7) uses real Binance data under the pre-registered protocol of `docs/PROTOCOL.md`, with Holm correction. The evaluation model is in `docs/MATHEMATICAL_MODEL.md`, "Evaluation Model".

## Bottom line

* **Phase 7, real data (held-out): the improved QUBO (exact binary encoding, fixed SA) and the adaptive hybrid do not beat TWAP, VWAP or discretized Almgren–Chriss.** On 12 held-out days of Binance BTCUSDT and LINKUSDT (72 windows per symbol, pre-registered protocol, evaluated once), none of the 12 primary comparisons is significant: every Holm-adjusted p = 1.00. Mean differences are -0.27 to +0.22 bps; on BTCUSDT all six 95% CIs lie within +-0.5 bps (practically equivalent), on LINKUSDT the CIs are about +-1-2 bps wide. The result holds at 0.5x and 2x the calibrated impact coefficient (no significant comparison in the sensitivity family). The reason is structural: at these order sizes spread + impact are 0.16 bps (BTC) and 0.6 bps (LINK) for every schedule, the QUBO's model cost is within 0.09 bps of AC's on average, and realized shortfall (5-5.5 bps mean, std ~50 bps) is dominated by price moves and clean-up cost that no schedule controls. See section 0.
* **Phase 6, synthetic data: the SA-QUBO and hybrid schedules do not beat TWAP or VWAP.** In every strategy experiment (one hour, a full day at three order sizes, four stress scenarios, walk-forward) the paired shortfall difference of SA-QUBO or Hybrid against TWAP has a CI that includes zero. Against VWAP the only significant difference is *against* the hybrid (market outage: +9.5 bps worse). The QUBO schedules pay slightly *more* market impact than TWAP (+0.15 to +0.20 bps at 1% of ADV, CI excludes zero), because their discrete quantity levels make the schedule lumpier.
* **QAOA does not beat uniform random sampling at finding the optimum** (re-checked in Phase 7 on the binary-encoded execution QUBO: P_opt advantage +0.007 [0.000, 0.016], p = 0.28; best sample optimal in 87% of runs vs 100% for uniform). With the same total shot budget, uniform sampling found the optimal bitstring in 100% of runs at every size tested (n <= 12); ideal QAOA's best sample found it in 95% (toy) and 88% (random QUBOs). QAOA's final distribution does put modestly more mass on the optimum than uniform sampling (e.g. +0.036 [0.015, 0.064] on the toy family), and its mean energy is better (<H>-ratio +0.053 [0.043, 0.063]), but the advantage fades with n: at n = 10-12 the probability on the optimum is 0.0001-0.004, comparable to uniform (0.0002-0.001).
* **Simulated annealing, fixed in Phase 7, now finds the exact optimum up to n = 16 in every seed** (1,000 sweeps honoured, 16 restarts; 50-80% at n = 18-20 on the toy and binary-encoded execution QUBOs). In Phase 6 its schedule stopped after 135 sweeps and it reached the optimum in 10% of seeds on the toy QUBO at n = 8-12. On the real-data QUBOs (12 variables) SA hit the exact integer optimum in every held-out cell.
* **Latency.** The fast path's own work per tick is microseconds (median 16 us, p99 28 us), but because the pure-Python SA thread holds the GIL, ticks scheduled every 5 ms start a median 35 ms late. None of this is kernel-bypass or sub-microsecond engineering.

## 0. Real data (held-out), Phase 7

Pre-registered protocol: `docs/PROTOCOL.md` (deviations listed there). Data: public Binance spot `aggTrades` (SHA-256 verified), 1-minute bars, BTCUSDT and LINKUSDT (LINK's quote volume is 1.11% of BTC's over the development days: 9.7 M vs 873 M USDT/day). Development days 2026-07-22 to 2026-08-06 (16) for all calibration and tuning; held-out test days 2026-08-07 to 2026-08-18 (12), evaluated **once** from the clean frozen commit `d2f227c` (`results/real_data_test/manifest.json`). Orders: 6 start times x 12 days x {0.1%, 0.5% of development ADV} x {60, 240 min} per symbol; unit of analysis = window (72 per symbol), difference averaged over the 4 cells. Fill model: bar-VWAP mid, measured half spread, linear temporary impact beta x participation, 25% participation cap, carry-forward, clean-up at the last bar charged as opportunity cost (`docs/MATHEMATICAL_MODEL.md`, "Phase 7").

**Calibration (development days, `results/real_data_dev/calibration.csv`).**

| Symbol | ADV | Lot | Impact beta (bps per unit participation) [SE], R^2 | Mean half spread | Mean sigma (1 min) |
|---|---|---|---|---|---|
| BTCUSDT | 13,578 BTC | 0.001 BTC | 1.52 [0.16], 0.17 | 0.0008 bps (one 0.01 USD tick) | 3.75 bps |
| LINKUSDT | 1.157 M LINK | 0.1 LINK | 1.19 [0.10], 0.08 | 0.56 bps | 5.97 bps |

**Tuning (development days only, `results/real_data_tune`).** Among 7 QUBO settings, only the two smallest encodings (12 and 15 binary variables) let SA reach the exact integer optimum (DP) in >= 95% of the 48 development cells; larger encodings (18-32 variables) reached it in 0-73% of cells, although their optimum is slightly closer to AC. Selected: 4 slices x 3 bits, 16 units (12 variables), SA 1,000 sweeps x 16 restarts; its model objective is 0.044 bps above the discretized AC optimum on average (max 0.42). Hybrid: 1 re-planning checkpoint (dev mean shortfall 2.85 bps vs 3.21-3.32 for 3 or 6 checkpoints; every hybrid variant was 0.23-0.69 bps *worse* than TWAP on development days).

**Development days (in-sample, `results/real_data_dev`, 96 windows per symbol).** No primary comparison is significant (all Holm p = 1). BTCUSDT: QUBO - TWAP +0.16 [-0.22, 0.55], QUBO - AC -0.03 [-0.13, 0.07]; LINKUSDT: QUBO - TWAP +0.02 [-1.10, 1.18], QUBO - AC -0.99 [-2.75, 0.64], Hybrid - TWAP +0.29 [-0.77, 1.36] bps.

**Held-out primary result (12 pre-registered comparisons, impact x1, lambda = 0; `results/real_data_test/comparisons.csv`, fig_real_primary_comparisons).** Mean paired difference in shortfall (bps, negative = strategy cheaper), 95% bootstrap CI over windows, day-clustered CI, Wilcoxon p, Holm-adjusted p:

| Symbol | Comparison | Mean | 95% CI | Day-cluster CI | p | p (Holm) | Decision |
|---|---|---|---|---|---|---|---|
| BTCUSDT | QUBO - TWAP | -0.10 | [-0.38, 0.18] | [-0.29, 0.08] | 0.53 | 1.00 | no difference (practically equivalent) |
| BTCUSDT | QUBO - VWAP | -0.03 | [-0.09, 0.03] | [-0.09, 0.04] | 0.44 | 1.00 | no difference (practically equivalent) |
| BTCUSDT | QUBO - AC | -0.03 | [-0.09, 0.03] | [-0.09, 0.04] | 0.45 | 1.00 | no difference (practically equivalent) |
| BTCUSDT | Hybrid - TWAP | +0.00 | [-0.25, 0.27] | [-0.21, 0.21] | 0.92 | 1.00 | no difference (practically equivalent) |
| BTCUSDT | Hybrid - VWAP | +0.08 | [0.00, 0.17] | [-0.01, 0.17] | 0.10 | 1.00 | no difference (practically equivalent) |
| BTCUSDT | Hybrid - AC | +0.08 | [0.00, 0.17] | [-0.01, 0.17] | 0.10 | 1.00 | no difference (practically equivalent) |
| LINKUSDT | QUBO - TWAP | -0.15 | [-1.64, 1.16] | [-1.69, 1.09] | 0.51 | 1.00 | no difference |
| LINKUSDT | QUBO - VWAP | -0.07 | [-1.04, 0.89] | [-0.75, 0.62] | 0.99 | 1.00 | no difference |
| LINKUSDT | QUBO - AC | +0.22 | [-1.76, 2.03] | [-1.71, 2.22] | 0.16 | 1.00 | no difference |
| LINKUSDT | Hybrid - TWAP | -0.27 | [-2.02, 1.13] | [-2.03, 1.09] | 0.37 | 1.00 | no difference |
| LINKUSDT | Hybrid - VWAP | -0.19 | [-1.28, 0.78] | [-0.92, 0.48] | 0.80 | 1.00 | no difference |
| LINKUSDT | Hybrid - AC | +0.10 | [-1.78, 1.86] | [-1.90, 2.05] | 0.18 | 1.00 | no difference |

Mean held-out shortfall (bps, 288 orders per symbol): BTCUSDT TWAP 4.93, VWAP 4.85, AC 4.85, QUBO 4.83, Hybrid 4.93; LINKUSDT TWAP 5.51, VWAP 5.43, AC 5.14, QUBO 5.36, Hybrid 5.25 (CIs about +-5 bps; per-order std 43-58 bps). Spread + impact is 0.16 bps (BTC) and 0.60 bps (LINK) for every strategy; the rest is timing and opportunity cost (BTC fill rate before clean-up 95-96%: the 25% cap binds in thin minutes), which the schedule barely changes (fig_real_cost_components).

**Sensitivity (section 8; fig_real_sensitivity).** With the evaluator's impact at x0.5 or x2 (optimisers at x1, or at the same multiple) no comparison is significant (every jointly Holm-adjusted p = 1.00; unadjusted, the only CIs excluding zero are Hybrid - VWAP/AC on BTCUSDT at x2, +0.14 [0.02, 0.28], i.e. against the hybrid). Mean differences stay within [-0.55, +0.26] bps. On BTCUSDT the x0.5/x2 optimiser variants give the same schedules as x1: with a one-tick spread the water-filling optimum is proportional to expected volume whatever beta is.

**Secondary (exploratory, unadjusted unless stated).**
* QUBO - AC cost gap in the model (held-out cells): mean 0.0005 bps (BTC), 0.088 bps (LINK, max 0.42); SA hit the exact integer optimum in every cell (gap to DP 0), so the gap is the coarse 4-slice encoding, not the solver. In realized shortfall the gap is -0.03 (BTC) and +0.22 (LINK) bps, both CIs covering zero. The 5-seed SA robustness run gives identical schedules (spread of mean differences 0).
* AC - TWAP: -0.08 [-0.35, 0.21] (BTC), -0.37 [-2.65, 1.95] (LINK); VWAP - TWAP -0.08 / -0.08. On BTCUSDT AC and VWAP coincide.
* Model-predicted vs realized spread + impact: BTC 0.08 vs 0.16 bps (realized minute volume varies around its expected value, and impact is convex in 1/V); LINK 0.59 vs 0.60 bps (QUBO).
* Risk-averse variant (lambda with lambda Var = E for TWAP, front-loaded schedules): on LINKUSDT QUBO - TWAP -3.8 [-8.1, -0.4] bps and QUBO - AC(lambda) +2.1 [-0.7, 5.0]; Holm-adjusted p >= 0.59. Not significant, and not part of the primary family.

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

Rerun in Phase 7 with the fixed SA (geometric schedule that runs exactly `num_sweeps` = 1,000 sweeps, `neal`-style temperature bounds, 16 vectorised restarts; `SA_1` = one restart) and the new `slice` family (exact binary encoding of the Phase 7 cost model). Fraction of seeds at the exact optimum:

| Family | n | SA (16 restarts) | SA_1 | Greedy |
|---|---|---|---|---|
| random Gaussian | 4-20 | 1.0 at every n | 1.0 at every n | 0.6-0.9 |
| toy (hardware problem) | 4-16 / 18 / 20 | 1.0 / 0.8 / 0.5 | 1.0 (n <= 12), 0.6 / 0.4 / 0.2 / 0.0 (n = 14-20) | 0.0-0.2 |
| execution (Phase 6 level encoding) | 6-18 | 1.0 | 1.0 | 0.0-0.6 |
| slice (Phase 7 binary encoding) | 4-16 / 18 / 20 | 1.0 / 0.7 / 0.5 | 1.0 (n <= 12), 0.4 / 0.3 / 0.1 / 0.1 (n = 15-20) | 0.0-0.1 |

(Phase 6, before the SA fix: toy 0.1 at n = 8-12 and 0.0 at n >= 14; execution 0.0 at n >= 12.) SA time is now 0.03-0.16 s per solve (1,000 sweeps x 16 restarts in NumPy); exhaustive search takes 1.8 s at n = 20, greedy 0.05-0.08 ms.

### QAOA vs SA vs uniform sampling (`qaoa_benchmark`, 5 seeds, p = 1-3, fig07, fig_hw_*)

Ideal Aer, COBYLA with at most 50 evaluations of 1,000 shots, then 5,000 final shots; the uniform baseline gets the same total budget (31k-55k shots). Noisy Aer uses a generic depolarizing + readout model (p = 0.02), not a device calibration, for n <= 10.

| Family / solver | Runs | P_opt QAOA - uniform [CI], p | <H>-ratio QAOA - uniform [CI] | Best sample optimal: QAOA / uniform |
|---|---|---|---|---|
| toy, ideal | 75 | +0.036 [0.015, 0.064], 0.010 | +0.053 [0.043, 0.063] | 95% / 100% |
| toy, noisy | 60 | +0.018 [0.005, 0.034], 0.09 | +0.020 [0.014, 0.027] | 100% / 100% |
| random, ideal | 75 | +0.026 [0.011, 0.044], 0.006 | +0.046 [0.031, 0.064] | 88% / 100% |
| random, noisy | 60 | +0.006 [0.003, 0.011], 0.011 | +0.013 [0.007, 0.022] | 100% / 100% |
| fig07 execution QUBO (n = 12), ideal | 15 | +0.023 [0.006, 0.045], 0.17 | +0.040 [0.027, 0.054] | 100% / 100% |
| slice (Phase 7 binary encoding, n = 4-12), ideal | 90 | +0.007 [0.000, 0.016], 0.28 | +0.051 [0.042, 0.060] | 87% / 100% |
| slice, noisy (n <= 10) | 75 | +0.002 [-0.001, 0.005], 0.23 | +0.009 [0.006, 0.014] | 100% / 100% |

Rerun in Phase 7 (toy, random and fig07 rows are unchanged: same seeds, same QAOA code). On the Phase 7 binary encoding QAOA's advantage in P_opt over uniform sampling is not significant (ideal +0.007, p = 0.28; noisy +0.002, p = 0.23); its mean energy is better (<H> ratio +0.05), but its best sample misses the optimum in 13% of ideal runs while uniform sampling with the same budget never does. Per size, ideal P_opt vs uniform: n = 4 0.096 vs 0.0625, n = 8 0.0044 vs 0.0039, n = 12 0.0002 vs 0.0002. Largest per-cell advantage: toy n = 4, p = 2, P_opt 0.34 vs 0.0625 uniform (slice family: best single run 0.34 at n = 4, p = 2). At n = 12 the ideal P_opt is 0.0003-0.0012 (uniform 0.0002). On the fig07 instance the fixed SA finds the optimum in 5 of 5 seeds in 0.21 s (Phase 6: 1 of 5 in 3.7 ms); ideal QAOA took 1.7-3.6 s (13x slower, not >1000x). Noise does not uniformly reduce P_opt by "30-50%": the noisy/ideal ratio ranges from 0.37 (n = 4, p = 2) to 1.6 (n = 4, p = 1).

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

Real market data only for two Binance spot pairs over 28 days, with a bar-level linear-impact fill model (no order book, no queue position, no fees, no permanent impact); no IBM hardware results (the committed `results/bench_hw_*.json` are unverifiable, claims audit F10; `experiments/hardware_analysis.py` is ready for recovered raw counts in `results/ibm_fez_recovered.jsonl`, which did not exist at the time of writing, and is tested only on a synthetic fixture); no ablation of the adverse-selection or leakage terms; no multi-venue fill model (venue routing in fig27 is the QUBO's allocation, not an executed outcome); no permanent market impact. See `docs/CLAIMS_AUDIT.md`, section 6, for the status of each paper claim.
