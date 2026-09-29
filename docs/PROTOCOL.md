# Phase 7 evaluation protocol (pre-registered)

Written and committed before any real-market data was downloaded or any evaluation was run.
Phase 6 found, on synthetic data, that the SA-QUBO and hybrid schedules do not beat TWAP or
VWAP and that QAOA does not beat uniform sampling at finding the optimum. Phase 7 improves
the method and evaluates it once on held-out real data. This document fixes, in advance,
what is measured, on which data, and how the result is decided. Any later change is listed
under "Deviations" with its reason; nothing below is edited silently.

## 1. Data

* **Source.** Binance public spot data, <https://data.binance.vision>, daily `aggTrades`
  zip files (`data/spot/daily/aggTrades/<SYMBOL>/`), each verified against its published
  `.CHECKSUM` (SHA-256) before use. No API keys, no private data. Raw files are stored under
  `data/` (git-ignored); only summaries are committed.
* **Symbols.**
  * `BTCUSDT`: the most liquid USDT pair.
  * `LINKUSDT`: the less liquid pair. Chosen because its quote volume was 1.55% of
    BTCUSDT's (mean of 2026-08-05 and 2026-08-12, from the published 1-minute klines,
    ~13 M USDT/day vs ~845 M USDT/day), inside the targeted 1-3% band, while still trading
    ~100k trades/day so that 1-minute bars are rarely empty. Among the pairs checked
    (TRX 3.1%, UNI 2.9%, ADA 2.2%, NEAR 1.7%, LINK 1.6%, AVAX 1.3%, LTC 1.1%, DOT 0.35%)
    it is near the middle of the band and is a long-listed, widely traded asset.
    The volume ratio over the development days is recomputed and reported. (2026-08-12
    falls in the test period: only the day's total quote volume and trade count of these
    pairs were looked at, for choosing the symbol, before the split below was fixed.)
* **Period.** 28 consecutive UTC days, 2026-07-22 to 2026-08-18 inclusive (crypto trades
  24/7, so a "day" is a UTC calendar day). Expected download: about 225 MB (BTCUSDT) + 15 MB
  (LINKUSDT), within the 50-300 MB budget. If a day is missing or fails its checksum after
  retries, it is dropped and recorded here as a deviation (no replacement day is chosen).
* **Split (chronological, no overlap).**
  * **Development days:** 2026-07-22 to 2026-08-06 (16 days). Used for everything that is
    estimated or tuned: intraday profiles, impact calibration, lot sizes, ADV, the SA
    schedule, QUBO sizes and penalty margin, hybrid re-planning rules, and debugging of the
    evaluation pipeline (a "dev evaluation" on these days is reported, labelled in-sample).
  * **Held-out test days:** 2026-08-07 to 2026-08-18 (12 days). Bars are built for these
    days at download time, but no statistic of them is computed, plotted or inspected until
    the code is frozen. The test evaluation is run exactly once, from a clean committed tree
    (the commit hash is recorded in the manifest); its results are reported whatever they are.

## 2. Bars and market inputs

1-minute bars per symbol from the aggregated trades (`qexec.market.binance`): open, high,
low, close, base volume, quote volume, VWAP = quote/base volume, trade count (sum of
`last_trade_id - first_trade_id + 1`), taker-buy volume (`is_buyer_maker` false) and signed
volume = taker buy - taker sell, and a trade-sign half-spread estimate: half the median
absolute price difference between consecutive aggregated trades of opposite taker side at
most 100 ms apart, in bps of price (NaN when a bar has fewer than 3 such pairs). Bars with no
trades have zero volume, carry the previous close, and fill nothing.

From the development days only:

* ADV: mean daily base volume.
* Intraday profiles on 96 fifteen-minute buckets of the UTC day: expected volume per minute
  (mean), half spread (median of bar estimates), per-minute volatility (RMS of 1-minute
  close-to-close log returns, bps).
* Impact coefficient beta (bps of price per unit participation): OLS through the origin of
  `r_k = beta * SV_k / Vbar_k + e_k` over all development bars, with `r_k` the 1-minute
  close-to-close log return in bps, `SV_k` signed volume and `Vbar_k` the expected volume of
  bar k's bucket (Kyle-style regression). A day-block bootstrap standard error is reported.
  The regression slope is used as the temporary impact coefficient of the fill model
  (`docs/MATHEMATICAL_MODEL.md`); the x0.5 sensitivity below covers the alternative reading
  that a child order pays only half the within-bar price response.

## 3. Execution simulation (evaluator)

The Phase 6 engine rules are reused (`ExecutionEngine`): one child order per minute,
unfilled shares carried forward to the next minute, the remainder after the last minute
charged as opportunity cost via a clean-up order at the last bar. Only the fill model
changes (no order book exists for these data):

* mid proxy `m_k` = bar VWAP; realized half spread `h_k` = the bar's estimate, or the bucket
  profile value when it is NaN;
* a child order for `q` lots fills `min(q, floor(0.25 * V_k))` (participation cap 25% of the
  realized bar volume `V_k`; nothing fills when `V_k = 0`);
* fill price `m_k * (1 + side * (h_k + beta * q_filled / V_k) / 1e4)` (linear temporary
  impact, no permanent impact, no fees: fees are the same per unit for every strategy);
* clean-up price for `U` unfilled lots at the last bar `T`: the same formula without the cap,
  with `V_T` replaced by the bucket's expected volume if `V_T = 0`.

Quantities are integer lots; the lot is a power of ten chosen on development data so that
the smallest order is at least 10,000 lots.

## 4. Orders

For each symbol, each day and each start time 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC
(a "window"; 96 development and 72 test windows per symbol) one order in each of four
cells: size {0.1%, 0.5%} of development ADV x horizon {60, 240} minutes. Side alternates by
start time (buy at 00, 08, 16; sell at 04, 12, 20) so that price drift does not favour
front- or back-loading on average. Arrival price `P0` = open of the first bar of the order.

## 5. Strategies

All strategies see only development-day profiles and, for the hybrid, bars already
observed in the current order (no look-ahead).

| Name | Schedule |
|---|---|
| TWAP | uniform over the horizon |
| VWAP | proportional to the development expected-volume profile over the horizon |
| AC | discretized Almgren-Chriss: the exact minimiser of the cost model below on the minute grid, with the time-varying development profiles (convex QP, scipy) |
| QUBO | the improved QUBO (exact binary encoding of per-slice lot counts, cost model below) solved by the improved simulated annealing, decoded to a schedule |
| Hybrid | starts from the QUBO schedule; at checkpoints re-solves the remaining lots over the remaining minutes with the same QUBO/SA, using profiles rescaled by what has been observed in the order so far |

Cost model optimised by AC, QUBO and Hybrid (bps of arrival notional):
`E[IS] = sum_k (q_k/N) (hbar_k + beta q_k / Vbar_k)` and
`Var[IS] = sum_k sigma_k^2 (R_k/N)^2`, `R_k` = lots not yet traded before minute k;
objective `E + lambda * Var`. **Primary: lambda = 0** (risk-neutral: the primary metric is
mean shortfall, so the optimiser targets the expected shortfall). Secondary: lambda set per
cell on development data so that `lambda * Var = E` for the TWAP schedule.

Method hyperparameters (QUBO slices, bits, units, penalty margin, SA sweeps/restarts/
temperatures, hybrid checkpoints and clipping) are tuned on synthetic QUBOs and development
days only, and frozen at the freeze commit.

## 6. Metrics

* **Primary:** implementation shortfall in bps of arrival notional, including opportunity
  cost, for each order (positive = cost). Per window, the paired difference strategy -
  baseline is averaged over the four cells.
* **Primary comparisons (12):** {QUBO, Hybrid} vs {TWAP, VWAP, AC} for each of the two
  symbols, at lambda = 0 and impact x1.
* **Secondary:** per-cell differences; spread, impact, timing and opportunity components;
  fill rate before clean-up; standard deviation of shortfall; QUBO-AC gap in the model
  objective (bps) and in realized shortfall; model-predicted vs realized cost; solver time;
  exact-integer-optimum (DP) schedule as a diagnostic; the dev evaluation (in-sample).

## 7. Statistics and decision rule

* Unit of observation: the window (72 per symbol on test days).
* For each primary comparison: mean paired difference, 95% percentile bootstrap CI over
  windows (10,000 seeded resamples), a day-clustered bootstrap CI as a robustness check, and
  the two-sided Wilcoxon signed-rank p-value over windows.
* Multiplicity: Holm correction over the 12 primary p-values (family-wise alpha = 0.05).
  Secondary analyses are reported with unadjusted p-values and labelled exploratory (the
  sensitivity family of section 8 is Holm-corrected within itself).
* Decision: a strategy **beats** a baseline on a symbol if the Holm-adjusted p < 0.05 and the
  mean difference is negative; it is **worse** if adjusted p < 0.05 and the mean is positive;
  otherwise **no detectable difference**, and we additionally say "practically equivalent"
  if the 95% CI lies within +-0.5 bps.
* Stochastic components: SA is seeded per order (seed derived from symbol, day, start and
  cell). The primary analysis uses one seed; a robustness run repeats QUBO with 5 seeds and
  reports the spread of the mean paired difference. The evaluator itself is deterministic.

## 8. Sensitivity (impact calibration)

The test evaluation is repeated with the evaluator's impact coefficient at 0.5x, 1x and 2x
beta, (a) with the optimisers still using 1x beta (misspecified model), and (b) with the
optimisers using the same multiple (correctly specified). Conclusions are stated as robust
only if the direction of every significant primary result is unchanged across multiples.

## 9. QAOA and hardware

* QAOA vs SA vs uniform random sampling is re-run on the improved encoding (small instances
  of the same cost model, n up to what exhaustive bounds allow on the simulator), reporting
  success probability, <H> ratio and the same-shot-budget uniform baseline, as in Phase 6.
* IBM ibm_fez raw counts, if recovered into `results/ibm_fez_recovered.jsonl`, are analysed
  against `toy_execution_qubo(n)` (`experiments/hardware_analysis.py`); if not recovered,
  the pipeline is tested on a clearly fake fixture and no hardware claim is made.

## Deviations

(none yet)
