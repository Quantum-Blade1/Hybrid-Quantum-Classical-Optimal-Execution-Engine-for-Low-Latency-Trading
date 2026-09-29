# Quantum Optimization for Optimal Trade Execution: research artifact

This repository is the code, data pipeline, results and manuscript for a reproducible study of
a hybrid quantum-classical trade-execution system. The system runs a fast execution loop that
never waits for the optimizer, and a slow path that solves an execution QUBO (simulated
annealing; exhaustive search and QAOA on Qiskit Aer offline) and publishes schedules through a
single latest-value slot. The study evaluates the resulting schedules against TWAP, VWAP and
discretized Almgren-Chriss under a pre-registered protocol on held-out Binance data. **The
result is negative:** the QUBO and hybrid schedules do not execute more cheaply than the
classical baselines, and QAOA on simulators does not beat uniform random sampling at finding
the optimum of the binary execution QUBO. The repository is built so that anyone can rerun
every number and check it.

## Headline findings

Full numbers with confidence intervals are in [`docs/RESULTS.md`](docs/RESULTS.md).

* **Held-out real data (12 days, BTCUSDT and LINKUSDT, evaluated once from a frozen commit):**
  none of the 12 pre-registered comparisons (QUBO and hybrid vs TWAP, VWAP, AC) is significant
  after Holm correction (every adjusted p = 1.00; mean differences -0.27 to +0.22 bps), and on
  BTCUSDT they are practically equivalent (all CIs within +-0.5 bps).
* **Why:** spread plus impact, the cost a schedule controls, is 0.16 bps (BTC) and 0.60 bps
  (LINK) for every strategy, while per-order shortfall has a standard deviation of 43-58 bps.
* **Synthetic market:** no setting shows a significant improvement over TWAP or VWAP.
* **Solvers:** the fixed simulated annealer finds the exact optimum in every seed up to 16
  variables; QAOA's small advantage over same-budget uniform sampling is not significant on the
  binary execution encoding, and best-of-shots metrics cannot separate solvers at these sizes.
* **Latency:** fast-path work is microseconds, but ticks start tens of milliseconds late
  because the pure-Python solver holds the GIL. This is CPython, not kernel bypass.
* **IBM hardware:** on `ibm_fez`, p = 1 QAOA at n = 4 and n = 10 (three runs each, recomputed
  from the recovered raw counts of every job) gives a better-than-uniform mean energy in every
  run, but a significantly higher probability of the optimum in only one of six; SA solves
  both sizes exactly.

The manuscript is [`paper/ieee/main.tex`](paper/ieee/main.tex) (IEEE Transactions on Quantum
Engineering format). [`docs/CLAIMS_AUDIT.md`](docs/CLAIMS_AUDIT.md) records every claim of the
earlier draft (archived in `paper/springer_qip_old/`) and what happened to it;
[`docs/PROTOCOL.md`](docs/PROTOCOL.md) is the pre-registered protocol;
[`docs/MATHEMATICAL_MODEL.md`](docs/MATHEMATICAL_MODEL.md) documents what the code computes.

## Install

Python 3.10 or newer.

```bash
git clone https://github.com/Quantum-Blade1/Hybrid-Quantum-Classical-Optimal-Execution-Engine-for-Low-Latency-Trading.git
cd Hybrid-Quantum-Classical-Optimal-Execution-Engine-for-Low-Latency-Trading
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # library, experiments, tests
pip install -e ".[hardware]"     # optional: IBM Quantum access (qiskit-ibm-runtime)
pip install -e ".[app]"          # optional: Streamlit dashboard
```

Building the PDF needs `latexmk` (TeX Live) or [`tectonic`](https://tectonic-typesetting.github.io).

## Reproduce

```bash
make quick          # every experiment at tiny sizes + every figure -> build/quick/ (about a minute; CI)
make all            # download Binance data (~255 MB, checksum-verified), run every experiment,
                    # draw every figure, write paper/ieee/numbers.tex (about 45 min of
                    # experiment time on an Apple M5, plus the download)
make paper          # check the manuscript sources, then build paper/ieee/main.pdf
make check-paper    # numbers.tex up to date; figures, macros, references and claims check
make check-figures  # every PDF in paper/figures/ has a registered producer
make lint test      # ruff, mypy, pytest (fast and slow suites)
```

Use `make PYTHON=.venv/bin/python ...` if `python` is not the environment's interpreter. One
experiment can be run on its own, e.g. `python -m experiments.walk_forward --quick`.

Provenance: every experiment writes `results/<name>/manifest.json` (config, seeds, git commit
and dirty flag, package versions, platform, wall time, SHA-256 of each output). Figures are
drawn only from `results/` by functions registered in `figures/registry.py` (a test enforces
that figure code reads results and computes nothing). Every number in the manuscript is a
macro in `paper/ieee/numbers.tex`, generated from `results/` by
`python -m experiments.paper_numbers`.

## Repository layout

```
src/qexec/           library: market data and calibration, execution engine and strategies,
                     cost model, QUBO encodings and solvers (exact, SA, greedy, QAOA),
                     runtime (fast/slow path, policy slot), statistics, IBM job analysis
experiments/         one script per result group -> results/<name>/ ; run_all runs them all;
                     paper_numbers.py and check_paper.py serve the manuscript
figures/             plot functions reading results/ only; registry.py maps PDFs to producers
results/             committed outputs with manifests (results/bench_*.json are historical,
                     unverifiable records and are not used)
paper/ieee/          IEEE TQE manuscript (main.tex, generated numbers.tex)
paper/figures/       figure PDFs (generated)
paper/springer_qip_old/  archived earlier draft (superseded; see the claims audit)
docs/                RESULTS, PROTOCOL, MATHEMATICAL_MODEL, CLAIMS_AUDIT
tests/               unit, property and regression tests mirroring src/qexec/
examples/, apps/     short API examples and a Streamlit dashboard
```

## IBM hardware counts

The `ibm_fez` records committed during the original run (`results/bench_hw_*.json`) are not
usable (empty count tables, inconsistent derived fields) and are kept only as superseded
history. The paper's hardware results come from the raw counts recovered from IBM Quantum:

1. `results/ibm_fez_recovered.jsonl` holds one JSON object per job record (job ID, status,
   creation time, number of measured bits, shots and counts in Qiskit bit order); the format
   is described in `src/qexec/hardware/jobs.py`.
2. `python -m experiments.hardware_analysis` (part of `make experiments`) recomputes every
   energy from the counts with the known toy QUBO, separates loop and final jobs by shot
   count, assigns loop jobs to runs by creation time, tests each final job against uniform
   sampling and compares it with the Aer results, and writes `results/hardware/` (its
   `manifest.json` lists every job ID).
3. `make figures` draws `fig_hw_ibm_success_prob.pdf`, `fig_hw_ibm_approx_ratio.pdf` and
   `fig_hw_ibm_trajectory.pdf`; `make paper-numbers` writes the hardware macros of Section VI-E.

## Citation

If you use this code or its results, please cite the paper (details to be updated on
publication):

```bibtex
@article{sharma_quantum_execution,
  author  = {Sharma, Krish Kumar and Rajarajeswari, S. and Chaudhari, Shilpa},
  title   = {Quantum Optimization for Optimal Trade Execution: A Reproducible Hybrid
             Architecture and a Pre-Registered Benchmark on Real Market Data},
  journal = {IEEE Transactions on Quantum Engineering},
  note    = {Manuscript in preparation},
  year    = {2026}
}
```

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
