# One-command reproduction. Full runs write results/ and paper/figures/ (committed);
# quick runs use tiny sizes and write build/quick/ (ignored), for CI and smoke tests.
PYTHON ?= python
export PYTHONPATH := src:.

.PHONY: all data experiments figures quick experiments-quick figures-quick check-figures test lint

all: data experiments figures

# Public Binance aggTrades for docs/PROTOCOL.md (~255 MB, checksum-verified, git-ignored).
# Experiments that need them skip with a message when they are absent.
data:
	$(PYTHON) -m experiments.fetch_binance

experiments:
	$(PYTHON) -m experiments.run_all

figures:
	$(PYTHON) -m figures.make_figures --results-dir results --output-dir paper/figures

quick: experiments-quick figures-quick

experiments-quick:
	$(PYTHON) -m experiments.run_all --quick --results-dir build/quick/results

figures-quick:
	$(PYTHON) -m figures.make_figures --results-dir build/quick/results --output-dir build/quick/figures

check-figures:
	$(PYTHON) -m figures.make_figures --check --output-dir paper/figures

lint:
	ruff check . && ruff format --check . && mypy src/qexec

test:
	$(PYTHON) -m pytest && $(PYTHON) -m pytest -m slow
