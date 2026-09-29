# One-command reproduction. Full runs write results/ and paper/figures/ (committed);
# quick runs use tiny sizes and write build/quick/ (ignored), for CI and smoke tests.
PYTHON ?= python
export PYTHONPATH := src:.

.PHONY: all experiments figures quick experiments-quick figures-quick check-figures test lint

all: experiments figures

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
