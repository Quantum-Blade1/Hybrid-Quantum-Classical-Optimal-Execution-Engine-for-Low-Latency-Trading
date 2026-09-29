# Full runs write results/ and paper/figures/ (committed); quick runs write build/quick/ (ignored).
PYTHON ?= python
export PYTHONPATH := src:.

.PHONY: all data experiments figures quick experiments-quick figures-quick check-figures test lint \
	paper-numbers check-paper paper-todos paper

all: data experiments figures paper-numbers

# Public Binance aggTrades (~255 MB, git-ignored); experiments that need them skip without them.
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

paper-numbers:
	$(PYTHON) -m experiments.paper_numbers

check-paper:
	$(PYTHON) -m experiments.paper_numbers --check
	$(PYTHON) -m experiments.check_paper

paper-todos:
	$(PYTHON) -m experiments.check_paper --todos

paper: check-paper
	cd paper/ieee && if command -v latexmk >/dev/null 2>&1; then \
		latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex; \
	elif command -v tectonic >/dev/null 2>&1; then \
		tectonic --keep-logs main.tex; \
	else \
		echo "No TeX toolchain (latexmk or tectonic) found; ran the source checks only."; \
	fi
