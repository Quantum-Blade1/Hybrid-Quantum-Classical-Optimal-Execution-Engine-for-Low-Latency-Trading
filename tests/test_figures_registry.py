"""Figures are produced only from results/: registry coverage and the import boundary."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "figures"
# Modules a figure may use to *read* results. Anything under qexec/qiskit computes.
FORBIDDEN_PREFIXES = ("qexec", "qiskit", "qiskit_aer", "experiments")
ILLUSTRATIVE_MODULES = {"diagrams"}

registry = pytest.importorskip("figures.registry")


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize(
    "module", sorted(p for p in FIGURES_DIR.glob("*.py")), ids=lambda p: p.stem
)
def test_empirical_figure_modules_do_not_compute(module):
    if module.stem in ILLUSTRATIVE_MODULES:
        return
    bad = {n for n in _imports(module) if n.split(".")[0] in FORBIDDEN_PREFIXES}
    assert not bad, f"{module.name} imports {bad}; figures must read results/ only"


def test_every_paper_figure_has_a_producer_or_is_listed_as_removed():
    pdfs = {p.name for p in (ROOT / "paper" / "figures").glob("*.pdf")}
    registered = set(registry.by_file())
    assert pdfs <= registered, f"unregistered PDFs: {sorted(pdfs - registered)}"
    assert not set(registry.REMOVED) & registered


def test_paper_includegraphics_are_registered_or_removed():
    tex = (ROOT / "paper" / "main.tex").read_text()
    referenced = {
        line.split("{")[-1].rstrip("}").strip()
        for line in tex.splitlines()
        if "\\includegraphics" in line
    }
    unknown = referenced - set(registry.by_file()) - set(registry.REMOVED)
    assert not unknown, f"paper references figures with no producer: {sorted(unknown)}"


def test_illustrative_figures_have_no_inputs_and_empirical_ones_do():
    from experiments.run_all import EXPERIMENTS

    for spec in registry.FIGURES:
        if spec.kind == "illustrative":
            assert not spec.inputs
        else:
            assert spec.inputs and set(spec.experiments) <= set(EXPERIMENTS), spec.file
