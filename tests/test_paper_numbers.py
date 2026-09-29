"""paper/ieee/numbers.tex is generated from results/ and must match a fresh regeneration."""

import re
from pathlib import Path

import pytest

from experiments import paper_numbers

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
NUMBERS = ROOT / "paper" / "ieee" / "numbers.tex"


@pytest.mark.skipif(
    not (RESULTS / "real_data_test" / "comparisons.csv").exists(),
    reason="committed real-data results not present",
)
def test_numbers_tex_matches_results():
    assert NUMBERS.read_text() == paper_numbers.build(RESULTS), (
        "paper/ieee/numbers.tex is stale: run `python -m experiments.paper_numbers`"
    )


def test_macro_names_are_letters_only_and_unique():
    names = re.findall(r"\\newcommand\{\\([^}]*)\}", NUMBERS.read_text())
    assert names and all(n.isalpha() for n in names)
    assert len(names) == len(set(names))


@pytest.mark.parametrize(
    ("value", "kwargs", "expected"),
    [
        (0.1234, {}, "0.12"),
        (-0.1234, {}, "$-$0.12"),
        (0.1234, {"sign": True}, "+0.12"),
        (-0.001, {"sign": True}, "0.00"),
        (1.5, {"nd": 0}, "2"),
    ],
)
def test_fmt(value, kwargs, expected):
    assert paper_numbers.fmt(value, **kwargs) == expected


def test_spell_and_p_values():
    assert paper_numbers.spell(12) == "OneTwo"
    assert paper_numbers.fmt_p(0.999) == "1.00"
    assert paper_numbers.fmt_p(0.004) == "0.004"
    assert paper_numbers.fmt_p(1e-6).startswith("$<$")


def test_macros_reject_bad_names():
    macros = paper_numbers.Macros()
    macros["Good"] = "1"
    with pytest.raises(ValueError):
        macros["Bad1"] = "1"
    with pytest.raises(ValueError):
        macros["Good"] = "2"
