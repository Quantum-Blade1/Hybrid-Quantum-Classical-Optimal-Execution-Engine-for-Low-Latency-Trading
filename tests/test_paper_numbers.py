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


def test_manuscript_source_checks_pass():
    from experiments import check_paper

    assert check_paper.check(ROOT / "paper" / "ieee" / "main.tex") == []


def test_source_checks_catch_problems(tmp_path):
    from experiments import check_paper

    tex = tmp_path / "main.tex"
    (tmp_path / "numbers.tex").write_text("\\newcommand{\\Known}{1}\n")
    tex.write_text(
        "\\begin{document}\\begin{figure}\n"
        "\\includegraphics{fig25_policy_staleness.pdf}\\end{document}\n"
        "\\Known \\Unknown \\ref{nowhere} \\cite{ghost} production-ready\n"
    )
    problems = "\n".join(check_paper.check(tex))
    assert "does not close" in problems
    assert "removed as fabricated" in problems
    assert "\\Unknown" in problems and "\\Known" not in problems.replace("\\Unknown", "")
    assert "nowhere" in problems and "ghost" in problems
    assert "production-ready" in problems


def test_author_todos_are_listed_and_other_todos_fail(tmp_path):
    from experiments import check_paper

    raw = "text % TODO(author): confirm funding\nmore % TODO fix this\nTODO(author): visible\n"
    assert check_paper.author_todos(raw) == [(1, "confirm funding"), (3, "visible")]
    problems = check_paper.check_todos(raw)
    assert len(problems) == 2
    assert "line 2" in problems[0] and "line 3" in problems[1]
