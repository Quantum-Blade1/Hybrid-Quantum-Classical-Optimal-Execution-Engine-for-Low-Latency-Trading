"""Source-level checks of paper/ieee/main.tex that do not need a TeX installation."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from figures.registry import REMOVED, by_file

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEX = ROOT / "paper" / "ieee" / "main.tex"
FIGURES_DIR = ROOT / "paper" / "figures"

# Upper-case control words provided by LaTeX, IEEEtran or loaded packages.
BUILTIN = {
    "IEEEPARstart",
    "IEEEkeywords",
    "MakeLowercase",
    "LaTeX",
    "TeX",
    "Big",
    "Bigl",
    "Bigr",
    "Bigg",
    "Biggl",
    "Biggr",
    "Delta",
    "Gamma",
    "Lambda",
    "Omega",
    "Phi",
    "Pi",
    "Psi",
    "Sigma",
    "Theta",
    "Upsilon",
    "Xi",
    "State",
    "Statex",
    "If",
    "Else",
    "ElsIf",
    "EndIf",
    "For",
    "EndFor",
    "While",
    "EndWhile",
    "Comment",
    "Require",
    "Ensure",
    "Return",
}
# Phrases from the retracted Springer draft that must not come back (docs/CLAIMS_AUDIT.md).
RETRACTED = (
    "first ever",
    "production-ready",
    "production quantum hardware",
    "DPDK",
    "kernel bypass",
    "kernel-bypass",
    "perfect approximation ratio",
    "15--40",
    "15-40",
    "80~$\\mu$s",
    "80 us",
    "DQC",
    "lock-free policy queue",
)
ALLOWED_CONTEXT = ("not a kernel-bypass system",)


def _strip_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in text.splitlines())


def check_environments(text: str) -> list[str]:
    problems = []
    stack: list[str] = []
    for match in re.finditer(r"\\(begin|end)\{([^}]+)\}", text):
        kind, env = match.groups()
        if kind == "begin":
            stack.append(env)
        elif not stack or stack[-1] != env:
            problems.append(f"\\end{{{env}}} does not close {stack[-1] if stack else 'nothing'}")
        else:
            stack.pop()
    problems += [f"\\begin{{{env}}} is never closed" for env in stack]
    return problems


def check_figures(text: str) -> list[str]:
    registered = by_file()
    problems = []
    names = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text)
    if not names:
        problems.append("no \\includegraphics found")
    for name in names:
        file = name if name.endswith(".pdf") else f"{name}.pdf"
        if file in REMOVED:
            problems.append(f"{file} was removed as fabricated ({REMOVED[file]})")
        elif file not in registered:
            problems.append(f"{file} has no producer in figures/registry.py")
        elif not (FIGURES_DIR / file).exists():
            problems.append(f"{file} is registered but missing: run `make figures`")
    return problems


def check_macros(text: str, numbers: str) -> list[str]:
    defined = set(re.findall(r"\\(?:re)?newcommand\{\\([A-Za-z]+)\}", numbers + text))
    used = set(re.findall(r"\\([A-Z][A-Za-z]*)", text))
    return [f"undefined macro \\{name}" for name in sorted(used - defined - BUILTIN)]


def check_references(text: str) -> list[str]:
    labels = set(re.findall(r"\\label\{([^}]+)\}", text))
    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", text))
    bibitems = set(re.findall(r"\\bibitem\{([^}]+)\}", text))
    cites = {
        k.strip() for group in re.findall(r"\\cite\{([^}]+)\}", text) for k in group.split(",")
    }
    problems = [f"\\ref to undefined label {r}" for r in sorted(refs - labels)]
    problems += [f"\\cite of undefined bibitem {c}" for c in sorted(cites - bibitems)]
    problems += [f"bibitem {b} is never cited" for b in sorted(bibitems - cites)]
    return problems


def check_retracted(text: str) -> list[str]:
    lowered = " ".join(text.split())
    for allowed in ALLOWED_CONTEXT:
        lowered = lowered.replace(allowed, "")
    return [
        f"retracted claim reappears: {phrase!r}"
        for phrase in RETRACTED
        if phrase.lower() in lowered.lower()
    ]


AUTHOR_TODO = "TODO(author)"


def author_todos(raw: str) -> list[tuple[int, str]]:
    """(line number, text) of every ``TODO(author)`` note in the raw source."""
    return [
        (number, line.split(AUTHOR_TODO, 1)[1].lstrip(": ").strip())
        for number, line in enumerate(raw.splitlines(), start=1)
        if AUTHOR_TODO in line
    ]


def check_todos(raw: str) -> list[str]:
    problems = []
    for number, line in enumerate(raw.splitlines(), start=1):
        rest = line.replace(AUTHOR_TODO, "")
        if "TODO" in rest:
            problems.append(f"line {number}: TODO not written as {AUTHOR_TODO}")
        elif AUTHOR_TODO in line and AUTHOR_TODO not in line[line.find("%") :]:
            problems.append(f"line {number}: {AUTHOR_TODO} outside a LaTeX comment")
    return problems


def check(tex_path: Path) -> list[str]:
    raw = tex_path.read_text()
    text = _strip_comments(raw)
    numbers_path = tex_path.parent / "numbers.tex"
    numbers = numbers_path.read_text() if numbers_path.exists() else ""
    problems = []
    if not numbers:
        problems.append(f"{numbers_path} missing: run `python -m experiments.paper_numbers`")
    problems += check_environments(text)
    problems += check_figures(text)
    problems += check_macros(text, numbers)
    problems += check_references(text)
    problems += check_retracted(text)
    problems += check_todos(raw)
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--todos", action="store_true", help="only list TODO(author) notes")
    args = parser.parse_args()
    todos = author_todos(args.tex.read_text())
    if args.todos:
        for number, note in todos:
            print(f"{args.tex.name}:{number}: {note}")
        return
    problems = check(args.tex)
    for problem in problems:
        print(f"PAPER: {problem}")
    if problems:
        sys.exit(1)
    print(f"{args.tex}: environments, figures, macros, references and retracted claims OK")
    if todos:
        print(f"{len(todos)} {AUTHOR_TODO} notes (allowed; `make paper-todos` lists them):")
        for number, note in todos:
            print(f"  line {number}: {note}")


if __name__ == "__main__":
    main()
