"""Build every registered figure from results/ into paper/figures/.

Usage:
    python -m figures.make_figures [--results-dir results] [--output-dir paper/figures]
                                   [--only fig06 fig21] [--check]

`--check` only verifies the registry against the output directory: every PDF there must
have a producer. Exits non-zero if a figure fails, an input is missing, or an unregistered
PDF is found.
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

import matplotlib.pyplot as plt

from figures.common import Results
from figures.registry import FIGURES, REMOVED, FigureSpec

PDF_METADATA = {"CreationDate": None, "Creator": "figures.make_figures"}


def unregistered_pdfs(output_dir: Path) -> list[str]:
    registered = {spec.file for spec in FIGURES}
    return sorted(p.name for p in output_dir.glob("*.pdf") if p.name not in registered)


def build(spec: FigureSpec, results: Results, output_dir: Path) -> Path:
    missing = [i for i in spec.inputs if not (results.root / i).exists()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {', '.join(missing)}")
    fig = spec.plot(results)
    path = output_dir / spec.file
    fig.savefig(path, metadata=PDF_METADATA)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper/figures"))
    parser.add_argument("--only", nargs="*", default=None, help="file-name prefixes")
    parser.add_argument("--check", action="store_true", help="registry check only")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    orphans = unregistered_pdfs(args.output_dir)
    for name in orphans:
        print(f"UNREGISTERED: {args.output_dir / name} has no producer in figures/registry.py")
    for name, reason in sorted(REMOVED.items()):
        print(f"removed (no producer by design): {name} [{reason}]")
    if args.check:
        sys.exit(1 if orphans else 0)

    results = Results(args.results_dir)
    specs = [s for s in FIGURES if args.only is None or s.file.startswith(tuple(args.only))]
    failures = 0
    start = time.perf_counter()
    for spec in specs:
        t0 = time.perf_counter()
        # Keep building the other figures if one fails; the exit code reports it.
        try:
            path = build(spec, results, args.output_dir)
        except Exception:
            traceback.print_exc()
            print(f"FAILED {spec.file}")
            failures += 1
            continue
        print(f"{spec.file:<45} {time.perf_counter() - t0:5.1f}s  [{spec.kind}] -> {path}")
    print(f"{len(specs) - failures}/{len(specs)} figures in {time.perf_counter() - start:.1f}s")
    if failures or orphans:
        sys.exit(1)


if __name__ == "__main__":
    main()
