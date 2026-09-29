"""Illustrative figures: they depict structure and make no empirical claim.

These are the only figure functions allowed to compute from qexec directly (fig02 builds
a QUBO matrix to show its sparsity pattern; no solver or simulator is run).
"""

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

from figures.common import Results, new_figure
from qexec.optimization.qubo import ExecutionQUBO, QUBOConfig


def fig01_system_architecture(_: Results) -> Figure:
    """Block diagram of the fast/slow path architecture. Timing labels are design targets."""
    fig, ax = new_figure(figsize=(7, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 6)
    ax.axis("off")
    ax.set_title("Hybrid Quantum-Classical Execution Architecture", fontsize=11, fontweight="bold")
    boxes = [
        (0.5, 4.5, 2.2, 1.0, "Market Data\nTick Stream", "#E3F2FD"),
        (0.5, 2.5, 2.2, 1.0, "Microstructure\nAnalyzer", "#E8F5E9"),
        (0.5, 0.5, 2.2, 1.0, "Adaptive Risk\nManager", "#FFF3E0"),
        (3.5, 3.5, 2.2, 1.2, "QUBO\nFormulation\n(6-term cost)", "#F3E5F5"),
        (3.5, 1.2, 2.2, 1.2, "Classical Solver\n(SA; QAOA offline)", "#FCE4EC"),
        (6.5, 4.0, 2.8, 0.8, "Fast Path (tick loop)", "#E0F7FA"),
        (6.5, 2.5, 2.8, 0.8, "Policy Queue (latest value)", "#FFF9C4"),
        (6.5, 1.0, 2.8, 0.8, "Slow Path (optimizer thread)", "#FFEBEE"),
        (6.5, -0.2, 2.8, 0.8, "Execution Orders", "#E8EAF6"),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.1", facecolor=color, edgecolor="#333", lw=0.8
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7, fontweight="bold")
    arrows = [
        (1.6, 4.5, 0, -0.4),
        (1.6, 2.5, 0, -0.4),
        (2.7, 3.0, 0.7, 0.8),
        (2.7, 1.0, 0.7, 0.5),
        (5.7, 4.0, 0.7, 0.2),
        (5.7, 2.0, 0.7, 0.2),
        (7.9, 4.0, 0, -0.5),
        (7.9, 2.5, 0, -0.5),
        (7.9, 1.0, 0, -0.5),
    ]
    for x, y, dx, dy in arrows:
        ax.annotate(
            "", xy=(x + dx, y + dy), xytext=(x, y), arrowprops={"arrowstyle": "->", "color": "#555"}
        )
    return fig


def fig02_qubo_matrix_structure(_: Results) -> Figure:
    """|Q| and sparsity of a 20-variable execution QUBO (5 slices x 4 levels)."""
    qubo = ExecutionQUBO(
        QUBOConfig(
            total_shares=1000,
            num_time_slices=5,
            num_venues=1,
            quantity_levels=[0, 100, 200, 300],
            equality_penalty=100.0,
            capacity_penalty=50.0,
        )
    )
    Q = qubo.build_qubo_matrix()
    fig, (ax1, ax2) = new_figure(1, 2, figsize=(7, 3))
    im = ax1.imshow(np.abs(Q), cmap="YlOrRd", aspect="equal")
    ax1.set_title(f"|Q| ({Q.shape[0]} variables)")
    ax1.set_xlabel("Variable index")
    ax1.set_ylabel("Variable index")
    fig.colorbar(im, ax=ax1, shrink=0.8)
    ax2.spy(Q, markersize=1.5, color="#1565C0")
    ax2.set_title(f"Sparsity pattern (nnz = {np.count_nonzero(Q)})")
    ax2.set_xlabel("Variable index")
    fig.tight_layout()
    return fig
