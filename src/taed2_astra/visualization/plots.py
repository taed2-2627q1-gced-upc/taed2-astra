"""Figures for the report and the model card.

Plots are written to FIGURES_DIR so every figure in the report has a single
known origin and can be regenerated from the repository.

TODO(team): implement the plots the report needs.
"""

from pathlib import Path

from taed2_astra.config import FIGURES_DIR


def save_figure(fig, name: str) -> Path:
    """Write a matplotlib figure to reports/figures and return its path."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path
