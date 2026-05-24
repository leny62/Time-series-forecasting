"""Matplotlib defaults used across EDA figures."""

from __future__ import annotations

import matplotlib as mpl

PALETTE = {
    "primary": "#1f4e79",
    "accent": "#c1502e",
    "muted": "#7d8a99",
    "weekend": "#dde6ed",
    "annot": "#3a5a40",
}


def apply() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#444",
            "axes.labelcolor": "#222",
            "axes.titleweight": "regular",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e8e8ec",
            "grid.linestyle": "-",
            "grid.linewidth": 0.7,
            "xtick.color": "#444",
            "ytick.color": "#444",
            "font.size": 10,
            "axes.titlesize": 11,
            "legend.frameon": False,
        }
    )
