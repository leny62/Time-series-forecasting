"""Spatial heatmap of total internet activity (Task 2.VI)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mtraffic.eda.style import PALETTE, apply
from mtraffic.data.loaders import area_totals


def to_grid(totals: pd.Series, grid_side: int = 100) -> np.ndarray:
    """Place per-area totals in a 100x100 numpy array.

    Convention used by the TIM dataset: square id 1 is the bottom-left corner; ids fill the
    grid by row (left-to-right) bottom-to-top. We render with origin lower so the city looks
    upright on the resulting heatmap.
    """
    grid = np.full((grid_side, grid_side), np.nan, dtype=np.float64)
    for sid, total in totals.items():
        idx = int(sid) - 1
        if 0 <= idx < grid_side * grid_side:
            r, c = divmod(idx, grid_side)
            grid[r, c] = float(total)
    return grid


def plot_spatial(totals: pd.Series, out_png: Path) -> dict[str, float]:
    apply()
    grid = to_grid(totals)
    with np.errstate(invalid="ignore"):
        log_grid = np.log10(np.where(grid > 0, grid, np.nan))

    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    im0 = ax[0].imshow(grid, origin="lower", cmap="magma", interpolation="nearest")
    ax[0].set_title("Total internet activity per area (linear)")
    ax[0].set_xlabel("grid column")
    ax[0].set_ylabel("grid row")
    fig.colorbar(im0, ax=ax[0], shrink=0.8, label="sum")

    im1 = ax[1].imshow(log_grid, origin="lower", cmap="viridis", interpolation="nearest")
    ax[1].set_title("log10 of total internet activity")
    ax[1].set_xlabel("grid column")
    ax[1].set_ylabel("grid row")
    fig.colorbar(im1, ax=ax[1], shrink=0.8, label="log10(sum)")

    # mark the top-traffic area
    top = int(totals.idxmax())
    idx = top - 1
    r, c = divmod(idx, 100)
    for a in (ax[0], ax[1]):
        a.plot(c, r, marker="o", markersize=8, mfc="none", mec="white", mew=1.6)

    fig.suptitle("Spatial distribution of two-month internet activity")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    return {
        "top_area": float(top),
        "top_value": float(totals.max()),
        "n_nonzero": float(np.sum(grid > 0)),
    }
