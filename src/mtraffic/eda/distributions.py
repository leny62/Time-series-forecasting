"""City-wide traffic distribution (Task 2.I)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from mtraffic.eda.style import PALETTE, apply
from mtraffic.data.loaders import area_totals


def compute_city_pdf(interim_dir: Path) -> pd.Series:
    return area_totals(interim_dir)


def plot_city_pdf(
    totals: pd.Series,
    out_png: Path,
    *,
    annotate_areas: list[int] | None = None,
) -> dict[str, float]:
    apply()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    x = totals.values.astype(float)
    x_pos = x[x > 0]
    logx = np.log10(x_pos)

    ax[0].hist(x_pos, bins=80, color=PALETTE["primary"], alpha=0.85, edgecolor="white")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("Total internet activity over the 50-day window")
    ax[0].set_ylabel("Number of areas (log scale)")
    ax[0].set_title("Histogram of per-area totals (linear x, log y)")

    kde = stats.gaussian_kde(logx)
    xs = np.linspace(logx.min(), logx.max(), 400)
    ys = kde(xs)
    ax[1].fill_between(xs, ys, color=PALETTE["primary"], alpha=0.35)
    ax[1].plot(xs, ys, color=PALETTE["primary"], linewidth=1.5)
    ax[1].set_xlabel("log10(total internet activity)")
    ax[1].set_ylabel("Estimated density")
    ax[1].set_title("Kernel density on log10(total)")

    if annotate_areas:
        for a in annotate_areas:
            if a in totals.index and totals.loc[a] > 0:
                v = np.log10(float(totals.loc[a]))
                ax[1].axvline(v, color=PALETTE["accent"], alpha=0.8, linewidth=1.0, linestyle="--")
                ax[1].text(v, ax[1].get_ylim()[1] * 0.9, f"  {a}", color=PALETTE["accent"], fontsize=9, va="top")

    fig.suptitle("City-wide distribution of total internet traffic across 10,000 areas")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)

    stats_out = {
        "n_areas": int(x.size),
        "n_positive": int(x_pos.size),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "std": float(x.std(ddof=0)),
        "min": float(x.min()),
        "max": float(x.max()),
        "skew": float(stats.skew(x_pos, bias=False)),
        "kurtosis": float(stats.kurtosis(x_pos, bias=False)),
        "p1": float(np.percentile(x, 1)),
        "p5": float(np.percentile(x, 5)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
    }
    return stats_out
