"""STL decomposition (Task 2.IV)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from mtraffic.eda.style import PALETTE, apply


def stl_decompose(series: pd.Series, period: int, robust: bool = True) -> pd.DataFrame:
    s = series.astype(float).interpolate(limit_direction="both")
    res = STL(s, period=period, robust=robust).fit()
    return pd.DataFrame(
        {"observed": s.values, "trend": res.trend.values, "seasonal": res.seasonal.values, "resid": res.resid.values},
        index=s.index,
    )


def strengths(decomp: pd.DataFrame) -> dict[str, float]:
    r = decomp["resid"]
    s = decomp["seasonal"]
    t = decomp["trend"]
    def var(x: pd.Series) -> float:
        return float(np.nanvar(x.values))
    fs = max(0.0, 1.0 - var(r) / max(var(s + r), 1e-12))
    ft = max(0.0, 1.0 - var(r) / max(var(t + r), 1e-12))
    return {"seasonal_strength": fs, "trend_strength": ft}


def plot_decomposition(decomp: pd.DataFrame, out_png: Path, title: str) -> None:
    apply()
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    for ax, col, color in zip(axes, ["observed", "trend", "seasonal", "resid"], [PALETTE["primary"], PALETTE["accent"], PALETTE["primary"], PALETTE["muted"]]):
        ax.plot(decomp.index, decomp[col].values, color=color, linewidth=0.7)
        ax.set_ylabel(col)
    axes[0].set_title(title)
    axes[-1].set_xlabel("date")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
