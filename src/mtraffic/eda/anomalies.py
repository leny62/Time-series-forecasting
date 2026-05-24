"""Anomaly screening using STL residuals and seasonal-naive deviation (Task 2.VII)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from mtraffic.eda.style import PALETTE, apply


def stl_residual_outliers(series: pd.Series, *, period: int = 144, z_threshold: float = 4.0) -> pd.DataFrame:
    s = series.astype(float).interpolate(limit_direction="both")
    res = STL(s, period=period, robust=True).fit()
    r = res.resid
    z = (r - r.mean()) / max(r.std(ddof=0), 1e-12)
    mask = z.abs() > z_threshold
    out = pd.DataFrame({"ts": s.index[mask], "value": s.values[mask.values], "residual": r.values[mask.values], "z": z.values[mask.values]})
    return out


def seasonal_naive_drops(series: pd.Series, *, period: int = 1008, threshold: float = 0.6, min_consecutive: int = 6) -> pd.DataFrame:
    s = series.astype(float).interpolate(limit_direction="both")
    baseline = s.shift(period)
    rel = (s - baseline) / np.where(baseline.abs() < 1e-6, np.nan, baseline)
    flagged = rel < -threshold
    # find runs of length >= min_consecutive
    runs: list[tuple[pd.Timestamp, pd.Timestamp, float]] = []
    run_start: int | None = None
    arr = flagged.values
    for i, v in enumerate(arr):
        if v and run_start is None:
            run_start = i
        elif not v and run_start is not None:
            if i - run_start >= min_consecutive:
                runs.append((s.index[run_start], s.index[i - 1], float(rel.iloc[run_start:i].mean())))
            run_start = None
    if run_start is not None and len(arr) - run_start >= min_consecutive:
        runs.append((s.index[run_start], s.index[-1], float(rel.iloc[run_start:].mean())))
    return pd.DataFrame(runs, columns=["start", "end", "mean_relative_drop"])


def plot_anomalies(series: pd.Series, outliers: pd.DataFrame, drops: pd.DataFrame, out_png: Path, title: str) -> None:
    apply()
    fig, ax = plt.subplots(1, 1, figsize=(12, 4.5))
    ax.plot(series.index, series.values, color=PALETTE["muted"], linewidth=0.5, alpha=0.7, label="observed")
    if not outliers.empty:
        ax.scatter(outliers["ts"], outliers["value"], color=PALETTE["accent"], s=10, label="STL outliers (|z| > 4)")
    for _, row in drops.iterrows():
        ax.axvspan(row["start"], row["end"], color="#ffe0b3", alpha=0.4, zorder=0)
    if not drops.empty:
        ax.axvspan(drops["start"].iloc[0], drops["end"].iloc[0], color="#ffe0b3", alpha=0.4, label="seasonal-naive drops")
    ax.set_title(title)
    ax.set_xlabel("date")
    ax.set_ylabel("internet activity")
    ax.legend(loc="upper right")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
