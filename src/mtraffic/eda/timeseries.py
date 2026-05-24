"""Per-area time-series plots (Task 2.II)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from mtraffic.eda.style import PALETTE, apply
from mtraffic.data.loaders import load_area_series


def plot_three_areas_two_weeks(
    interim_dir: Path,
    areas: list[int],
    out_png: Path,
    *,
    start: datetime,
    days: int = 14,
    labels: list[str] | None = None,
) -> None:
    apply()
    end = start + pd.Timedelta(days=days) - pd.Timedelta(minutes=10)
    fig, axes = plt.subplots(len(areas), 1, figsize=(12, 7), sharex=True)
    if len(areas) == 1:
        axes = [axes]
    for ax, area, name in zip(axes, areas, labels or [f"area {a}" for a in areas]):
        s = load_area_series(interim_dir, area, start=start, end=end)
        ax.plot(s.index, s.values, color=PALETTE["primary"], linewidth=0.8, alpha=0.95, label="10 minute traffic")
        smoothed = s.rolling(window=6, min_periods=1).mean()
        ax.plot(smoothed.index, smoothed.values, color=PALETTE["accent"], linewidth=1.2, alpha=0.9, label="1 hour moving average")
        ax.set_title(f"{name} (square id {area})")
        ax.set_ylabel("internet activity")
        # shade weekends
        for d in pd.date_range(start, end, freq="D"):
            if d.dayofweek >= 5:
                ax.axvspan(d, d + pd.Timedelta(days=1), color=PALETTE["weekend"], alpha=0.5, zorder=0)
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.tick_params(axis="x", labelrotation=0)
    axes[0].legend(loc="upper right", ncol=2)
    axes[-1].set_xlabel("date")
    fig.suptitle(f"Internet traffic across three areas, first {days} days")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
