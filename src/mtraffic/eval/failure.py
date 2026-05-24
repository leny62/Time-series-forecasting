"""Failure window detection (Task 3.VIII)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mtraffic.eda.style import PALETTE, apply


def rolling_mae(timestamps: pd.DatetimeIndex, y_true: np.ndarray, y_pred: np.ndarray, *, window: int) -> pd.Series:
    abs_err = np.abs(y_true - y_pred)
    return pd.Series(abs_err, index=timestamps).rolling(window=window, min_periods=window).mean()


def find_joint_failure(
    timestamps: pd.DatetimeIndex,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    *,
    window: int = 6,
    multiplier: float = 2.0,
) -> tuple[pd.Timestamp, pd.Timestamp, dict[str, float]] | None:
    """Find an overlapping window where every model exceeds its own per-area mean MAE by `multiplier`x.

    Returns (start_ts, end_ts, per_model_ratios) for the worst window, or None if no shared spike exists.
    """
    rolling: dict[str, pd.Series] = {}
    averages: dict[str, float] = {}
    for name, yp in predictions.items():
        ae = np.abs(y_true - yp)
        rolling[name] = pd.Series(ae, index=timestamps).rolling(window=window, min_periods=window).mean()
        averages[name] = float(ae.mean())
    if not rolling:
        return None
    # Boolean mask of windows that violate the threshold for every model.
    joint_mask: pd.Series | None = None
    for name, series in rolling.items():
        cur = series > averages[name] * multiplier
        joint_mask = cur if joint_mask is None else (joint_mask & cur)
    if joint_mask is None or not joint_mask.any():
        return None
    # Aggregate severity score: sum of normalized excess.
    excess = pd.DataFrame(
        {name: (rolling[name] / max(averages[name], 1e-9)) for name in rolling}
    )
    severity = excess.where(joint_mask, np.nan).sum(axis=1)
    ts_at_peak = severity.idxmax()
    half = pd.Timedelta(minutes=10 * window // 2)
    start, end = ts_at_peak - half, ts_at_peak + half
    ratios = {name: float(excess.loc[ts_at_peak, name]) for name in rolling}
    return pd.Timestamp(start), pd.Timestamp(end), ratios


def plot_failure_window(
    timestamps: pd.DatetimeIndex,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    window: tuple[pd.Timestamp, pd.Timestamp],
    out_png: Path,
    title: str,
) -> None:
    apply()
    half = pd.Timedelta(hours=2)
    start, end = window[0] - half, window[1] + half
    mask = (timestamps >= start) & (timestamps <= end)
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.plot(timestamps[mask], y_true[mask], color="#222", linewidth=1.2, label="actual")
    colors = [PALETTE["primary"], PALETTE["accent"], "#5b8e7d"]
    for (name, yp), c in zip(predictions.items(), colors):
        ax.plot(timestamps[mask], yp[mask], color=c, linewidth=1.0, alpha=0.9, label=name)
    ax.axvspan(window[0], window[1], color="#ffe0b3", alpha=0.5, label="joint failure window")
    ax.set_title(title)
    ax.set_xlabel("date")
    ax.set_ylabel("internet activity")
    ax.legend(loc="upper right", ncol=4)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
