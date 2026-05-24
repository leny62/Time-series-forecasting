"""Forecast plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mtraffic.eda.style import PALETTE, apply


def plot_forecast(
    timestamps: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_png: Path,
    title: str,
) -> None:
    apply()
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(timestamps, y_true, color=PALETTE["primary"], linewidth=1.0, label="actual")
    axes[0].plot(timestamps, y_pred, color=PALETTE["accent"], linewidth=1.0, alpha=0.95, label="predicted")
    axes[0].set_ylabel("internet activity")
    axes[0].set_title(title)
    axes[0].legend(loc="upper right")

    resid = y_true - y_pred
    axes[1].plot(timestamps, resid, color=PALETTE["muted"], linewidth=0.7)
    axes[1].axhline(0, color="#444", linewidth=0.6)
    axes[1].set_ylabel("residual")
    axes[1].set_xlabel("date")
    axes[1].xaxis.set_major_locator(mdates.DayLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def plot_combined(
    timestamps: pd.DatetimeIndex,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    out_png: Path,
    title: str,
) -> None:
    apply()
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.plot(timestamps, y_true, color="#222", linewidth=1.0, label="actual")
    colors = [PALETTE["primary"], PALETTE["accent"], "#5b8e7d"]
    for (name, yp), color in zip(predictions.items(), colors):
        ax.plot(timestamps, yp, color=color, linewidth=0.9, alpha=0.85, label=name)
    ax.set_title(title)
    ax.set_ylabel("internet activity")
    ax.set_xlabel("date")
    ax.legend(loc="upper right", ncol=4)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
