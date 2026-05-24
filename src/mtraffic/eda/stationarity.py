"""Stationarity: rolling stats, ADF and KPSS (Task 2.III)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

from mtraffic.eda.style import PALETTE, apply


def rolling_stats(series: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    mu = series.rolling(window=window, min_periods=window).mean()
    sigma = series.rolling(window=window, min_periods=window).std()
    return mu, sigma


def adf_summary(series: pd.Series) -> dict[str, float]:
    s = series.dropna().astype(float).values
    if s.size < 100:
        return {"statistic": float("nan"), "pvalue": float("nan"), "usedlag": float("nan")}
    stat, p, used, n, _, _ = adfuller(s, regression="ct", autolag="AIC")
    return {"statistic": float(stat), "pvalue": float(p), "usedlag": float(used), "nobs": float(n)}


def kpss_summary(series: pd.Series) -> dict[str, float]:
    s = series.dropna().astype(float).values
    if s.size < 100:
        return {"statistic": float("nan"), "pvalue": float("nan")}
    try:
        stat, p, lags, _ = kpss(s, regression="ct", nlags="auto")
    except Exception:
        return {"statistic": float("nan"), "pvalue": float("nan")}
    return {"statistic": float(stat), "pvalue": float(p), "lags": float(lags)}


def plot_rolling(
    series: pd.Series,
    window: int,
    out_png: Path,
    *,
    title: str = "Rolling mean and standard deviation",
) -> dict[str, float]:
    apply()
    mu, sigma = rolling_stats(series, window=window)
    fig, ax = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    ax[0].plot(series.index, series.values, color=PALETTE["muted"], linewidth=0.6, alpha=0.7, label="observed")
    ax[0].plot(mu.index, mu.values, color=PALETTE["primary"], linewidth=1.2, label=f"rolling mean (window={window})")
    ax[0].set_ylabel("internet activity")
    ax[0].set_title(title)
    ax[0].legend(loc="upper right")
    ax[1].plot(sigma.index, sigma.values, color=PALETTE["accent"], linewidth=1.0, label=f"rolling std (window={window})")
    ax[1].set_ylabel("rolling std")
    ax[1].set_xlabel("date")
    ax[1].legend(loc="upper right")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)

    adf = adf_summary(series)
    kp = kpss_summary(series)
    return {"adf_stat": adf["statistic"], "adf_p": adf["pvalue"], "kpss_stat": kp["statistic"], "kpss_p": kp["pvalue"]}


def plot_diff(series: pd.Series, lag: int, out_png: Path, *, title: str) -> dict[str, float]:
    """First difference (lag=1) or seasonal difference (lag=144) and plot."""
    apply()
    d = series.diff(lag).dropna()
    fig, ax = plt.subplots(1, 1, figsize=(11, 3.5))
    ax.plot(d.index, d.values, color=PALETTE["primary"], linewidth=0.6)
    ax.set_title(title)
    ax.set_xlabel("date")
    ax.set_ylabel("differenced value")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    adf = adf_summary(d)
    return {"adf_stat_diff": adf["statistic"], "adf_p_diff": adf["pvalue"]}
