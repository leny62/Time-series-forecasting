"""ACF and PACF plots (Task 2.V)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf

from mtraffic.eda.style import PALETTE, apply


def plot_acf_pacf(
    series: pd.Series,
    out_acf: Path,
    out_pacf: Path,
    *,
    acf_lags: int = 2016,
    pacf_lags: int = 60,
    title_prefix: str = "",
) -> dict[str, float]:
    """Plot ACF (line, fast) and PACF (statsmodels stems). Numeric acf values returned up to acf_lags."""
    apply()
    s = series.dropna().astype(float).values
    max_lag = int(min(acf_lags, len(s) - 1))

    # Fast ACF: compute the array once, draw as a line plot. Avoids statsmodels' stem rendering
    # which is O(n) matplotlib artists and very slow for n > 500.
    arr, confint = acf(s, nlags=max_lag, fft=True, alpha=0.05)
    fig_acf, ax_acf = plt.subplots(1, 1, figsize=(12, 4))
    lags = np.arange(max_lag + 1)
    band = (confint[:, 1] - confint[:, 0]) / 2.0
    ax_acf.fill_between(lags, -band, band, color=PALETTE["primary"], alpha=0.12, label="95% band")
    ax_acf.plot(lags, arr, color=PALETTE["primary"], linewidth=0.9)
    for marker, label in [(144, "daily"), (1008, "weekly")]:
        if marker <= max_lag:
            ax_acf.axvline(marker, color=PALETTE["accent"], alpha=0.5, linewidth=0.8, linestyle="--")
            ax_acf.text(marker, ax_acf.get_ylim()[1] * 0.9, f"  {label}", color=PALETTE["accent"], fontsize=9, va="top")
    ax_acf.axhline(0, color="#aaa", linewidth=0.6)
    ax_acf.set_xlim(0, max_lag)
    ax_acf.set_title(f"{title_prefix} Autocorrelation up to lag {max_lag}")
    ax_acf.set_xlabel("lag (10 minute steps)")
    ax_acf.set_ylabel("acf")
    ax_acf.legend(loc="upper right")
    out_acf.parent.mkdir(parents=True, exist_ok=True)
    fig_acf.savefig(out_acf)
    plt.close(fig_acf)

    fig_p, ax_p = plt.subplots(1, 1, figsize=(8, 4))
    plot_pacf(pd.Series(s), ax=ax_p, lags=min(pacf_lags, len(s) // 2 - 1), alpha=0.05, method="ywm")
    ax_p.set_title(f"{title_prefix} Partial autocorrelation up to lag {pacf_lags}")
    ax_p.set_xlabel("lag (10 minute steps)")
    out_pacf.parent.mkdir(parents=True, exist_ok=True)
    fig_p.savefig(out_pacf)
    plt.close(fig_p)

    return {
        "acf_lag_144": float(arr[144]) if len(arr) > 144 else float("nan"),
        "acf_lag_1008": float(arr[1008]) if len(arr) > 1008 else float("nan"),
    }
