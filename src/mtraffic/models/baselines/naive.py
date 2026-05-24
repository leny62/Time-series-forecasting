"""Naive baselines: last value and seasonal naive."""

from __future__ import annotations

import pandas as pd


class LastValue:
    name = "naive_last"

    def fit(self, history: pd.Series) -> None:
        return None

    def predict_one_step(self, history: pd.Series, target_ts: pd.Timestamp) -> float:
        return float(history.iloc[-1])


class SeasonalNaive:
    """Predict yhat_{t+1} = y_{t+1-period}. Period in 10 minute steps."""

    def __init__(self, period: int, name: str | None = None) -> None:
        self.period = period
        self.name = name or f"naive_period_{period}"

    def fit(self, history: pd.Series) -> None:
        return None

    def predict_one_step(self, history: pd.Series, target_ts: pd.Timestamp) -> float:
        lookup_ts = target_ts - pd.Timedelta(minutes=10 * self.period)
        if lookup_ts in history.index:
            return float(history.loc[lookup_ts])
        # fall back to the last available step
        return float(history.iloc[-1])
