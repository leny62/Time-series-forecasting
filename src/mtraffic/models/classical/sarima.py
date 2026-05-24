"""ARIMA with dual Fourier exogenous regressors.

The high-frequency daily seasonality (period 144 at 10 minute resolution) and the weekly
modulation (period 1008) are both encoded as deterministic Fourier sine/cosine pairs in the
exogenous matrix. The endogenous component is a small ARIMA(p,d,q), keeping the state space
compact and the fit cheap on CPU. This is the classical dynamic harmonic regression approach
recommended for high-frequency seasonal data; see Hyndman and Athanasopoulos, FPP3.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX


def fourier_block(ts: pd.DatetimeIndex, *, period: int, terms: int, anchor: pd.Timestamp | None = None) -> np.ndarray:
    """Sine/cosine pairs for a given period (in 10 minute steps), evaluated at `ts`."""
    a = anchor if anchor is not None else ts[0]
    steps = ((ts - a) // pd.Timedelta(minutes=10)).to_numpy().astype(np.float64)
    cols = []
    for k in range(1, terms + 1):
        cols.append(np.sin(2.0 * np.pi * k * steps / period))
        cols.append(np.cos(2.0 * np.pi * k * steps / period))
    return np.stack(cols, axis=1)


def build_exog(ts: pd.DatetimeIndex, *, daily_terms: int, weekly_terms: int, anchor: pd.Timestamp | None = None) -> np.ndarray:
    """Stack daily (period 144) and weekly (period 1008) Fourier pairs."""
    a = anchor if anchor is not None else ts[0]
    daily = fourier_block(ts, period=144, terms=daily_terms, anchor=a)
    weekly = fourier_block(ts, period=1008, terms=weekly_terms, anchor=a)
    return np.hstack([daily, weekly])


@dataclass(slots=True)
class SarimaModel:
    """Fourier-ARIMA dynamic harmonic regression model."""

    order: tuple[int, int, int] = (2, 0, 2)
    daily_terms: int = 4
    weekly_terms: int = 3
    use_log1p: bool = True
    name: str = "sarima"
    _result: Any | None = field(default=None, init=False, repr=False)
    _scaler_log: bool = field(default=False, init=False, repr=False)
    _history: pd.Series | None = field(default=None, init=False, repr=False)
    _anchor: pd.Timestamp | None = field(default=None, init=False, repr=False)

    def _transform(self, y: pd.Series) -> pd.Series:
        if self._scaler_log:
            return np.log1p(y.astype(float))
        return y.astype(float)

    def _inverse(self, y: np.ndarray) -> np.ndarray:
        if self._scaler_log:
            return np.expm1(y)
        return y

    def fit(self, history: pd.Series) -> None:
        self._scaler_log = bool(self.use_log1p and (history.min() >= 0))
        self._history = history.copy()
        self._anchor = pd.Timestamp(history.index[0])
        y = self._transform(history)
        exog = build_exog(history.index, daily_terms=self.daily_terms, weekly_terms=self.weekly_terms, anchor=self._anchor)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", category=ValueWarning)
            model = SARIMAX(
                y.values,
                exog=exog,
                order=self.order,
                seasonal_order=(0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False,
                trend=None,
            )
            self._result = model.fit(disp=False, maxiter=100, method="lbfgs")

    def predict_one_step(self, history: pd.Series, target_ts: pd.Timestamp) -> float:
        if self._result is None or self._history is None or self._anchor is None:
            raise RuntimeError("Call fit() before predict_one_step().")
        last_fitted_ts = self._history.index[-1]
        if history.index[-1] > last_fitted_ts:
            new = history.loc[history.index > last_fitted_ts]
            if len(new) > 0:
                y_new = self._transform(new).values
                exog_new = build_exog(
                    pd.DatetimeIndex(new.index),
                    daily_terms=self.daily_terms,
                    weekly_terms=self.weekly_terms,
                    anchor=self._anchor,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=ConvergenceWarning)
                    warnings.simplefilter("ignore", category=ValueWarning)
                    self._result = self._result.append(y_new, exog=exog_new, refit=False)
                self._history = pd.concat([self._history, new])
        exog_next = build_exog(
            pd.DatetimeIndex([target_ts]),
            daily_terms=self.daily_terms,
            weekly_terms=self.weekly_terms,
            anchor=self._anchor,
        )
        yhat = self._result.get_forecast(steps=1, exog=exog_next).predicted_mean
        return float(self._inverse(np.asarray(yhat))[0])
