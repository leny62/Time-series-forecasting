"""One-step-ahead walk-forward evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd


class OneStepModel(Protocol):
    """Protocol all models implement to plug into walk-forward."""

    def fit(self, history: pd.Series) -> None: ...

    def predict_one_step(self, history: pd.Series, target_ts: pd.Timestamp) -> float: ...


@dataclass(slots=True)
class WalkForwardResult:
    timestamps: pd.DatetimeIndex
    y_true: np.ndarray
    y_pred: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"ts": self.timestamps, "y_true": self.y_true, "y_pred": self.y_pred})


def run(
    model: OneStepModel,
    series: pd.Series,
    test_start: datetime,
    test_end: datetime,
) -> WalkForwardResult:
    """Roll a one-step-ahead forecast across the test window.

    At each test timestamp t the model sees the actual history up to (but not including) t and
    predicts y_t. The window slides one step at a time.
    """
    s = series.sort_index()
    test_idx = s.loc[test_start:test_end].index
    if len(test_idx) == 0:
        raise ValueError(f"Empty test window between {test_start} and {test_end}")
    y_true = np.empty(len(test_idx), dtype=np.float64)
    y_pred = np.empty(len(test_idx), dtype=np.float64)
    for i, ts in enumerate(test_idx):
        history = s.loc[: ts - pd.Timedelta(minutes=10)]
        y_pred[i] = float(model.predict_one_step(history, pd.Timestamp(ts)))
        y_true[i] = float(s.loc[ts])
    return WalkForwardResult(timestamps=pd.DatetimeIndex(test_idx), y_true=y_true, y_pred=y_pred)
