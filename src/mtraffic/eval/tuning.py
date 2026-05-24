"""Hyperparameter sweep harness used by the experiment scripts.

Each sweep entry trains a single model on the training window and reports one-step-ahead
performance on the validation window. The test window is never touched, which keeps the
final reported numbers honest.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from mtraffic.eval.metrics import compute_all
from mtraffic.eval.walkforward import run as walk_run


@dataclass(slots=True)
class ExperimentResult:
    model: str
    area: int
    label: str
    params: dict[str, Any]
    train_seconds: float
    val_mae: float
    val_rmse: float
    val_mape_percent: float
    val_smape_percent: float
    notes: str = ""

    def as_row(self) -> dict[str, Any]:
        flat = {
            "model": self.model,
            "area": self.area,
            "label": self.label,
            "train_seconds": round(self.train_seconds, 3),
            "val_MAE": round(self.val_mae, 4),
            "val_RMSE": round(self.val_rmse, 4),
            "val_MAPE_percent": round(self.val_mape_percent, 4),
            "val_sMAPE_percent": round(self.val_smape_percent, 4),
            "notes": self.notes,
        }
        for k, v in self.params.items():
            flat[f"param.{k}"] = v
        return flat


def evaluate_on_val(
    model: Any,
    series: pd.Series,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    *,
    mape_epsilon: float = 1e-3,
) -> dict[str, float]:
    res = walk_run(model, series, val_start, val_end)
    return compute_all(res.y_true, res.y_pred, epsilon=mape_epsilon)


def run_one(
    *,
    model_name: str,
    area: int,
    label: str,
    params: dict[str, Any],
    train: pd.Series,
    val: pd.Series,
    series: pd.Series,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    model_factory: Callable[[], Any],
    notes: str = "",
) -> ExperimentResult:
    """Train a model from `model_factory()`, evaluate one-step-ahead on the validation window."""
    m = model_factory()
    t0 = time.perf_counter()
    if hasattr(m, "fit") and getattr(m, "name", "").startswith(("lstm", "cnn")):
        m.fit(train, val)
    else:
        m.fit(train)
    elapsed = time.perf_counter() - t0
    metrics = evaluate_on_val(m, series, val_start, val_end)
    return ExperimentResult(
        model=model_name,
        area=area,
        label=label,
        params=params,
        train_seconds=elapsed,
        val_mae=metrics["MAE"],
        val_rmse=metrics["RMSE"],
        val_mape_percent=metrics["MAPE_percent"],
        val_smape_percent=metrics["sMAPE_percent"],
        notes=notes,
    )
