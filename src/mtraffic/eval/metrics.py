"""Forecast metrics. All metrics are computed on the original scale."""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def mape_percent(y_true: np.ndarray, y_pred: np.ndarray, *, epsilon: float = 1e-3) -> float:
    denom = np.maximum(np.abs(y_true), epsilon)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def smape_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    safe = np.where(denom > 0, denom, np.nan)
    return float(np.nanmean(np.abs(y_true - y_pred) / safe) * 100.0)


def compute_all(y_true: np.ndarray, y_pred: np.ndarray, *, epsilon: float = 1e-3) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE_percent": mape_percent(y_true, y_pred, epsilon=epsilon),
        "sMAPE_percent": smape_percent(y_true, y_pred),
    }
