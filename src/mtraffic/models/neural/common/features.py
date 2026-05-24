"""Feature engineering shared by LSTM and CNN: time of day, day of week, normalization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def time_features(index: pd.DatetimeIndex) -> np.ndarray:
    """Return (T, 5) array: sin/cos of time-of-day, sin/cos of day-of-week, is_weekend."""
    tod = (index.hour * 6 + index.minute // 10).to_numpy(dtype=np.float64)  # 0..143
    dow = index.dayofweek.to_numpy(dtype=np.float64)  # 0..6
    is_weekend = (dow >= 5).astype(np.float64)
    out = np.stack(
        [
            np.sin(2.0 * np.pi * tod / 144.0),
            np.cos(2.0 * np.pi * tod / 144.0),
            np.sin(2.0 * np.pi * dow / 7.0),
            np.cos(2.0 * np.pi * dow / 7.0),
            is_weekend,
        ],
        axis=1,
    )
    return out


@dataclass(slots=True)
class StandardScaler1D:
    """Fit on training data only, then z-score everything else."""

    mean: float = 0.0
    std: float = 1.0

    @classmethod
    def fit(cls, x: np.ndarray) -> "StandardScaler1D":
        mu = float(np.nanmean(x))
        sd = float(np.nanstd(x, ddof=0))
        if sd < 1e-9:
            sd = 1.0
        return cls(mean=mu, std=sd)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

    def to_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "StandardScaler1D":
        return cls(mean=float(d["mean"]), std=float(d["std"]))


def make_features_array(series: pd.Series, scaler: StandardScaler1D) -> np.ndarray:
    """Return (T, F) feature array: standardized value plus 5 time features."""
    values = series.astype(float).to_numpy()
    z = scaler.transform(values).reshape(-1, 1)
    tf = time_features(pd.DatetimeIndex(series.index))
    return np.hstack([z, tf]).astype(np.float32)


def make_windows(
    features: np.ndarray,
    targets_standardized: np.ndarray,
    seq_len: int,
    *,
    target_start_idx: int = 0,
    target_end_idx: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding windows. X has shape (N, seq_len, F); y has shape (N,).

    The first valid target index in features is `seq_len` (since we predict t given t-seq_len..t-1).
    target_start_idx and target_end_idx restrict which targets appear in the dataset.
    """
    T, F = features.shape
    valid_start = max(seq_len, target_start_idx)
    valid_end = T if target_end_idx is None else target_end_idx
    n = max(0, valid_end - valid_start)
    if n <= 0:
        return np.zeros((0, seq_len, F), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    X = np.empty((n, seq_len, F), dtype=np.float32)
    y = np.empty((n,), dtype=np.float32)
    for j, t in enumerate(range(valid_start, valid_end)):
        X[j] = features[t - seq_len : t]
        y[j] = targets_standardized[t]
    return X, y
