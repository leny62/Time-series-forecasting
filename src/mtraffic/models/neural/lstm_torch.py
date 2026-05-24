"""PyTorch LSTM forecaster for univariate one-step-ahead prediction."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from mtraffic.models.neural.common.features import (
    StandardScaler1D,
    make_features_array,
    make_windows,
)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _LSTMNet(nn.Module):
    def __init__(self, in_features: int, hidden: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(self.norm(last)).squeeze(-1)


@dataclass(slots=True)
class LstmConfig:
    seq_len: int = 288
    hidden: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    batch_size: int = 64
    max_epochs: int = 50
    patience: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    huber_delta: float = 1.0


class LstmForecaster:
    name = "lstm"

    def __init__(self, cfg: LstmConfig) -> None:
        self.cfg = cfg
        self.scaler: StandardScaler1D | None = None
        self.model: _LSTMNet | None = None
        self.device = _device()
        self.n_features: int = 6

    def fit(
        self,
        train: pd.Series,
        val: pd.Series,
        *,
        seed: int = 20251201,
        verbose: bool = False,
    ) -> dict[str, float]:
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.scaler = StandardScaler1D.fit(train.astype(float).to_numpy())
        train_arr = make_features_array(train, self.scaler)
        val_arr = make_features_array(val, self.scaler)
        # For validation we want X built from val features and y from val targets, but the first
        # seq_len timestamps need history. Concatenate train tail + val for that.
        if len(train_arr) >= self.cfg.seq_len:
            stitched = np.concatenate([train_arr[-self.cfg.seq_len :], val_arr], axis=0)
            val_y_targets = stitched[:, 0]
            Xv, yv = make_windows(stitched, val_y_targets, self.cfg.seq_len)
        else:
            Xv, yv = make_windows(val_arr, val_arr[:, 0], self.cfg.seq_len)

        train_y_targets = train_arr[:, 0]
        Xt, yt = make_windows(train_arr, train_y_targets, self.cfg.seq_len)
        self.n_features = train_arr.shape[1]

        device = self.device
        self.model = _LSTMNet(self.n_features, self.cfg.hidden, self.cfg.num_layers, self.cfg.dropout).to(device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg.max_epochs)
        loss_fn = nn.HuberLoss(delta=self.cfg.huber_delta)

        Xt_t = torch.from_numpy(Xt).to(device)
        yt_t = torch.from_numpy(yt).to(device)
        Xv_t = torch.from_numpy(Xv).to(device)
        yv_t = torch.from_numpy(yv).to(device)

        best_val = float("inf")
        best_state: dict | None = None
        patience = 0
        history: list[dict[str, float]] = []

        n = Xt_t.shape[0]
        for epoch in range(self.cfg.max_epochs):
            self.model.train()
            perm = torch.randperm(n, device=device)
            running = 0.0
            for i in range(0, n, self.cfg.batch_size):
                idx = perm[i : i + self.cfg.batch_size]
                xb = Xt_t[idx]
                yb = yt_t[idx]
                opt.zero_grad(set_to_none=True)
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                opt.step()
                running += float(loss.item()) * xb.shape[0]
            sched.step()
            train_loss = running / max(n, 1)

            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(Xv_t)
                val_loss = float(loss_fn(val_pred, yv_t).item())
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            if verbose:
                print(f"epoch {epoch:02d}  train={train_loss:.5f}  val={val_loss:.5f}")
            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= self.cfg.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return {"best_val_loss": best_val, "epochs": len(history)}

    def predict_one_step(self, history: pd.Series, target_ts: pd.Timestamp) -> float:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Call fit() or load() before predict_one_step().")
        features = make_features_array(history, self.scaler)
        if features.shape[0] < self.cfg.seq_len:
            return float(history.iloc[-1])
        window = features[-self.cfg.seq_len :][np.newaxis, ...]
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(window).to(self.device)
            z = float(self.model(x).cpu().numpy()[0])
        return float(self.scaler.inverse(np.array([z]))[0])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict() if self.model is not None else None,
                "scaler": self.scaler.to_dict() if self.scaler is not None else None,
                "cfg": asdict(self.cfg),
                "n_features": self.n_features,
            },
            str(path),
        )

    @classmethod
    def load(cls, path: Path) -> "LstmForecaster":
        blob = torch.load(str(path), map_location="cpu", weights_only=False)
        cfg = LstmConfig(**blob["cfg"])
        m = cls(cfg)
        m.n_features = int(blob.get("n_features", 6))
        m.scaler = StandardScaler1D.from_dict(blob["scaler"])
        m.model = _LSTMNet(m.n_features, cfg.hidden, cfg.num_layers, cfg.dropout)
        m.model.load_state_dict(blob["state_dict"])
        m.model.to(m.device).eval()
        return m
