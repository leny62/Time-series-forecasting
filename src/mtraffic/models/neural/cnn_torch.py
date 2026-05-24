"""Dilated 1D CNN forecaster (TCN style) for univariate one-step-ahead prediction.

The receptive field grows exponentially with the dilation schedule, so a stack of 8 dilated
causal convolutions covers windows much longer than the input sequence at modest cost. This
is a strong counterpoint to the LSTM, which carries information forward via a recurrent
state. Same loss, same scaler, same one-step-ahead protocol.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
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


class _CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int) -> None:
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class _TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = _CausalConv1d(channels, channels, kernel, dilation)
        self.conv2 = _CausalConv1d(channels, channels, kernel, dilation)
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=channels)
        self.norm2 = nn.GroupNorm(num_groups=1, num_channels=channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.norm1(self.conv1(x)))
        h = self.drop(h)
        h = torch.relu(self.norm2(self.conv2(h)))
        return self.drop(h) + x


class _TCNNet(nn.Module):
    def __init__(self, in_features: int, channels: int, kernel: int, dilations: list[int], dropout: float) -> None:
        super().__init__()
        self.proj = nn.Conv1d(in_features, channels, kernel_size=1)
        self.blocks = nn.ModuleList([_TCNBlock(channels, kernel, d, dropout) for d in dilations])
        self.head_norm = nn.LayerNorm(channels)
        self.head = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)  ->  (B, F, T)
        h = x.transpose(1, 2)
        h = self.proj(h)
        for blk in self.blocks:
            h = blk(h)
        # take the last time step
        last = h[:, :, -1]
        return self.head(self.head_norm(last)).squeeze(-1)


@dataclass(slots=True)
class CnnConfig:
    seq_len: int = 1008
    filters: int = 32
    kernel_size: int = 3
    dilations: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 128])
    dropout: float = 0.05
    batch_size: int = 64
    max_epochs: int = 40
    patience: int = 6
    lr: float = 1e-3
    huber_delta: float = 1.0


class CnnForecaster:
    name = "cnn"

    def __init__(self, cfg: CnnConfig) -> None:
        self.cfg = cfg
        self.scaler: StandardScaler1D | None = None
        self.model: _TCNNet | None = None
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
        if len(train_arr) >= self.cfg.seq_len:
            stitched = np.concatenate([train_arr[-self.cfg.seq_len :], val_arr], axis=0)
            Xv, yv = make_windows(stitched, stitched[:, 0], self.cfg.seq_len)
        else:
            Xv, yv = make_windows(val_arr, val_arr[:, 0], self.cfg.seq_len)
        Xt, yt = make_windows(train_arr, train_arr[:, 0], self.cfg.seq_len)
        self.n_features = train_arr.shape[1]

        device = self.device
        self.model = _TCNNet(
            self.n_features, self.cfg.filters, self.cfg.kernel_size, list(self.cfg.dilations), self.cfg.dropout
        ).to(device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg.max_epochs)
        loss_fn = nn.HuberLoss(delta=self.cfg.huber_delta)

        Xt_t = torch.from_numpy(Xt).to(device)
        yt_t = torch.from_numpy(yt).to(device)
        Xv_t = torch.from_numpy(Xv).to(device)
        yv_t = torch.from_numpy(yv).to(device)

        best_val = float("inf")
        best_state: dict | None = None
        patience = 0

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
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                running += float(loss.item()) * xb.shape[0]
            sched.step()
            train_loss = running / max(n, 1)

            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(Xv_t)
                val_loss = float(loss_fn(val_pred, yv_t).item())
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
        return {"best_val_loss": best_val}

    def predict_one_step(self, history: pd.Series, target_ts: pd.Timestamp) -> float:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Call fit() or load() before predict_one_step().")
        features = make_features_array(history, self.scaler)
        if features.shape[0] < self.cfg.seq_len:
            # not enough history; fall back to last observation
            return float(history.iloc[-1])
        window = features[-self.cfg.seq_len :][np.newaxis, ...]
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(window).to(self.device)
            z = float(self.model(x).cpu().numpy()[0])
        return float(self.scaler.inverse(np.array([z]))[0])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        cfg_dict = asdict(self.cfg)
        cfg_dict["dilations"] = list(cfg_dict.get("dilations", []))
        torch.save(
            {
                "state_dict": self.model.state_dict() if self.model is not None else None,
                "scaler": self.scaler.to_dict() if self.scaler is not None else None,
                "cfg": cfg_dict,
                "n_features": self.n_features,
            },
            str(path),
        )

    @classmethod
    def load(cls, path: Path) -> "CnnForecaster":
        blob = torch.load(str(path), map_location="cpu", weights_only=False)
        raw = dict(blob["cfg"])
        raw["dilations"] = list(raw.get("dilations", [1, 2, 4, 8, 16, 32, 64, 128]))
        cfg = CnnConfig(**raw)
        m = cls(cfg)
        m.n_features = int(blob.get("n_features", 6))
        m.scaler = StandardScaler1D.from_dict(blob["scaler"])
        m.model = _TCNNet(m.n_features, cfg.filters, cfg.kernel_size, list(cfg.dilations), cfg.dropout)
        m.model.load_state_dict(blob["state_dict"])
        m.model.to(m.device).eval()
        return m
