"""CLI for training models. Currently exposes a placeholder; per-model commands appear below."""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import pandas as pd
import typer

from mtraffic.config import Config
from mtraffic.data.loaders import area_totals, load_area_series
from mtraffic.models.classical.sarima import SarimaModel
from mtraffic.utils import logging as mtlog
from mtraffic.utils import seed as mtseed

train_app = typer.Typer(add_completion=False, help="Train forecasting models.")
log = mtlog.get_logger(__name__)


def _resolve_area_labels(areas_arg: str, cfg: Config) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for token in areas_arg.split(","):
        token = token.strip()
        if not token:
            continue
        if token.lower() == "top":
            totals = area_totals(cfg.paths.interim_dir)
            area = int(totals.idxmax())
            out.append(("top", area))
        else:
            out.append((token, int(token)))
    return out


def _ensure_dirs(cfg: Config) -> tuple[Path, Path, Path]:
    models_dir = cfg.paths.reports_dir / "models"
    figs = cfg.paths.reports_dir / "figures"
    tabs = cfg.paths.reports_dir / "tables"
    for p in (models_dir, figs, tabs):
        p.mkdir(parents=True, exist_ok=True)
    return models_dir, figs, tabs


@train_app.command("sarima")
def train_sarima(
    areas: str = typer.Option("top,4159,4556", "--areas"),
    interim_dir: Path = typer.Option(None, "--interim-dir"),
    report_dir: Path = typer.Option(None, "--report-dir"),
    config: Path = typer.Option(None, "--config"),
) -> None:
    cfg = Config.load(config) if config else Config.load()
    if interim_dir is not None:
        cfg.paths.interim_dir = interim_dir.resolve()
    if report_dir is not None:
        cfg.paths.reports_dir = report_dir.resolve()
    models_dir, _, tabs = _ensure_dirs(cfg)
    timings: list[dict] = []

    for label, area in _resolve_area_labels(areas, cfg):
        log.info("Fitting SARIMA on area %s (square %d)", label, area)
        s = load_area_series(cfg.paths.interim_dir, area)
        train = s.loc[cfg.eval.train_start : cfg.eval.train_end]
        m = SarimaModel(
            order=cfg.models.sarima.order,
            daily_terms=cfg.models.sarima.daily_terms,
            weekly_terms=cfg.models.sarima.weekly_terms,
            use_log1p=cfg.models.sarima.use_log1p_if_helpful,
        )
        t0 = time.perf_counter()
        m.fit(train)
        elapsed = time.perf_counter() - t0
        out_dir = models_dir / str(area)
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "sarima.pkl").open("wb") as f:
            pickle.dump(m, f)
        log.info("Area %d: fit in %.2fs, %d training steps", area, elapsed, len(train))
        timings.append({"model": "sarima", "area": area, "train_seconds": round(elapsed, 3), "train_steps": len(train)})

    _persist_timings(timings, "sarima", tabs)
    typer.echo(json.dumps({"trained": len(timings), "timing_csv": str(tabs / "task3_timing.csv")}))


def _persist_timings(rows: list[dict], model_name: str, tabs: Path) -> None:
    timing_csv = tabs / "task3_timing.csv"
    df_new = pd.DataFrame(rows)
    if timing_csv.exists():
        existing = pd.read_csv(timing_csv)
        mask = ~((existing["model"].astype(str) == model_name) & (existing["area"].isin(df_new["area"])))
        merged = pd.concat([existing[mask], df_new], ignore_index=True)
    else:
        merged = df_new
    merged.to_csv(timing_csv, index=False)


@train_app.command("lstm")
def train_lstm(
    areas: str = typer.Option("top,4159,4556", "--areas"),
    interim_dir: Path = typer.Option(None, "--interim-dir"),
    report_dir: Path = typer.Option(None, "--report-dir"),
    config: Path = typer.Option(None, "--config"),
) -> None:
    from mtraffic.models.neural.lstm_torch import LstmConfig, LstmForecaster

    cfg = Config.load(config) if config else Config.load()
    if interim_dir is not None:
        cfg.paths.interim_dir = interim_dir.resolve()
    if report_dir is not None:
        cfg.paths.reports_dir = report_dir.resolve()
    models_dir, _, tabs = _ensure_dirs(cfg)
    mtseed.set_all(cfg.seed)
    timings: list[dict] = []

    for label, area in _resolve_area_labels(areas, cfg):
        log.info("Training LSTM on area %s (square %d)", label, area)
        s = load_area_series(cfg.paths.interim_dir, area).astype(float).interpolate(limit_direction="both")
        train = s.loc[cfg.eval.train_start : cfg.eval.train_end]
        val = s.loc[cfg.eval.val_start : cfg.eval.val_end]
        lc = LstmConfig(
            seq_len=cfg.models.lstm.seq_len,
            hidden=cfg.models.lstm.hidden,
            num_layers=cfg.models.lstm.num_layers,
            dropout=cfg.models.lstm.dropout,
            batch_size=cfg.models.lstm.batch_size,
            max_epochs=cfg.models.lstm.max_epochs,
            patience=cfg.models.lstm.patience,
            lr=cfg.models.lstm.lr,
            weight_decay=cfg.models.lstm.weight_decay,
            grad_clip=cfg.models.lstm.grad_clip,
            huber_delta=cfg.models.lstm.huber_delta,
        )
        m = LstmForecaster(lc)
        t0 = time.perf_counter()
        info = m.fit(train, val, seed=cfg.seed)
        elapsed = time.perf_counter() - t0
        out = models_dir / str(area) / "lstm.pt"
        m.save(out)
        log.info("Area %d: fit in %.2fs (%d epochs, best val loss %.5f)", area, elapsed, int(info.get("epochs", 0)), float(info["best_val_loss"]))
        timings.append({"model": "lstm", "area": area, "train_seconds": round(elapsed, 3), "epochs": int(info.get("epochs", 0)), "best_val_loss": round(float(info["best_val_loss"]), 6)})

    _persist_timings(timings, "lstm", tabs)
    typer.echo(json.dumps({"trained": len(timings), "timing_csv": str(tabs / "task3_timing.csv")}))


@train_app.command("cnn")
def train_cnn(
    areas: str = typer.Option("top,4159,4556", "--areas"),
    interim_dir: Path = typer.Option(None, "--interim-dir"),
    report_dir: Path = typer.Option(None, "--report-dir"),
    config: Path = typer.Option(None, "--config"),
) -> None:
    from mtraffic.models.neural.cnn_torch import CnnConfig, CnnForecaster

    cfg = Config.load(config) if config else Config.load()
    if interim_dir is not None:
        cfg.paths.interim_dir = interim_dir.resolve()
    if report_dir is not None:
        cfg.paths.reports_dir = report_dir.resolve()
    models_dir, _, tabs = _ensure_dirs(cfg)
    mtseed.set_all(cfg.seed)
    timings: list[dict] = []

    for label, area in _resolve_area_labels(areas, cfg):
        log.info("Training CNN on area %s (square %d)", label, area)
        s = load_area_series(cfg.paths.interim_dir, area).astype(float).interpolate(limit_direction="both")
        train = s.loc[cfg.eval.train_start : cfg.eval.train_end]
        val = s.loc[cfg.eval.val_start : cfg.eval.val_end]
        cc = CnnConfig(
            seq_len=cfg.models.cnn.seq_len,
            filters=cfg.models.cnn.filters,
            kernel_size=cfg.models.cnn.kernel_size,
            dilations=list(cfg.models.cnn.dilations),
            dropout=cfg.models.cnn.dropout,
            batch_size=cfg.models.cnn.batch_size,
            max_epochs=cfg.models.cnn.max_epochs,
            patience=cfg.models.cnn.patience,
            lr=cfg.models.cnn.lr,
            huber_delta=cfg.models.cnn.huber_delta,
        )
        m = CnnForecaster(cc)
        t0 = time.perf_counter()
        info = m.fit(train, val, seed=cfg.seed)
        elapsed = time.perf_counter() - t0
        out = models_dir / str(area) / "cnn.pt"
        m.save(out)
        log.info("Area %d: fit in %.2fs (best val loss %.5f)", area, elapsed, float(info["best_val_loss"]))
        timings.append({"model": "cnn", "area": area, "train_seconds": round(elapsed, 3), "best_val_loss": round(float(info["best_val_loss"]), 6)})

    _persist_timings(timings, "cnn", tabs)
    typer.echo(json.dumps({"trained": len(timings), "timing_csv": str(tabs / "task3_timing.csv")}))


