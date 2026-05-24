"""CLI for forecasting and evaluation."""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any

import pandas as pd
import typer

from mtraffic.config import Config
from mtraffic.data.loaders import area_totals, load_area_series
from mtraffic.eval import metrics as mtr
from mtraffic.eval import plots as mplots
from mtraffic.eval import walkforward
from mtraffic.models.baselines.naive import LastValue, SeasonalNaive
from mtraffic.utils import logging as mtlog

forecast_app = typer.Typer(add_completion=False, help="Run one-step-ahead forecasts.")
eval_app = typer.Typer(add_completion=False, help="Aggregate metrics, tables and plots.")
log = mtlog.get_logger(__name__)


def _resolve_area_labels(areas_arg: str, cfg: Config) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for token in areas_arg.split(","):
        token = token.strip()
        if not token:
            continue
        if token.lower() == "top":
            totals = area_totals(cfg.paths.interim_dir)
            out.append(("top", int(totals.idxmax())))
        else:
            out.append((token, int(token)))
    return out


def _load_model(report_dir: Path, area: int, model_kind: str) -> Any:
    if model_kind == "sarima":
        with (report_dir / "models" / str(area) / "sarima.pkl").open("rb") as f:
            return pickle.load(f)
    if model_kind == "naive_last":
        return LastValue()
    if model_kind == "naive_d144":
        return SeasonalNaive(period=144, name="naive_d144")
    if model_kind == "naive_w1008":
        return SeasonalNaive(period=1008, name="naive_w1008")
    if model_kind == "lstm":
        from mtraffic.models.neural.lstm_torch import LstmForecaster
        ckpt = report_dir / "models" / str(area) / "lstm.pt"
        return LstmForecaster.load(ckpt)
    if model_kind == "cnn":
        from mtraffic.models.neural.cnn_torch import CnnForecaster
        ckpt = report_dir / "models" / str(area) / "cnn.pt"
        return CnnForecaster.load(ckpt)
    raise ValueError(f"Unknown model kind {model_kind!r}")


@forecast_app.callback(invoke_without_command=True)
def forecast(
    model: str = typer.Option(..., "--model", help="sarima|lstm|cnn|naive_last|naive_d144|naive_w1008"),
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
    forecasts_dir = cfg.paths.reports_dir / "tables" / "forecasts"
    forecasts_dir.mkdir(parents=True, exist_ok=True)
    timings: list[dict] = []

    for label, area in _resolve_area_labels(areas, cfg):
        log.info("Forecast %s on area %s (square %d)", model, label, area)
        s = load_area_series(cfg.paths.interim_dir, area)
        train_val_history = s.loc[cfg.eval.train_start : cfg.eval.val_end]
        m = _load_model(cfg.paths.reports_dir, area, model)
        # Models that are stateless (naive) or already fitted (sarima/lstm/cnn after train CLI).
        if hasattr(m, "fit") and model.startswith("naive"):
            m.fit(train_val_history)
        # SARIMA: bulk-append validation observations to the state space before walk-forward.
        if model == "sarima" and getattr(m, "_history", None) is not None and m._history.index[-1] < cfg.eval.val_end:
            history_for_append = s.loc[: cfg.eval.val_end]
            # Single call: pass the full history up to val_end as the target_ts argument that the model uses
            # to detect new observations beyond what it has seen.
            m.predict_one_step(history_for_append, history_for_append.index[-1] + pd.Timedelta(minutes=10))
        t0 = time.perf_counter()
        result = walkforward.run(m, s, cfg.eval.test_start, cfg.eval.test_end)
        elapsed = time.perf_counter() - t0
        df = result.to_frame()
        df.to_parquet(forecasts_dir / f"{model}_{area}.parquet")
        per_step = elapsed / max(len(df), 1)
        timings.append({"model": model, "area": area, "predict_seconds_total": round(elapsed, 3), "predict_seconds_per_step_mean": round(per_step, 4), "test_steps": len(df)})
        log.info("Area %d: %d test steps in %.2fs (%.4fs/step)", area, len(df), elapsed, per_step)

    timing_csv = cfg.paths.reports_dir / "tables" / "task3_timing.csv"
    df_new = pd.DataFrame(timings)
    if timing_csv.exists():
        existing = pd.read_csv(timing_csv)
        for col in df_new.columns:
            if col not in existing.columns:
                existing[col] = pd.NA
        keep = existing[~existing.set_index(["model", "area"]).index.isin(df_new.set_index(["model", "area"]).index)]
        merged = pd.concat([keep, df_new], ignore_index=True)
    else:
        merged = df_new
    merged.to_csv(timing_csv, index=False)
    typer.echo(json.dumps({"forecast": model, "areas": [a for _, a in _resolve_area_labels(areas, cfg)], "forecasts_dir": str(forecasts_dir)}))


@eval_app.command("task3")
def evaluate_task3(
    areas: str = typer.Option("top,4159,4556", "--areas"),
    report_dir: Path = typer.Option(None, "--report-dir"),
    config: Path = typer.Option(None, "--config"),
    models: str = typer.Option("sarima,lstm,cnn,naive_last,naive_d144,naive_w1008", "--models"),
) -> None:
    cfg = Config.load(config) if config else Config.load()
    if report_dir is not None:
        cfg.paths.reports_dir = report_dir.resolve()
    figs = cfg.paths.reports_dir / "figures"
    tabs = cfg.paths.reports_dir / "tables"
    forecasts_dir = tabs / "forecasts"
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)
    area_labels = _resolve_area_labels(areas, cfg)
    model_list = [m.strip() for m in models.split(",") if m.strip()]

    rows: list[dict] = []
    for label, area in area_labels:
        predictions: dict[str, "pd.Series"] = {}
        timestamps: pd.DatetimeIndex | None = None
        y_true: pd.Series | None = None
        for model_name in model_list:
            f = forecasts_dir / f"{model_name}_{area}.parquet"
            if not f.exists():
                log.warning("Missing forecast file %s, skipping", f)
                continue
            df = pd.read_parquet(f)
            if y_true is None:
                timestamps = pd.DatetimeIndex(df["ts"])
                y_true = df["y_true"].to_numpy()
            yp = df["y_pred"].to_numpy()
            predictions[model_name] = yp
            m = mtr.compute_all(y_true, yp, epsilon=cfg.eval.mape_epsilon)
            rows.append({"area": area, "model": model_name, **{k: round(v, cfg.eval.metric_decimals) for k, v in m.items()}})

            # Per-model figure
            out_png = figs / f"forecast_{area}_{model_name}.png"
            mplots.plot_forecast(timestamps, y_true, yp, out_png, title=f"Forecast vs actual, area {area}, {model_name}")

        if predictions and timestamps is not None and y_true is not None:
            mplots.plot_combined(timestamps, y_true, predictions, figs / f"forecast_{area}_combined.png", title=f"All models, area {area}")

    if not rows:
        typer.echo("No forecast files found; nothing to evaluate.")
        return

    metrics_df = pd.DataFrame(rows).sort_values(["area", "MAE"]).reset_index(drop=True)
    metrics_csv = tabs / "task3_metrics_all.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    for _, area in area_labels:
        sub = metrics_df[metrics_df["area"] == area]
        if not sub.empty:
            sub.to_csv(tabs / f"task3_metrics_{area}.csv", index=False)

    # comparison.md
    lines = ["# Task 3 Comparison\n"]
    for _, area in area_labels:
        sub = metrics_df[metrics_df["area"] == area]
        if sub.empty:
            continue
        lines.append(f"\n## Area {area}\n")
        lines.append(sub.drop(columns=["area"]).to_markdown(index=False) + "\n")
    (cfg.paths.reports_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    typer.echo(json.dumps({"metrics_csv": str(metrics_csv), "comparison_md": str(cfg.paths.reports_dir / "comparison.md")}))
