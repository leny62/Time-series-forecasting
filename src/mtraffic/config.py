"""Typed configuration loaded from YAML, overridable from CLI flags."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


@dataclass(slots=True)
class Paths:
    raw_dir: Path
    interim_dir: Path
    reports_dir: Path
    geojson: Path


@dataclass(slots=True)
class IngestCfg:
    expected_first_date: str
    expected_last_date: str
    chunk_rows: int
    workers: int
    area_stripe_size: int
    quarantine: bool
    tz: str


@dataclass(slots=True)
class EdaCfg:
    areas: list[str | int]
    rolling_window_steps: int
    preview_days: int
    acf_max_lag: int
    pacf_max_lag: int


@dataclass(slots=True)
class EvalCfg:
    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime
    test_start: datetime
    test_end: datetime
    mape_epsilon: float
    metric_decimals: int


@dataclass(slots=True)
class SarimaCfg:
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    daily_terms: int
    weekly_terms: int
    use_log1p_if_helpful: bool
    optimizer: str


@dataclass(slots=True)
class LstmCfg:
    seq_len: int
    hidden: int
    num_layers: int
    dropout: float
    batch_size: int
    max_epochs: int
    patience: int
    lr: float
    weight_decay: float
    grad_clip: float
    huber_delta: float


@dataclass(slots=True)
class CnnCfg:
    seq_len: int
    filters: int
    kernel_size: int
    dilations: list[int]
    dropout: float
    batch_size: int
    max_epochs: int
    patience: int
    lr: float
    huber_delta: float


@dataclass(slots=True)
class ModelsCfg:
    sarima: SarimaCfg
    lstm: LstmCfg
    cnn: CnnCfg


@dataclass(slots=True)
class Config:
    seed: int
    paths: Paths
    ingest: IngestCfg
    eda: EdaCfg
    eval: EvalCfg
    models: ModelsCfg
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str | None = None) -> Config:
        cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        return cls._from_dict(data, cfg_path.parent.parent)

    @staticmethod
    def _to_dt(value: Any, tz: str = "Europe/Rome") -> datetime:
        if isinstance(value, datetime):
            dt = value
        else:
            dt = datetime.fromisoformat(str(value))
        if dt.tzinfo is None:
            # Attach project tz so comparisons against tz-aware series indices work.
            import pandas as _pd

            return _pd.Timestamp(dt).tz_localize(tz).to_pydatetime()
        return dt

    @classmethod
    def _from_dict(cls, data: dict[str, Any], repo_root: Path) -> Config:
        paths_raw = data["paths"]
        paths = Paths(
            raw_dir=(repo_root / paths_raw["raw_dir"]).resolve(),
            interim_dir=(repo_root / paths_raw["interim_dir"]).resolve(),
            reports_dir=(repo_root / paths_raw["reports_dir"]).resolve(),
            geojson=(repo_root / paths_raw["geojson"]).resolve(),
        )
        ingest = IngestCfg(**data["ingest"])
        eda_raw = data["eda"]
        eda = EdaCfg(
            areas=list(eda_raw["areas"]),
            rolling_window_steps=int(eda_raw["rolling_window_steps"]),
            preview_days=int(eda_raw["preview_days"]),
            acf_max_lag=int(eda_raw["acf_max_lag"]),
            pacf_max_lag=int(eda_raw["pacf_max_lag"]),
        )
        ev = data["eval"]
        eval_cfg = EvalCfg(
            train_start=cls._to_dt(ev["train_start"]),
            train_end=cls._to_dt(ev["train_end"]),
            val_start=cls._to_dt(ev["val_start"]),
            val_end=cls._to_dt(ev["val_end"]),
            test_start=cls._to_dt(ev["test_start"]),
            test_end=cls._to_dt(ev["test_end"]),
            mape_epsilon=float(ev["mape_epsilon"]),
            metric_decimals=int(ev["metric_decimals"]),
        )
        m = data["models"]
        sarima = SarimaCfg(
            order=tuple(m["sarima"]["order"]),
            seasonal_order=tuple(m["sarima"]["seasonal_order"]),
            daily_terms=int(m["sarima"]["daily_terms"]),
            weekly_terms=int(m["sarima"]["weekly_terms"]),
            use_log1p_if_helpful=bool(m["sarima"]["use_log1p_if_helpful"]),
            optimizer=str(m["sarima"]["optimizer"]),
        )
        lstm = LstmCfg(**m["lstm"])
        cnn = CnnCfg(**m["cnn"])
        return cls(
            seed=int(data["seed"]),
            paths=paths,
            ingest=ingest,
            eda=eda,
            eval=eval_cfg,
            models=ModelsCfg(sarima=sarima, lstm=lstm, cnn=cnn),
        )
