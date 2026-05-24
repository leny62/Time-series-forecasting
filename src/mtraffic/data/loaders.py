"""Lazy loaders that read narrow slices from the Parquet store."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads

from mtraffic.io.paths import partition_path


def _scan_dataset(interim_dir: Path) -> pads.Dataset:
    return pads.dataset(interim_dir, format="parquet", partitioning="hive")


def _ts_scalar(value: datetime, tz: str = "Europe/Rome") -> pa.Scalar:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(tz)
    else:
        ts = ts.tz_convert(tz)
    return pa.scalar(ts.to_pydatetime(), type=pa.timestamp("ms", tz=tz))


def load_area_series(
    interim_dir: Path,
    area: int,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.Series:
    """Return the 10 minute internet series for one square_id as a pandas Series indexed by ts."""
    dataset = _scan_dataset(interim_dir)
    filt = pc.field("square_id") == pa.scalar(area, type=pa.uint16())
    if start is not None:
        filt = filt & (pc.field("ts") >= _ts_scalar(start))
    if end is not None:
        filt = filt & (pc.field("ts") <= _ts_scalar(end))
    table = dataset.to_table(columns=["square_id", "ts", "internet"], filter=filt)
    df = table.to_pandas()
    df = df.sort_values("ts").reset_index(drop=True)
    series = pd.Series(df["internet"].to_numpy(), index=pd.DatetimeIndex(df["ts"]), name=f"area_{area}")
    series.index.name = "ts"
    return series


def load_areas_frame(
    interim_dir: Path,
    areas: Iterable[int],
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    """Return a DataFrame with one column per area_id (column names: area_<id>)."""
    areas = list(areas)
    dataset = _scan_dataset(interim_dir)
    filt = pc.field("square_id").isin(pa.array(areas, type=pa.uint16()))
    if start is not None:
        filt = filt & (pc.field("ts") >= _ts_scalar(start))
    if end is not None:
        filt = filt & (pc.field("ts") <= _ts_scalar(end))
    table = dataset.to_table(columns=["square_id", "ts", "internet"], filter=filt)
    df = table.to_pandas()
    out = df.pivot_table(index="ts", columns="square_id", values="internet").sort_index()
    out.columns = [f"area_{int(c)}" for c in out.columns]
    return out


def area_totals(interim_dir: Path) -> pd.Series:
    """Sum internet activity per square_id across the entire store."""
    dataset = _scan_dataset(interim_dir)
    table = dataset.to_table(columns=["square_id", "internet"])
    grouped = table.group_by("square_id").aggregate([("internet", "sum")])
    df = grouped.to_pandas()
    df = df.rename(columns={"internet_sum": "total"})
    s = pd.Series(df["total"].to_numpy(), index=df["square_id"].astype(int).to_numpy(), name="total")
    s.index.name = "square_id"
    return s.sort_index()


def list_partitions(interim_dir: Path) -> list[Path]:
    return sorted(interim_dir.rglob("day=*.parquet"))


def has_partition(interim_dir: Path, day: date) -> bool:
    return partition_path(interim_dir, day).exists()
