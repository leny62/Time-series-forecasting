"""Parquet writers for the canonical interim store."""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from mtraffic.io.paths import partition_path


def sha256_of_file(path: Path, *, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


def write_daily_parquet(
    table: pa.Table,
    interim_dir: Path,
    day: date,
    *,
    compression: str = "snappy",
) -> Path:
    """Write one daily Arrow table to data/interim/year_month=YYYY-MM/day=YYYY-MM-DD.parquet.

    Atomic: write to a sibling .tmp file and rename on success.
    """
    out = partition_path(interim_dir, day)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    pq.write_table(
        table,
        tmp,
        compression=compression,
        use_dictionary=False,
        write_statistics=True,
        row_group_size=200_000,
    )
    tmp.replace(out)
    return out
