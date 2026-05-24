"""Streaming TSV reader for daily Milan files.

Source rows have eight tab separated fields (no header):

    square_id  time_interval  country_code  sms_in  sms_out  call_in  call_out  internet

Only square_id, time_interval and internet are kept. Per (square_id, time_interval) we sum
the internet activity across country codes so each (area, 10 minute bin) has one row.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pcsv

from mtraffic.transform.dtypes import canonical_schema

SOURCE_COLUMNS = (
    "square_id",
    "time_interval",
    "country_code",
    "sms_in",
    "sms_out",
    "call_in",
    "call_out",
    "internet",
)


def _read_options(block_size: int) -> pcsv.ReadOptions:
    return pcsv.ReadOptions(
        block_size=block_size,
        autogenerate_column_names=False,
        column_names=list(SOURCE_COLUMNS),
        skip_rows=0,
    )


def _parse_options() -> pcsv.ParseOptions:
    return pcsv.ParseOptions(delimiter="\t", invalid_row_handler=lambda row: "skip")


def _convert_options() -> pcsv.ConvertOptions:
    return pcsv.ConvertOptions(
        include_columns=["square_id", "time_interval", "internet"],
        column_types={
            "square_id": pa.uint32(),
            "time_interval": pa.int64(),
            "internet": pa.float64(),
        },
        null_values=[""],
        strings_can_be_null=True,
    )


def iter_batches(path: Path, block_size: int = 4 * 1024 * 1024) -> Iterator[pa.RecordBatch]:
    """Yield Arrow record batches from one daily TSV file (selected columns only)."""
    with pcsv.open_csv(
        str(path),
        read_options=_read_options(block_size),
        parse_options=_parse_options(),
        convert_options=_convert_options(),
    ) as reader:
        while True:
            try:
                batch = reader.read_next_batch()
            except StopIteration:
                break
            if batch is None:
                break
            yield batch


def _aggregate_country(table: pa.Table) -> pa.Table:
    """Group by (square_id, time_interval) and sum internet across country codes."""
    grouped = table.group_by(["square_id", "time_interval"]).aggregate([("internet", "sum")])
    grouped = grouped.rename_columns(["square_id", "time_interval", "internet"])
    grouped = grouped.sort_by([("square_id", "ascending"), ("time_interval", "ascending")])
    return grouped


def read_one_file(path: Path, *, tz: str = "Europe/Rome") -> pa.Table:
    """Read one daily file, aggregate by (square_id, ts), return the canonical Arrow table.

    The optimized streaming path: pyarrow CSV reader with column pruning, then a single
    group-by-sum across all batches.
    """
    batches: list[pa.RecordBatch] = list(iter_batches(path))
    if not batches:
        return pa.table({"square_id": [], "ts": [], "internet": []}, schema=canonical_schema(tz=tz))
    raw = pa.Table.from_batches(batches)
    raw = raw.filter(pc.is_valid(raw["internet"]))
    raw = _aggregate_country(raw)
    raw = raw.rename_columns(["square_id", "ts", "internet"])
    # Cast columns to the canonical schema.
    square = raw["square_id"].cast(pa.uint16())
    ts = raw["ts"].cast(pa.timestamp("ms", tz="UTC")).cast(pa.timestamp("ms", tz=tz))
    internet = raw["internet"].cast(pa.float32())
    return pa.table({"square_id": square, "ts": ts, "internet": internet}, schema=canonical_schema(tz=tz))
