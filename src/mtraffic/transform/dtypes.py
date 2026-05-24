"""Dtype optimization for the canonical schema."""

from __future__ import annotations

import pyarrow as pa

CANONICAL_FIELDS = ["square_id", "ts", "internet"]


def canonical_schema(tz: str = "Europe/Rome") -> pa.Schema:
    """Return the canonical Arrow schema used in the Parquet store."""
    return pa.schema(
        [
            pa.field("square_id", pa.uint16(), nullable=False),
            pa.field("ts", pa.timestamp("ms", tz=tz), nullable=False),
            pa.field("internet", pa.float32(), nullable=True),
        ]
    )


def to_canonical(table: pa.Table, tz: str = "Europe/Rome") -> pa.Table:
    """Cast an arrow table containing square_id, ts (ms epoch int64) and internet to the canonical schema."""
    if "square_id" not in table.column_names or "ts" not in table.column_names or "internet" not in table.column_names:
        missing = [c for c in CANONICAL_FIELDS if c not in table.column_names]
        raise ValueError(f"Cannot cast to canonical schema, missing columns: {missing}")
    square = table["square_id"].cast(pa.uint16())
    ts = table["ts"]
    if pa.types.is_integer(ts.type):
        ts = ts.cast(pa.timestamp("ms", tz="UTC")).cast(pa.timestamp("ms", tz=tz))
    elif pa.types.is_timestamp(ts.type) and ts.type.tz != tz:
        ts = ts.cast(pa.timestamp("ms", tz=tz))
    internet = table["internet"].cast(pa.float32())
    return pa.table({"square_id": square, "ts": ts, "internet": internet}, schema=canonical_schema(tz=tz))
