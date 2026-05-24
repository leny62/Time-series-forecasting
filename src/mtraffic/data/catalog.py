"""Dataset discovery and manifest management for the interim Parquet store."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pyarrow.parquet as pq

from mtraffic.io.paths import list_daily_files, manifest_path, missing_path, partition_path


@dataclass(slots=True)
class PartitionRecord:
    day: str
    path: str
    rows: int
    bytes: int
    sha256: str


def expected_dates(start: date, end: date) -> list[date]:
    days: list[date] = []
    cur = start
    while cur <= end:
        days.append(cur)
        cur += timedelta(days=1)
    return days


def discover_raw(raw_dir: Path) -> dict[date, Path]:
    """Map calendar dates to the raw TSV file present for that date."""
    return {d: p for d, p in list_daily_files(raw_dir)}


def write_manifest(
    interim_dir: Path,
    records: list[PartitionRecord],
    expected_start: date,
    expected_end: date,
) -> Path:
    interim_dir.mkdir(parents=True, exist_ok=True)
    out = manifest_path(interim_dir)
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "expected_start": expected_start.isoformat(),
        "expected_end": expected_end.isoformat(),
        "partitions": [asdict(r) for r in records],
        "n_partitions": len(records),
        "total_rows": sum(r.rows for r in records),
        "total_bytes": sum(r.bytes for r in records),
    }
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(out)
    return out


def write_missing(interim_dir: Path, missing: list[date]) -> Path:
    out = missing_path(interim_dir)
    payload = {"missing_dates": [d.isoformat() for d in missing], "count": len(missing)}
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(out)
    return out


def scan_existing_partitions(interim_dir: Path, days: list[date]) -> list[PartitionRecord]:
    records: list[PartitionRecord] = []
    for d in days:
        p = partition_path(interim_dir, d)
        if not p.exists():
            continue
        rows = pq.ParquetFile(p).metadata.num_rows
        from mtraffic.io.writers import sha256_of_file
        records.append(
            PartitionRecord(
                day=d.isoformat(),
                path=str(p.relative_to(interim_dir)),
                rows=rows,
                bytes=p.stat().st_size,
                sha256=sha256_of_file(p),
            )
        )
    return records
